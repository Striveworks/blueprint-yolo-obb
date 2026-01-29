import base64
import gzip
import itertools
import json
import os
import shutil
import tempfile
import unittest
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from blueprint_toolkit import (
    Checkpoint,
    CheckpointEngineSelector,
    CheckpointInferenceServerSettingsDict,
    CheckpointNotFoundError,
    DatasetSnapshot,
    MemoryMetricSaver,
    MemoryProgressSaver,
    ModelConfigDict,
    RunContext,
    local_run_context,
)
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from blueprint_yolo_obb.src.config import Config, Datasets, YoloDatasets
from blueprint_yolo_obb.src.train import (
    CHARIOT_VALOR_EVALUATIONS_FILE_NAME,
    MODEL_PT_NAME,
    Callbacks,
    fetch_datums,
    load_checkpoint_or_init_model,
    load_model,
    load_start_checkpoint,
    load_start_model,
    save_model,
    train,
)

_BLACK_10x10_PNG = base64.decodebytes(
    b"""
iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAGElEQ
VQY02NkYGD4z0AEYGIgEowqpI5CAKYXARPJvDplAAAAAElFTkSuQmCC
"""
)


def mock_get_datum_at_index_object_detection(index: int):
    return (
        _BLACK_10x10_PNG,
        {
            "id": str(index),
            "annotations": [
                {
                    "bbox": {"xmin": 1, "xmax": 10, "ymin": 2, "ymax": 9},
                    "class_label": "dog",
                }
            ],
        },
    )


def mock_get_datum_at_index_oriented_object_detection(index: int):
    return (
        _BLACK_10x10_PNG,
        {
            "id": str(index),
            "annotations": [
                {
                    # Equivalent to the object detection annotation, just in cxcywhr format
                    "oriented_bbox": {
                        "cx": 0.55,
                        "cy": 0.55,
                        "w": 0.9,
                        "h": 0.7,
                        "r": 0,
                    },
                    "class_label": ["cat", "dog", "chicken"][index % 3],
                }
            ],
        },
    )


def mock_get_datum_at_index_image_segmentation(index: int):
    return (
        _BLACK_10x10_PNG,
        {
            "id": str(index),
            "annotations": [
                {
                    # Equivalent to the object detection annotation, just in segmentation format
                    "contour": [
                        [
                            {"x": 10, "y": 9},
                            {"x": 1, "y": 9},
                            {"x": 1, "y": 2},
                            {"x": 10, "y": 2},
                        ],
                    ],
                    "class_label": ["cat", "dog", "chicken"][index % 3],
                }
            ],
        },
    )


def mock_get_datum_at_index_image_segmentation_multiple_contour(index: int):
    return (
        _BLACK_10x10_PNG,
        {
            "id": str(index),
            "annotations": [
                {
                    "contour": [
                        [{"x": 10, "y": 9}, {"x": 1, "y": 9}],
                        [{"x": 1, "y": 2}, {"x": 10, "y": 2}],
                    ],
                    "class_label": ["cat", "dog", "chicken"][index % 3],
                    "id": "annotation-id",
                }
            ],
        },
    )


def mock_prepare_snapshot_split(
    fetcher: MagicMock, snapshot_id: str, split: str | None
):
    if split == "train":
        fetcher.num_datums = 35
    elif split == "val":
        fetcher.num_datums = 18
    else:
        raise ValueError(f"unexpected split {split}")


def get_mock_fetcher(
    get_datum_at_index: Callable[[int], tuple[bytes, dict[str, Any]]],
):
    fetcher = MagicMock()
    fetcher.get_datum_at_index.side_effect = get_datum_at_index
    fetcher.get_class_labels.side_effect = [{"cat", "dog"}, {"chicken"}]
    fetcher.prepare_snapshot_split.side_effect = partial(
        mock_prepare_snapshot_split, fetcher
    )
    return fetcher


def mock_get_fetcher(
    task_type: Literal[
        "Object Detection", "Oriented Object Detection", "Image Segmentation"
    ],
):
    if task_type == "Object Detection":
        get_datum_at_index = mock_get_datum_at_index_object_detection
    elif task_type == "Oriented Object Detection":
        get_datum_at_index = mock_get_datum_at_index_oriented_object_detection
    elif task_type == "Image Segmentation":
        get_datum_at_index = mock_get_datum_at_index_image_segmentation
    else:
        raise ValueError(f"unsupported task type {task_type}")

    return get_mock_fetcher(get_datum_at_index)


def test_fetch_datums_multiple_contours():
    mock_run_context = MagicMock()
    mock_run_context.dataset_fetcher = get_mock_fetcher(
        mock_get_datum_at_index_image_segmentation_multiple_contour
    )

    cfg = Config(
        datasets=Datasets(
            train_data=[DatasetSnapshot(snapshot_id="snapshot id", split="train")],
            val_data=[DatasetSnapshot(snapshot_id="snapshot id", split="val")],
        )
    )
    with pytest.raises(ValueError) as ve:
        fetch_datums(mock_run_context, cfg, None)
    assert (
        str(ve.value)
        == "contour for annotation annotation-id invalid: must contain one polygon"
    )


@pytest.mark.parametrize(
    "task_type",
    ["Object Detection", "Oriented Object Detection", "Image Segmentation"],
)
def test_fetch_datums(
    task_type: Literal[
        "Object Detection", "Oriented Object Detection", "Image Segmentation"
    ],
):
    def assert_images_populated(tmpdir: Path, train_or_val: str, count: int):
        num_imgs = 0
        for img in (tmpdir / "images" / train_or_val).iterdir():
            assert img.read_bytes() == _BLACK_10x10_PNG
            num_imgs += 1
        assert num_imgs == count

    def assert_labels_populated(tmpdir: Path, train_or_val: str, count: int):
        num_labels = 0
        for label in (tmpdir / "labels" / train_or_val).iterdir():
            label = list(map(float, label.read_text().strip("\n").split(" ")))
            assert label[0] in {0, 1, 2}  # cat, dog, or chicken
            assert {
                (label[i + 1], label[i + 2]) for i in range(0, len(label[1:]), 2)
            } == {(1.0, 0.9), (1.0, 0.2), (0.1, 0.2), (0.1, 0.9)}
            num_labels += 1
        assert num_labels == count

    mock_run_context = MagicMock()
    mock_run_context.dataset_fetcher = mock_get_fetcher(task_type)
    cfg = Config(
        datasets=Datasets(
            train_data=[DatasetSnapshot(snapshot_id="snapshot id", split="train")],
            val_data=[DatasetSnapshot(snapshot_id="snapshot id", split="val")],
        )
    )
    tmpdir, class_to_idx = fetch_datums(mock_run_context, cfg, None)
    try:
        assert class_to_idx == {"cat": 0, "chicken": 1, "dog": 2}
        assert_images_populated(tmpdir, "train", 35)
        assert_images_populated(tmpdir, "val", 18)
        assert_labels_populated(tmpdir, "train", 35)
        assert_labels_populated(tmpdir, "val", 18)
    finally:
        shutil.rmtree(tmpdir)


def test_callbacks_on_train_batch_start():
    mock_run_context = MagicMock()
    callbacks = Callbacks(mock_run_context, 10)
    mock_trainer = MagicMock()
    mock_trainer.epochs = 100
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 35
    mock_trainer.train_loader.dataset = mock_dataset
    mock_trainer.batch_size = 16
    # 300 steps total: takes 3 steps to complete an epoch (35/16), times 100 epochs
    expected_progress = {
        "operation": "Training",
        "final_value": 300,
        "units": "steps",
        "value": 1,
    }
    callbacks.on_train_batch_start(mock_trainer)
    mock_run_context.save_progress.assert_called_with([expected_progress])
    expected_progress.update({"value": 2})
    callbacks.on_train_batch_start(mock_trainer)
    mock_run_context.save_progress.assert_called_with([expected_progress])
    expected_progress.update({"value": 3})
    callbacks.on_train_batch_start(mock_trainer)
    mock_run_context.save_progress.assert_called_with([expected_progress])
    assert callbacks.global_step == 3


@patch("blueprint_yolo_obb.src.train.YOLO")
@pytest.mark.parametrize(
    "cfg,expected_num_metrics,expected_num_legacy_metrics",
    [
        (
            Config(
                datasets=YoloDatasets(data_root=Path("/tmp")),
                class_to_idx={"cat": 0, "dog": 1},
            ),
            1,
            21,
        ),
        (
            Config(
                datasets=Datasets(
                    train_data=[DatasetSnapshot(snapshot_id="sid", split="train")],
                    val_data=[
                        DatasetSnapshot(snapshot_id="sid", split="val"),
                        DatasetSnapshot(snapshot_id="sid", split="test"),
                    ],
                )
            ),
            2,
            33,
        ),
    ],
)
def test_callback_on_model_save(
    mock_yolo_model,
    cfg: Config,
    expected_num_metrics: int,
    expected_num_legacy_metrics: int,
):
    def model_results(value):
        mock_result = MagicMock()
        mock_result.orig_img = _BLACK_10x10_PNG
        mock_result.path = value
        mock_result.names = {0: "cat", 1: "dog"}
        mock_obb = MagicMock()
        mock_obb.xywhr = [
            torch.tensor([4.419, 1.666, 3.767, 2.502, 0.0000], device="cpu")
        ]
        mock_obb.orig_shape = torch.tensor([10, 10], device="cpu")
        mock_obb.data = torch.tensor(
            [[4.419, 1.666, 3.767, 2.502, 0.0000, 0.9]], device="cpu"
        )
        mock_obb.cls = torch.tensor([0], device="cpu")
        mock_result.obb = [mock_obb]
        return [mock_result]

    mock_model = MagicMock(side_effect=model_results)
    mock_model.device = torch.device("cpu")
    mock_yolo_model.return_value = mock_model

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pt") as last_pt_file:
        torch.save(ModelClass(), last_pt_file.name)
        mock_trainer = MagicMock()
        mock_trainer.epoch = 9
        mock_trainer.batch_size = 8
        mock_trainer.last = Path(last_pt_file.name)
        mock_validator = MagicMock()
        mock_validator.dataloader = [
            {
                "im_file": [
                    "sid.train.img1.jpg",
                    "sid.val.img2.jpg",
                    "sid.test.img3.jpg",
                ],
                "batch_idx": torch.tensor([0, 1, 2]),
                "cls": torch.tensor([[0], [1], [0]]),
                "bboxes": torch.tensor(
                    [
                        [0.4419, 0.1666, 0.3767, 0.2502, 0.0000],
                        [0.7088, 0.1429, 0.1990, 0.1144, 1.5708],
                        [0.4419, 0.1666, 0.3767, 0.2502, 0.0000],
                    ]
                ),
            }
        ]
        mock_trainer.validator = mock_validator
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            mock_checkpoint = MagicMock()
            mock_checkpoint.dir = Path(checkpoint_dir)
            mock_run_context = MagicMock()
            mock_run_context.save_checkpoint.return_value.__enter__.return_value = (
                mock_checkpoint
            )
            mock_run_context.load_config.return_value = cfg

            idx_to_class = {0: "cat", 1: "dog"}
            global_step = 10
            callbacks = Callbacks(mock_run_context, 5, global_step)
            callbacks.on_model_save(mock_trainer, idx_to_class)

            mock_run_context.save_metrics.assert_called_once()
            metrics = mock_run_context.save_metrics.call_args[0][0]
            assert len(metrics) == expected_num_legacy_metrics
            tags = []
            for m in metrics:
                assert m["global_step"] == global_step
                tags.append(m["tag"].split("/")[2])
            assert set(tags) == set(
                [
                    "cat_AP_IOU0.1",
                    "dog_AP_IOU0.1",
                    "cat_AP_IOU0.5",
                    "dog_AP_IOU0.5",
                    "cat_AP_IOU0.75",
                    "dog_AP_IOU0.75",
                    "mAP_IOU0.1",
                    "mAP_IOU0.5",
                    "mAP_IOU0.75",
                    "cat_Precision_IOU0.1_p0.5",
                    "cat_Precision_IOU0.5_p0.5",
                    "cat_Precision_IOU0.75_p0.5",
                    "dog_Precision_IOU0.1_p0.5",
                    "dog_Precision_IOU0.5_p0.5",
                    "dog_Precision_IOU0.75_p0.5",
                    "cat_Recall_IOU0.1_p0.5",
                    "cat_Recall_IOU0.5_p0.5",
                    "cat_Recall_IOU0.75_p0.5",
                    "dog_Recall_IOU0.1_p0.5",
                    "dog_Recall_IOU0.5_p0.5",
                    "dog_Recall_IOU0.75_p0.5",
                ]
            )

            mock_run_context.save_checkpoint.assert_called_once_with(global_step)
            assert os.path.exists(
                Path(checkpoint_dir) / CHARIOT_VALOR_EVALUATIONS_FILE_NAME
            )
            with gzip.open(
                Path(checkpoint_dir) / CHARIOT_VALOR_EVALUATIONS_FILE_NAME,
                "rb",
            ) as f:
                data = json.load(f)
                assert len(data) == expected_num_metrics
                for d in data:
                    if type(cfg.datasets) is YoloDatasets:
                        assert d["snapshot_id"] == "local"
                        assert d["split"] == "val"
                    else:
                        assert d["snapshot_id"] == "sid"
                        assert d["split"] in ["val", "test"]
                    for m in d["metrics"] or []:
                        assert m["type"] in [
                            "AP",
                            "mAP",
                            "APAveragedOverIOUs",
                            "mAPAveragedOverIOUs",
                            "AR",
                            "mAR",
                            "ARAveragedOverScores",
                            "mARAveragedOverScores",
                            "Accuracy",
                            "PrecisionRecallCurve",
                            "Counts",
                            "Precision",
                            "Recall",
                            "F1",
                            "ConfusionMatrix",
                            "AggregatedPrecisionRecallCurve",
                            "AggregatedPrecision",
                            "AggregatedRecall",
                            "AggregatedF1",
                            "AggregatedCounts",
                        ]


def test_callbacks_on_val_end():
    mock_run_context = MagicMock()
    callbacks = Callbacks(mock_run_context, 10)
    mock_validator = MagicMock()
    mock_validator.metrics.results_dict = {
        "metrics/precision(B)": 0.5,
        "metrics/recall(B)": 0.5,
        "metrics/mAP50(B)": 0.5,
        "metrics/mAP50-95(B)": 0.5,
        "fitness": 0.5,
    }
    mock_validator.metrics.plot = False
    callbacks.global_step = 5
    callbacks.on_val_end(mock_validator)
    mock_run_context.save_metrics.assert_called_once_with(
        [
            {
                "global_step": 5,
                "tag": "metrics/precision(B)",
                "value": 0.5,
            },
            {
                "global_step": 5,
                "tag": "metrics/recall(B)",
                "value": 0.5,
            },
            {
                "global_step": 5,
                "tag": "metrics/mAP50(B)",
                "value": 0.5,
            },
            {
                "global_step": 5,
                "tag": "metrics/mAP50-95(B)",
                "value": 0.5,
            },
            {
                "global_step": 5,
                "tag": "fitness",
                "value": 0.5,
            },
        ]
    )


class ModelClass(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)


@pytest.mark.parametrize(
    "idx_to_class,expected_class_labels",
    [
        ({0: "cat", 1: "dog"}, {"cat": 0, "dog": 1}),
        ({0: "human"}, {"human": 0}),
    ],
)
def test_save_model_load_model(
    idx_to_class: dict[int, str],
    expected_class_labels: dict[str, int],
):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pt") as last_pt_file:
        model = YOLO("yolov8n-obb.yaml", task="obb")
        assert isinstance(model.model, DetectionModel)
        model.model.names = {int(id): label for id, label in idx_to_class.items()}
        model.save(last_pt_file.name)
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            save_model(
                Path(last_pt_file.name),
                Path(checkpoint_dir),
            )

            # confirm a valid `model.pt` was saved
            model_pt_path = Path(checkpoint_dir) / MODEL_PT_NAME
            assert model_pt_path.exists()
            assert YOLO(model_pt_path, task="obb")

            # confirm the correct `chariot_model_config.json` was saved
            engine_selector = CheckpointEngineSelector(
                org_name="Chariot",
                project_name="Common",
                engine_name="yolo-obb",
            )
            assert json.loads(
                (Path(checkpoint_dir) / "chariot_model_config.json").read_text()
            ) == ModelConfigDict(
                artifact_type="custom-engine",
                class_labels=expected_class_labels,
                copy_key_suffixes=[MODEL_PT_NAME],
                isvc_settings=CheckpointInferenceServerSettingsDict(
                    engine_selector=engine_selector
                ),
                supported_engines=[engine_selector],
            )

            with load_model(Path(checkpoint_dir)) as temp_pt_file:
                assert YOLO(temp_pt_file.name, "obb")


@pytest.mark.slow  # TODO(scohen): Consider this an integration/medium test?
def test_train_local_run_context():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_config = Config(
            epochs=5,
            datasets=Datasets(
                train_data=[DatasetSnapshot(snapshot_id="snapshot id", split="train")],
                val_data=[DatasetSnapshot(snapshot_id="snapshot id", split="val")],
            ),
            ckpt_epoch_period=2,
            resize_to=128,  # resize just large enough to actually 'learn' (nonzero AP, recall, etc)
        )
        config = base_config.model_dump()
        with local_run_context(run_id="my-run-id", config=config, base_dir=tmpdir) as r:
            with patch.object(
                r, "dataset_fetcher", mock_get_fetcher("Oriented Object Detection")
            ):
                train(r)
                assert type(r.progress_saver) is MemoryProgressSaver
                assert r.progress_saver.current_progress == [
                    {
                        "operation": "Training",
                        "final_value": 15,
                        "units": "steps",
                        "value": 15,
                    }
                ]
                assert type(r.metric_saver) is MemoryMetricSaver
                assert (
                    len(r.metric_saver.metrics) == 175
                )  # 5 epochs, 5 YOLO metrics and 30 valor metrics per epoch
                assert any(
                    map(
                        lambda m: type(m["value"]) is np.float64 and m["value"] > 0.0,
                        r.metric_saver.metrics,
                    )
                )
                metrics_by_global_step = itertools.groupby(
                    r.metric_saver.metrics, key=lambda m: m["global_step"]
                )
                # 35 images in the train set, with a 16 batch size, means we should save every 3 global steps
                assert set(dict(metrics_by_global_step).keys()) == {
                    3,
                    6,
                    9,
                    12,
                    15,
                }
                with r.load_checkpoint() as ckpt:
                    assert (ckpt.dir / MODEL_PT_NAME).exists()
                    # Save checkpoint every 2 epochs, each epoch is 3 global steps, so we
                    # should have checkpoints at global step == 6 and 12
                    assert int(ckpt.dir.name.split("-")[0]) == 12
                    assert any(
                        p.is_dir() and p.name.startswith("6-")
                        for p in ckpt.dir.parent.iterdir()
                    ), "No checkpoint found for global step 6"

        # Start another run, resuming from last checkpoint (only one epoch remaining)
        last_ckpt_id = ""
        with local_run_context(run_id="my-run-id", config=config, base_dir=tmpdir) as r:
            with patch.object(
                r, "dataset_fetcher", mock_get_fetcher("Oriented Object Detection")
            ):
                train(r)
                assert type(r.progress_saver) is MemoryProgressSaver
                assert r.progress_saver.current_progress == [
                    {
                        "operation": "Training",
                        "final_value": 15,
                        "units": "steps",
                        "value": 15,
                    }
                ]
                assert type(r.metric_saver) is MemoryMetricSaver
                assert (
                    len(r.metric_saver.metrics) == 35
                )  # Single epoch trained, 35 metrics per epoch
                assert any(
                    map(
                        lambda m: type(m["value"]) is np.float64 and m["value"] > 0.0,
                        r.metric_saver.metrics,
                    )
                )

                with r.load_checkpoint() as ckpt:
                    last_ckpt_id = ckpt.id

        # Start another run, using the last checkpoint but with a new run_id
        base_config.start_model = Checkpoint(checkpoint_id=last_ckpt_id)
        base_config.epochs = 2
        config = base_config.model_dump()
        with local_run_context(
            run_id="new-run-id", config=config, base_dir=tmpdir
        ) as r:
            with patch.object(
                r, "dataset_fetcher", mock_get_fetcher("Oriented Object Detection")
            ):
                train(r)
                assert type(r.progress_saver) is MemoryProgressSaver
                assert r.progress_saver.current_progress == [
                    {
                        "operation": "Training",
                        "final_value": 6,
                        "units": "steps",
                        "value": 6,
                    }
                ]
                assert type(r.metric_saver) is MemoryMetricSaver
                assert (
                    len(r.metric_saver.metrics) == 70
                )  # All epochs trained, 35 metrics per epoch
                assert any(
                    map(
                        lambda m: type(m["value"]) is np.float64 and m["value"] > 0.0,
                        r.metric_saver.metrics,
                    )
                )
                with r.load_checkpoint() as ckpt:
                    assert (ckpt.dir / MODEL_PT_NAME).exists()
                    assert int(ckpt.dir.name.split("-")[0]) == 6


@pytest.mark.slow  # TODO(scohen): Consider this an integration/medium test?
def test_train_local_run_context_class_alias():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            epochs=5,
            datasets=Datasets(
                train_data=[DatasetSnapshot(snapshot_id="snapshot id", split="train")],
                val_data=[DatasetSnapshot(snapshot_id="snapshot id", split="val")],
            ),
            ckpt_epoch_period=2,
            resize_to=32,  # keep small image_size for better performance
            class_to_idx={"cat": 0, "dog": 0, "chicken": 1},
            idx_to_class={0: "pet", 1: "chicken"},
        ).model_dump()
        with local_run_context(run_id="my-run-id", config=config, base_dir=tmpdir) as r:
            with patch.object(
                r, "dataset_fetcher", mock_get_fetcher("Oriented Object Detection")
            ):
                train(r)


class TestLoadCheckpointOrInitModel(unittest.TestCase):
    @patch("blueprint_yolo_obb.src.train.get_latest_checkpoint")
    def test_init_model(self, mock_get_latest_checkpoint):
        """Start from scratch with no latest checkpoint."""
        run_context = MagicMock(spec=RunContext)
        mock_get_latest_checkpoint.return_value = (
            "latest_checkpoint_file",
            "mock_model",
        )

        checkpoint_resume_file, model = load_checkpoint_or_init_model(run_context, 2)

        mock_get_latest_checkpoint.assert_called_once_with(run_context, 2)
        self.assertEqual(checkpoint_resume_file, "latest_checkpoint_file")
        self.assertEqual(model, "mock_model")

    @patch("blueprint_yolo_obb.src.train.get_latest_checkpoint")
    @patch("blueprint_yolo_obb.src.train.CallbackYOLO")
    @patch("blueprint_yolo_obb.src.train.Callbacks")
    def test_init_model_when_no_checkpoint(
        self, mock_callbacks, mock_yolo, mock_get_latest_checkpoint
    ):
        """Test case for initializing the model when no checkpoint is found."""
        run_context = MagicMock(spec=RunContext)
        mock_get_latest_checkpoint.side_effect = CheckpointNotFoundError(
            "No checkpoint found"
        )

        mock_callbacks.return_value = MagicMock()
        mock_yolo.return_value = "mock_model"

        checkpoint_resume_file, model = load_checkpoint_or_init_model(run_context, 10)

        assert checkpoint_resume_file is None
        assert model == "mock_model"
        mock_yolo.assert_called_once_with(
            "yolov8n-obb.yaml", callbacks=mock_callbacks.return_value
        )


class TestLoadStartCheckpoint(unittest.TestCase):
    @patch("blueprint_yolo_obb.src.train.get_latest_checkpoint")
    @patch("blueprint_yolo_obb.src.train.load_model")
    @patch("ultralytics.engine.model.checks.check_file")
    def test_checkpoint_from_same_run(
        self, mock_check_file, mock_load_model, mock_get_latest_checkpoint
    ):
        """Start from a checkpoint from the same run, start from checkpoint."""
        run_context = MagicMock(spec=RunContext)
        run_context.run_id = "run_123"
        ckpt = MagicMock()
        ckpt.run_id = "run_123"
        ckpt.global_step = 20

        run_context.load_checkpoint.return_value.__enter__.return_value = ckpt
        mock_load_model.return_value.name = "checkpoint_model_file"
        mock_check_file.return_value = True

        checkpoint_resume_file, _, pretrained = load_start_checkpoint(
            run_context, "checkpoint_1", 10
        )

        mock_get_latest_checkpoint.assert_not_called()
        mock_load_model.assert_called_once_with(ckpt.dir)
        self.assertEqual(checkpoint_resume_file.name, "checkpoint_model_file")
        self.assertFalse(
            pretrained
        )  # pretrained should be False since global_step != 0

    @patch("blueprint_yolo_obb.src.train.get_latest_checkpoint")
    @patch("blueprint_yolo_obb.src.train.load_model")
    @patch("blueprint_yolo_obb.src.train.CallbackYOLO")
    @patch("ultralytics.engine.model.checks.check_file")
    def test_checkpoint_from_different_run_from_scratch(
        self,
        mock_check_file,
        mock_chariot_yolo,
        mock_load_model,
        mock_get_latest_checkpoint,
    ):
        """Start from a checkpoint from a different run, no latest checkpoint."""
        run_context = MagicMock(spec=RunContext)
        run_context.run_id = "run_123"
        ckpt = MagicMock()
        ckpt.run_id = "run_456"
        ckpt.global_step = 100

        run_context.load_checkpoint.return_value.__enter__.return_value = ckpt
        mock_get_latest_checkpoint.side_effect = CheckpointNotFoundError(
            "No checkpoint found"
        )
        mock_checkpoint = MagicMock()
        mock_checkpoint.name = "checkpoint_model_file"
        mock_load_model.return_value = mock_checkpoint
        mock_chariot_yolo.return_value = "checkpoint_model_file"
        mock_check_file.return_value = True

        checkpoint_resume_file, model, pretrained = load_start_checkpoint(
            run_context, "checkpoint_1", 10
        )

        mock_get_latest_checkpoint.assert_called_once_with(run_context, 10)
        self.assertEqual(checkpoint_resume_file, mock_checkpoint)
        self.assertEqual(model, "checkpoint_model_file")
        self.assertTrue(
            pretrained
        )  # True using another run's checkpoint and no latest checkpoint

    @patch("blueprint_yolo_obb.src.train.get_latest_checkpoint")
    def test_checkpoint_from_different_run_from_latest(
        self, mock_get_latest_checkpoint
    ):
        """Start from a checkpoint from a different run, no latest checkpoint."""
        run_context = MagicMock(spec=RunContext)
        run_context.run_id = "run_123"
        ckpt = MagicMock()
        ckpt.run_id = "run_456"

        run_context.load_checkpoint.return_value.__enter__.return_value = ckpt
        mock_get_latest_checkpoint.return_value = (
            "latest_checkpoint_file",
            "mock_model",
        )

        checkpoint_resume_file, model, pretrained = load_start_checkpoint(
            run_context, "checkpoint_1", 10
        )

        mock_get_latest_checkpoint.assert_called_once_with(run_context, 10)
        self.assertEqual(checkpoint_resume_file, "latest_checkpoint_file")
        self.assertEqual(model, "mock_model")
        self.assertFalse(
            pretrained
        )  # False using another run's checkpoint and but restarting on latest checkpoint


class TestLoadStartModel(unittest.TestCase):
    @patch("blueprint_yolo_obb.src.train.get_latest_checkpoint")
    def test_start_model_with_latest_checkpoint(self, mock_get_latest_checkpoint):
        """Start from scratch with no latest checkpoint."""
        run_context = MagicMock(spec=RunContext)
        mock_get_latest_checkpoint.return_value = (
            "latest_checkpoint_file",
            "mock_model",
        )

        checkpoint_resume_file, model = load_start_model(run_context, "model_id", 2)

        mock_get_latest_checkpoint.assert_called_once_with(run_context, 2)
        self.assertIsNotNone(checkpoint_resume_file)
        self.assertEqual(model, "mock_model")

    @patch("blueprint_yolo_obb.src.train.get_latest_checkpoint")
    @patch("blueprint_yolo_obb.src.train.load_model")
    @patch("blueprint_yolo_obb.src.train.CallbackYOLO")
    @patch("pathlib.Path.unlink")
    def test_checkpoint_from_different_run_from_scratch(
        self,
        mock_unlink,
        mock_chariot_yolo,
        mock_load_model,
        mock_get_latest_checkpoint,
    ):
        """Start from scratch with no latest checkpoint."""
        run_context = MagicMock(spec=RunContext)
        mock_get_latest_checkpoint.side_effect = CheckpointNotFoundError(
            "No checkpoint found"
        )
        mock_load_model.return_value.name = "checkpoint_model_file"
        mock_model = MagicMock()
        mock_chariot_yolo.return_value = mock_model

        checkpoint_resume_file, model = load_start_model(run_context, "model_id", 2)

        assert checkpoint_resume_file is None
        assert model == mock_model
