import gzip
import io
import json
import logging
import math
import shutil
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from blueprint_toolkit import (
    Checkpoint,
    CheckpointEngineSelector,
    CheckpointInferenceServerSettingsDict,
    CheckpointNotFoundError,
    Model,
    ModelConfigDict,
    RunContext,
    SaveMetricDict,
)
from chariot_valor.lite import Loader as ValorLoader
from chariot_valor.lite import TaskType
from PIL import Image
from ultralytics import YOLO
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops
from ultralytics.utils.callbacks import get_default_callbacks

from blueprint_yolo_obb.src.config import Config, Datasets, YoloDatasets
from blueprint_yolo_obb.src.util import convert_yolo_results
from blueprint_yolo_obb.src.yolo_yaml import get_yolo_file

MODEL_PT_NAME = "model.pt"
CHARIOT_VALOR_EVALUATIONS_FILE_NAME = "chariot_valor_evaluations.json.gz"


def save_model(
    last_pt_file: Path,
    checkpoint_dir: Path,
):
    if not last_pt_file.exists():
        raise ValueError(f"last_pt_file {str(last_pt_file)!r} does not exist")
    model = YOLO(last_pt_file, task="obb")
    model.save(str(checkpoint_dir / MODEL_PT_NAME))
    idx_to_class = model.names
    with open(checkpoint_dir / "chariot_model_config.json", "w") as fp:
        engine_selector = CheckpointEngineSelector(
            org_name="Chariot",
            project_name="Common",
            engine_name="yolo-obb",
        )
        json.dump(
            ModelConfigDict(
                artifact_type="custom-engine",
                class_labels={cls: idx for idx, cls in idx_to_class.items()},
                copy_key_suffixes=[MODEL_PT_NAME],
                isvc_settings=CheckpointInferenceServerSettingsDict(
                    engine_selector=engine_selector
                ),
                supported_engines=[engine_selector],
            ),
            fp,
        )


def load_model(dir: Path):
    # TODO(s.maddox): simplify model loading to not make a copy?
    temp_model_pt_file = tempfile.NamedTemporaryFile("wb", suffix=".pt", delete=False)
    temp_model_pt_file.write((dir / MODEL_PT_NAME).read_bytes())
    return temp_model_pt_file


def convert_yolo_results_to_valor_metrics(
    validator: OBBValidator,
    model: YOLO,
    idx_to_class: dict[int, str],
    snapshot_id: str | None,
    split: str | None,
):
    valor_evaluations = []

    if validator.dataloader:
        gt_ann = defaultdict(list)
        pred_ann = defaultdict(dict)

        for gts in iter(validator.dataloader):
            for i, img_idx in enumerate(gts["batch_idx"].tolist()):
                image_path = gts["im_file"][int(img_idx)]
                img_id = Path(image_path).stem
                if snapshot_id and not img_id.startswith(
                    f"{snapshot_id}.{split or ''}."
                ):
                    continue

                gt_ann[img_id].append(
                    {
                        "oriented_bbox": {
                            "cx": float(gts["bboxes"][i][0].item()),
                            "cy": float(gts["bboxes"][i][1].item()),
                            "w": float(gts["bboxes"][i][2].item()),
                            "h": float(gts["bboxes"][i][3].item()),
                            "r": float(gts["bboxes"][i][4].item()),
                        },
                        "class_label": idx_to_class[int(gts["cls"][i].item())],
                    }
                )

                if img_id not in pred_ann:
                    results = model(image_path)
                    results_json = convert_yolo_results(model.device, results)
                    if len(results_json) > 0:
                        pred_ann[img_id] = results_json[0]

        class_to_idx = {cls: idx for idx, cls in idx_to_class.items()}
        loader = ValorLoader(
            TaskType.ORIENTED_OBJECT_DETECTION,
            "detect",
            class_to_idx,
        )
        loader.add_data([(k, gt_ann[k], pred_ann[k]) for k in pred_ann])
        valor_evaluations = loader.evaluate()

    return {
        "snapshot_id": snapshot_id if snapshot_id else "local",
        "split": split if split else "val",
        "metrics": valor_evaluations or [],
    }


def to_legacy_teddy_metric(valor_metric: dict[str, Any]) -> tuple[str, float] | None:
    match valor_metric:
        case {
            "type": metric_type,
            "parameters": {
                "iou_threshold": iou_threshold,
                "label": label,
                **extra_params,
            },
            "value": value,
        } if metric_type in {"AP", "Precision", "Recall"}:
            if (p_threshold := extra_params.get("score_threshold")) is not None:
                p_suffix = f"_p{p_threshold}"
            else:
                p_suffix = ""
            return (f"{label}_{metric_type}_IOU{iou_threshold}{p_suffix}", value)
        case {
            "type": "mAP",
            "parameters": {"iou_threshold": iou_threshold},
            "value": value,
        }:
            return (f"mAP_IOU{iou_threshold}", value)
        case _:
            return None


class Callbacks:
    def __init__(
        self, run_context: RunContext, ckpt_epoch_period: int, global_step: int = 0
    ):
        self.run_context = run_context
        self.ckpt_epoch_period = ckpt_epoch_period
        self.global_step = global_step
        self.latest_val_step = None

    def on_train_batch_start(self, trainer: OBBTrainer):
        self.global_step += 1
        self.run_context.save_progress(
            [
                {
                    "operation": "Training",
                    "final_value": trainer.epochs
                    * math.ceil(len(trainer.train_loader.dataset) / trainer.batch_size),
                    "units": "steps",
                    "value": self.global_step,
                }
            ]
        )

    def on_val_end(self, validator: OBBValidator):
        if self.global_step == self.latest_val_step:
            # Sometimes validation is done at the end of training
            # on `best.pt`, the best model of the training process.
            # This is a simple way to avoid saving those metrics
            # on the final step
            return
        self.latest_val_step = self.global_step
        self.run_context.save_metrics(
            [
                {"global_step": self.global_step, "tag": tag, "value": val}
                for tag, val in validator.metrics.results_dict.items()
            ]
        )

    def on_model_save(
        self,
        trainer: OBBTrainer,
        idx_to_class: dict[int, str],
    ):
        # If args.save=True (default), then every epoch, `last.pt`
        # will get saved, and this callback will be called. There is also a `save_period`
        # config which will ensure that every `save_period` epochs, the checkpoint file
        # is saved separately as `epoch<num>.pt`. If args.save_period=-1 (default),
        # then these `epoch<num>.pt` files will not be saved
        if not trainer.args.save or trainer.last is None or trainer.validator is None:
            logging.warning(
                "Skipping saving model, if args.save=False or no last.pt or validator"
            )
            return

        # save metrics
        valor_metrics = []
        model = YOLO(trainer.last, task="obb")
        config = Config.model_validate(self.run_context.load_config())

        if type(config.datasets) is YoloDatasets:
            assert trainer.validator  # to appease the the type checker
            valor_metrics.append(
                convert_yolo_results_to_valor_metrics(
                    trainer.validator,
                    model,
                    idx_to_class,
                    None,
                    None,
                )
            )

        if type(config.datasets) is Datasets:
            for t in config.datasets.val_data or []:
                assert trainer.validator  # to appease the the type checker
                valor_metrics.append(
                    convert_yolo_results_to_valor_metrics(
                        trainer.validator,
                        model,
                        idx_to_class,
                        t.snapshot_id,
                        t.split,
                    )
                )

        del model, config

        teddy_metrics = [
            SaveMetricDict(
                global_step=self.global_step,
                tag=f"{m[0]}/{m[1]}/{m[2][0]}",
                value=m[2][1],
            )
            for m in [
                (
                    metrics["snapshot_id"],
                    metrics["split"],
                    to_legacy_teddy_metric(v_m),
                )
                for metrics in valor_metrics or []
                for v_m in metrics["metrics"] or []
            ]
            if m[2]
        ]
        self.run_context.save_metrics(teddy_metrics)

        if (trainer.epoch + 1) % self.ckpt_epoch_period == 0:
            logging.info(f"Saving checkpoint at step {self.global_step}")

            with self.run_context.save_checkpoint(self.global_step) as ckpt:
                save_model(trainer.last, ckpt.dir)

                if len(valor_metrics) > 0:
                    with open(
                        ckpt.dir / CHARIOT_VALOR_EVALUATIONS_FILE_NAME, "w+b"
                    ) as fp:
                        cp_metrics_bytes = json.dumps(valor_metrics, indent=4).encode(
                            "utf-8"
                        )
                        fp.write(gzip.compress(cp_metrics_bytes))

            logging.info(f"Done saving checkpoint at step {self.global_step}")


class CallbackYOLO(YOLO):
    def __init__(
        self,
        model="yolov8n.pt",
        task=None,
        verbose=False,
        callbacks: Callbacks | None = None,
    ):
        if not callbacks:
            raise ValueError("must specify callbacks")
        self.custom_callbacks = callbacks
        super().__init__(model=model, task=task, verbose=verbose)  # type: ignore[reportArgumentType] bad annotations in YOLO

    @property
    def task_map(self):
        task_map = super().task_map
        task_map["obb"]["trainer"] = OBBTrainerWithCallbacks.with_callbacks(
            self.custom_callbacks
        )
        return task_map


class OBBTrainerWithCallbacks(OBBTrainer):
    def __init__(
        self,
        callbacks: Callbacks,
        cfg=DEFAULT_CFG,
        overrides=None,
        _callbacks=None,
    ):
        self.custom_callbacks = callbacks
        super().__init__(cfg, overrides=overrides, _callbacks=_callbacks)

    def get_validator(self):
        callbacks = get_default_callbacks()
        # Note: OBBTrainer doesn't propagate callbacks to the validator, so need to subclass
        # in order to setup this on_val_end callback
        callbacks["on_val_end"].append(self.custom_callbacks.on_val_end)
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        validator = OBBValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=callbacks,
        )
        validator.metrics.plot = False
        return validator

    @classmethod
    def with_callbacks(cls, callbacks: Callbacks):
        return partial(cls, callbacks=callbacks)


def fetch_datums(
    run_context: RunContext, config: Config, class_to_idx: dict[str, int] | None
) -> tuple[Path, dict[str, int]]:
    # TODO(scohen): Determine if it's possible to not pre-cache all
    # data here. It is safer to download all the data and put it in
    # the proper YOLO directories, rather than subclass the YOLO Dataset
    # for just-in-time data loading, given the YOLO Dataset includes some image
    # transformations in it and doesn't seem to be a part of the public API
    assert type(config.datasets) is Datasets
    data_root = Path(tempfile.TemporaryDirectory().name)
    logging.info(f"Downloading chariot datums to {data_root}")

    train_val_data = [d for d in config.datasets.train_data] + [
        v for v in config.datasets.val_data or []
    ]
    dataset_fetcher = run_context.dataset_fetcher

    cl = []
    for t in train_val_data:
        dataset_fetcher.prepare_snapshot_split(t.snapshot_id, t.split)
        for c in dataset_fetcher.get_class_labels():
            cl.append(c)
    class_labels = set(cl)

    class_to_idx = class_to_idx or {c: i for i, c in enumerate(sorted(class_labels))}
    (data_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (data_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (data_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (data_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    def fetch_datum_at_index(snapshot_id: str, split: str | None, index: int):
        assert data_root
        assert snapshot_id
        assert split
        datum, target = dataset_fetcher.get_datum_at_index(index)
        datum_img = Image.open(io.BytesIO(datum))
        assert datum_img.format

        width, height = datum_img.size
        (
            data_root
            / "images"
            / split
            / f"{snapshot_id}.{split or ''}.{target['id']}.{datum_img.format.lower()}"
        ).write_bytes(datum)
        detection_anns = [
            a
            for a in target.get("annotations", [])
            # For initial testing, support bbox/Object Detection and contour/Image Segmentation in addition to oriented
            if a.get("bbox") or a.get("oriented_bbox") or a.get("contour")
        ]
        label_file = (
            data_root
            / "labels"
            / split
            / f"{snapshot_id}.{split or ''}.{target['id']}.txt"
        )
        with open(
            label_file,
            "w",
        ) as f:
            for detection_ann in detection_anns:
                assert "class_label" in detection_ann
                class_label = detection_ann["class_label"]
                class_idx = class_to_idx[class_label]
                if "bbox" in detection_ann:
                    bbox = detection_ann["bbox"]
                    top_left = (bbox["xmin"] / width, bbox["ymin"] / height)
                    top_right = (bbox["xmax"] / width, bbox["ymin"] / height)
                    bottom_right = (bbox["xmax"] / width, bbox["ymax"] / height)
                    bottom_left = (bbox["xmin"] / width, bbox["ymax"] / height)
                    annotation = (
                        f"{class_idx}"
                        + f" {bottom_right[0]:.5g}"
                        + f" {bottom_right[1]:.5g}"
                        + f" {top_right[0]:.5g}"
                        + f" {top_right[1]:.5g}"
                        + f" {top_left[0]:.5g}"
                        + f" {top_left[1]:.5g}"
                        + f" {bottom_left[0]:.5g}"
                        + f" {bottom_left[1]:.5g}\n"
                    )
                    f.write(annotation)
                elif "oriented_bbox" in detection_ann:
                    oriented_bbox = detection_ann["oriented_bbox"]
                    xyxyxyxy = ops.xywhr2xyxyxyxy(
                        torch.tensor(
                            [
                                oriented_bbox["cx"],
                                oriented_bbox["cy"],
                                oriented_bbox["w"],
                                oriented_bbox["h"],
                                oriented_bbox["r"],
                            ]
                        )
                    )
                    assert type(xyxyxyxy) is torch.Tensor
                    xyxyxyxy = torch.flatten(xyxyxyxy)
                    annotation = (
                        f"{class_idx}"
                        + f" {xyxyxyxy[0].item():.5g}"
                        + f" {xyxyxyxy[1].item():.5g}"
                        + f" {xyxyxyxy[2].item():.5g}"
                        + f" {xyxyxyxy[3].item():.5g}"
                        + f" {xyxyxyxy[4].item():.5g}"
                        + f" {xyxyxyxy[5].item():.5g}"
                        + f" {xyxyxyxy[6].item():.5g}"
                        + f" {xyxyxyxy[7].item():.5g}\n"
                    )
                    f.write(annotation)
                elif "contour" in detection_ann:
                    contour = detection_ann["contour"]
                    if len(contour) != 1:
                        raise ValueError(
                            f"contour for annotation {detection_ann['id']} invalid: must contain one polygon"
                        )
                    box_points = cv2.boxPoints(
                        cv2.minAreaRect(
                            np.array(
                                [(c["x"], c["y"]) for c in contour[0]],
                                dtype=np.int32,
                            )
                        )
                    )
                    annotation = (
                        f"{class_idx}"
                        + f" {box_points[0][0] / width:.5g}"
                        + f" {box_points[0][1] / height:.5g}"
                        + f" {box_points[1][0] / width:.5g}"
                        + f" {box_points[1][1] / height:.5g}"
                        + f" {box_points[2][0] / width:.5g}"
                        + f" {box_points[2][1] / height:.5g}"
                        + f" {box_points[3][0] / width:.5g}"
                        + f" {box_points[3][1] / height:.5g}\n"
                    )
                    f.write(annotation)

    for t in train_val_data:
        dataset_fetcher.prepare_snapshot_split(t.snapshot_id, t.split)
        with ThreadPoolExecutor() as tp:
            list(
                tp.map(
                    partial(fetch_datum_at_index, t.snapshot_id, t.split),
                    range(dataset_fetcher.num_datums),
                )
            )

    logging.info(f"Done downloading chariot datums to {data_root}")
    return data_root, class_to_idx


def get_latest_checkpoint(run_context: RunContext, ckpt_epoch_period: int):
    logging.info("Checking for existing checkpoints for this run")
    with run_context.load_checkpoint() as ckpt:
        logging.info(f"Found checkpoint from global step {ckpt.global_step}")
        callbacks = Callbacks(
            run_context, ckpt_epoch_period, global_step=ckpt.global_step
        )
        checkpoint_resume_file = load_model(ckpt.dir)
        model = CallbackYOLO(
            checkpoint_resume_file.name,
            callbacks=callbacks,
        )

    return checkpoint_resume_file, model


def load_checkpoint_or_init_model(run_context: RunContext, ckpt_epoch_period: int):
    try:
        checkpoint_resume_file, model = get_latest_checkpoint(
            run_context, ckpt_epoch_period
        )
    except CheckpointNotFoundError:
        logging.info("No existing checkpoint for this run found")
        callbacks = Callbacks(run_context, ckpt_epoch_period)
        model = CallbackYOLO(
            "yolov8n-obb.yaml", callbacks=callbacks
        )  # Ensure this is a .yaml file, not a .pt file, so that weights are not downloaded
        checkpoint_resume_file = None
    return checkpoint_resume_file, model


def load_start_checkpoint(
    run_context: RunContext, checkpoint_id: str, ckpt_epoch_period: int
):
    logging.info("loading start checkpoint")
    with run_context.load_checkpoint(checkpoint_id) as ckpt:
        # If the provided checkpoint is from the current run, start from checkpoint
        if run_context.run_id == ckpt.run_id:
            logging.warning(
                f"Provided checkpoint is from current run {ckpt.run_id}, starting from checkpoint."
            )
            callbacks = Callbacks(
                run_context, ckpt_epoch_period, global_step=ckpt.global_step
            )
            checkpoint_resume_file = load_model(ckpt.dir)
            model = CallbackYOLO(
                checkpoint_resume_file.name,
                callbacks=callbacks,
            )
            return checkpoint_resume_file, model, False
        else:
            logging.warning(
                f"Current run_id is {run_context.run_id}, but checkpoint run_id {ckpt.run_id} is from a different run."
            )
            # Check for existing latest checkpoint for this run to resume from
            try:
                checkpoint_resume_file, model = get_latest_checkpoint(
                    run_context, ckpt_epoch_period
                )
                return checkpoint_resume_file, model, False
            # If not found, start from global step 0 as pretrained
            except CheckpointNotFoundError:
                ckpt.global_step = 0
                callbacks = Callbacks(
                    run_context, ckpt_epoch_period, global_step=ckpt.global_step
                )
                checkpoint_resume_file = load_model(ckpt.dir)
                model = CallbackYOLO(
                    checkpoint_resume_file.name,
                    callbacks=callbacks,
                )
                return checkpoint_resume_file, model, True


def load_start_model(run_context: RunContext, model_id: str, ckpt_epoch_period: int):
    # Check for existing latest checkpoint for this run to resume from
    try:
        checkpoint_resume_file, model = get_latest_checkpoint(
            run_context, ckpt_epoch_period
        )
        return checkpoint_resume_file, model
    # If not found, start from model from global step 0
    except CheckpointNotFoundError:
        with run_context.load_model(model_id) as model:
            callbacks = Callbacks(run_context, ckpt_epoch_period)
            checkpoint_resume_file = load_model(model.dir)
            model = CallbackYOLO(
                checkpoint_resume_file.name,
                callbacks=callbacks,
            )
            Path(checkpoint_resume_file.name).unlink()
        # Return None as the checkpoint resume file. When starting from a model,
        # we just initialize the model with the model's weights, but don't want to resume
        # using the checkpointed epoch or optimizer weights
        return None, model


def train(run_context: RunContext):
    config = Config.model_validate(run_context.load_config())

    logging.basicConfig(level=config.log_level)
    LOGGER.setLevel(config.log_level)

    checkpoint_resume_file, pretrained = None, False
    if config.start_model is None:
        logging.info("No start model provided - checking for latest checkpoint")
        checkpoint_resume_file, model = load_checkpoint_or_init_model(
            run_context, config.ckpt_epoch_period
        )
    elif type(config.start_model) is Checkpoint:
        logging.info(f"Checkpoint provided: {config.start_model.checkpoint_id}")
        checkpoint_resume_file, model, pretrained = load_start_checkpoint(
            run_context, config.start_model.checkpoint_id, config.ckpt_epoch_period
        )
    elif type(config.start_model) is Model:
        logging.info(f"Model provided: {config.start_model.model_id}")
        checkpoint_resume_file, model = load_start_model(
            run_context, config.start_model.model_id, config.ckpt_epoch_period
        )
        if checkpoint_resume_file is None:
            pretrained = True
    else:
        raise ValueError(f"unhandled start_model type: {type(config.start_model)}")

    logging.info(f"Starting training with pretrained={pretrained}")

    temp_datums_dir = None
    if type(config.datasets) is Datasets:
        temp_datums_dir, datums_class_to_idx = fetch_datums(
            run_context, config, config.class_to_idx
        )
        data_root = temp_datums_dir
        if not config.class_to_idx:
            config.class_to_idx = datums_class_to_idx
    elif type(config.datasets) is YoloDatasets:
        data_root = config.datasets.data_root
    else:
        raise ValueError(f"unexpected dataset type {type(config.datasets)}")

    model.add_callback(
        "on_train_batch_start", model.custom_callbacks.on_train_batch_start
    )
    assert config.class_to_idx
    idx_to_class = config.idx_to_class or {i: c for c, i in config.class_to_idx.items()}
    model.add_callback(
        "on_model_save",
        partial(
            model.custom_callbacks.on_model_save,
            idx_to_class=idx_to_class,
        ),
    )

    try:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
            f.write(get_yolo_file(data_root.absolute(), idx_to_class))
            f.flush()
            device = "cuda" if torch.cuda.is_available() else "cpu"

            train_kwargs = {
                # TODO: do we need to be specifying names, here?
                "data": f.name,
                "epochs": config.epochs,
                "batch": config.batch_size,
                "imgsz": config.resize_to,
                "device": device,
            }

            # If we have weights and starting from scratch, set pretrained.
            # Otherwise, we can resume training.
            if checkpoint_resume_file:
                key = "pretrained" if pretrained else "resume"
                logging.info(
                    f"{'Starting from pretrained weights' if pretrained else 'Resuming training from checkpoint'} {checkpoint_resume_file}"
                )
                train_kwargs[key] = checkpoint_resume_file.name

            results = model.train(**train_kwargs)
            logging.info(results)
    finally:
        if temp_datums_dir:
            shutil.rmtree(temp_datums_dir)
        if checkpoint_resume_file:
            Path(checkpoint_resume_file.name).unlink()
