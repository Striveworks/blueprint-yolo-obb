import base64
import json
import tempfile
from pathlib import Path

import pytest
import torch
from pytest import fixture
from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext

from blueprint_yolo_obb.src.data.handler import Handler, prepare_data

_BLACK_10x10_PNG = base64.decodebytes(
    b"""
iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAGElEQ
VQY02NkYGD4z0AEYGIgEowqpI5CAKYXARPJvDplAAAAAElFTkSuQmCC
"""
)


@fixture(scope="module")
def model_file(mutable_datadir: str) -> str:
    return str(Path(mutable_datadir) / "s3" / "yolov8n-obb.pt")


def handle_inference(
    model_file: str,
    img_file: str,
    model_config_json: dict | None = None,
):
    torch.manual_seed(42 * 42)

    with tempfile.TemporaryDirectory() as tmpdir:
        if model_config_json is not None:
            with (Path(tmpdir) / "model_config.json").open("w") as f:
                json.dump(model_config_json, f)

        ctx = MockContext(
            model_pt_file=model_file,
            model_dir=str(tmpdir),
            model_file="",
        )

        handler = Handler()
        handler.initialize(ctx)
        handler.context = ctx

        data = prepare_data(img_file, 1, [0.25])

        return handler.handle(data, ctx)


@pytest.mark.requires_s3
def test_handler_no_detections(model_file: str):
    with tempfile.NamedTemporaryFile() as tf:
        tf.write(_BLACK_10x10_PNG)
        tf.flush()
        results = handle_inference(model_file, tf.name)
        assert len(results) == 1
        assert results[0] == {
            "detection_classes": [],
            "detection_scores": [],
            "oriented_detection_boxes": [],
        }


@pytest.mark.requires_s3
def test_handler_detections(mutable_datadir: Path, model_file: str):
    results = handle_inference(model_file, str(mutable_datadir / "s3" / "planes.jpg"))
    assert len(results) == 1
    result = results[0]
    assert type(result) is dict
    assert len(result["detection_classes"]) == 31
    assert len(result["detection_scores"]) == 31
    assert len(result["oriented_detection_boxes"]) == 31

    assert set(result["detection_classes"]) == {"plane"}
    assert result["oriented_detection_boxes"][0] == pytest.approx(
        {
            "cx": 0.5976253151893616,
            "cy": 0.8829976320266724,
            "h": 0.30599817633628845,
            "r": 0.1294691413640976,
            "w": 0.17874747514724731,
        },
        rel=1e-3,
    )
    assert result["detection_scores"][0] == pytest.approx(0.8989537954330444, rel=1e-3)


@pytest.mark.requires_s3
def test_handler_idx_to_class(mutable_datadir: Path, model_file: str):
    results = handle_inference(
        model_file,
        str(mutable_datadir / "s3" / "planes.jpg"),
        model_config_json={"idx_to_class": {0: "ufo"}},
    )
    assert len(results) == 1
    result = results[0]
    assert type(result) is dict
    assert len(result["detection_classes"]) == 31
    assert len(result["detection_scores"]) == 31
    assert len(result["oriented_detection_boxes"]) == 31

    assert set(result["detection_classes"]) == {"ufo"}
    assert result["oriented_detection_boxes"][0] == pytest.approx(
        {
            "cx": 0.5976253151893616,
            "cy": 0.8829976320266724,
            "h": 0.30599817633628845,
            "r": 0.1294691413640976,
            "w": 0.17874747514724731,
        },
        rel=1e-3,
    )
    assert result["detection_scores"][0] == pytest.approx(0.8989537954330444, rel=1e-3)


@pytest.mark.requires_s3
def test_handler_resize_to(mutable_datadir: Path, model_file: str):
    scores = []
    for resize_to in [128, 256, 512, 640, 1280]:
        results = handle_inference(
            model_file,
            str(mutable_datadir / "s3" / "planes.jpg"),
            model_config_json={"idx_to_class": {0: "ufo"}, "resize_to": resize_to},
        )
        assert len(results) == 1
        result = results[0]
        assert type(result) is dict
        assert len(result["detection_scores"]) > 0
        assert set(result["detection_classes"]) == {"ufo"}
        scores.append(result["detection_scores"][0])
    assert scores == pytest.approx(
        [
            0.8805660605430603,
            0.8758456707000732,
            0.8941929936408997,
            0.89892578125,
            0.9011396169662476,
        ],
        rel=1e-3,
    )
