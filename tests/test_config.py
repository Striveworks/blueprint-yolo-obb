import json

import pytest
from pydantic import ValidationError

from blueprint_yolo_obb.src.config import Config


def test_config_empty():
    with pytest.raises(ValidationError) as ve:
        Config.model_validate({})
    errors = ve.value.errors()
    assert len(errors) == 1
    error = errors[0]
    assert error["type"] == "missing"
    assert error["loc"] == ("datasets",)


def test_chariot_dataset_success_no_val_data():
    Config.model_validate(
        {"datasets": {"train_data": [{"snapshot_id": "foo"}], "val_data": None}}
    )


def test_chariot_dataset_no_train_data():
    with pytest.raises(ValidationError) as ve:
        Config.model_validate({"datasets": {"train_data": []}})
    error_str = str(ve.value)
    assert "List should have at least 1 item after validation" in error_str
    assert "train_data" in error_str


def test_yolo_dataset_no_class_to_idx():
    with pytest.raises(ValidationError) as ve:
        Config.model_validate({"datasets": {"data_root": "/tmp"}})
    assert "if YoloDatasets are specified, class_to_idx must be specified " in str(ve)


def test_yolo_dataset_class_to_idx():
    Config.model_validate(
        {"datasets": {"data_root": "/tmp"}, "class_to_idx": {"cat": 0}}
    )


def test_config_json_schema(mutable_datadir, enable_self_correction):
    """File-driven, self-correcting test for the config JSON schema"""
    actual = json.dumps(Config.model_json_schema(), indent=2)
    expected_file = mutable_datadir / "config_json_schema.json"
    if not expected_file.exists():
        expected_file.touch()
        raise AssertionError(f"Created '{expected_file}'.")
    with expected_file.open() as fp:
        expected = fp.read()
    if enable_self_correction and actual != expected:
        print(f"Mismatch in '{expected_file}'. Overwriting.")
        with expected_file.open("w") as fp:
            fp.write(actual)

    assert actual == expected
