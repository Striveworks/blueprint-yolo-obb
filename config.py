from dataclasses import dataclass
from typing import Annotated, Literal

from annotated_types import MinLen
from blueprint_toolkit import Checkpoint, DatasetSnapshot, Model
from pydantic import (
    BaseModel,
    DirectoryPath,
    PositiveInt,
    model_validator,
)
from typing_extensions import Self


@dataclass
class Datasets:
    train_data: Annotated[list[DatasetSnapshot], MinLen(1)]
    """Training datasets."""
    val_data: Annotated[list[DatasetSnapshot], MinLen(1)] | None
    """Validation datasets."""


@dataclass
class YoloDatasets:
    data_root: DirectoryPath
    """If using local datasets, the root path to
    the data directory. This directory should be in YOLO format.
    If specified, provide `class_to_idx`."""


class Config(BaseModel):
    datasets: Datasets | YoloDatasets
    """Datasets to train on."""
    class_to_idx: dict[str, int] | None = None
    """Mapping from class label name to integer id
    By default, this is built from all sorted class labels in the datasets."""
    idx_to_class: dict[int, str] | None = None
    """Mapping from integer id to class label name, used at inference time
    By default, this is constructed from class_to_idx."""
    start_model: Model | Checkpoint | None = None
    """The model or checkpoint to start from"""
    ckpt_epoch_period: PositiveInt = 5
    """Epoch frequency to save checkpoints"""
    log_level: Literal[
        "CRITICAL", "FATAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG", "NOTSET"
    ] = "INFO"
    """Use verbose logging"""

    # Configurations below here directly map to YOLO configs
    epochs: int = 100
    """Total number of training epochs. Each epoch represents
    a full pass over the entire dataset.
    Adjusting this value can affect training duration and model performance."""
    batch_size: int | float = 16
    """Batch size, with three modes: set as an integer (e.g., batch=16),
    auto mode for 60% GPU memory utilization (batch=-1), or auto mode
    with specified utilization fraction (batch=0.70)."""
    resize_to: int = 640
    """Target image size for training. All images are resized
    to this dimension before being fed into the model.
    Affects model accuracy and computational complexity."""

    @model_validator(mode="after")
    def check_data(self) -> Self:
        if type(self.datasets) is YoloDatasets and not self.class_to_idx:
            raise ValueError(
                "if YoloDatasets are specified, class_to_idx must be specified"
            )
        return self
