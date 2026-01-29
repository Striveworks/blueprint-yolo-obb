# blueprint-yolo-obb

The YOLO OBB Blueprint is for oriented object detection using Ultralytics YOLO. This Blueprint wraps the YOLO v8 model to train models that can detect non-axis aligned objects (oriented bounding boxes).

## Overview

This Blueprint provides a complete training pipeline for oriented object detection tasks. It integrates with the Blueprint Toolkit to locally manage Training Runs, checkpoints, metrics, and datasets.

## Features

- **Oriented Object Detection:** Train YOLO models to detect rotated objects using oriented bounding boxes
- **Checkpoint Management:** Automatic checkpoint saving and resuming
- **Metrics Tracking:** Integration with training metrics and evaluation
- **Model Export:** Exports trained models in TorchServe format for deployment

## Components

### Training (`train.py`)

The main training module that:
- Initializes and configures YOLO OBB models
- Manages training loops with custom callbacks
- Handles dataset fetching and conversion
- Saves checkpoints and models
- Tracks metrics and progress

Key classes:
- `CallbackYOLO`: Extended YOLO class with `run_context` integration
- `OBBTrainerWithCallbacks`: Custom trainer with `run_context` callbacks
- `Callbacks`: `run_context` callbacks for checkpoint saving and metric tracking

### Configuration (`config.py`)

Pydantic-based configuration management:
- `Config`: Main configuration model with validation

Supports configuration options including:
- Model initialization (from checkpoint or pretrained model)
- Training hyperparameters (epochs, batch size, image size)
- Class mappings

### Utilities

- `data/util.py`: Converts YOLO results to standardized oriented bounding box format
- `yolo_yaml.py`: Generates YOLO dataset configuration `yaml` files

## Usage

This Blueprint is designed to be run as part of a training system that provides a `RunContext`. The main entry point is the `train()` function that accepts a `RunContext` and executes the training loop. This library leverages a local run context but can be extended with custom `RunContext` implementations.

The following example demonstrates how to start training from a local_run_context
where `config` is the training run configuration expected by the blueprint below:
```python
  with open(config_path) as fp:
      config = json.load(fp)
  run_context = local_run_context(
      run_id=<your Run ID>, config=config, base_dir=<your base dir>
  )

  with run_context:
      train(run_context)
```

Example configuration structure:
```json
{
  "datasets": {
    "train_data": [
      {
        "snapshot_id": 123,
        "split": "train"
      }
    ],
    "val_data": [
      {
        "snapshot_id": 123,
        "split": "val"
      }
    ]
  },
  "class_to_idx": null,
  "idx_to_class": null,
  "start_model": null,
  "ckpt_epoch_period": 5,
  "log_level": "INFO",
  "epochs": 100,
  "batch_size": 16,
  "resize_to": 640
}
```
