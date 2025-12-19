from blueprint_yolo_obb.src.yolo_yaml import get_yolo_file


def test_get_yolo_file():
    file = get_yolo_file("/root/path", {0: "cat", 1: "dog"})
    assert (
        file
        == """# This file was copied from https://github.com/ultralytics/ultralytics/blob/2d332a1/ultralytics/cfg/datasets/dota8.yaml#L34-L35

# Ultralytics YOLO 🚀, AGPL-3.0 license
# Example usage: yolo train model=yolov8n-obb.pt data=dota8.yaml
# parent
# ├── ultralytics
# └── datasets
#     └── dota8  ← downloads here (1MB)

# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: /root/path # dataset root dir
train: images/train # train images (relative to 'path') 4 images
val: images/val # val images (relative to 'path') 4 images

names:
  0: cat
  1: dog
"""
    )
