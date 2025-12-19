from pathlib import Path

from jinja2 import Environment

_TEMPLATE_STR = """# This file was copied from https://github.com/ultralytics/ultralytics/blob/2d332a1/ultralytics/cfg/datasets/dota8.yaml#L34-L35

# Ultralytics YOLO 🚀, AGPL-3.0 license
# Example usage: yolo train model=yolov8n-obb.pt data=dota8.yaml
# parent
# ├── ultralytics
# └── datasets
#     └── dota8  ← downloads here (1MB)

# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: {{ path }} # dataset root dir
train: images/train # train images (relative to 'path') 4 images
val: images/val # val images (relative to 'path') 4 images

names:
{%- for idx, class in idx_to_class.items() %}
  {{ idx }}: {{ class -}}
{% endfor %}

"""
_TEMPLATE = Environment().from_string(_TEMPLATE_STR)


def get_yolo_file(root_path: str | Path, idx_to_class: dict[int, str]) -> str:
    return _TEMPLATE.render(path=str(root_path), idx_to_class=idx_to_class)
