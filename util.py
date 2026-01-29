import logging

import torch
from ultralytics.engine.results import Results

logger = logging.getLogger(__name__)


def convert_yolo_results(device, data: list[Results]) -> list[dict]:
    results_json = []
    for result in data:
        result_json = {
            "detection_classes": [],
            "detection_scores": [],
            "oriented_detection_boxes": [],
        }
        if not result.obb:
            results_json.append(result_json)
            continue
        for box in result.obb:
            try:
                score = box.data[0][5].item()
            except IndexError:
                logger.warning(f"cannot find detection score in data: {box.data}")
                score = 0.0

            xywhr = box.xywhr[0].to(device)
            orig_height = box.orig_shape[0]
            orig_width = box.orig_shape[1]
            xywhr_scaled = xywhr / torch.tensor(
                [orig_width, orig_height, orig_width, orig_height, 1],
                device=device,
            )
            result_json["oriented_detection_boxes"].append(
                {
                    "cx": float(xywhr_scaled[0].item()),
                    "cy": float(xywhr_scaled[1].item()),
                    "w": float(xywhr_scaled[2].item()),
                    "h": float(xywhr_scaled[3].item()),
                    "r": float(xywhr_scaled[4].item()),
                }
            )
            result_json["detection_classes"].append(result.names[box.cls.item()])
            result_json["detection_scores"].append(score)

        results_json.append(result_json)

    return results_json
