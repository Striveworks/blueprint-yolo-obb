import base64
import io
import json
import logging
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import TypeAlias

import torch
from PIL import Image, ImageDraw
from ts.torch_handler.object_detector import ObjectDetector
from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import ops

try:
    from blueprint_yolo_obb.src.data.util import convert_yolo_results
except ImportError:
    from util import convert_yolo_results


PreprocessedInput: TypeAlias = list[tuple[torch.FloatTensor | Image.Image, float]]

DEFAULT_SCORE_THRESHOLD = 0.5

logger = logging.getLogger(__name__)


class Handler(ObjectDetector):
    def __init__(self):
        super().__init__()
        self.resize_to = 640

    def _get_predictor(self, max_nms: int = 30000):
        predictor_cls = self.model._smart_load("predictor")

        def postprocess(self, preds, img, orig_imgs, **kwargs):
            """Post-processes predictions and returns a list of Results objects."""
            preds = ops.non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                self.args.classes,
                self.args.agnostic_nms,
                max_det=self.args.max_det,
                max_nms=max_nms,
                nc=len(self.model.names),
                end2end=getattr(self.model, "end2end", False),
                rotated=self.args.task == "obb",
            )

            if not isinstance(
                orig_imgs, list
            ):  # input images are a torch.Tensor, not a list
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

            return self.construct_results(preds, img, orig_imgs, **kwargs)

        predictor_cls.postprocess = postprocess
        return predictor_cls

    def initialize(self, context):
        logger.info("initializing handler")

        properties = context.system_properties

        self.device = torch.device(
            "cuda:" + str(properties["gpu_id"])
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        self.manifest = context.manifest
        model_dir = Path(properties.get("model_dir"))

        serialized_file = self.manifest["model"]["serializedFile"]
        self.model_pt_path = model_dir / serialized_file
        model_config_file_path = model_dir / "model_config.json"
        model_config = {}
        if model_config_file_path.exists():
            with model_config_file_path.open("r") as fp:
                model_config = json.load(fp)

        self.model = YOLO(self.model_pt_path, task="obb").to(self.device)

        idx_to_class = model_config.get("idx_to_class")
        if idx_to_class:
            assert isinstance(self.model.model, DetectionModel)
            self.model.model.names = {
                int(id): label for id, label in idx_to_class.items()
            }

        resize_to = model_config.get("resize_to")
        if resize_to:
            self.resize_to = resize_to

        self.score_thresholds: list[float] = []
        self.initialized = True
        logger.info("handler is initialized")

    def preprocess(self, data) -> PreprocessedInput:  # type: ignore[reportIncompatibleMethodOverride] returning a list here is valid
        image_threshold_pairs = []
        for row in data:
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                image = base64.b64decode(image)

            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
            else:
                image = torch.FloatTensor(image)

            try:
                score_threshold = float(
                    row.get("parameters", {}).get(
                        "score_threshold", DEFAULT_SCORE_THRESHOLD
                    )
                )
            except ValueError:
                score_threshold = DEFAULT_SCORE_THRESHOLD

            image_threshold_pairs.append((image, score_threshold))

        return image_threshold_pairs

    def inference(self, data: PreprocessedInput, *args, **kwargs):
        # Group consecutive images that have the same `score_threshold` together.  It may
        # often be the case that all score thresholds are the same, in which case this is
        # equivalent to just returning the value of `self.model.predict(...)`.
        return [
            model_output
            for score_threshold, image_threshold_pairs in groupby(
                data, key=itemgetter(1)
            )
            for model_output in self.model.predict(
                source=[model_inputs for model_inputs, _ in image_threshold_pairs],
                *args,
                **{**kwargs, **{"conf": score_threshold}},
                predictor=self._get_predictor(max_nms=2048),
                imgsz=self.resize_to,
                save_conf=True,
                device=self.device,
            )
        ]

    def postprocess(self, data: list[Results]) -> list[dict]:  # type: ignore[reportIncompatibleMethodOverride] returning a list here is valid
        return convert_yolo_results(self.device, data)


def prepare_data(local_data_file: str, batch_size: int, score_thresholds: list[float]):
    """
    Function to prepare data based on the desired batch size
    """
    f = open(local_data_file, "rb", buffering=0)
    read_data = f.read()
    data = []
    for i in range(batch_size):
        tmp = {}
        tmp["data"] = read_data
        tmp["parameters"] = {"score_threshold": score_thresholds[i]}
        data.append(tmp)
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        required=True,
        type=str,
        help="path to an image file (e.g., png) stored locally to run inference on",
    )
    parser.add_argument(
        "--model-file",
        required=False,
        type=str,
        help="path to a model file (e.g., yolov8n-obb.pt)",
        default="yolov8n-obb.pt",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        help="score threshold to use",
        default=0.5,
    )
    args = parser.parse_args()

    ctx = MockContext(
        model_pt_file=args.model_file,
        model_dir=".",
        model_file="",
    )

    torch.manual_seed(42 * 42)
    handler = Handler()
    handler.initialize(ctx)
    handler.context = ctx

    data = prepare_data(args.image, 1, [args.score_threshold])

    results = handler.handle(data, ctx)
    result = results[0]  # Batch size of 1
    print(results)
    img = Image.open(args.image)

    draw = ImageDraw.ImageDraw(img)
    assert type(result) is dict
    with torch.inference_mode():
        for box, cls in zip(
            result["oriented_detection_boxes"], result["detection_classes"]
        ):
            box_corners = (
                ops.xywhr2xyxyxyxy(
                    torch.tensor(
                        [
                            box["cx"] * img.width,
                            box["cy"] * img.height,
                            box["w"] * img.width,
                            box["h"] * img.height,
                            box["r"],
                        ]
                    )
                )
                .reshape(8)
                .tolist()
            )
            draw.polygon(box_corners, outline="red", width=5)
            top_left = box_corners[4:6]
            draw.text((top_left[0] + 10, top_left[1] + 10), cls, fill="red")
    img.show()
