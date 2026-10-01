"""Select and draw one correct and one incorrect fruit prediction."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

from fruit_harvest.decision import HarvestPolicy
from fruit_harvest.metrics import box_iou, ground_truth_for, match_detections
from fruit_harvest.model import load_detector, predict_image


NAMES = {"unripe": "não maduro", "semi_ripe": "intermediário", "ripe": "maduro"}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


@dataclass(frozen=True)
class Example:
    image: Path
    truth: str
    truth_box: tuple[float, float, float, float]
    prediction: str
    prediction_box: tuple[float, float, float, float]
    confidence: float
    iou: float

    def as_dict(self) -> dict:
        return {
            "image": self.image.name,
            "truth": self.truth,
            "truth_bbox_xyxy": self.truth_box,
            "prediction": self.prediction,
            "prediction_bbox_xyxy": self.prediction_box,
            "confidence": self.confidence,
            "iou": self.iou,
        }


def select_examples(images: list[Path], weights: Path, image_size: int,
                    min_confidence: float) -> tuple[Example, Example]:
    detector = load_detector(weights)
    correct = incorrect = None
    correct_score = incorrect_score = (-1.0, -1.0)
    for image in images:
        result = predict_image(detector, image, HarvestPolicy(), image_size, min_confidence)
        truths = ground_truth_for(image, result.width, result.height)
        match = match_detections(truths, result.detections, 0.5)
        for truth_index, prediction_index in match.pairs:
            truth, truth_box = truths[truth_index]
            prediction = result.detections[prediction_index]
            item = Example(image, truth, truth_box, prediction.class_name,
                           prediction.bbox_xyxy, prediction.confidence,
                           box_iou(truth_box, prediction.bbox_xyxy))
            if truth == prediction.class_name:
                score = (item.confidence, item.iou)
                if score > correct_score:
                    correct, correct_score = item, score
            else:
                # A missed ripe fruit is more informative for this project's goal.
                score = (float(truth == "ripe"), item.confidence)
                if score > incorrect_score:
                    incorrect, incorrect_score = item, score
    if correct is None or incorrect is None:
        raise RuntimeError("Need at least one correctly and one incorrectly classified matched fruit")
    return correct, incorrect


def draw_example(example: Example, target: Path, correct: bool) -> None:
    source = Image.open(example.image).convert("RGB")
    union = (
        min(example.truth_box[0], example.prediction_box[0]),
        min(example.truth_box[1], example.prediction_box[1]),
        max(example.truth_box[2], example.prediction_box[2]),
        max(example.truth_box[3], example.prediction_box[3]),
    )
    center_x, center_y = (union[0] + union[2]) / 2, (union[1] + union[3]) / 2
    crop_width = min(source.width, max(420, (union[2] - union[0]) * 5))
    crop_height = min(source.height, max(300, (union[3] - union[1]) * 5))
    left = max(0, min(source.width - crop_width, center_x - crop_width / 2))
    top = max(0, min(source.height - crop_height, center_y - crop_height / 2))
    cropped = source.crop((int(left), int(top), int(left + crop_width), int(top + crop_height)))
    fitted = ImageOps.contain(cropped, (920, 520), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (960, 660), "#F8FAF9")
    origin_x, origin_y = (960 - fitted.width) // 2, 80 + (520 - fitted.height) // 2
    canvas.paste(fitted, (origin_x, origin_y))
    draw = ImageDraw.Draw(canvas)
    heading = ImageFont.truetype(FONT, 30)
    body = ImageFont.truetype(FONT, 19)
    status = "ACERTO" if correct else "ERRO"
    status_color = "#147D64" if correct else "#BA3A3A"
    draw.text((22, 15), status, font=heading, fill=status_color)
    draw.text((190, 21), f"Real: {NAMES[example.truth]} | Previsto: {NAMES[example.prediction]}",
              font=body, fill="#183E3A")

    scale = fitted.width / cropped.width
    def mapped(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return tuple(
            origin_x + (value - left) * scale if index % 2 == 0
            else origin_y + (value - top) * scale
            for index, value in enumerate(box)
        )

    draw.rectangle(mapped(example.truth_box), outline="#15AE6B", width=5)
    draw.rectangle(mapped(example.prediction_box), outline="#E26443", width=5)
    draw.text((22, 615), "Verde: anotação real | Laranja: previsão", font=body, fill="#183E3A")
    draw.text((630, 615), f"Conf.: {example.confidence:.2f} | IoU: {example.iou:.2f}",
              font=body, fill="#183E3A")
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Processed three-class dataset.yaml")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--credit", required=True, help="Image creator, dataset, and redistribution license")
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--license-url", required=True)
    parser.add_argument("--output", type=Path, default=Path("output/examples"))
    args = parser.parse_args()
    descriptor = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    images_dir = Path(descriptor["path"]) / descriptor[args.split]
    images = sorted(image for image in images_dir.iterdir() if image.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise ValueError(f"No images in {images_dir}")
    correct, incorrect = select_examples(images, args.weights, args.image_size, args.confidence)
    draw_example(correct, args.output / "correct.jpg", True)
    draw_example(incorrect, args.output / "incorrect.jpg", False)
    (args.output / "examples.json").write_text(json.dumps({
        "split": args.split,
        "image_credit": args.credit,
        "source_url": args.source_url,
        "license_url": args.license_url,
        "modifications": "Cropped and overlaid ground-truth/prediction boxes and text",
        "confidence_floor": args.confidence,
        "matching_iou": 0.5,
        "correct": correct.as_dict(),
        "incorrect": incorrect.as_dict(),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output / "ATTRIBUTION.md").write_text(
        "# Image attribution\n\n"
        f"`correct.jpg` and `incorrect.jpg` use images from [{args.credit}]({args.source_url}). "
        f"The source is licensed under [CC BY 4.0]({args.license_url}). "
        "The images were cropped and annotated with ground-truth boxes, model predictions, "
        "labels, confidence, and IoU for this project. The authors of the source dataset "
        "do not endorse this project.\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
