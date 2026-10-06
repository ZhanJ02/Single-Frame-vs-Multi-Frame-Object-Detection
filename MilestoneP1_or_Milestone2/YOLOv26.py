import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


# Paths
DEFAULT_IMAGES = Path(
    r"C:\Users\zhanj\Single-Frame-vs-Multi-Frame-Object-Detection\MilestoneP1_or_Milestone2\BDDK_Dataset\bdd100k\bdd100k\images\100k\val"
)

DEFAULT_LABELS = Path(
    r"C:\Users\zhanj\Single-Frame-vs-Multi-Frame-Object-Detection\MilestoneP1_or_Milestone2\BDDK_Dataset\bdd100k_labels_release\bdd100k\labels"
)

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "bdd100k_yolo26_val_results"


def iou_matrix(predicted_boxes, ground_truth_boxes):
    """Calculate IoU between every prediction and ground-truth box."""

    predicted_boxes = np.asarray(
        predicted_boxes, dtype=float
    ).reshape(-1, 4)

    ground_truth_boxes = np.asarray(
        ground_truth_boxes, dtype=float
    ).reshape(-1, 4)

    intersection_top_left = np.maximum(
        predicted_boxes[:, None, :2],
        ground_truth_boxes[None, :, :2],
    )

    intersection_bottom_right = np.minimum(
        predicted_boxes[:, None, 2:],
        ground_truth_boxes[None, :, 2:],
    )

    intersection = np.maximum(
        0,
        intersection_bottom_right - intersection_top_left,
    ).prod(axis=2)

    prediction_area = np.maximum(
        0,
        predicted_boxes[:, 2:] - predicted_boxes[:, :2],
    ).prod(axis=1)

    ground_truth_area = np.maximum(
        0,
        ground_truth_boxes[:, 2:] - ground_truth_boxes[:, :2],
    ).prod(axis=1)

    union = (
        prediction_area[:, None]
        + ground_truth_area[None, :]
        - intersection
    )

    return intersection / np.maximum(union, 1e-12)


def match_counts(boxes, scores, ground_truth, crowds, threshold):
    """
    Match predictions to ground truth in descending confidence order.

    Each regular ground-truth box can match only one prediction.
    Unmatched predictions overlapping crowd regions are ignored.
    """

    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    ground_truth = np.asarray(
        ground_truth, dtype=float
    ).reshape(-1, 4)

    used_ground_truth = set()
    tp = 0
    fp = 0
    ignored = 0

    overlaps = iou_matrix(boxes, ground_truth)

    prediction_order = np.argsort(
        -np.asarray(scores),
        kind="stable",
    )

    for prediction_index in prediction_order:
        available = [
            index
            for index in range(len(ground_truth))
            if index not in used_ground_truth
        ]

        if available:
            best_match = max(
                available,
                key=lambda index: overlaps[prediction_index, index],
            )

            if overlaps[prediction_index, best_match] >= threshold:
                used_ground_truth.add(best_match)
                tp += 1
                continue

        absorbed_by_crowd = False

        for crowd in crowds:
            crowd = np.asarray(crowd, dtype=float)

            intersection = np.maximum(
                0,
                np.minimum(boxes[prediction_index, 2:], crowd[2:])
                - np.maximum(boxes[prediction_index, :2], crowd[:2]),
            ).prod()

            prediction_area = np.maximum(
                0,
                boxes[prediction_index, 2:]
                - boxes[prediction_index, :2],
            ).prod()

            # COCO crowd matching uses intersection / prediction area.
            overlap = intersection / max(prediction_area, 1e-12)

            if overlap >= threshold:
                absorbed_by_crowd = True
                break

        if absorbed_by_crowd:
            ignored += 1
        else:
            fp += 1

    fn = len(ground_truth) - len(used_ground_truth)

    return tp, fp, fn, ignored


def annotation_records(label_path):
    """Read native BDD100K JSON annotations."""

    if label_path.is_dir():
        files = sorted(label_path.rglob("*.json"))
    else:
        files = [label_path]

    if not files:
        raise ValueError("No annotation JSON files were found.")

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))

        if not isinstance(data, list):
            raise ValueError(
                f"{path}: expected a native BDD100K JSON list."
            )

        for frame in data:
            if "labels" not in frame:
                raise ValueError(
                    f"{path}: a frame is missing labels. "
                    "Unlabeled images cannot be used for evaluation."
                )

            yield path.stem, frame


def load_samples(images_path, labels_path, limit):
    """Match annotated frames to image files and collect car boxes."""

    samples = []
    seen = set()

    for stem, frame in annotation_records(labels_path):
        name = frame["name"]
        video = frame.get("videoName", stem)

        candidates = [
            images_path / video / name,
            images_path / name,
        ]

        image = next(
            (path for path in candidates if path.is_file()),
            None,
        )

        if image is None:
            raise FileNotFoundError(
                f"Could not find image: {name}\n"
                f"Searched: {candidates}"
            )

        image_key = str(image.resolve())

        if image_key in seen:
            raise ValueError(
                f"Duplicate annotated image: {image_key}"
            )

        seen.add(image_key)
        labels = []

        for label in frame["labels"]:
            if label.get("category") != "car":
                continue

            box = label.get("box2d")

            if not box:
                raise ValueError(
                    f"Car annotation without box2d in {name}"
                )

            xyxy = [
                float(box[key])
                for key in ("x1", "y1", "x2", "y2")
            ]

            if not all(np.isfinite(xyxy)):
                raise ValueError(
                    f"Non-finite bounding box in {name}: {xyxy}"
                )

            if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                raise ValueError(
                    f"Invalid bounding box in {name}: {xyxy}"
                )

            attributes = label.get("attributes", {})

            if (
                label.get("ignore", False)
                or attributes.get("ignore", False)
            ):
                raise ValueError(
                    "Generic ignore regions need a "
                    "dataset-specific evaluator."
                )

            crowd = bool(
                attributes.get("crowd", False)
                or label.get("iscrowd", False)
            )

            labels.append((xyxy, crowd))

        # Include explicitly annotated images with no cars.
        # Predictions on these images can count as false positives.
        samples.append((image, labels))

        if limit > 0 and len(samples) >= limit:
            break

    has_regular_cars = any(
        not crowd
        for _, labels in samples
        for _, crowd in labels
    )

    if not samples or not has_regular_cars:
        raise ValueError(
            "No labeled, non-crowd cars found. "
            "Check your image and annotation paths."
        )

    return samples


def valid_mean(values):
    values = np.asarray(values)
    valid = values[values >= 0]
    return float(valid.mean()) if len(valid) else 0.0


def save_plots(output_path, tp, fp, fn, conf, match_iou, curve, ap50):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Rows: predicted class. Columns: ground-truth class.
    # Bottom-right is undefined for object detection.
    matrix = np.array([
        [tp, fp],
        [fn, 0],
    ])

    with (output_path / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["Prediction / Ground truth", "Car", "Background"])
        writer.writerow(["Car", tp, fp])
        writer.writerow(["Background", fn, "N/A"])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")

    for row in range(2):
        for column in range(2):
            text = (
                "N/A"
                if row == 1 and column == 1
                else str(matrix[row, column])
            )

            ax.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                color="black",
            )

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Car", "Background"],
        yticklabels=["Car", "Background"],
        xlabel="Ground truth",
        ylabel="Prediction",
        title=f"Car detection: confidence={conf}, IoU={match_iou}",
    )

    fig.tight_layout()
    fig.savefig(
        output_path / "confusion_matrix.png",
        dpi=200,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        np.linspace(0, 1, len(curve)),
        curve,
        linewidth=2,
    )

    ax.set(
        xlabel="Recall",
        ylabel="Interpolated precision",
        xlim=(0, 1),
        ylim=(0, 1),
        title=f"Car precision-recall at IoU=0.50: AP={ap50:.4f}",
    )

    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        output_path / "precision_recall_curve.png",
        dpi=200,
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_IMAGES,
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
    )
    parser.add_argument("--model", default="yolo26m.pt")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--device",
        default=None,
        help="0 for the first CUDA GPU, or cpu",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for F1 and confusion matrix",
    )
    parser.add_argument(
        "--ap-conf",
        type=float,
        default=0.001,
        help="Low confidence cutoff for AP calculation",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.50,
    )
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 evaluates all annotated images",
    )

    args = parser.parse_args()

    if not args.images.is_dir():
        parser.error(
            f"Image directory does not exist:\n{args.images}"
        )

    if not args.labels.exists():
        parser.error(
            f"Annotation path does not exist:\n{args.labels}"
        )

    if not (
        0 <= args.ap_conf <= args.conf <= 1
        and 0 < args.match_iou <= 1
    ):
        parser.error(
            "Require 0 <= ap-conf <= conf <= 1 "
            "and 0 < match-iou <= 1."
        )

    if args.limit < 0 or args.max_det < 10:
        parser.error(
            "limit must be nonnegative and max-det must be at least 10."
        )

    from ultralytics import YOLO
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    print(f"Images: {args.images}")
    print(f"Labels: {args.labels}")
    print("Reading annotations...", flush=True)

    samples = load_samples(
        args.images,
        args.labels,
        args.limit,
    )

    print(f"Evaluating {len(samples)} images.")
    print(f"Loading model: {args.model}", flush=True)

    model = YOLO(args.model)

    car_ids = [
        int(class_id)
        for class_id, name in model.names.items()
        if name.lower() == "car"
    ]

    if len(car_ids) != 1:
        raise ValueError(
            "The checkpoint must contain one class named 'car'."
        )

    args.output.mkdir(parents=True, exist_ok=True)

    dataset = {
        "info": {},
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "car"}],
    }

    detections = []
    per_image = []
    totals = np.zeros(4, dtype=np.int64)
    inference_times = []

    start = time.perf_counter()

    # IMPORTANT:
    # Pass one image path per call.
    # Passing all paths as a Python list can create a huge batch.
    for image_id, (image_path, labels) in enumerate(samples, 1):
        result = model.predict(
            source=str(image_path),
            batch=1,
            classes=car_ids,
            conf=args.ap_conf,
            imgsz=args.imgsz,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )[0]

        height, width = result.orig_shape

        dataset["images"].append({
            "id": image_id,
            "file_name": str(image_path),
            "height": height,
            "width": width,
        })

        ground_truth = []
        crowds = []

        for box, crowd in labels:
            x1, y1, x2, y2 = box

            dataset["annotations"].append({
                "id": len(dataset["annotations"]) + 1,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": int(crowd),
            })

            if crowd:
                crowds.append(box)
            else:
                ground_truth.append(box)

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        # Keep low-confidence predictions for AP.
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(float, box)

            detections.append({
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })

        # Use the fixed confidence threshold for F1 and the matrix.
        keep = scores >= args.conf

        counts = match_counts(
            boxes[keep],
            scores[keep],
            ground_truth,
            crowds,
            args.match_iou,
        )

        totals += counts
        per_image.append([str(image_path), *counts])
        inference_times.append(float(result.speed["inference"]))

        # Release this image's result before processing the next image.
        del result

        if image_id == 1 or image_id % 100 == 0:
            elapsed = time.perf_counter() - start
            rate = image_id / elapsed if elapsed > 0 else 0.0

            print(
                f"Evaluated {image_id}/{len(samples)} images "
                f"| Pipeline speed: {rate:.2f} images/sec",
                flush=True,
            )

    elapsed = time.perf_counter() - start
    tp, fp, fn, ignored = map(int, totals)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn
        else 0.0
    )

    print("Computing COCO-style AP...", flush=True)

    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()

    if detections:
        evaluator = COCOeval(
            coco,
            coco.loadRes(detections),
            "bbox",
        )

        evaluator.params.maxDets = [1, 10, args.max_det]
        evaluator.evaluate()
        evaluator.accumulate()

        # Dimensions:
        # IoU threshold, recall threshold, class, area, max detections.
        values = evaluator.eval["precision"][:, :, 0, 0, -1]

        ap = valid_mean(values)
        ap50 = valid_mean(values[0])
        ap75 = valid_mean(values[5])

        average_recall = valid_mean(
            evaluator.eval["recall"][:, 0, 0, -1]
        )

        curve = np.maximum(values[0], 0).tolist()

    else:
        ap = 0.0
        ap50 = 0.0
        ap75 = 0.0
        average_recall = 0.0
        curve = [0.0] * 101

    summary = {
        "model": args.model,
        "split": args.split,
        "class": "car",
        "images": len(samples),
        "partial_run": bool(args.limit),
        "confidence": args.conf,
        "matching_iou": args.match_iou,
        "ap_confidence_floor": args.ap_conf,
        "max_detections_per_image": args.max_det,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "ignored_crowd_predictions": ignored,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "AP50": ap50,
        "AP75": ap75,
        "AP50_95": ap,
        "AR50_95": average_recall,
        "mean_inference_ms": float(np.mean(inference_times)),
        "pipeline_images_per_second": len(samples) / elapsed,
        "note": (
            "Car-only COCO-style AP. Fixed-threshold F1 uses "
            "confidence-ordered, one-to-one matching. "
            "TN and classification accuracy are undefined. "
            "Timing includes cold start and excludes AP calculation. "
            "This is not an official BDD100K benchmark score."
        ),
    }

    (args.output / "metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    (args.output / "predictions_coco.json").write_text(
        json.dumps(detections),
        encoding="utf-8",
    )

    with (args.output / "per_image.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "TP", "FP", "FN", "ignored"])
        writer.writerows(per_image)

    save_plots(
        args.output,
        tp,
        fp,
        fn,
        args.conf,
        args.match_iou,
        curve,
        ap50,
    )

    print("\nResults")
    print(f"Images:       {len(samples)}")
    print(f"TP:           {tp}")
    print(f"FP:           {fp}")
    print(f"FN:           {fn}")
    print(f"Precision:    {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:       {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1:           {f1:.4f} ({f1 * 100:.2f}%)")
    print(f"AP50:         {ap50:.4f} ({ap50 * 100:.2f}%)")
    print(f"AP75:         {ap75:.4f} ({ap75 * 100:.2f}%)")
    print(f"AP50-95:      {ap:.4f} ({ap * 100:.2f}%)")
    print(
        f"Inference:    "
        f"{summary['mean_inference_ms']:.2f} ms/image"
    )
    print(f"\nResults saved to:\n{args.output.resolve()}")


if __name__ == "__main__":
    main()