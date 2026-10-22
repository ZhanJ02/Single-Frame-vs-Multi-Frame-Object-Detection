from pathlib import Path
import re

import cv2
import torch
from ultralytics import YOLO


# ================= EDIT SETTINGS HERE =================

SEQUENCE_DIR = Path(
    r"C:\Users\zhanj\Single-Frame-vs-Multi-Frame-Object-Detection"
    r"\MilestoneP1_or_Milestone2\DETRAC\DETRAC-Images\MVI_40855"
)

OUTPUT_VIDEO = (
    SEQUENCE_DIR.parent.parent
    / "YOLOv26_Videos"
    / "MVI_40855_detections.mp4"
)

MODEL_WEIGHTS = "yolo26m.pt"
CONFIDENCE = 0.25
IMAGE_SIZE = 640

# Playback speed; lower values produce slower playback.
OUTPUT_FPS = 25

# COCO classes: car=2, bus=5, truck=7.
VEHICLE_CLASSES = [2, 5, 7]

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ======================================================


def numerical_sort_key(path):
    """Sort frames as img1, img2, ..., img10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def main():
    if not SEQUENCE_DIR.is_dir():
        raise FileNotFoundError(
            f"Sequence folder does not exist:\n{SEQUENCE_DIR}"
        )

    frames = sorted(
        [
            path
            for path in SEQUENCE_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ],
        key=numerical_sort_key,
    )

    if not frames:
        raise RuntimeError(f"No images found in:\n{SEQUENCE_DIR}")

    first_frame = cv2.imread(str(frames[0]))
    if first_frame is None:
        raise RuntimeError(f"Cannot read first frame: {frames[0]}")

    height, width = first_frame.shape[:2]
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_WEIGHTS)

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        OUTPUT_FPS,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video:\n{OUTPUT_VIDEO}")

    print(f"Sequence: {SEQUENCE_DIR.name}")
    print(f"Total frames: {len(frames)}")
    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_VIDEO}")

    written = 0

    try:
        for index, frame_path in enumerate(frames, start=1):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Cannot read frame: {frame_path}")

            # VideoWriter requires a consistent frame size.
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))

            result = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                classes=VEHICLE_CLASSES,
                device=DEVICE,
                max_det=300,
                verbose=False,
            )[0]

            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        1,
                    )

                    cv2.putText(
                        frame,
                        "vehicle",
                        (max(0, x1), max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

            cv2.putText(
                frame,
                f"{SEQUENCE_DIR.name} | Frame {index}/{len(frames)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            writer.write(frame)
            written += 1

            if index % 100 == 0 or index == len(frames):
                print(f"Processed {index}/{len(frames)} frames")

    finally:
        writer.release()

    print(f"\nSaved {written} frames to:\n{OUTPUT_VIDEO}")
    print(f"Video duration: {written / OUTPUT_FPS:.2f} seconds")


if __name__ == "__main__":
    main()