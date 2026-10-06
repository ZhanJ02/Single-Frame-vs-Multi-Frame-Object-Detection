# Reproducibility

This document explains how to reproduce the car-only YOLO26 baseline and record future experiments. It distinguishes functionality already present in the evaluation script from controls that must still be added or recorded separately.

## 1\. Current Reproducibility Status

The script uses a fixed pretrained checkpoint, explicit annotation records, one-image-at-a-time inference, and configurable evaluation thresholds. It does not train the model or randomly sample evaluation images. With `--limit 100`, it evaluates the first 100 annotation records rather than a random subset.

The script saves selected settings in `metrics.json`, but it does **not** automatically record the random seed, package versions, GPU details, Git commit, checkpoint hash, or dataset checksum. Seed control is not implemented in the previously supplied script; the function below is an addition.

The original run's environment has not been supplied. Capturing the environment now records the current state; it should not be represented as proof of the original run's versions if packages have changed.

## 2\. Record the Code Version

Commit the evaluation script and documentation before a reproducibility run. From the repository root, record:

```cmd
git rev-parse HEAD
git status --short
```

Save the output with the run. If the working tree contains changes, either commit the relevant changes or preserve a patch:

```cmd
git diff > code\_changes.patch
```

A commit hash alone does not identify uncommitted changes.

## 3\. Record the Environment

Run these commands using the same Python environment used for evaluation:

```cmd
python --version > python\_version.txt
python -m pip freeze > requirements.txt
python -c "import platform, torch, ultralytics; print(platform.platform()); print('PyTorch:', torch.\_\_version\_\_); print('Ultralytics:', ultralytics.\_\_version\_\_); print('PyTorch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is\_available()); print('GPU:', torch.cuda.get\_device\_name(0) if torch.cuda.is\_available() else 'CPU')" > runtime\_environment.txt
nvidia-smi > gpu\_environment.txt
```

Keep copies of these files with the experiment. `nvidia-smi` applies to NVIDIA GPU runs; note CPU hardware separately for CPU runs.

Restore recorded packages in a compatible environment with:

```cmd
python -m pip install -r requirements.txt
```

A package freeze is useful but not sufficient by itself: Python version, operating system, GPU driver, CUDA compatibility, and package availability also affect reproducibility. Do not upgrade dependencies during a reproduction run.

## 4\. Identify the Model and Dataset

Record the checkpoint's SHA-256 hash:

```cmd
certutil -hashfile yolo26m.pt SHA256
```

Hash the exact annotation file used:

```cmd
certutil -hashfile "C:\\Users\\zhanj\\Single-Frame-vs-Multi-Frame-Object-Detection\\MilestoneP1\\BDDK\_Dataset\\bdd100k\\bdd100k\\labels\\bdd100k\_labels\_images\_val.json" SHA256
```

Record the dataset source, download date, extracted image directory, annotation path, split, and evaluated image count. For stronger dataset verification, retain the downloaded archive's checksum or a per-image checksum manifest.

Changing a local path does not change the experiment if it points to identical data. Pair validation images with validation labels; `--split val` is metadata and does not select those paths automatically.

## 5\. Add Seed Control

Add this function to `YOLOv26.py` and call it before constructing the model:

```python
def set\_seed(seed=42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual\_seed(seed)

    if torch.cuda.is\_available():
        torch.cuda.manual\_seed\_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
```

In `main()`, before `model = YOLO(args.model)`:

```python
set\_seed(42)
```

Also record `42` in the run metadata. These controls reduce randomness but do not guarantee bitwise-identical results across hardware, library versions, or all operations. For stricter auditing, `torch.use\_deterministic\_algorithms(True)` can be enabled; it may fail when an operation has no deterministic implementation and can affect performance.

During future training, also control data-loader worker seeds, shuffle generators, augmentation randomness, and the sequence-split procedure. Preserve temporal order inside each window.

## 6\. Preserve the Exact Run Command

First check the pipeline:

```cmd
python YOLOv26.py --split val --limit 100 --device 0 --output results\\yolo26m\_val\_smoke
```

Then run the full evaluation with explicit settings:

```cmd
python YOLOv26.py --split val --model yolo26m.pt --device 0 --imgsz 640 --conf 0.25 --ap-conf 0.001 --match-iou 0.50 --max-det 300 --output results\\yolo26m\_val\_run01
```

Store this command in a text file with the run. To capture console output in Command Prompt, append:

```cmd
> evaluation.log 2>\&1
```

Redirection stores progress in the log instead of displaying it in the terminal. Use a new output directory for each run.

## 7\. Preserve Evaluation Semantics

For this baseline, record:

|Item|Setting|
|-|-|
|Model|Pretrained YOLO26m|
|Fine-tuning|None|
|Class|Car only|
|Data split|BDD100K 100k validation|
|Full evaluation count|10,000 images|
|Image-size setting|640|
|Inference batch|1|
|F1/matrix confidence|0.25|
|Matching IoU|0.50|
|AP confidence floor|0.001|
|Maximum detections/image|300|
|F1 matching|Confidence-ordered, one-to-one|
|AP evaluator|pycocotools, car-only COCO-style AP|
|Proposed seed|42; add explicitly before future runs|

Confirm these settings against the original run files before assigning them to historical results. The AP maximum-detection setting is 300, so comparisons with protocols using 100 detections need to account for that difference.

Keep `metrics.json`, prediction JSON, per-image CSV, matrix, PR curve, environment files, model/annotation hashes, Git revision, and command together. Crowd handling and generic ignore-region behavior must remain consistent between compared runs.

## 8\. Timing and Resource Measurements

The reported 8.16 ms/image is the mean model inference time from the completed run. It is not full-pipeline latency and should not be compared directly with another model's end-to-end time.

For future comparisons, use the same machine, target frames, image settings, numerical precision, and timing scope. Warm up each model and synchronize CUDA around manual GPU timing. Report inference time and end-to-end throughput separately, along with the timed image count and repeated-run variation.

Record peak GPU memory separately; it is not currently captured by the script. Centered temporal windows need future frames, so account for buffering delay in any real-time deployment analysis.

## 9\. Fair Temporal Comparisons

Use exactly the same target frames and ground-truth cars for single-frame and multi-frame evaluation. A 100k still-image run cannot be directly compared with a different MOT sample as if the input-window length were the only change.

Keep training and validation sequences separate. If only pretrained checkpoints are compared, describe the result as a pretrained transfer comparison because architectures and pretraining data differ. To isolate temporal input, compare a temporal detector with its corresponding single-frame baseline under a matched training and evaluation protocol.

## Run Record Template

Copy this template into each experiment directory:

```text
Run ID:
Date:
Git commit:
Working tree clean / saved patch:
Script filename:
Exact command:
Python version:
Ultralytics version:
PyTorch version:
CUDA version and driver:
GPU / CPU:
Checkpoint filename:
Checkpoint SHA-256:
Dataset source and download date:
Images directory:
Annotation file:
Annotation SHA-256:
Split and image count:
Seed and seed-control implementation:
Image-size setting:
Confidence / AP confidence floor:
Matching IoU:
Maximum detections:
Numerical precision:
Timing scope and warm-up:
Peak GPU memory:
Results directory:
Known limitations:
```

