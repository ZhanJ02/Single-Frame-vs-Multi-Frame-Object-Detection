# Single-Frame vs. Multi-Frame Object Detection

**Author:** James Zhan  
**Course:** CSCE585

## Project Objective

This project studies whether temporal information improves vehicle detection enough to justify additional inference latency, GPU-memory usage, and computational cost. The current implementation evaluates a pretrained single-frame YOLO26m baseline on cars in BDD100K. Multi-frame evaluation is planned.

## Current Implementation

* Model: pretrained `yolo26m.pt`, without BDD100K fine-tuning.
* Dataset: BDD100K 100k **validation** images and their JSON annotations.
* Category: **car only**; buses and trucks are not merged into this class.
* Inference: one image per call to avoid loading the complete dataset as a large batch.
* Evaluation: project-defined fixed-threshold precision/recall/F1 and car-only COCO-style AP.
* These are not official BDD100K leaderboard scores.

The 100k image release alone does not provide neighboring-frame windows. Temporal evaluation requires the MOT subset or corresponding source videos. Standard 100k test images lack public bounding-box annotations, so this baseline uses validation.

## Suggested Code Organization

```text
MilestoneP1/
    YOLOv26.py
    README.md
    REPRODUCIBILITY.md
    requirements.txt                 # Generate from the working environment
    BDDK\_Dataset/                    # Local; exclude from Git
    results/                         # Separate directory for each run
```

The script separates annotation loading, IoU calculation, one-to-one matching, inference, COCO-style AP calculation, and plotting. Function names may differ between earlier script revisions.

## Installation

Use Windows Command Prompt in the folder containing `YOLOv26.py`:

```cmd
cd /d "C:\\Users\\zhanj\\Single-Frame-vs-Multi-Frame-Object-Detection\\MilestoneP1"
python -m pip install -U ultralytics pycocotools matplotlib numpy
```

For a compatible NVIDIA GPU, PyTorch must have CUDA support. Check your installation:

```cmd
python -c "import torch; print(torch.\_\_version\_\_); print('CUDA available:', torch.cuda.is\_available())"
```

After the environment works, record versions as described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The installation command above is initial setup; reproduce an existing run using its recorded package versions.

## Dataset Configuration

Download and extract the BDD100K image dataset and annotations. The current local paths, based on the reported folder structure, are:

```python
DEFAULT\_IMAGES = Path(
    r"C:\\Users\\zhanj\\Single-Frame-vs-Multi-Frame-Object-Detection"
    r"\\MilestoneP1\\BDDK\_Dataset\\bdd100k\\bdd100k\\images\\100k\\val"
)

DEFAULT\_LABELS = Path(
    r"C:\\Users\\zhanj\\Single-Frame-vs-Multi-Frame-Object-Detection"
    r"\\MilestoneP1\\BDDK\_Dataset\\bdd100k\\bdd100k"
    r"\\labels\\bdd100k\_labels\_images\_val.json"
)
```

Confirm these paths exist. Point the labels argument directly to the **validation JSON file**, rather than the parent labels folder, to avoid accidentally reading training annotations.

Paths can also be supplied through `--images` and `--labels`. Changing `--split` only changes the run metadata; it does not automatically change the image or annotation paths.

## Run Instructions

Check the first 100 annotated images:

```cmd
python YOLOv26.py --split val --limit 100 --device 0 --output results\\yolo26m\_val\_smoke
```

Evaluate all annotated validation images with explicit settings:

```cmd
python YOLOv26.py --split val --model yolo26m.pt --device 0 --imgsz 640 --conf 0.25 --ap-conf 0.001 --match-iou 0.50 --max-det 300 --output results\\yolo26m\_val\_run01
```

For CPU inference, replace `--device 0` with `--device cpu`. The first model load may download the checkpoint and requires internet access.

Use a new output directory for every experiment; reusing an output directory can overwrite earlier files. A run using `--limit` is a partial evaluation, not the full baseline.

## Evaluation Settings

|Setting|Baseline value|Purpose|
|-|-:|-|
|Image-size setting|640|Model resizing/letterboxing setting|
|Inference batch|1|Bound GPU memory usage|
|Confidence for F1/matrix|0.25|Select predictions for fixed-threshold counts|
|Matching IoU|0.50|Match each prediction to at most one regular ground-truth car|
|Confidence floor for AP|0.001|Retain low-confidence detections for ranking|
|Maximum detections/image|300|Prediction and AP evaluation cap|

F1 matching is confidence-ordered and one-to-one. Crowd annotations, when present, receive special handling. Generic ignore annotations are rejected by the supplied evaluator rather than silently mis-scored. Precision and recall use the fixed confidence threshold; AP integrates over prediction rankings and IoU thresholds.

True negatives are undefined here, so ordinary classification accuracy is not reported.

## Output Files

|File|Contents|
|-|-|
|`metrics.json`|Aggregate metrics and selected evaluation settings|
|`confusion\_matrix.png`|Car/background detection matrix|
|`confusion\_matrix.csv`|Matrix values|
|`precision\_recall\_curve.png`|Interpolated car PR curve at IoU 0.50|
|`per\_image.csv`|TP, FP, FN, and ignored prediction counts per image|
|`predictions\_coco.json`|Predictions with boxes and scores in COCO-style format|

Matrix rows are predictions; columns are ground truth:

|Prediction / Ground truth|Car|Background|
|-|-|-|
|Car|TP|FP|
|Background|FN|N/A|

## Observed Baseline Results

The completed run reported:

|Metric|Result|
|-|-:|
|Images|10,000|
|TP|52,130|
|FP|9,609|
|FN|50,376|
|Precision|84.44%|
|Recall|50.86%|
|F1|63.48%|
|AP50|64.29%|
|AP75|38.31%|
|AP50–95|38.41%|
|Mean model inference time|8.16 ms/image|

Results were transcribed from the completed-run screenshot. Confirm exact run settings using the original `metrics.json` and command. Inference timing excludes image loading and other pipeline overhead and includes cold-start effects; it is not end-to-end latency. No temporal-model results or measured successful-run peak memory are available yet.

## Troubleshooting

* **Missing pycocotools:** run `python -m pip install pycocotools`.
* **Missing yolo26m.pt:** update Ultralytics or pass a local checkpoint path with `--model`.
* **Missing image:** confirm validation images are paired with validation annotations.
* **CUDA out of memory:** use the revised one-image-at-a-time script and close other GPU workloads. Passing all images as a list can create a large batch.
* **CUDA unavailable:** use `--device cpu` or install a compatible CUDA-enabled PyTorch build.

## Versioning and Next Steps

Track code and documentation with Git. Exclude datasets, caches, environments, and model weights; record checkpoint hashes instead. Preserve the code commit and environment for each run.

Next, evaluate temporal models on identical target frames, analyze errors by size and scene condition, and measure end-to-end latency and peak GPU memory. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the run-record procedure.

## References

* [BDD100K dataset paper](https://arxiv.org/abs/1805.04687)
* [Ultralytics YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/)
* [TransVOD official implementation](https://github.com/SJTU-LuHe/TransVOD)

