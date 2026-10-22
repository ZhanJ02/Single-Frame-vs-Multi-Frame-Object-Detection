import argparse
import csv
import json
from pathlib import Path
import time
import re
import platform
import numpy as np

# Place this script in MilestoneP1_or_Milestone2, beside the DETRAC folder.
# Edit these constants if your dataset is somewhere else.
DEFAULT_ROOT = Path(__file__).resolve().parent / "DETRAC"
DEFAULT_IMAGES = DEFAULT_ROOT / "DETRAC-Images"
DEFAULT_LABELS = DEFAULT_ROOT / "DETRAC-Test-Annotations-MAT"
DEFAULT_PREFIX = "MVI_40"


def iou_matrix(pred, gt):
    pred = np.asarray(pred, dtype=float).reshape(-1, 4)
    gt = np.asarray(gt, dtype=float).reshape(-1, 4)
    intersection = np.maximum(0, np.minimum(pred[:, None, 2:], gt[None, :, 2:])
                              - np.maximum(pred[:, None, :2], gt[None, :, :2])).prod(2)
    pa = np.maximum(0, pred[:, 2:] - pred[:, :2]).prod(1)
    ga = np.maximum(0, gt[:, 2:] - gt[:, :2]).prod(1)
    return intersection / np.maximum(pa[:, None] + ga[None, :] - intersection, 1e-12)


def match_counts(boxes, scores, gt, crowds, threshold):
    """Confidence-ordered, one-to-one matching; crowds absorb unmatched detections."""
    boxes = np.asarray(boxes).reshape(-1, 4)
    gt = np.asarray(gt).reshape(-1, 4)
    used = set()
    tp = fp = ignored = 0
    overlaps = iou_matrix(boxes, gt)
    for pi in np.argsort(-np.asarray(scores), kind='stable'):
        available = [j for j in range(len(gt)) if j not in used]
        if available:
            best = max(available, key=lambda j: overlaps[pi, j])
            if overlaps[pi, best] >= threshold:
                used.add(best)
                tp += 1
                continue
        # COCO crowd matching uses intersection / detection area.
        absorbed = False
        for crowd in crowds:
            intersection = np.maximum(0, np.minimum(boxes[pi, 2:], crowd[2:])
                                      - np.maximum(boxes[pi, :2], crowd[:2])).prod()
            area = np.maximum(0, boxes[pi, 2:] - boxes[pi, :2]).prod()
            if intersection / max(area, 1e-12) >= threshold:
                absorbed = True
                break
        if absorbed:
            ignored += 1
        else:
            fp += 1
    return tp, fp, len(gt) - len(used), ignored


def load_detrac_mat(path):
    """Standard DETRAC gtInfo: X=center x, Y=bottom y, W/H=box size.

    Converts one-based MATLAB pixel origins to zero-based xyxy.
    Reference: cheind/py-motmetrics motmetrics/io.py load_detrac_mat.
    """
    from scipy.io import loadmat
    from scipy.sparse import issparse
    try:
        data = loadmat(path, struct_as_record=False, squeeze_me=False)
    except NotImplementedError as exc:
        raise ValueError(f"{path}: MATLAB v7.3/HDF5 is not supported. Export as -v7 MAT.") from exc
    if "gtInfo" not in data:
        raise ValueError(f"{path}: expected gtInfo, found {[k for k in data if not k.startswith('__')]}")
    info = data["gtInfo"].item()
    fields = getattr(info, "_fieldnames", [])
    names = {name.lower(): name for name in fields}
    required = ("x", "y", "w", "h", "framenums")
    if any(name not in names for name in required):
        raise ValueError(f"{path}: unsupported gtInfo fields {fields}; expected X,Y,W,H,frameNums.")
    def array(name):
        value = getattr(info, names[name])
        return np.asarray(value.toarray() if issparse(value) else value, dtype=float)
    x, y, w, h = [array(name) for name in ("x", "y", "w", "h")]
    if len({v.shape for v in (x, y, w, h)}) != 1 or x.ndim != 2:
        raise ValueError(f"{path}: X/Y/W/H must have identical 2D shapes.")
    frames_float = array("framenums").reshape(-1)
    if not len(frames_float) or not np.isfinite(frames_float).all():
        raise ValueError(f"{path}: empty or nonfinite frameNums.")
    frames = frames_float.astype(int)
    if not np.array_equal(frames, frames_float) or len(set(frames)) != len(frames) or frames.min() < 1:
        raise ValueError(f"{path}: frameNums must be unique positive integers.")
    # Standard toolkit indexes the matrices by absolute frame ID minus one.
    if frames.max() > x.shape[0]:
        raise ValueError(f"{path}: frameNums exceeds matrix rows; inspect this MAT variant.")
    frame_boxes = {}
    for frame in sorted(frames):
        row = frame - 1
        valid = (x[row] > 0) & (w[row] > 0) & (h[row] > 0)
        boxes = []
        for column in np.flatnonzero(valid):
            left = x[row, column] - w[row, column] / 2 - 1
            top = y[row, column] - h[row, column] - 1
            box = [left, top, left + w[row, column], top + h[row, column]]
            if not np.isfinite(box).all():
                raise ValueError(f"{path}: invalid box in frame {frame}.")
            boxes.append((box, False))
        frame_boxes[int(frame)] = boxes
    return frame_boxes


def load_samples(images_root, annotations_root, prefix, limit=0):
    """Match exact sequence names, then process ascending annotated frame IDs."""
    if not prefix.startswith("MVI_"):
        prefix = "MVI_" + prefix
    folders = sorted(p for p in images_root.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if not folders:
        raise ValueError(f"No {prefix}* sequence folders found in {images_root}")
    mat_index = {}
    for path in annotations_root.rglob("*.mat"):
        if path.stem in mat_index:
            raise ValueError(f"Duplicate annotation name {path.stem} in {annotations_root}")
        mat_index[path.stem] = path
    samples, sequences = [], []
    for folder in folders:
        if folder.name not in mat_index:
            raise FileNotFoundError(f"Missing {folder.name}.mat in {annotations_root}. "
                                    "Use annotations matching all selected sequences.")
        frame_boxes = load_detrac_mat(mat_index[folder.name])
        frame_paths = {}
        for path in folder.iterdir():
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            match = re.search(r"(\d+)$", path.stem)
            if match:
                frame = int(match.group(1))
                if frame in frame_paths:
                    raise ValueError(f"Duplicate image frame ID {frame} in {folder}")
                frame_paths[frame] = path
        sequences.append(folder.name)
        for frame, boxes in frame_boxes.items():
            if frame not in frame_paths:
                raise FileNotFoundError(f"{folder.name}: missing annotated frame {frame}")
            samples.append((frame_paths[frame], boxes))
            if limit and len(samples) >= limit:
                return samples, sequences
    if not samples:
        raise ValueError("No annotated frames found.")
    return samples, sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=DEFAULT_IMAGES)
    parser.add_argument('--labels', type=Path, default=DEFAULT_LABELS)
    parser.add_argument('--split', default='test', help='Metadata only; paths choose actual split')
    parser.add_argument('--prefix', default=DEFAULT_PREFIX, help='MVI_40 or 40')
    parser.add_argument('--check-data', action='store_true', help='Check MAT/image matching without inference')
    parser.add_argument('--model', default='yolo26m.pt')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'YOLOv26_DETRAC_40_Results')
    parser.add_argument('--device', default=None, help='0 for first CUDA GPU, or cpu')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.25, help='Fixed threshold for F1/matrix')
    parser.add_argument('--ap-conf', type=float, default=0.001, help='Low prediction cutoff for AP')
    parser.add_argument('--match-iou', type=float, default=0.5)
    parser.add_argument('--max-det', type=int, default=300)
    parser.add_argument('--limit', type=int, default=0, help='Smoke test only; 0 means all labeled frames')
    parser.add_argument('--warmup', type=int, default=10, help='Untimed warm-up predictions before evaluation')
    args = parser.parse_args()
    if not args.images.is_dir() or not args.labels.exists():
        parser.error('Images directory and annotation path must exist.')
    if not (0 <= args.ap_conf <= args.conf <= 1 and 0 < args.match_iou <= 1):
        parser.error('Require 0 <= ap-conf <= conf <= 1 and 0 < match-iou <= 1.')
    if args.warmup < 1:
        parser.error('warmup must be at least 1 to initialize the backend before timing.')
    if args.limit < 0 or args.max_det < 10:
        parser.error('limit must be nonnegative and max-det must be at least 10.')
    samples, sequences = load_samples(args.images, args.labels, args.prefix, args.limit)
    print('Sequences:', ', '.join(sequences), flush=True)
    print(f'Frames: {len(samples)}; vehicle boxes: {sum(len(b) for _, b in samples)}', flush=True)
    if args.check_data:
        print('MAT/image matching passed. No model inference performed.')
        return
    import torch
    import cv2
    import ultralytics
    from ultralytics import YOLO
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model = YOLO(args.model)
    vehicle_names = {'car', 'bus', 'truck'}
    car_ids = [int(k) for k, name in model.names.items() if name.lower() in vehicle_names]
    if not car_ids:
        raise ValueError('Checkpoint needs car, bus, or truck classes.')
    print('YOLO classes merged into vehicle:', [model.names[k] for k in car_ids], flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = {'info': {}, 'images': [], 'annotations': [],
               'categories': [{'id': 1, 'name': 'vehicle'}]}
    detections, per_image = [], []
    totals = np.zeros(4, dtype=np.int64)
    speed_ms = []
    prediction_ms, frame_ms, timing_rows = [], [], []
    predict_options = dict(batch=1, classes=car_ids, conf=args.ap_conf,
                           imgsz=args.imgsz, max_det=args.max_det,
                           device=args.device, verbose=False)
    # Initialize/fuse the backend and warm it using real input, without adding
    # warm-up predictions to accuracy metrics or timed frame counts.
    first_image = cv2.imread(str(samples[0][0]))
    if first_image is None:
        raise RuntimeError(f'Cannot decode {samples[0][0]}')
    print(f'Warming up with {args.warmup} predictions...', flush=True)
    for _ in range(args.warmup):
        warm_results = model.predict(source=first_image, **predict_options)
        del warm_results
    del first_image
    actual_device = torch.device(model.predictor.device)
    use_cuda = actual_device.type == 'cuda'
    def synchronize():
        if use_cuda:
            torch.cuda.synchronize(actual_device)
    synchronize()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(actual_device)
        memory_before = torch.cuda.memory_allocated(actual_device)
    else:
        memory_before = None
    start = time.perf_counter()
    count = 0
    for image_id, (path, labels) in enumerate(samples, 1):
        # Full frame latency: disk read/decode + predict + CPU box transfer.
        # Matching/AP/plotting are deliberately outside this measurement.
        synchronize()
        frame_start = time.perf_counter()
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f'Cannot decode {path}')
        synchronize()
        prediction_start = time.perf_counter()
        result = model.predict(source=image, **predict_options)[0]
        synchronize()
        prediction_duration = (time.perf_counter() - prediction_start) * 1000
        boxes = result.boxes.xyxy.cpu().numpy().copy()
        scores = result.boxes.conf.cpu().numpy().copy()
        synchronize()
        frame_duration = (time.perf_counter() - frame_start) * 1000
        prediction_ms.append(prediction_duration)
        frame_ms.append(frame_duration)
        speed_ms.append(float(result.speed['inference']))
        timing_rows.append([str(path), prediction_duration, frame_duration,
                            float(result.speed['inference'])])
        count += 1
        height, width = result.orig_shape
        dataset['images'].append({'id': image_id, 'file_name': str(path),
                                  'height': height, 'width': width})
        gt, crowds = [], []
        for box, crowd in labels:
            x1, y1, x2, y2 = box
            dataset['annotations'].append({'id': len(dataset['annotations']) + 1,
                'image_id': image_id, 'category_id': 1, 'bbox': [x1, y1, x2-x1, y2-y1],
                'area': (x2-x1)*(y2-y1), 'iscrowd': int(crowd)})
            (crowds if crowd else gt).append(box)
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(float, box)
            detections.append({'image_id': image_id, 'category_id': 1,
                'bbox': [x1, y1, x2-x1, y2-y1], 'score': float(score)})
        keep = scores >= args.conf
        counts = match_counts(boxes[keep], scores[keep], gt, crowds, args.match_iou)
        totals += counts
        per_image.append([str(path), *counts])
        del result, image
        if image_id % 100 == 0:
            print(f'Evaluated {image_id}/{len(samples)} frames', flush=True)
    if count != len(samples):
        raise RuntimeError('Prediction count does not match annotated frame count.')
    synchronize()
    elapsed = time.perf_counter() - start
    # Capture before CPU COCO evaluation or plot generation.
    memory = {
        'device': str(actual_device),
        'gpu_name': torch.cuda.get_device_name(actual_device) if use_cuda else None,
        'gpu_total_mib': torch.cuda.get_device_properties(actual_device).total_memory / 2**20 if use_cuda else None,
        'allocated_after_warmup_mib': memory_before / 2**20 if use_cuda else None,
        'peak_allocated_mib': torch.cuda.max_memory_allocated(actual_device) / 2**20 if use_cuda else None,
        'peak_reserved_mib': torch.cuda.max_memory_reserved(actual_device) / 2**20 if use_cuda else None,
        'peak_allocated_above_warmup_mib': (torch.cuda.max_memory_allocated(actual_device) - memory_before) / 2**20 if use_cuda else None,
        'scope': 'PyTorch CUDA allocator for this process, measured after warm-up. Includes resident model tensors; excludes other processes, driver allocations and non-PyTorch CUDA allocations. CPU runs report null.'
    }
    def latency_stats(values):
        return {'mean': float(np.mean(values)), 'median': float(np.median(values)),
                'p95': float(np.percentile(values, 95)), 'min': float(np.min(values)),
                'max': float(np.max(values))}
    systems = {
        'warmup_predictions_excluded': args.warmup, 'timed_frames': count,
        'batch_size': 1, 'imgsz': args.imgsz,
        'backend_forward_ms': latency_stats(speed_ms),
        'predict_call_latency_ms': latency_stats(prediction_ms),
        'frame_latency_ms': latency_stats(frame_ms),
        'backend_forward_equivalent_fps': 1000 / float(np.mean(speed_ms)),
        'predict_call_throughput_fps': 1000 * count / sum(prediction_ms),
        'frame_processing_throughput_fps': 1000 * count / sum(frame_ms),
        'evaluation_loop_throughput_fps': count / elapsed,
        'evaluation_loop_seconds': elapsed,
        'gpu_memory': memory,
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
                        'opencv': cv2.__version__, 'torch_cuda': torch.version.cuda,
                        'backend_fp16': bool(getattr(model.predictor.model, 'fp16', False))},
        'measurement_notes': {
            'backend_forward': 'Ultralytics-reported model forward only. Equivalent FPS is not end-to-end throughput.',
            'predict_call': 'Synchronized wall time for model.predict, including preprocessing, transfers, forward and postprocessing. Disk read and final CPU box copy excluded.',
            'frame': 'Synchronized wall time from disk read/decode through prediction and CPU box copy. Excludes metric matching, AP, output writing and plotting. OS disk cache may be warm.',
            'evaluation_loop': 'Includes frame processing plus Python annotation/matching/collection and progress printing. Excludes warm-up, annotation loading, model loading, final AP and output files.',
            'comparison': 'Serial batch-1 measurement with per-frame synchronization. Use matching hardware, inputs and settings. Temporal capture/buffering delay is not measured.'
        }
    }
    (args.output/'systems_metrics.json').write_text(json.dumps(systems, indent=2), encoding='utf-8')
    with (args.output/'per_image_timing.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['image', 'predict_call_ms', 'frame_processing_ms', 'backend_forward_ms'])
        writer.writerows(timing_rows)
    tp, fp, fn, ignored = map(int, totals)
    precision = tp / (tp+fp) if tp+fp else 0.0
    recall = tp / (tp+fn) if tp+fn else 0.0
    f1 = 2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    if detections:
        evaluator = COCOeval(coco, coco.loadRes(detections), 'bbox')
        evaluator.params.maxDets = [1, 10, args.max_det]
        evaluator.evaluate()
        evaluator.accumulate()
        # Avoid summarize()'s hard-coded maxDets=100 assumption.
        values = evaluator.eval['precision'][:, :, 0, 0, -1]
        valid = values[values >= 0]
        ap = float(valid.mean()) if len(valid) else 0.0
        ap50 = float(values[0][values[0] >= 0].mean())
        ap75 = float(values[5][values[5] >= 0].mean())
        ar = float(evaluator.eval['recall'][:, 0, 0, -1].mean())
        curve = values[0].tolist()
    else:
        ap = ap50 = ap75 = ar = 0.0
        curve = [0.0] * 101
    summary = {'model': args.model, 'split': args.split, 'class': 'vehicle',
        'images': count, 'sequences': sequences, 'prefix': args.prefix,
        'imgsz': args.imgsz, 'dataset': 'UA-DETRAC', 'yolo_classes': [model.names[k] for k in car_ids],
        'mat_coordinates': 'X=center x, Y=bottom y, one-based pixel origin',
        'ignore_regions_applied': False, 'partial_run': bool(args.limit), 'confidence': args.conf,
        'matching_iou': args.match_iou, 'ap_confidence_floor': args.ap_conf,
        'max_detections_per_image': args.max_det, 'TP': tp, 'FP': fp, 'FN': fn,
        'ignored_crowd_predictions': ignored, 'precision': precision, 'recall': recall,
        'f1': f1, 'AP50': ap50, 'AP75': ap75, 'AP50_95': ap, 'AR50_95': ar,
        'mean_inference_ms': float(np.mean(speed_ms)),
        'pipeline_images_per_second': count / elapsed, 'systems': systems,
        'note': 'All gtInfo vehicles evaluated against merged YOLO car/bus/truck. Ignore regions not applied. COCO-style AP; fixed-threshold F1 uses confidence-ordered matching. '
                'TN/accuracy undefined for object detection. Timing excludes warm-up; see systems_metrics.json for scopes; '
                'not an official UA-DETRAC benchmark score.'}
    (args.output/'metrics.json').write_text(json.dumps(summary, indent=2))
    (args.output/'predictions_coco.json').write_text(json.dumps(detections))
    with (args.output/'per_image.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['image', 'TP', 'FP', 'FN', 'ignored'])
        writer.writerows(per_image)
    # Per-sequence fixed-threshold counts, preserving each sequence's identity.
    sequence_totals = {}
    for row in per_image:
        name = Path(row[0]).parent.name
        sequence_totals.setdefault(name, np.zeros(4, dtype=np.int64))
        sequence_totals[name] += row[1:]
    with (args.output/'per_sequence.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sequence', 'TP', 'FP', 'FN', 'ignored', 'precision', 'recall', 'f1'])
        for name, counts in sequence_totals.items():
            stp, sfp, sfn, signored = map(int, counts)
            writer.writerow([name, stp, sfp, sfn, signored,
                             stp/(stp+sfp) if stp+sfp else 0,
                             stp/(stp+sfn) if stp+sfn else 0,
                             2*stp/(2*stp+sfp+sfn) if 2*stp+sfp+sfn else 0])
    matrix = np.array([[tp, fp], [fn, 0]])
    np.savetxt(args.output/'confusion_matrix.csv', matrix, fmt='%d', delimiter=',')
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap='Blues')
    for row in range(2):
        for col in range(2):
            ax.text(col, row, 'N/A' if row == col == 1 else str(matrix[row, col]),
                    ha='center', va='center')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Vehicle', 'Background'],
           yticklabels=['Vehicle', 'Background'], xlabel='Ground truth', ylabel='Prediction',
           title=f'Vehicle detection: conf={args.conf}, IoU={args.match_iou}')
    fig.tight_layout()
    fig.savefig(args.output/'confusion_matrix.png', dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 1, 101), curve)
    ax.set(xlabel='Recall', ylabel='Interpolated precision', xlim=(0, 1), ylim=(0, 1),
           title=f'Vehicle precision-recall at IoU=0.50 (AP={ap50:.4f})')
    fig.tight_layout()
    fig.savefig(args.output/'precision_recall_curve.png', dpi=200)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f'Precision {precision:.4f} | Recall {recall:.4f} | F1 {f1:.4f}')
    print('\nSYSTEMS RESULTS (after warm-up)')
    print(f"Predict latency: {systems['predict_call_latency_ms']['mean']:.2f} ms mean, "
          f"{systems['predict_call_latency_ms']['p95']:.2f} ms p95")
    print(f"Frame latency (read to CPU boxes): {systems['frame_latency_ms']['mean']:.2f} ms mean")
    print(f"Frame processing throughput: {systems['frame_processing_throughput_fps']:.2f} frames/sec")
    print(f"Evaluation loop throughput: {systems['evaluation_loop_throughput_fps']:.2f} frames/sec")
    if use_cuda:
        print(f"Peak GPU allocated: {memory['peak_allocated_mib']:.2f} MiB")
        print(f"Peak GPU reserved: {memory['peak_reserved_mib']:.2f} MiB")
    else:
        print('Peak CUDA GPU memory: N/A (CPU/non-CUDA run)')
    print(f'Results saved to {args.output.resolve()}')


if __name__ == '__main__':
    main()
