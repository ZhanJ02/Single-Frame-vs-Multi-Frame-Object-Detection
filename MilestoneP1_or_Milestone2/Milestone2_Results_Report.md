# Milestone 2: Generating and Analyzing Partial Results
**James Zhan — Single-Frame vs. Multi-Frame Vehicle Detection**

## Project Goal

My project investigates whether using consecutive video frames improves vehicle detection enough to justify the additional computational cost. The ML component compares single-frame and multi-frame object detection; the systems component measures latency, throughput, and GPU-memory requirements. The main question is whether temporal information provides a useful accuracy–efficiency trade-off for traffic monitoring.

For this milestone, I completed the pretrained YOLO26m baseline and added systems measurements to the evaluation pipeline. The multi-frame comparison is the next stage; no temporal-model results are reported yet.

## Dataset and Experimental Setup

I initially evaluated the model on 10,000 BDD100K validation images. However, that release contains separate still images rather than ordered neighboring frames. I changed the main experimental dataset to UA-DETRAC so that both models can be evaluated on the same sequential traffic scenes.

The current evaluation selects image-sequence folders beginning with `MVI_40` and loads the matching MATLAB `.mat` annotations. Frames are processed in numerical order within each sequence. Each selected sequence requires a corresponding annotation file; the sequence prefix alone does not determine whether a sequence belongs to the official training or test split.

The baseline uses pretrained **YOLO26m (`yolo26m.pt`)**, without fine-tuning on UA-DETRAC. YOLO processes one frame at a time; processing sequential images does not make the model temporal. The evaluator merges YOLO's car, bus, and truck predictions into one `vehicle` class and compares them with the annotated vehicle boxes.

| Setting | Evaluation-script default |
|---|---|
| Model | YOLO26m |
| Input | One image per prediction |
| Image-size setting | 640 |
| Confidence threshold for precision, recall, and F1 | 0.25 |
| Matching IoU | 0.50 |
| Low confidence cutoff for AP | 0.001 |
| Maximum detections per image | 300 |
| Warm-up | 10 predictions, excluded from timing |

These are the script defaults; the saved run configuration should be used to confirm any command-line overrides. The exact evaluated frame count, sequence list, GPU model, software versions, and inference precision should accompany the final experiment. They were not available in the result summary used for this update, so I have not assigned values to them.

## UA-DETRAC Detection Results

| Metric | Result |
|---|---:|
| Precision | 60.85% |
| Recall | 84.83% |
| F1 score | 70.87% |

The baseline detects approximately 85 out of every 100 labeled vehicles; however, only about 61 out of every 100 predictions match a labeled vehicle under the evaluation criteria. This gives a missed-vehicle rate of approximately 15.17% and a false-positive share of approximately 39.15% among predictions. The latter is not the classification false-positive rate, which requires a defined true-negative count.

The strongest result is recall; the pretrained model finds a large proportion of the annotated vehicles without dataset-specific training. Precision remains the main limitation, and the F1 score reflects the balance between these two metrics. These results establish a baseline rather than showing an improvement over another model.

Precision is calculated as `TP / (TP + FP)`, recall as `TP / (TP + FN)`, and F1 as `2 × precision × recall / (precision + recall)`. The evaluator uses one-to-one box matching at the selected IoU threshold. True negatives are not defined in this object-detection setup, so ordinary classification accuracy is not reported.

AP values and exact TP, FP, and FN counts were not included in the latest available summary; they should be copied from the completed run's saved metrics before submission if required. Rounded precision and recall alone are insufficient to reconstruct those counts.

## Computational Efficiency Results

| Measurement | Result | Scope |
|---|---:|---|
| Mean prediction-call latency | 10.94 ms/frame | Preprocessing, prediction, and postprocessing |
| 95th-percentile prediction latency | 20.11 ms/frame | Approximately 95% of prediction calls finish within this time |
| Mean frame-processing latency | 13.18 ms/frame | Image loading through predicted boxes copied to CPU |
| Frame-processing throughput | 75.88 frames/sec | Timed frame-processing work |
| Evaluation-loop throughput | 74.01 frames/sec | Includes matching and result-collection overhead |
| Peak allocated GPU memory | Approximately 159 MiB | Peak PyTorch tensor allocation |
| Peak reserved GPU memory | Approximately 216 MiB | Peak memory held by the PyTorch allocator |

The average prediction call takes 10.94 ms; including image loading and copying boxes to CPU increases the average to 13.18 ms. The difference is approximately 2.24 ms per frame. Evaluation-loop throughput is slightly lower because the loop also performs metric-related work.

The frame-processing throughput agrees with the mean latency: `1000 / 13.18 ≈ 75.87 FPS`. The small difference from 75.88 FPS is consistent with rounding. These measurements describe offline processing on the evaluated hardware; they do not include model startup or establish live-camera performance.

### Measurement Procedure

The script performs 10 warm-up predictions before collecting timing results. CUDA is synchronized around the timed prediction call so that elapsed wall-clock time includes completed GPU work. Python's `time.perf_counter()` measures elapsed time; latency is converted from seconds to milliseconds.

For N evaluated frames, throughput is calculated as `N / sum(frame times in seconds)`. Prediction-call and frame-processing throughput use their corresponding timing boundaries. Evaluation-loop throughput uses the total elapsed loop time; it excludes model loading, annotation loading, warm-up, final AP computation, plotting, and file saving.

After warm-up, the script resets PyTorch's CUDA peak-memory counters and records peak allocated and reserved memory during evaluation. The values include the resident model and are converted to MiB using `bytes / 1024²`. They do not measure total device memory usage; driver overhead, other applications, and some non-PyTorch allocations are outside this scope.

These measurements quantify runtime and memory efficiency. **FLOPs/GFLOPs, energy consumption, and monetary cost have not been measured.** Latency is hardware dependent and should not be presented as an arithmetic-operation count.

## Positive Outcomes

I now have a baseline on sequential traffic data and a working path for loading MATLAB annotations. This supports a consistent comparison where YOLO26 receives one target frame and a temporal detector receives neighboring frames from the same sequence.

The baseline combines 84.83% recall with approximately 75.88 processed frames per second. This provides a useful reference for assessing whether the temporal model's detection gains justify additional processing time and memory. The updated evaluator also separates prediction latency from frame-processing latency; therefore, the comparison can account for pipeline overhead instead of relying only on model-forward timing.

## Limitations and Interpretation

The current evaluator does not apply UA-DETRAC's separate ignore regions. Predictions inside those regions may be counted as false positives; therefore, the reported precision and F1 are custom evaluation scores rather than official UA-DETRAC benchmark scores. Ignore-region handling should be implemented consistently for both models before interpreting their final comparison.

The one-class vehicle mapping also needs care. YOLO's car, bus, and truck categories do not exactly reproduce UA-DETRAC's vehicle taxonomy. Combining them supports a practical vehicle-detection baseline; however, the results do not measure separate accuracy for cars, buses, vans, or other vehicle types.

Small vehicles, occlusion, lighting, and domain differences are possible sources of errors, but aggregate metrics do not establish their individual effects. I still need to inspect prediction examples and analyze performance by sequence and object size. Temporal consistency and detection flicker have not been quantified yet.

Finally, sequential frames are correlated. Any training, validation, and test split must keep complete sequences separate to avoid neighboring-frame leakage. The `MVI_40` selection is a chosen subset; its results should not be generalized to the entire dataset without broader evaluation.

## Earlier BDD100K Baseline

| Measurement | Earlier BDD100K result |
|---|---:|
| Validation images | 10,000 |
| Precision, car only | 84.44% |
| Recall, car only | 50.86% |
| F1 score, car only | 63.48% |
| AP50 | 64.29% |
| AP75 | 38.31% |
| AP50–95 | 38.41% |
| Mean model-forward time | 8.16 ms/image |

These results document the earlier implementation stage. They are not directly comparable with the UA-DETRAC scores because the dataset and target-category mapping changed. The earlier 8.16 ms model-forward measurement also has a narrower timing scope than the current 10.94 ms prediction-call latency; their difference cannot be interpreted as a model slowdown.

## Next Steps

1. Add and verify ignore-region handling; confirm the exact sequence list, frame count, run settings, and hardware metadata.
2. Inspect false positives and false negatives; summarize results by sequence and object size where supported.
3. Implement the multi-frame comparison using TransVOD and evaluate exactly the same target frames with the same vehicle mapping and matching criteria.
4. Measure both systems on the same hardware with matching precision, warm-up, and timing boundaries. Include reference-frame processing and disclose feature caching; report any delay caused by waiting for future frames.
5. Compare detection quality against latency, target-frame throughput, and memory usage. Add GFLOPs separately if an operation-count comparison is required.

YOLO26 versus TransVOD compares two complete systems with different architectures and potentially different pretraining. It cannot isolate the effect of temporal input alone. A stronger controlled experiment would also compare TransVOD with its corresponding single-frame Deformable DETR baseline under matched training and evaluation conditions.

At this stage, the single-frame detection and systems baseline is complete; the evidence does not yet establish whether multi-frame detection offers a better accuracy–efficiency trade-off.
