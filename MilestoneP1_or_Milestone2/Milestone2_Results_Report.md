# Milestone 2: Generating and Analyzing Partial Results
**James Zhan — Single-Frame vs. Multi-Frame Object Detection**

## Introduction and Experimental Setup

The goal of my project is to determine whether using multiple video frames improves vehicle detection enough to justify the additional latency, GPU memory, and computational cost. For this milestone, I implemented and evaluated the single-frame baseline; this establishes the reference point for later temporal-model comparisons.

I evaluated pretrained YOLO26m on all 10,000 images in the BDD100K 100k validation split. The evaluation focused on cars rather than combining all vehicle categories. I used the pretrained checkpoint without fine-tuning on BDD100K. Assuming the supplied script defaults were unchanged, inference used an image-size setting of 640; precision, recall, and F1 used confidence 0.25 and matching IoU 0.50. AP used predictions down to confidence 0.001 and a maximum of 300 detections per image. These settings should be confirmed against the saved metrics.json and run command.

The results below are car-only COCO-style metrics from my evaluator; they are not official BDD100K leaderboard scores. These are validation results, not official test-set results. Only the single-frame model has been evaluated so far; therefore, these results cannot yet establish whether temporal input improves detection.

## Results Summary

| Metric | Result |
|---|---:|
| Validation images | 10,000 |
| True positives | 52,130 |
| False positives | 9,609 |
| False negatives | 50,376 |
| Precision | 84.44% |
| Recall | 50.86% |
| F1 score | 63.48% |
| AP50 | 64.29% |
| AP75 | 38.31% |
| AP50–95 | 38.41% |
| Mean model inference time | 8.16 ms/image |

The fixed-threshold confusion matrix has prediction classes as rows and ground-truth classes as columns:

| Prediction / Ground truth | Car | Background |
|---|---:|---:|
| Car | 52,130 | 9,609 |
| Background | 50,376 | N/A |

True negatives are undefined in this object-detection setup; consequently, ordinary classification accuracy would not be an appropriate summary metric. The precision, recall, and F1 values describe one confidence threshold, whereas AP summarizes performance across confidence levels.

## Positive Outcomes

The most encouraging accuracy result is the precision of 84.44%. At the evaluated threshold, approximately 84 out of every 100 car predictions matched an annotated car. This suggests that the baseline produces relatively few false detections compared with its number of correct detections; however, this precision must be interpreted alongside the lower recall.

The model also correctly detected 52,130 cars across the validation images. This demonstrates that the pretrained detector can provide a useful baseline on BDD100K without dataset-specific fine-tuning. AP50 reached 64.29%, which establishes a measurable reference for the next experiments. It does not represent an improvement because no alternative model has been evaluated under the same conditions yet.

The reported mean inference time was 8.16 ms per image. Its reciprocal is approximately 122.5 images per second for model inference alone; this suggests a promising starting point for studying the cost of temporal processing. It does not establish end-to-end throughput because image loading, preprocessing, postprocessing, and other pipeline overhead are not included in that inference-only figure.

## Negative Results and Possible Explanations

The main limitation is recall. The evaluator counted 50,376 false negatives and 52,130 true positives; thus, approximately 49.14% of the evaluated non-crowd cars were missed at the selected threshold. The F1 score of 63.48% reflects this imbalance between precision and recall. For traffic monitoring, missed vehicles can lead to incomplete counts and unreliable observations even when the detections that are produced are usually correct.

Possible explanations include small or distant cars, occlusion, low lighting, and differences between the model's pretraining data and BDD100K. BDD100K was designed to include varied geographic, environmental, and weather conditions [1]. However, the aggregate results do not identify which conditions caused the misses; I need to inspect false-negative examples and calculate results by object size and scene attributes before treating these explanations as established findings.

AP decreased from 64.29% at IoU 0.50 to 38.31% at IoU 0.75, a difference of 25.98 percentage points. This indicates that performance is substantially weaker under stricter box-overlap requirements. It suggests that accurate localization is another limitation; however, the gap alone cannot distinguish localization errors from other factors such as small-object difficulty.

During implementation, an attempt to process the full image list caused a CUDA out-of-memory error on the 8 GB GPU. The failed allocation requested 27.47 GiB. Changing inference to process one image at a time allowed the evaluation to complete. This was an input-batching problem rather than evidence that a single YOLO26m image requires that amount of memory. The successful run's peak GPU-memory usage has not yet been measured.

## Connection to Temporal Detection Research

TransVOD aggregates feature memories and object queries across frames and reports improvements over its Deformable DETR baseline on ImageNet VID [2]. Flow-Guided Feature Aggregation also reports benefits from combining neighboring-frame features, particularly for difficult fast-moving objects [3]. These studies support investigating temporal information; their reported gains cannot be transferred directly to this experiment because the datasets, architectures, and evaluation protocols differ.

My current results identify recall and localization as areas to investigate. They do not prove that temporal information will resolve those limitations.

## Next Steps

I will first inspect representative false positives and false negatives, using the saved confusion matrix and precision–recall curve alongside annotated prediction examples. I will also analyze detection performance by car size, lighting, and occlusion where annotations support it.

Next, I will obtain the BDD100K MOT sequences or corresponding source videos. The 100k image release alone does not supply neighboring-frame windows. I will evaluate YOLO26 and TransVOD on exactly the same target frames and annotations; YOLO26 will receive one frame, while TransVOD will receive the target frame and its reference frames. Both will use the same car category and matching criteria.

A pretrained-model comparison will measure practical transfer performance, but different architectures and pretraining datasets prevent it from isolating the effect of temporal input. A stronger temporal ablation would compare a temporal detector against its corresponding single-frame baseline and, where feasible, fine-tune both using the same training sequences.

Finally, I will measure full-pipeline latency, throughput, and peak GPU memory after warm-up. Centered windows also require future frames; their buffering delay should be reported separately from inference time. These measurements will allow me to assess whether any detection gains justify the additional systems cost.

## References

1. Yu et al. *BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning*. CVPR 2020. https://arxiv.org/abs/1805.04687
2. He et al. *End-to-End Video Object Detection with Spatial-Temporal Transformers*. ACM Multimedia 2021. https://arxiv.org/abs/2105.10920
3. Zhu et al. *Flow-Guided Feature Aggregation for Video Object Detection*. ICCV 2017. https://arxiv.org/abs/1703.10025

