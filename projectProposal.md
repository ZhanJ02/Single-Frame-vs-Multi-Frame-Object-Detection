# Project Repository

**GitHub repository:** TODO: Add the URL of the team's GitHub repository.

All source code, configuration files, experiment logs, results, and documentation will be maintained in this repository. Each team member will work through separate branches or assigned issues and submit regular commits and pull requests so that individual contributions remain visible throughout the semester.

# Introduction

Object detectors typically process each video frame as an independent still image. This approach is efficient, but it ignores temporal information available in nearby frames. A single frame may contain motion blur, partial occlusion, poor lighting, water reflections, or an object that is too small to recognize reliably. Neighboring frames may provide a clearer view or reveal motion patterns that help distinguish a real object from background noise.

This project will compare **single-frame object detection** with **multi-frame object detection**. The single-frame baseline will use one RGB image as input. The multi-frame variants will use short sequences containing 3, 5, or 7 consecutive RGB frames and predict the bounding boxes for the center frame. The main research question is:

> **Does multi-frame input improve object-detection accuracy and robustness enough to justify its additional latency, GPU-memory usage, and computational cost?**

This is an ML systems problem because the best model is not necessarily the model with the highest detection accuracy. A multi-frame model may detect difficult objects more reliably while also reducing throughput and requiring more memory. The project will study this quality-versus-cost trade-off and determine which input configuration provides the most practical balance.

# Problem Statement

The project focuses on video-based maritime object detection, where targets are often small, distant, partially occluded, or visually similar to waves and reflections. The experimental dataset will be an existing sequence-based maritime dataset assembled from:

- in-house video collected at Lake Murray and Charleston, South Carolina;
- the Singapore Maritime Dataset, which contains on-shore, on-board, and near-infrared maritime video recorded under multiple environmental conditions [1]; and
- the LaRS maritime dataset, which provides temporally adjacent frames from diverse lakes, rivers, and seas [2].

The current organized dataset contains approximately 15,969 training frames, 3,187 validation frames, and 2,818 test frames. These counts may change slightly after final validation and removal of corrupted, duplicate, or incorrectly ordered samples. Data will be split by complete video sequence rather than by individual frame. This prevents nearly identical neighboring frames from appearing in both training and testing and producing overly optimistic results.

Each sample will contain an ordered temporal window and the object-detection labels for its center frame. The study will compare four input configurations:

- 1 RGB frame, or 3 input channels;
- 3 RGB frames, or 9 stacked channels;
- 5 RGB frames, or 15 stacked channels; and
- 7 RGB frames, or 21 stacked channels.

The expected result is that multi-frame models will improve detection of small, blurred, and partially occluded objects because they can use information from adjacent frames. However, additional frames are also expected to increase data-loading cost, inference latency, training time, and GPU-memory usage. Performance may show diminishing returns after a certain number of frames. The goal is to measure these effects rather than assume that more frames are always better.

# Technical Approach

The project will use a pretrained YOLO object detector as the common detection backbone. YOLO is appropriate because it provides a practical real-time baseline and can be evaluated using both task-quality and system-performance metrics. The implementation will reuse the Ultralytics training and inference pipeline where possible [3].

The single-frame baseline will use the standard three-channel RGB input. For the multi-frame variants, 3, 5, or 7 consecutive frames will be stacked along the channel dimension. A small learnable convolutional input adapter will transform the stacked channels into a three-channel representation that can be passed into the same pretrained detector. This keeps the detector backbone consistent across experiments while allowing the multi-frame system to learn temporal patterns from the input sequence.

The multi-frame model will be supervised using only the center frame's bounding-box labels. Every experiment will use the same dataset split, target-frame labels, detector size, image resolution, augmentation policy, hardware, and overall training procedure. The main independent variable will be the number of input frames. Secondary experiments may vary frame spacing to compare consecutive frames with frames sampled at a larger temporal interval.

The repository will include:

- a sequence-aware dataset loader that preserves frame order;
- single-frame and configurable multi-frame model definitions;
- training, validation, and testing scripts;
- reproducible configuration files for every experiment;
- profiling code for latency, throughput, memory, and training time;
- scripts for calculating detection metrics and confidence intervals; and
- scripts for generating the final plots, tables, and qualitative examples.

The project will not attempt to design a completely new detector, build an object tracker, or deploy the system to an embedded device. These features are outside the semester scope. The primary contribution will be a controlled and reproducible systems comparison of input-window sizes.

# Evaluation

## Research Questions

1. How does using 3, 5, or 7 frames affect detection quality compared with a single frame?
2. How much latency, GPU memory, throughput, and training-time overhead does each additional input configuration introduce?
3. Do multi-frame inputs provide larger improvements for small, blurred, or partially occluded objects?
4. At what point do additional frames produce diminishing returns?
5. Which model lies on the best quality-cost trade-off for the available hardware?

## Quantitative Evaluation

Detection quality will be measured using:

- mean average precision at IoU 0.50 (**mAP50**);
- mean average precision averaged from IoU 0.50 to 0.95 (**mAP50-95**);
- precision;
- recall;
- F1 score; and
- false positives and false negatives from the confusion matrix.

System performance will be measured using:

- median and 95th-percentile inference latency per target frame;
- throughput in frames per second;
- peak allocated GPU memory;
- model parameter count and saved-model size;
- training time per epoch and total training time; and
- preprocessing and data-loading time.

All models will be tested on the same hardware with the same batch size and image resolution for the primary comparison. Software versions, GPU model, CUDA version, precision mode, and inference settings will be recorded. Inference profiling will include warm-up iterations before timed trials. Each primary training configuration will be repeated with three random seeds. The report will show the mean and standard deviation across runs. Where appropriate, paired bootstrap confidence intervals over test sequences will be used to estimate the uncertainty of performance differences.

The primary summary will be a Pareto plot with detection quality on one axis and latency or GPU memory on the other. This plot will show whether a multi-frame model improves quality without being dominated by a cheaper configuration.

## Qualitative Evaluation

The final report will include side-by-side prediction examples from all four input configurations. Examples will be selected from challenging scenes involving:

- small or distant objects;
- motion blur;
- partial occlusion;
- glare and water reflections;
- low contrast or poor lighting; and
- camera movement.

Example videos will also be generated with synchronized predictions from the single-frame and best-performing multi-frame models. These examples will help explain where temporal information corrects a missed detection, removes a false positive, or fails to improve the result.

## Expected Evidence

The final evaluation will contain:

- a table comparing all detection and system metrics;
- a quality-versus-latency Pareto plot;
- a quality-versus-memory Pareto plot;
- training and validation loss curves;
- an ablation plot showing performance as the number of frames increases;
- confusion matrices for the main baseline and best multi-frame model; and
- qualitative images and videos showing successes and failures.

# Related Work

Prior video object-detection research has shown that nearby frames can improve a detector when an individual frame has motion blur, defocus, occlusion, or an unusual object appearance. Flow-Guided Feature Aggregation aligns and combines feature maps from nearby frames using estimated motion, improving upon single-frame baselines, particularly for fast-moving objects [4]. Deep Feature Flow addresses video-processing cost by running an expensive network on selected key frames and propagating features to other frames [5]. These methods demonstrate both the value of temporal information and the importance of considering efficiency.

YOLOV extends a one-stage YOLOX detector to video by selecting useful object regions and aggregating information between target and reference frames. Its authors specifically evaluate both detection accuracy and inference speed, which closely matches the quality-cost focus of this project [6]. More recent work has also investigated a simpler multi-frame YOLO approach that stacks consecutive frames and supervises a target frame. That work reports improved robustness while keeping architectural changes limited [7].

Temporal modeling has also been studied specifically in maritime environments. WaSR-T uses recent frames to learn temporal differences between true obstacles and reflections on water, reducing false positives compared with single-frame maritime methods [8]. LaRS was designed with preceding frames so that temporal texture can be studied in maritime perception [2].

This project differs from work proposing a new state-of-the-art video detector. Its contribution is a controlled ML systems study of **how many input frames are worthwhile** when the model, data split, hardware, and evaluation procedure are held constant. The project will jointly report detection quality, robustness, latency, throughput, memory, and training cost so that the result supports an operational model-selection decision.

# References

[1] D. K. Prasad, D. Rajan, L. Rachmawati, E. Rajabally, and C. Quek, “Video Processing From Electro-Optical Sensors for Object Detection and Tracking in a Maritime Environment: A Survey,” *IEEE Transactions on Intelligent Transportation Systems*, 2017. Singapore Maritime Dataset: https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset

[2] L. Žust, J. Perš, and M. Kristan, “LaRS: A Diverse Panoptic Maritime Obstacle Detection Dataset and Benchmark,” 2023. https://arxiv.org/abs/2308.09618

[3] Ultralytics, “Ultralytics YOLO Documentation and Source Code.” https://github.com/ultralytics/ultralytics

[4] X. Zhu, Y. Wang, J. Dai, L. Yuan, and Y. Wei, “Flow-Guided Feature Aggregation for Video Object Detection,” *IEEE International Conference on Computer Vision*, 2017. https://arxiv.org/abs/1703.10025

[5] X. Zhu, Y. Xiong, J. Dai, L. Yuan, and Y. Wei, “Deep Feature Flow for Video Recognition,” *IEEE Conference on Computer Vision and Pattern Recognition*, 2017. https://arxiv.org/abs/1611.07715

[6] Y. Shi, N. Wang, and X. Guo, “YOLOV: Making Still Image Object Detectors Great at Video Object Detection,” 2022. https://arxiv.org/abs/2208.09686

[7] Y. Quan, B. Kiefer, M. Messmer, and A. Zell, “Lightweight Multi-Frame Integration for Robust YOLO Object Detection in Videos,” 2025. https://arxiv.org/abs/2506.20550

[8] L. Žust and M. Kristan, “Temporal Context for Robust Maritime Obstacle Detection,” 2022. https://arxiv.org/abs/2203.05352
