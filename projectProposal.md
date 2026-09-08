# Project Repository

# Team and Responsibilities

James Zhan is the sole team member. He will dedicate time every week to work on the project and update the GitHub accordingly. 

# Introduction

Object detectors typically process each video frame as an independent still image. This approach is efficient, but it ignores temporal information available in nearby frames. In traffic videos, a single frame may contain motion blur, partial vehicle occlusion, poor lighting, shadows, or a car that is too small to recognize reliably. Neighboring frames may provide a clearer view or reveal motion patterns that help distinguish a vehicle from the road and surrounding background.

This project will compare **single-frame object detection** with **multi-frame object detection**. The single-frame baseline will use one RGB image as input. The multi-frame variants will use short sequences containing 3, 5, or 7 consecutive RGB frames and predict the bounding boxes for the center frame. The main research question is:

> **Does multi-frame input improve object-detection accuracy and robustness enough to justify its additional latency, GPU-memory usage, and computational cost?**

This is an ML systems problem because the best model is not necessarily the model with the highest detection accuracy. A multi-frame model may detect difficult objects more reliably while also reducing throughput and requiring more memory. The project will study this quality-versus-cost trade-off and determine which input configuration provides the most practical balance.

# Problem Statement

The project focuses on detecting cars and other vehicles in traffic-camera videos. Vehicles may appear small or distant, become partially hidden behind other vehicles, experience motion blur, or be difficult to distinguish under rain, nighttime lighting, shadows, and heavy traffic. These are situations where information from adjacent frames may improve a detector's decision.

The primary benchmark will be **UA-DETRAC**, a traffic video dataset containing 100 challenging sequences and more than 140,000 annotated frames. It includes approximately 1.21 million vehicle bounding boxes and attributes describing vehicle category, illumination, occlusion, and truncation [1]. The videos contain traffic intersections, highways, and other road settings recorded under conditions such as sunny weather, cloudy weather, rain, and nighttime lighting.

The official UA-DETRAC division contains 60 training sequences and 40 testing sequences. A subset of the training sequences will be reserved for validation. Data will be split by complete video sequence rather than by individual frame. This prevents nearly identical neighboring frames from appearing in both training and testing and producing overly optimistic results. The primary task will treat cars, buses, and vans as a single **vehicle** class so the experiment focuses on temporal input rather than fine-grained vehicle classification. If time permits, BDD100K may be used as an additional generalization dataset because it contains geographically and environmentally diverse driving videos [2].

Each sample will contain an ordered temporal window and the object-detection labels for its center frame. The study will compare four input configurations:

- 1 RGB frame, or 3 input channels;
- 3 RGB frames, or 9 stacked channels;
- 5 RGB frames, or 15 stacked channels; and
- 7 RGB frames, or 21 stacked channels.

The expected result is that multi-frame models will improve detection of small, blurred, and partially occluded vehicles because they can use information from adjacent frames. However, additional frames are also expected to increase data-loading cost, inference latency, training time, and GPU-memory usage. Performance may show diminishing returns after a certain number of frames. The goal is to measure these effects rather than assume that more frames are always better.

# Technical Approach

The project will use a pretrained YOLO object detector as the common detection backbone. YOLO is appropriate for traffic monitoring because it provides a practical real-time baseline and can be evaluated using both task-quality and system-performance metrics. The implementation will reuse the Ultralytics training and inference pipeline where possible [3].

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
3. Do multi-frame inputs provide larger improvements for small, blurred, or partially occluded vehicles?
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

- small or distant vehicles;
- motion blur;
- partial vehicle occlusion in dense traffic;
- rain, road glare, and strong shadows;
- low contrast or nighttime lighting; and
- crowded intersections and highways.

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

Temporal modeling has also been studied specifically in traffic environments. A recent spatiotemporal vehicle-detection study introduced a sequential traffic dataset and modified a YOLO-based detector to use temporal information, reporting improvements over a single-frame baseline [7]. UA-DETRAC provides a suitable controlled benchmark for this project because its traffic sequences contain frame-level vehicle boxes and challenging attributes such as occlusion, illumination changes, and truncation [1]. D2-City provides additional evidence that large-scale driving videos with frame-level detection and tracking annotations are useful for studying difficult traffic conditions [8].

This project differs from work proposing a new state-of-the-art video detector. Its contribution is a controlled ML systems study of **how many input frames are worthwhile** when the model, data split, hardware, and evaluation procedure are held constant. The project will jointly report detection quality, robustness, latency, throughput, memory, and training cost so that the result supports an operational model-selection decision.

# References

[1] L. Wen, D. Du, Z. Cai, Z. Lei, M.-C. Chang, H. Qi, J. Lim, M.-H. Yang, and S. Lyu, “UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking,” 2015. https://arxiv.org/abs/1511.04136

[2] F. Yu, H. Chen, X. Wang, W. Xian, Y. Chen, F. Liu, V. Madhavan, and T. Darrell, “BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning,” 2018. https://arxiv.org/abs/1805.04687

[3] Ultralytics, “Ultralytics YOLO Documentation and Source Code.” https://github.com/ultralytics/ultralytics

[4] X. Zhu, Y. Wang, J. Dai, L. Yuan, and Y. Wei, “Flow-Guided Feature Aggregation for Video Object Detection,” *IEEE International Conference on Computer Vision*, 2017. https://arxiv.org/abs/1703.10025

[5] X. Zhu, Y. Xiong, J. Dai, L. Yuan, and Y. Wei, “Deep Feature Flow for Video Recognition,” *IEEE Conference on Computer Vision and Pattern Recognition*, 2017. https://arxiv.org/abs/1611.07715

[6] Y. Shi, N. Wang, and X. Guo, “YOLOV: Making Still Image Object Detectors Great at Video Object Detection,” 2022. https://arxiv.org/abs/2208.09686

[7] K. Telegraph and C. Kyrkou, “Spatiotemporal Object Detection for Improved Aerial Vehicle Detection in Traffic Monitoring,” 2024. https://arxiv.org/abs/2410.13616

[8] Z. Che, G. Li, T. Li, B. Jiang, X. Shi, X. Zhang, Y. Lu, G. Wu, Y. Liu, and J. Ye, “D2-City: A Large-Scale Dashcam Video Dataset of Diverse Traffic Scenarios,” 2019. https://arxiv.org/abs/1904.01975
