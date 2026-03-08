# Computer Vision Algorithms for Human Activity Recognition (HAR)


## Abstract
This project investigates vision-based Human Activity Recognition (HAR) for smart home scenarios. It performs a comparative analysis between **RGB-based (appearance)** and **Skeleton-based (motion)** approaches.

The models were evaluated on a curated subset of the Kinetics dataset (7,000 clips, 400 actions). The study explores whether lightweight spatial encoders or skeleton representations are more suitable for limited-data settings and privacy-conscious applications.

## Key Features
* **Multi-Modality Analysis:** Comparison of RGB video features vs. Skeleton keypoints.
* **RGB Pipelines:** Implementation of CNN-based learners including ResNet, MobileNet, and VGG.
* **Skeleton Pipelines:** Pose estimation using MediaPipe, HRNet, and PIXIE.
* **Temporal Modeling:** Utilization of BiLSTMs, Transformers, and 3D CNNs (R3D-18).
* **Smart Home Focus:** Optimized for detecting daily living activities and safety-critical events (e.g., falls).

## Model Architectures

### 1. RGB-Based Models (CNN)
We evaluated five distinct architectures that treat video as a sequence of RGB frames:
* **MobileNet-V2 + MLP:** (Best Performer) Lightweight 2D encoder with temporal averaging.
* **ResNet-18 + BiLSTM:** Frame extraction followed by a bidirectional LSTM.
* **VGG-16 + Transformer:** Deep feature extraction with an attention-based temporal encoder.
* **TSM (ResNet-18):** Temporal Shift Module for efficient motion sensing.
* **R3D-18:** A 3D-CNN approach capturing spatiotemporal features simultaneously.

### 2. Skeleton-Based Models
We extracted joint coordinates to learn motion patterns while abstracting appearance (privacy-preserving):
* **MediaPipe Pose:** 33 3D landmarks + visibility scores.
* **HRNet (COCO-17):** High-Resolution Network providing 17 2D keypoints.
* **PIXIE / SMPL-X:** Extracts 145 3D body joints (dense mesh).
* *Temporal Head:* All skeleton inputs are processed by a shared BiLSTM.

## Dataset
* **Source:** A curated subset of the **Kinetics** dataset.
* **Size:** 7,000 short video clips.
* **Classes:** 400 distinct action classes.
* **Preprocessing:**
    * **RGB:** Frames sampled at stride 4, resized to 224x224.
    * **Skeleton:** Sequences truncated/padded to $T=300$ frames.

## Experimental Results
Top-1 and Top-5 Accuracy on the test split:

| Modality | Model | Top-1 Accuracy | Top-5 Accuracy |
| :--- | :--- | :--- | :--- |
| **RGB** | **MobileNet-V2 + MLP** | **12.0%** | **35.0%** |
| RGB | VGG-16 + Transformer | 10.0% | 30.0% |
| RGB | ResNet-18 + BiLSTM | 9.9% | 23.0% |
| RGB | R3D-18 (3D CNN) | 9.0% | 27.0% |
| RGB | TSM (ResNet-18) | 7.0% | 17.0% |
| **Skeleton** | **HRNet + BiLSTM** | **4.0%** | 10.5% |
| Skeleton | **MediaPipe + BiLSTM** | 3.8% | **12.0%** |
| Skeleton | PIXIE + BiLSTM | 3.8% | 10.5% |

**Key Findings:**
1.  **MobileNet-V2** performed best overall, suggesting that compact 2D encoders with simple temporal aggregation generalize better on smaller datasets.
2.  **MediaPipe** offered the best Top-5 accuracy among skeleton models due to its 3D depth and visibility scores.
3.  **Skeleton methods** lagged in absolute accuracy but remain vital for privacy-preserving monitoring.

## Installation & Requirements
*(Note: Please ensure you have Python 3.8+ installed)*

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name
