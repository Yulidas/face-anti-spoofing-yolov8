# Face Anti-Spoofing with YOLOv8n — RGB, HSV & MiDaS

**Real-time face presentation attack detection | Academic computer vision project**

Developed during the **Master's program in Biometrics and Intelligent Vision at Université Paris-Est Créteil (UPEC)**, this project investigates whether two YOLOv8-based detectors trained on different color representations can improve the identification of face presentation attacks.

The work covers **dataset assembly, automatic face localization and YOLO annotation, dataset preparation, model fine-tuning, webcam inference, and decision-level fusion**. An optional **MiDaS monocular depth-estimation module** provides an experimental additional cue.

> **Research prototype, not a production authentication system.** The dataset and trained checkpoints are not included in this public repository. Model confidence scores are not calibrated liveness probabilities.


### Third-party attribution — MiDaS

The optional monocular depth-estimation component uses **[MiDaS by Intel Intelligent Systems Lab (Intel ISL / @isl-org)](https://github.com/isl-org/MiDaS)**. MiDaS was developed independently of this academic project. The MiDaS source code, pretrained models, and original research are **not my work**; my contribution is their experimental integration into the RGB/HSV anti-spoofing decision pipeline.

- **Original repository:** [github.com/isl-org/MiDaS](https://github.com/isl-org/MiDaS)
- **Authors to credit (foundational paper):** René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun.
- **Paper:** Ranftl et al., *Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer*, IEEE TPAMI, 2022. See the [official project's citation instructions](https://github.com/isl-org/MiDaS#citation).
- **License:** [MIT License — Copyright (c) 2019 Intel ISL](https://github.com/isl-org/MiDaS/blob/master/LICENSE).

If a DPT-based MiDaS model is used, also cite Ranftl, Bochkovskiy, and Koltun, *Vision Transformers for Dense Prediction* (ICCV 2021), as requested by the original project. This repository links to MiDaS as a separate dependency rather than redistributing its source or weights.

## 1. Project goals

The primary task is to classify detected face regions into two classes:

| Class | YOLO class ID | Meaning |
| --- | ---: | --- |
| **FAKE** | `0` | Face presentation attack, such as a printed picture or an image displayed on a screen |
| **REAL** | `1` | Bona fide face presentation |

The project explores three inference configurations:

1. **HSV only:** a YOLOv8 model fine-tuned using HSV-transformed images.
2. **RGB only:** a separately fine-tuned YOLOv8 model using the original color-image dataset.
3. **RGB + HSV:** spatial matching of detections followed by a conservative decision rule. Optional MiDaS pseudo-depth estimation can be enabled for additional analysis.

Here, **RGB** denotes the conventional color-image branch. The inference code uses OpenCV's native **BGR** frames for that branch to preserve the historical model pipeline, and applies `BGR → HSV` before sending frames to the HSV model.

## 2. Dataset and preparation

I assembled a **balanced dataset of 12,090 face images**, organized into two classes:

| Class | Images | Share |
| --- | ---: | ---: |
| REAL | 6,045 | 50% |
| FAKE | 6,045 | 50% |
| **Total** | **12,090** | **100%** |

The same original examples were used for both experiments:

- **Color dataset:** original face images for the standard-color model.
- **HSV dataset:** the same images transformed into HSV color space for the second model.

These are **two representations of the same underlying dataset**, not 24,180 independently collected examples.

### Annotation and dataset splitting

`collect.py` automates face localization and writes YOLO-format bounding-box annotations. The operator specifies the class of the source material (`0` for FAKE or `1` for REAL); the script does **not** independently determine ground-truth authenticity. It can collect frames from a webcam or process existing image files, with optional confidence and blur filters.

`prepare_dataset.py` pairs images with their `.txt` labels, splits them into subsets, and automatically generates `data.yaml` containing the image paths and class mapping.

The configured split is **70% training, 15% validation, and 15% test**:

| Subset | Proportion | Expected images for 12,090 examples |
| --- | ---: | ---: |
| Training | 70% | 8,463 |
| Validation | ~15% | 1,813 |
| Test | ~15% | 1,814 |

These are counts implied by the current splitting algorithm (which assigns the rounding remainder to the test set), **not independently verified counts from the original training runs**. The script performs a seeded random split and does **not** enforce identity-disjoint or session-disjoint partitions. A rigorous biometric evaluation should separate individuals and acquisition sessions, and use matching original-image partitions for RGB and HSV to avoid leakage.

## 3. Model training

Two detectors were developed by **fine-tuning pretrained YOLOv8 models** on the color and HSV datasets separately.

The final architecture reported for the project was **YOLOv8 Nano (`yolov8n`)**, chosen for its relatively small size and suitability for experimentation with webcam inference. A larger YOLOv8 configuration was also explored, but its memory requirements and unsuccessful training behavior under the available conditions motivated the use of Nano.

`train.py` provides a configurable Ultralytics training entry point. Its defaults include 640-pixel inputs, 160 epochs, and automatic batch selection; these are **repository settings**, not proof of the exact parameters or final performance achieved in every historical run.

**Important:** The original checkpoints and training logs are not available in this repository. Accuracy figures suggested by historical checkpoint filenames have therefore **not** been presented as independently verified evaluation results.

## 4. Real-time inference and decision fusion

```text
                       Live webcam (OpenCV)
                                |
                     +----------+----------+
                     |                     |
              BGR -> HSV conversion     Color input
                     |                     |
                YOLOv8 HSV             YOLOv8 RGB
                     |                     |
                     +------ IoU matching--+
                                |
                    Decision-level fusion
                     |                  |
               FAKE takes precedence   REAL requires
               if either model         agreement from
               predicts FAKE           both detectors
                                |
                    Temporal confirmation
                                |
                    Final on-screen verdict
                                |
                  Optional MiDaS depth cue
```

### `detect.py` — individual and combined detectors

- `--mode hsv`: runs only the HSV checkpoint.
- `--mode rgb`: runs only the standard-color checkpoint.
- `--mode both`: associates detections using bounding-box **intersection over union (IoU)** and combines their decisions.

In combined mode, **FAKE takes priority if either matched detector predicts FAKE**. A **REAL** result requires both detectors to predict REAL; their reported detection confidences are then averaged **50/50**. Missing/mismatched detections do not automatically produce REAL.

The webcam stream is displayed continuously, while model inference can be spaced using `--interval` (default: 0.35 seconds). By default, two consistent positive observations are required before displaying REAL (`--stable-real 2`). A large face bounding box can also trigger a **TOO CLOSE** rejection; this is an apparent-size heuristic, **not** a calibrated distance measurement.

The compact overlay presents the **final verdict** and each available model's **predicted class and raw confidence**. Different rejection or verification states reflect the rules of the application rather than ground-truth labels.

### `fusion_yolo_midas.py` — optional pseudo-depth cue

This script reuses the same RGB/HSV decision rules and optionally loads **MiDaS**, an external monocular depth-estimation model. It estimates a relative contrast between central and lateral facial regions and may flag low pseudo-relief.

MiDaS is **not a physical depth sensor**. It can estimate apparent facial relief even when the camera is viewing a printed photograph or a screen. Its output is therefore used as a **heuristic caution/rejection cue, not positive proof of liveness**. In selective mode, depth estimation is performed periodically and in difficult cases rather than on every frame. Default thresholds and fusion rules are experimental and require evaluation with real attack examples.

## 5. Repository structure

```text
face-anti-spoofing-yolov8/
├── collect.py               # Face collection and YOLO label generation
├── prepare_dataset.py       # Dataset splitting and data.yaml generation
├── train.py                 # YOLOv8 fine-tuning entry point
├── detect.py                # Live webcam: HSV, RGB, or BOTH
├── fusion_yolo_midas.py     # Live webcam: RGB + HSV + optional MiDaS
├── antispoof/
│   ├── __init__.py
│   └── core.py              # Reusable detection/fusion utilities
├── requirements.txt
├── .gitignore
└── README.md
```

**Not distributed:** personal face images, private dataset, trained `.pt` checkpoints, experiment outputs, and the third-party MiDaS source checkout.

## 6. Installation (Ubuntu/Linux)

Python **3.10 or 3.11** is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The public repository does not contain trained checkpoints. To run the webcam demos, supply your own compatible YOLOv8 models with **class `0 = Fake`, `1 = Real`**, for example:

```text
models/
├── Tra_hsvversion_acc90.pt
└── Tra_version_acc88.pt
```

These are **historical example filenames**, not verified model performance values. The `models/` folder is ignored by Git.

### Test the models separately

**HSV only**

```bash
python detect.py --mode hsv \
  --hsv-model models/Tra_hsvversion_acc90.pt --input 0
```

**Standard-color (RGB) model only**

```bash
python detect.py --mode rgb \
  --rgb-model models/Tra_version_acc88.pt --input 0
```

**Both models — 50/50 fusion, FAKE priority**

```bash
python detect.py --mode both \
  --hsv-model models/Tra_hsvversion_acc90.pt \
  --rgb-model models/Tra_version_acc88.pt \
  --input 0 --interval 0.4 --stable-real 2
```

Press **Q** or **Esc** to close the webcam window. If camera index `0` is unavailable, try `--input 1`.

### Optional MiDaS integration

MiDaS is maintained separately: [isl-org/MiDaS](https://github.com/isl-org/MiDaS). To experiment with it, install the additional dependencies and obtain a local copy:

```bash
python -m pip install torch torchvision timm einops
git clone https://github.com/isl-org/MiDaS.git MiDaS
```

Then run:

```bash
python fusion_yolo_midas.py \
  --hsv-model models/Tra_hsvversion_acc90.pt \
  --rgb-model models/Tra_version_acc88.pt \
  --midas-repo ./MiDaS \
  --midas-model MiDaS_small \
  --source 0 --depth-mode selective --show-depth
```

MiDaS may download its pretrained weights on first use. The local `MiDaS/` checkout is ignored by this project's Git configuration. If you want to compare the two detectors **without** depth estimation while using the fusion script, specify `--depth-mode off` and omit `--show-depth`.

## 7. Data collection and training examples

Install the optional collection dependencies:

```bash
python -m pip install cvzone mediapipe
```

Example commands on datasets you are permitted to use:

```bash
# Collect bona fide webcam frames and produce labels
python collect.py --source 0 --class-id 1 --output dataset/All

# Annotate existing presentation-attack images
python collect.py --source /path/to/fake_images --class-id 0 --output dataset/All

# Split image/label pairs and generate data.yaml
python prepare_dataset.py --input dataset/All --output dataset/SplitData

# Fine-tune a YOLOv8 Nano model
python train.py --model yolov8n.pt \
  --data dataset/SplitData/data.yaml --epochs 160
```

For an HSV experiment, a **separately transformed HSV dataset** must first be prepared using the **same underlying examples and partition assignments**. The original full offline HSV conversion workflow is not included in this compact repository.

## 8. Limitations and suggested evaluation

This project is an academic proof of concept. Before making quantitative accuracy or security claims, it would be important to:

- Restore and inspect the final training logs and checkpoint metadata.
- Evaluate HSV, RGB, and fusion variants on an independent, identity-disjoint test set.
- Measure **APCER**, **BPCER**, and **ACER**, alongside precision/recall and inference latency where appropriate.
- Test printed photos, display-based replay attacks, variations in lighting, pose, distance, and camera hardware.
- Calibrate thresholds on a separate validation set; confidence scores from different YOLO detectors are not inherently comparable probabilities.
- Run an ablation study comparing the two-detector system **with and without MiDaS**.

No dataset images, real biometric measurements, independently verified benchmark metrics, or validated anti-spoofing accuracy are claimed to be included here.

## 9. Credits and responsible use

**Academic contribution:** assembling the dataset, setting up collection/automatic bounding-box annotation, preparing the two image representations, fine-tuning YOLOv8 models, and developing real-time inference and experimental fusion.

**External tools and models:** [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [PyTorch](https://pytorch.org/), [OpenCV](https://opencv.org/), and [MiDaS by Intel ISL / @isl-org](https://github.com/isl-org/MiDaS). For MiDaS authorship, academic citations, and the MIT license, see **Third-party attribution — MiDaS** above. MiDaS source code and weights are **not authored by this project's developer** and are **not bundled in this repository**.

Only use facial images for which appropriate permission and data-protection safeguards exist. This demonstration must not be treated as a production-ready identity-verification or access-control solution.

## License

This project is released under the GNU Affero General
Public License v3.0 (AGPL-3.0).

The original project code and experimental integration
were developed as part of a Master's project at
Université Paris-Est Créteil (UPEC), France, in 2023.

Third-party components retain their respective licenses:
- Ultralytics YOLOv8: AGPL-3.0
- MiDaS (Intel ISL): MIT License

See the LICENSE file and MiDaS/LICENSE for details.
