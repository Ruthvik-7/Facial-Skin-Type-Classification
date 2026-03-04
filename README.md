# Facial Skin Type Classification

End-to-end ML pipeline for classifying facial skin as **oily**, **dry**, or **normal** from images. Uses a hybrid feature extraction approach combining HOG texture descriptors with deep CNN features, then classifies with ResNet50 and VGG16. Evaluated via 5-fold cross-validation on a manually curated dataset of 166 images.

**ResNet50 achieved 84.16% average accuracy (±4.58) — outperforming VGG16 at 80.39%.**

---

## Pipeline overview

```
Raw facial image
      │
      ▼
  Preprocessing
  ├── Gaussian Blur          → noise reduction
  ├── Bilateral Filtering    → edge-preserving smoothing
  └── CLAHE                  → contrast enhancement
      │
      ▼
  Feature Extraction (hybrid)
  ├── HOG descriptors        → texture & structural features
  └── MobileNetV2            → deep CNN features (ImageNet pretrained, fine-tuned)
      │
      ▼
  Feature concatenation (HOG + CNN → single vector)
      │
      ├──► ResNet50 classifier  → softmax → {Oily, Normal, Dry}   [84.16% acc]
      └──► VGG16 classifier     → softmax → {Oily, Normal, Dry}   [80.39% acc]
```

---

## Dataset

Manually curated using a 50MP phone camera. All images annotated by hand.

| Property | Value |
|----------|-------|
| Total images | 166 |
| Classes | Oily, Dry, Normal |
| Age range | 9–40 years |
| Annotation method | Manual |
| Split strategy | 5-fold cross-validation |

**Labeling logic (brightness-based clustering via VGG16 + K-Means):**
- Darker images → Oily skin
- Medium brightness → Normal skin
- Brighter images → Dry skin

---

## Preprocessing

Three techniques are applied in sequence before feature extraction:

**Gaussian Blur** — smooths the image, reduces high-frequency noise without destroying edge information.

**Bilateral Filter** — edge-preserving noise reduction. Unlike Gaussian, it considers both spatial distance and pixel intensity, keeping pore and texture boundaries intact.

**CLAHE (Contrast Limited Adaptive Histogram Equalization)** — boosts local contrast in low-light or low-contrast regions. Particularly useful for distinguishing subtle skin texture variations across different skin types.

---

## Feature extraction

**HOG (Histogram of Oriented Gradients)**
Captures local gradient orientations across image patches — good for encoding texture patterns like pore density and surface roughness that differ across skin types.

**MobileNetV2**
Lightweight CNN pretrained on ImageNet, fine-tuned for this task. Extracts hierarchical deep features from the skin images. Chosen for efficiency without sacrificing representation quality.

Both feature sets are concatenated into a single vector before being passed to the classifier. This fusion captures both low-level texture and high-level semantic patterns.

---

## Classification models

**ResNet50**
Uses residual connections to mitigate vanishing gradient issues in deep networks. A Global Average Pooling layer reduces dimensionality before the fully connected layers. Softmax output over 3 classes. Optimizer: Adam. Loss: categorical cross-entropy.

**VGG16**
Deep CNN with 16 weight layers. Also uses GAP before FC layers for dimensionality reduction. Showed higher stability across folds (lower std dev: ±1.59 vs ResNet's ±4.58), despite slightly lower mean accuracy.

---

## Results

**5-fold cross-validation accuracy:**

| Model | Mean Accuracy | Std Dev |
|-------|--------------|---------|
| ResNet50 | **84.16%** | ±4.58% |
| VGG16 | 80.39% | ±1.59% |

ResNet50 wins on average accuracy — residual connections help it learn deeper skin texture hierarchies. VGG16 is more stable across folds — worth considering if consistency matters more than peak performance.

Common misclassifications occur between **normal** and **dry** skin types, which share similar brightness and texture properties.

---

## Project structure

```
├── dataset/
│   ├── oily/                     # raw oily skin images
│   ├── dry/                      # raw dry skin images
│   └── normal/                   # raw normal skin images
├── preprocessing.py              # Gaussian Blur, Bilateral Filter, CLAHE
├── feature_extraction.py         # HOG + MobileNetV2 hybrid pipeline
├── train_resnet50.py             # ResNet50 training + 5-fold CV
├── train_vgg16.py                # VGG16 training + 5-fold CV
├── predict.py                    # inference on new images
├── results/
│   ├── confusion_matrices/       # per-fold confusion matrices
│   └── accuracy_plots.png
└── requirements.txt
```

---

## Setup

```bash
pip install tensorflow keras opencv-python scikit-learn numpy matplotlib pillow
```

GPU training recommended. Verify TensorFlow detects your GPU:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## Run

**Train ResNet50:**
```bash
python train_resnet50.py --dataset ./dataset --epochs 50 --folds 5
```

**Train VGG16:**
```bash
python train_vgg16.py --dataset ./dataset --epochs 50 --folds 5
```

**Predict on a new image:**
```python
from predict import classify_skin

result = classify_skin("path/to/face_image.jpg", model="resnet50")
print(result)
# → {'skin_type': 'Oily', 'confidence': 0.87}
```

---

## Tech stack

| Component | Tool |
|-----------|------|
| Image preprocessing | OpenCV (Gaussian Blur, Bilateral Filter, CLAHE) |
| Texture features | HOG (Scikit-learn / OpenCV) |
| Deep features | MobileNetV2, ResNet50, VGG16 (TensorFlow/Keras) |
| Clustering | K-Means (Scikit-learn) |
| Evaluation | 5-fold cross-validation, confusion matrices |
| Optimizer | Adam |
| Loss | Categorical cross-entropy |

---

## Limitations & future work

- Dataset size (166 images) is small — model generalizability across diverse skin tones, ethnicities, and lighting conditions is limited
- Normal/dry misclassifications need a larger, more balanced dataset to resolve
- Future directions: self-supervised learning, transformer-based architectures (ViT), spectral imaging input, real-world clinical validation

---

*IIIT Dharwad · Dept. of Data Science & Artificial Intelligence*
