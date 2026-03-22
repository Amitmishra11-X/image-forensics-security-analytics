# 🤚 Gesture Recognition for HCI Using CNN & Transfer Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

A real-time **hand gesture recognition system** for Human-Computer Interaction (HCI), enabling touchless control of applications using CNN-based deep learning and transfer learning.

---

## 📌 Project Overview

Traditional input devices (keyboard, mouse) create barriers for accessibility and touchless interaction. This project builds a real-time gesture recognition system that:

- Recognises hand gestures from **live webcam feed** in real-time
- Uses **transfer learning** (ResNet-50, VGG-16) for high accuracy
- Provides **Grad-CAM** visualisation to validate model attention
- Deployed as a live **Streamlit app** with confidence scores per gesture
- Trained on **ASL** (American Sign Language) and **HaGRID** datasets

---

## 🏗️ System Architecture

```
Webcam / Image Input
        │
        ▼
┌──────────────────────────────┐
│      Preprocessing           │
│  - Hand Detection (MediaPipe)│
│  - ROI Extraction            │
│  - Resize to 224×224         │
│  - Normalise                 │
└──────────────┬───────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌────────────┐   ┌─────────────┐
│ ResNet-50  │   │   VGG-16    │
│ (ImageNet  │   │ (ImageNet   │
│  weights)  │   │  weights)   │
│ Fine-tuned │   │ Fine-tuned  │
└─────┬──────┘   └──────┬──────┘
       └────────┬────────┘
                ▼
      ┌─────────────────┐
      │  Classification │
      │  Head           │
      │  (N gestures)   │
      └────────┬────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
┌────────┐         ┌────────────┐
│Gesture │         │  Grad-CAM  │
│ Label  │         │  Heatmap   │
│+ Conf. │         │ Validation │
└────────┘         └────────────┘
```

---

## 🔑 Key Features

| Feature | Description |
|---|---|
| **Real-time Recognition** | Live webcam feed with per-frame prediction |
| **Transfer Learning** | ResNet-50 and VGG-16 fine-tuned on gesture datasets |
| **Data Augmentation** | Random flips, rotations, colour jitter, affine transforms |
| **Grad-CAM Validation** | Validates model focuses on hand regions, not background |
| **Multi-dataset Support** | ASL alphabet + HaGRID (18 dynamic gestures) |
| **Benchmark Comparison** | Frozen vs full fine-tuning strategy analysis |
| **Streamlit Dashboard** | Live webcam feed with confidence bar charts |

---

## 📁 Project Structure

```
gesture-recognition-hci/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── model.py              # ResNet-50 and VGG-16 gesture models
│   ├── dataset.py            # ASL and HaGRID dataset loaders
│   ├── train.py              # Training pipeline with augmentation
│   ├── evaluate.py           # Evaluation metrics and confusion matrix
│   ├── gradcam.py            # Grad-CAM for attention validation
│   ├── predict.py            # Single image prediction
│   ├── hand_detector.py      # MediaPipe hand detection + ROI extraction
│   └── augmentation.py       # torchvision augmentation pipeline
│
├── notebooks/
│   ├── 01_Data_Exploration.ipynb       # Dataset visualisation
│   ├── 02_Model_Training.ipynb         # Training walkthrough
│   ├── 03_GradCAM_Validation.ipynb     # Attention map validation
│   └── 04_Benchmark_Comparison.ipynb   # ResNet vs VGG comparison
│
├── app.py                    # Streamlit real-time webcam app
├── data/                     # Dataset directory
├── models/                   # Saved checkpoints
└── results/                  # Output confusion matrices, plots
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Amitmishra11-X/gesture-recognition-hci.git
cd gesture-recognition-hci

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Launch Real-Time Webcam App
```bash
streamlit run app.py
```
- Opens webcam feed
- Shows predicted gesture + confidence score in real time
- Displays Grad-CAM attention overlay

### 2. Train the Model
```bash
# Train on ASL dataset
python src/train.py --model resnet50 --dataset asl --epochs 30

# Train on HaGRID dataset
python src/train.py --model vgg16 --dataset hagrid --epochs 30

# Frozen backbone (feature extraction only)
python src/train.py --model resnet50 --freeze --epochs 50
```

### 3. Evaluate Model
```bash
python src/evaluate.py --model resnet50 --checkpoint models/resnet50_best.pth
```

### 4. Predict on Single Image
```bash
python src/predict.py --image path/to/hand_image.jpg --model resnet50
```

---

## 📊 Results

### ASL Dataset (26 classes — A to Z)

| Model | Strategy | Accuracy | F1-Score | Inference |
|---|---|---|---|---|
| ResNet-50 | Full fine-tuning | **94.2%** | 0.941 | 18ms/frame |
| VGG-16 | Full fine-tuning | 91.8% | 0.916 | 24ms/frame |
| ResNet-50 | Frozen backbone | 87.3% | 0.871 | 16ms/frame |
| VGG-16 | Frozen backbone | 84.1% | 0.839 | 22ms/frame |
| Custom CNN | From scratch | 78.6% | 0.782 | 8ms/frame |

### HaGRID Dataset (18 dynamic gesture classes)

| Model | Accuracy | F1-Score |
|---|---|---|
| ResNet-50 (fine-tuned) | **91.7%** | 0.914 |
| VGG-16 (fine-tuned) | 88.4% | 0.881 |

---

## 🧪 Data Augmentation Pipeline

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.3, contrast=0.3,
        saturation=0.2, hue=0.1
    ),
    transforms.RandomAffine(
        degrees=10,
        translate=(0.1, 0.1),
        scale=(0.85, 1.15)
    ),
    transforms.RandomErasing(p=0.2),   # Simulate occlusion
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

## 🔬 Grad-CAM Validation

Grad-CAM confirms the model attends to **hand regions** and not background:

- ✅ Correct attention: Model focuses on finger positions and hand shape
- ❌ Background attention: Indicates overfitting — triggers re-training

---

## 📚 Datasets

| Dataset | Classes | Images | Description |
|---|---|---|---|
| [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) | 26 | 87,000 | American Sign Language A-Z |
| [HaGRID](https://github.com/hukenovs/hagrid) | 18 | 554,800 | Dynamic gesture recognition |

---

## 🔮 Future Work

- [ ] Extend to **dynamic/temporal gestures** using CNN + LSTM
- [ ] Add **MediaPipe Holistic** for full body gesture recognition
- [ ] Deploy as **mobile app** using TensorFlow Lite / ONNX export
- [ ] Add **custom gesture recording** for personalised HCI

---

## 📚 References

1. He, K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
2. Simonyan, K., Zisserman, A. (2015). Very Deep CNNs for Large-Scale Image Recognition. ICLR.
3. Selvaraju, R.R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV.
4. Kapitanov, A. et al. (2022). HaGRID — HAnd Gesture Recognition Image Dataset. arXiv.

---

## 👨‍💻 Author

**Amit Mishra**
B.Tech CSE (3rd Year) | L.D.A.H Rajkiya Engineering College, Mainpuri | AKTU
- GitHub: [@Amitmishra11-X](https://github.com/Amitmishra11-X)
- Email: am0651465@gmail.com

---

## 📄 License

This project is licensed under the MIT License.
