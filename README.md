# 🔍 AI-Driven Image Forensics & Security Analytics

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

A deep learning pipeline for **image forensics and security analytics** — detecting image tampering, splicing, copy-move forgery, and morphed document detection using CNN-based spatial feature analysis, Error Level Analysis (ELA), and Grad-CAM explainability.

---

## 📌 Project Overview

Digital image manipulation is increasingly used in fake news, identity fraud, and legal evidence tampering. This project builds an end-to-end forensic analysis system that:

- Detects **tampered regions** in images using deep learning
- Performs **Error Level Analysis (ELA)** to find re-compression inconsistencies
- Uses **Grad-CAM** to produce interpretable heatmaps of manipulated areas
- Tests **adversarial robustness** against JPEG compression and Gaussian noise attacks
- Deploys as a **real-time Streamlit dashboard** for forensic analysis

---

## 🏗️ System Architecture

```
Input Image
     │
     ▼
┌─────────────────────────────────────┐
│         Preprocessing Module        │
│  - Resize, Normalize, Convert       │
│  - ELA Generation                   │
│  - Noise Residual Extraction        │
└──────────────────┬──────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌──────────────┐     ┌─────────────────┐
│  CNN Feature │     │  Signal-Based   │
│  Extraction  │     │  Forensic       │
│  (ResNet-50/ │     │  Features (ELA, │
│   VGG-16)    │     │  Frequency DOM) │
└──────┬───────┘     └────────┬────────┘
       └──────────┬───────────┘
                  ▼
        ┌─────────────────┐
        │  Classification  │
        │  Head (Authentic │
        │  vs Manipulated) │
        └────────┬─────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
  ┌──────────┐     ┌────────────┐
  │ Grad-CAM │     │  Forensic  │
  │ Heatmap  │     │   Report   │
  └──────────┘     └────────────┘
```

---

## 🔑 Key Features

| Feature | Description |
|---|---|
| **CNN Transfer Learning** | Fine-tuned ResNet-50 and VGG-16 on image forensics datasets |
| **Error Level Analysis** | Detects re-compression inconsistencies from JPEG artifacts |
| **Noise Residual Analysis** | Identifies source camera mismatches via noise patterns |
| **Frequency Domain Analysis** | DCT-based spectral anomaly detection |
| **Grad-CAM Explainability** | Visual heatmaps of tampered regions for human-auditable decisions |
| **Adversarial Robustness** | Tested against JPEG compression (Q=70,80,90) and Gaussian noise |
| **Streamlit Dashboard** | Real-time forensic analysis with ELA visualisation |

---

## 📁 Project Structure

```
image-forensics-security-analytics/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── ela.py                  # Error Level Analysis module
│   ├── noise_residual.py       # Noise residual extraction
│   ├── frequency_analysis.py   # DCT/frequency domain features
│   ├── model.py                # ResNet-50 and VGG-16 models
│   ├── train.py                # Training pipeline
│   ├── evaluate.py             # Evaluation metrics
│   ├── gradcam.py              # Grad-CAM implementation
│   ├── predict.py              # Single image prediction
│   └── dataset.py              # Dataset loader
│
├── notebooks/
│   ├── 01_ELA_Analysis.ipynb           # ELA exploration notebook
│   ├── 02_Model_Training.ipynb         # Training walkthrough
│   ├── 03_GradCAM_Visualisation.ipynb  # Grad-CAM demo
│   └── 04_Adversarial_Testing.ipynb    # Robustness evaluation
│
├── app.py                      # Streamlit dashboard
├── data/
│   └── sample_images/          # Sample authentic and tampered images
│
├── models/                     # Saved model checkpoints
└── results/                    # Output heatmaps and reports
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Amitmishra11-X/image-forensics-security-analytics.git
cd image-forensics-security-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
Upload any image and instantly get:
- ELA visualisation
- Grad-CAM heatmap of tampered regions
- Authenticity prediction with confidence score
- Forensic analysis report

### 2. Train the Model
```bash
python src/train.py --model resnet50 --epochs 30 --batch_size 32
```

### 3. Predict on a Single Image
```bash
python src/predict.py --image path/to/image.jpg --model resnet50
```

### 4. Run Adversarial Robustness Test
```bash
python src/evaluate.py --attack jpeg --quality 80
python src/evaluate.py --attack gaussian --sigma 25
```

---

## 🧪 Techniques Used

### Error Level Analysis (ELA)
When a JPEG image is saved, compression creates uniform error levels. If a region was edited and re-saved, it shows **different error levels** — appearing brighter in ELA output.

```python
# How ELA works (simplified)
original = Image.open("image.jpg")
resaved = original.save("temp.jpg", quality=90)
ela_image = ImageChops.difference(original, resaved)
# Bright areas = potentially tampered
```

### Grad-CAM
Gradient-weighted Class Activation Mapping highlights which regions the CNN focused on:

```python
# Grad-CAM pipeline
gradients = model.get_gradients(image, target_class)
weights = torch.mean(gradients, dim=[2, 3])
cam = torch.sum(weights[:, :, None, None] * feature_maps, dim=1)
heatmap = F.relu(cam)
```

### Adversarial Robustness Testing
```python
# JPEG compression attack
attacked = apply_jpeg_compression(image, quality=80)
score_original = model.predict(image)
score_attacked = model.predict(attacked)
degradation = score_original - score_attacked
```

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| ResNet-50 (fine-tuned) | 91.3% | 89.7% | 92.1% | 90.9% | 0.96 |
| VGG-16 (fine-tuned) | 88.6% | 87.2% | 89.4% | 88.3% | 0.94 |
| Custom CNN (baseline) | 83.1% | 81.5% | 84.7% | 83.1% | 0.91 |

**Adversarial Robustness (ResNet-50):**

| Attack | Clean Accuracy | Attacked Accuracy | Degradation |
|---|---|---|---|
| JPEG Q=90 | 91.3% | 89.1% | -2.2% |
| JPEG Q=80 | 91.3% | 85.4% | -5.9% |
| Gaussian σ=15 | 91.3% | 87.2% | -4.1% |
| Gaussian σ=25 | 91.3% | 82.6% | -8.7% |

---

## 🔬 Research Connections

This project is directly related to:
- **Image Manipulation Detection** — detecting splicing, copy-move forgery
- **Morphed Document Detection** — identifying tampered passport/ID images
- **Source Camera Identification** — noise residual fingerprinting
- **Assistive Technology** — applying forensics pipeline to scene understanding for visually impaired users

---

## 📚 References

1. Fridrich, J., Goljan, M., & Du, R. (2003). Detecting LSB steganography in color and grayscale images.
2. Zhou, P., Han, X., Morariu, V.I., Davis, L.S. (2018). Learning Rich Features for Image Manipulation Detection. CVPR.
3. Selvaraju, R.R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV.
4. Rossler, A. et al. (2019). FaceForensics++: Learning to Detect Manipulated Facial Images. ICCV.

---

## 👨‍💻 Author

**Amit Mishra**
B.Tech CSE (3rd Year) | L.D.A.H Rajkiya Engineering College, Mainpuri | AKTU
- GitHub: [@Amitmishra11-X](https://github.com/Amitmishra11-X)
- Email: am0651465@gmail.com

---

## 📄 License

This project is licensed under the MIT License.
