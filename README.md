# 🌾 Paddy Doctor AI: Disease Recognition & Quantitative Severity Segmentation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flutter](https://img.shields.io/badge/Flutter-3.24-02569B.svg)](https://flutter.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Computer Vision and Mobile Health diagnostic system that combines **EfficientNet-B0**, **Grad-CAM++ Explainable AI (XAI)**, **Meta's Segment Anything Model (SAM)**, and **Dual-Spectrum Digital Image Processing (DIP)** to deliver real-time pathogen classification and precise lesion severity quantification.

---

## 🌟 Key Features & Pipeline Architecture

```
[ Farmer's Smartphone (Flutter UI / Web PWA) ]
                      │
                      ▼ (POST /predict)
   [ 1. Lightweight Classifier: EfficientNet-B0 ]
        • 97.60% Accuracy | 97.80% Macro F1 | 5.3M Parameters
                      │
                      ▼
   [ 2. Explainable AI: Grad-CAM++ Attribution Map ]
        • 224x224 Bilinear Feature Upsampling
        • Connected Components & Salient Peak Extraction
                      │
                      ▼
   [ 3. Foundation Segmentation: Zero-Shot Prompted SAM ]
        • Box Prompt + Positive Lesion Point + Negative Background Point
        • Multi-mask alignment selection via activation-weighted score
                      │
                      ▼
   [ 4. Dual-Spectrum Canopy DIP & Severity Quantification ]
        • ExG (>5) + Green (H:25-95) + Necrotic (H:10-25) Leaf Extraction
        • Severity % = (Lesion Pixels / Canopy Pixels) × 100
                      │
                      ▼
[ Live Diagnosis + Severity Gauge + Pathological Overlays + Agronomic Advice ]
```

---

## 📊 Benchmark Test Performance (1,041 Unseen Samples)

| Metric | Score |
| :--- | :--- |
| **Overall Test Accuracy** | **97.60%** |
| **Macro F1-Score** | **97.80%** |
| **Weighted F1-Score** | **97.60%** |
| **Macro ROC-AUC** | **99.66%** |
| **Macro PR-AUC** | **98.73%** |

### Per-Class Pathogen Metrics:
* **Bacterial Leaf Blight**: 98.08% F1 | 99.87% ROC-AUC
* **Rice Blast**: 96.00% F1 | 99.58% ROC-AUC
* **Brown Spot**: 95.16% F1 | 99.41% ROC-AUC
* **Dead Heart**: 97.52% F1 | 99.78% ROC-AUC
* **Healthy (Normal)**: 98.92% F1 | 99.98% ROC-AUC
* *All 10 classes achieve F1 $\ge 95.16\%$!*

---

## 📱 Mobile App UI (Flutter Material 3)

The mobile client ([`flutter_app/`](flutter_app)) includes:
* **Camera & Gallery Image Input** with cross-platform in-memory byte rendering.
* **Severity Progress Meter** with animated color-coded health tiers:
  * Healthy (0%), Mild (0-10%), Moderate (10-25%), Severe (25-50%), Very Severe (>50%).
* **Multi-Tab Visualizer**:
  * 🌿 **SAM Segmented Composite** (Soft green canopy + Vivid red lesion overlay)
  * 🔥 **Grad-CAM++ Attribution Heatmap**
  * 🎯 **Clean Binary Lesion Mask**
  * 📷 **Original Field Leaf**
* **Agronomic Treatment Recommendation Card**: Dosages, chemical management, and cultural prevention tips.

---

## 🚀 Quick Start Guide

### 1. Start the FastAPI Backend
```bash
cd paddy_disease_detection
python server.py
```
*Server starts on `http://0.0.0.0:8000`.*

### 2. Launch the Public Tunnel (For Smartphone Testing)
```bash
.\cloudflared.exe tunnel --url http://127.0.0.1:8000
```
*Gives you an instant public HTTPS link (`https://xxxx.trycloudflare.com`) accessible from any smartphone.*

### 3. Run the Flutter Mobile App
```bash
cd flutter_app
flutter pub get
flutter run -d chrome     # Run in Chrome/Edge
# OR: flutter run          # Run on connected Android smartphone
```

---

## 📦 Project Structure

```
├── paddy_disease_detection/
│   ├── server.py                        # FastAPI Backend & Inference Engine
│   ├── paddy_disease_sam_pipeline.ipynb # Full Research & Training Notebook
│   ├── static/app.html                  # Responsive Mobile Web PWA App
│   ├── Dockerfile                       # Production Container Definition
│   ├── requirements.txt                 # Backend Python Dependencies
│   └── checkpoints/
│       └── best_efficientnet_b0.pth     # Trained Classifier Checkpoint
├── flutter_app/
│   ├── lib/main.dart                    # Complete Flutter Material 3 Code
│   ├── pubspec.yaml                     # Flutter Dependencies
│   └── android/                         # Android Scaffolding & Permissions
├── start_backend.bat                    # 1-Click Server Launcher
├── start_public_tunnel.bat             # 1-Click Cloudflare Tunnel Launcher
├── docker-compose.yml                   # Container Orchestration
└── .gitignore                           # Git Exclusion Rules
```

---

## 📄 License
This project is licensed under the MIT License.
