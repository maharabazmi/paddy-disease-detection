# 🌾 Paddy Doctor AI: Disease Recognition & Quantitative Severity Segmentation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flutter](https://img.shields.io/badge/Flutter-3.24-02569B.svg)](https://flutter.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com/)
[![MobileSAM](https://img.shields.io/badge/MobileSAM-39MB-brightgreen.svg)](https://github.com/ChaoningZhang/MobileSAM)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Computer Vision and Mobile Health diagnostic system that combines **EfficientNet-B0**, **Grad-CAM++ Explainable AI (XAI)**, **MobileSAM (Segment Anything Model)**, and **Dual-Spectrum Digital Image Processing (DIP)** to deliver real-time pathogen classification and precise lesion severity quantification.

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
   [ 3. Foundation Segmentation: Zero-Shot Prompted MobileSAM ]
        • 39 MB TinyViT weights | ~1.7s CPU inference
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

## 🛠️ How to Run This Project

### 📋 Prerequisites
* **Python 3.10+** (Tested on Python 3.10 – 3.13)
* **Git** installed on your system
* *(Optional)* **Flutter SDK 3.24+** (if running the native Flutter app)

---

### 1. Clone the Repository
```bash
git clone https://github.com/maharabazmi/paddy-disease-detection.git
cd paddy-disease-detection
```

---

### 2. Install Python Dependencies
Install the required machine learning and server packages:
```bash
cd paddy_disease_detection
pip install -r requirements.txt
```
> **Note on Model Weights**: Both pre-trained checkpoints are bundled in the repository:
> * `checkpoints/best_efficientnet_b0.pth` (16 MB - Fine-tuned Classifier)
> * `checkpoints/mobile_sam.pt` (39 MB - Lightweight MobileSAM)

---

### 3. Run the Backend AI Server

You have two convenient ways to start the server:

#### Option A: 1-Click Launch (Windows)
Double-click **`start_backend.bat`** in the project root directory.

#### Option B: Manual Command Line
```bash
cd paddy_disease_detection
python server.py
```
The server will start listening at:
* **Local Web App**: `http://localhost:8000`
* **Network IP**: `http://<your-laptop-ip>:8000`
* **Interactive Swagger API Docs**: `http://localhost:8000/docs`

---

### 4. Access the Application on Your Smartphone

#### Method 1: Local Wi-Fi / Hotspot (Instant)
1. Ensure your smartphone and laptop are connected to the **same Wi-Fi router** or **mobile hotspot**.
2. Find your laptop's local IP (run `ipconfig` in PowerShell, e.g., `192.168.0.100`).
3. Open Chrome or Safari on your phone and go to:
   ```
   http://<your-laptop-ip>:8000
   ```
4. The mobile web application will load immediately with direct camera support!

#### Method 2: Global Public HTTPS Tunnel (Cloudflare)
If you want to access the app over **4G/5G mobile data** or present in a classroom with client-isolated Wi-Fi:
1. Double-click **`start_public_tunnel.bat`** *(or run `.\cloudflared.exe tunnel --url http://127.0.0.1:8000`)*.
2. Cloudflare will print a live public HTTPS link:
   ```text
   https://xxxx-xxxx-xxxx.trycloudflare.com
   ```
3. Open this link on **any phone anywhere in the world**!

---

### 5. Running the Flutter Mobile Client (Optional)

If you wish to run the cross-platform Flutter application:

#### In Browser (Chrome / Edge):
```bash
cd flutter_app
flutter pub get
flutter run -d chrome
```

#### On Physical Android Smartphone:
1. Connect your Android device via USB cable and enable **USB Debugging** in Developer Options.
2. Verify device is detected:
   ```bash
   flutter devices
   ```
3. Build and launch:
   ```bash
   flutter run
   ```
4. In the app, tap the **Settings icon (`⚙️`)** in the top right to configure your laptop IP or Cloudflare tunnel URL!

---

### 6. Running with Docker Compose (Optional)

To spin up the containerized microservice:
```bash
docker compose up --build
```
The container exposes port `8000` automatically.

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
│       ├── best_efficientnet_b0.pth     # Trained Classifier Checkpoint (16 MB)
│       └── mobile_sam.pt                # MobileSAM TinyViT Checkpoint (39 MB)
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
