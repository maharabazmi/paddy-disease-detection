# 🌾 Paddy Doctor AI - Mobile Application (Option 2)

A cross-platform Flutter mobile application connected to the **FastAPI + PyTorch (EfficientNet-B0 + Grad-CAM++ + SAM + DIP)** backend microservice.

---

## 📱 App Highlights & Features

1. **Camera & Gallery Input**: Capture live paddy leaves directly in the field or pick images from the device gallery.
2. **Dynamic Server Configuration**: Tap the Settings gear icon in the top right to configure your laptop/server IP address anytime (`http://<laptop-ip>:8000`). Pre-configured by default to `http://192.168.0.100:8000`.
3. **Multi-Tab Pathological Visualization**:
   - **SAM Segmented Overlay**: Composite image showing detected canopy in green tint and segmented disease lesions in red.
   - **Grad-CAM++ Heatmap**: Visual explanation highlighting the precise anatomical regions driving the classification.
   - **Binary Lesion Mask**: High-contrast segmentation mask isolating diseased pixels.
   - **Original Leaf**: Raw captured photo for reference.
4. **Severity Meter**: Interactive progress gauge displaying exact severity percentage (e.g., `32.18%`) and classification tier (`Mild`, `Moderate`, `Severe`, `Very Severe`).
5. **Agronomic Treatment Recommendation**: Real-time actionable agronomic advice tailored specifically for the diagnosed pathogen.

---

## 🚀 Quick Start Guide

### Step 1: Start the Backend Microservice
Double-click `start_backend.bat` in the root folder, or run in terminal:
```bash
cd e:\paddy\paddy_disease_detection\paddy_disease_detection
python server.py
```
> The server will load EfficientNet-B0, Grad-CAM++, and SAM, listening on `0.0.0.0:8000`.

### Step 2: Ensure Phone & Laptop Are on the Same Network
- Connect your phone and laptop to the **same Wi-Fi router** or **turn on Mobile Hotspot** on your phone and connect your laptop to it.
- Your laptop's current local Wi-Fi IP is: **`192.168.0.100`**.

### Step 3: Run the Flutter App
Open terminal in `e:\paddy\paddy_disease_detection\flutter_app` and run:
```bash
flutter pub get
flutter run
```
*(Or open the `flutter_app` folder in VS Code / Android Studio, select your connected Android device or emulator, and press F5 / Run).*

### Step 4: Test in the App
1. When the app opens, verify the server URL in Settings (top right gear icon) is set to `http://192.168.0.100:8000`.
2. Tap **"Camera"** or **"Gallery"** to load a paddy leaf image.
3. The app will send the image to the backend, run the 4-stage pipeline, and render the complete diagnosis, severity meter, and XAI/SAM overlays!
