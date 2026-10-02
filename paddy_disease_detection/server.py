import io
import os
import base64
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize FastAPI App
app = FastAPI(
    title="Paddy Disease Diagnosis & Severity Quantification API",
    description="Backend microservice serving EfficientNet-B0 + Grad-CAM++ + SAM + DIP for mobile field diagnostics.",
    version="1.0.0"
)

# Enable CORS for Flutter & Web Clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WORKDIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224

CLASSES = [
    'bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight',
    'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro'
]

DISPLAY_NAMES = {
    'bacterial_leaf_blight': 'Bacterial Leaf Blight (Xanthomonas oryzae)',
    'bacterial_leaf_streak': 'Bacterial Leaf Streak (Xanthomonas oryzae pv. oryzicola)',
    'bacterial_panicle_blight': 'Bacterial Panicle Blight (Burkholderia glumae)',
    'blast': 'Rice Blast (Magnaporthe oryzae)',
    'brown_spot': 'Brown Spot (Bipolaris oryzae)',
    'dead_heart': 'Dead Heart / Yellow Stem Borer (Scirpophaga incertulas)',
    'downy_mildew': 'Downy Mildew (Sclerophthora macrospora)',
    'hispa': 'Rice Hispa (Dicladispa armigera)',
    'normal': 'Healthy Paddy Leaf (No Pathogen Detected)',
    'tungro': 'Rice Tungro Spherical / Bacilliform Virus'
}

TREATMENTS = {
    'normal': "Crop is healthy! Maintain regular irrigation schedule, monitor weekly, and apply standard balanced NPK fertilization.",
    'blast': "Foliar spray with Tricyclazole 75 WP @ 0.6 g/L or Azoxystrobin 25 SC @ 1 ml/L. Avoid excess urea/nitrogen fertilizer.",
    'brown_spot': "Apply Mancozeb 75 WP @ 2 g/L or Propiconazole 25 EC @ 1 ml/L. Correct potassium and silicon deficiency in soil.",
    'bacterial_leaf_blight': "Spray Copper Hydroxide 77 WP @ 2 g/L or Streptocycline @ 0.1 g/L. Drain field temporarily to reduce humidity.",
    'bacterial_leaf_streak': "Apply balanced potassium fertilizer. Avoid clipping seedling tips before transplanting. Spray Copper Oxychloride @ 2.5 g/L.",
    'bacterial_panicle_blight': "Plant resistant cultivars. Avoid high seeding rates and excessive nitrogen. Ensure proper drainage during flowering.",
    'hispa': "Sweep-net adult beetles. Apply Chlorpyrifos 20 EC @ 2 ml/L or Quinalphos 25 EC @ 1.5 ml/L. Clip and destroy damaged leaf tips.",
    'dead_heart': "Conserve egg parasitoids (Trichogramma). Apply Cartap Hydrochloride 4G @ 10 kg/acre or Chlorantraniliprole 18.5 SC @ 0.3 ml/L.",
    'downy_mildew': "Improve field drainage immediately. Seed treatment with Metalaxyl @ 3 g/kg seed. Rogue out severely infected stunted plants.",
    'tungro': "Control the Green Leafhopper vector using Imidacloprid 17.8 SL @ 0.3 ml/L or Thiamethoxam 25 WG @ 0.2 g/L. Rogue infected hills."
}

# Image Preprocessing
eval_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Global Model Containers
classifier_model = None
sam_predictor = None
gradcam_hook = None


class GradCAMPlusPlus:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.target_layer = model.features[-1]
        self.activations = []
        self.gradients = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(m, i, o): self.activations.append(o)
        def backward_hook(m, gi, go): self.gradients.append(go[0])
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        self.activations.clear()
        self.gradients.clear()
        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
        conf = F.softmax(logits, dim=1)[0, target_class].item()
        logits[0, target_class].backward(retain_graph=True)

        A = self.activations[-1].detach()
        g = self.gradients[-1].detach()
        g2, g3 = g.pow(2), g.pow(3)
        denom = 2.0 * g2 + torch.sum(A * g3, dim=(2, 3), keepdim=True)
        denom = torch.where(denom != 0.0, denom, torch.ones_like(denom) * 1e-8)
        w = g2 / denom
        alpha = torch.sum(w * F.relu(g), dim=(2, 3), keepdim=True)
        cam = torch.sum(alpha * A, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
        return cam.squeeze().cpu().numpy(), target_class, conf


def generate_sam_prompts_from_cam(cam_map, target_shape=(IMAGE_SIZE, IMAGE_SIZE), threshold=0.45, margin_ratio=0.10, min_area=80):
    h_target, w_target = target_shape[:2]
    cam_resized = cv2.resize(cam_map, (w_target, h_target), interpolation=cv2.INTER_LINEAR)
    binary_cam = (cam_resized >= threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_cam)

    valid_boxes = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            valid_boxes.append([x, y, x + bw, y + bh])

    if len(valid_boxes) > 0:
        boxes_arr = np.array(valid_boxes)
        x_min, y_min = int(np.min(boxes_arr[:, 0])), int(np.min(boxes_arr[:, 1]))
        x_max, y_max = int(np.max(boxes_arr[:, 2])), int(np.max(boxes_arr[:, 3]))
    else:
        high_act = np.where(cam_resized >= np.percentile(cam_resized, 85))
        if len(high_act[0]) > 0:
            y_min, y_max = int(np.min(high_act[0])), int(np.max(high_act[0]))
            x_min, x_max = int(np.min(high_act[1])), int(np.max(high_act[1]))
        else:
            x_min, y_min, x_max, y_max = 0, 0, w_target, h_target

    pad_x = int((x_max - x_min) * margin_ratio)
    pad_y = int((y_max - y_min) * margin_ratio)

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(w_target, x_max + pad_x)
    y_max = min(h_target, y_max + pad_y)

    bounding_box = np.array([x_min, y_min, x_max, y_max])
    peak_y, peak_x = np.unravel_index(np.argmax(cam_resized), cam_resized.shape)
    pos_point = np.array([[peak_x, peak_y]])

    low_candidates = np.where(cam_resized < 0.08)
    if len(low_candidates[0]) > 0:
        cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
        dists = (low_candidates[1] - cx)**2 + (low_candidates[0] - cy)**2
        furthest_idx = np.argmax(dists)
        neg_point = np.array([[low_candidates[1][furthest_idx], low_candidates[0][furthest_idx]]])
    else:
        neg_point = np.array([[0, 0]])

    return {
        "box": bounding_box,
        "pos_point": pos_point,
        "neg_point": neg_point,
        "cam_resized": cam_resized
    }


def run_sam_segmentation(image_rgb, prompts, predictor, cam_map):
    predictor.set_image(image_rgb)
    box = prompts["box"]
    pos_pt = prompts["pos_point"]
    neg_pt = prompts["neg_point"]

    point_coords = np.vstack([pos_pt, neg_pt])
    point_labels = np.array([1, 0])

    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box[None, :],
        multimask_output=True
    )

    cam_resized = cv2.resize(cam_map, (image_rgb.shape[1], image_rgb.shape[0]))
    alignment_scores = []
    for i in range(len(masks)):
        mask_i = masks[i]
        mask_area = np.sum(mask_i)
        cam_overlap = (np.sum(cam_resized[mask_i]) / (mask_area + 1e-6)) if mask_area > 0 else 0.0
        alignment_scores.append(0.5 * scores[i] + 0.5 * cam_overlap)

    best_idx = int(np.argmax(alignment_scores))
    return {
        "selected_mask": masks[best_idx].astype(np.uint8),
        "selected_score": scores[best_idx]
    }


def morphological_postprocess(mask, kernel_size=3, min_area=15):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    cleaned = np.zeros_like(closed)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(cleaned)
    for cnt in contours:
        cv2.drawContours(filled_mask, [cnt], 0, 1, thickness=cv2.FILLED)
    return filled_mask


def estimate_total_canopy_mask(image_rgb):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv, (25, 25, 25), (95, 255, 255))
    chlorotic_necrotic_mask = cv2.inRange(hsv, (10, 20, 20), (25, 255, 255))

    r = image_rgb[:, :, 0].astype(np.float32)
    g = image_rgb[:, :, 1].astype(np.float32)
    b = image_rgb[:, :, 2].astype(np.float32)
    exg = 2 * g - r - b
    exg_mask = (exg > 5).astype(np.uint8) * 255

    combined = cv2.bitwise_or(green_mask, chlorotic_necrotic_mask)
    combined = cv2.bitwise_or(combined, exg_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(leaf_closed)
    leaf_mask = np.zeros_like(leaf_closed)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = np.max(areas)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 0.03 * max_area:
                leaf_mask[labels == i] = 1
    else:
        leaf_mask = np.ones((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
    return leaf_mask


def compute_disease_severity(lesion_mask, canopy_mask, predicted_class):
    if predicted_class == "normal":
        return {
            "lesion_pixels": 0,
            "leaf_pixels": int(np.sum(canopy_mask)),
            "severity_percentage": 0.0,
            "severity_category": "Healthy (Normal)"
        }

    valid_lesion = cv2.bitwise_and(lesion_mask, canopy_mask)
    lesion_pixels = int(np.sum(valid_lesion))
    canopy_pixels = int(np.sum(canopy_mask))

    severity_pct = (lesion_pixels / (canopy_pixels + 1e-6)) * 100.0 if canopy_pixels > 0 else 0.0
    severity_pct = float(np.clip(severity_pct, 0.0, 100.0))

    if severity_pct == 0.0:
        category = "Early / Trace (<0.1%)"
    elif severity_pct <= 10.0:
        category = "Mild (0-10%)"
    elif severity_pct <= 25.0:
        category = "Moderate (10-25%)"
    elif severity_pct <= 50.0:
        category = "Severe (25-50%)"
    else:
        category = "Very Severe (>50%)"

    return {
        "lesion_pixels": lesion_pixels,
        "leaf_pixels": canopy_pixels,
        "severity_percentage": round(severity_pct, 2),
        "severity_category": category
    }


def to_base64_png(image_np_rgb):
    bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", bgr)
    return base64.b64encode(buffer).decode("utf-8")


@app.on_event("startup")
def load_models():
    global classifier_model, sam_predictor, gradcam_hook

    ckpt_path = os.path.join(WORKDIR, "checkpoints", "best_efficientnet_b0.pth")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"Classifier checkpoint not found at: {ckpt_path}")

    print(f"Loading EfficientNet-B0 on {DEVICE}...")
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(1280, 10)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    classifier_model = model
    gradcam_hook = GradCAMPlusPlus(classifier_model)
    print("EfficientNet-B0 & Grad-CAM++ loaded successfully.")

    # Load SAM ViT-B (or MobileSAM if present)
    mobile_sam_path = os.path.join(WORKDIR, "checkpoints", "mobile_sam.pt")
    standard_sam_path = os.path.join(WORKDIR, "checkpoints", "sam_vit_b_01ec64.pth")

    if os.path.exists(mobile_sam_path):
        from mobile_sam import sam_model_registry, SamPredictor
        print(f"Loading MobileSAM from {mobile_sam_path}...")
        sam = sam_model_registry["vit_t"](checkpoint=mobile_sam_path)
    elif os.path.exists(standard_sam_path):
        from segment_anything import sam_model_registry, SamPredictor
        print(f"Loading SAM ViT-B from {standard_sam_path}...")
        sam = sam_model_registry["vit_b"](checkpoint=standard_sam_path)
    else:
        raise RuntimeError("No SAM checkpoint found in checkpoints directory!")

    sam.to(device=DEVICE)
    sam.eval()
    sam_predictor = SamPredictor(sam)
    print("SAM Predictor initialized successfully.")


@app.get("/", response_class=HTMLResponse)
@app.get("/app", response_class=HTMLResponse)
def serve_mobile_app():
    html_path = os.path.join(WORKDIR, "static", "app.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h2>Paddy Doctor AI API is Online. Static app file not found.</h2>")


@app.get("/api/status")
def api_status():
    return {
        "status": "online",
        "service": "Paddy Disease Diagnosis API",
        "version": "1.0.0",
        "classes": CLASSES
    }


@app.post("/predict")
async def predict_leaf(file: UploadFile = File(...)):
    try:
        # 1. Read Image
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        rgb_224 = np.array(pil_img.resize((IMAGE_SIZE, IMAGE_SIZE)))

        # 2. Classifier Inference
        tensor_in = eval_transforms(pil_img).unsqueeze(0).to(DEVICE)
        cam_map, pred_idx, conf = gradcam_hook.generate_cam(tensor_in)
        pred_class = CLASSES[pred_idx]

        # 3. SAM Prompt Generation & Segmentation
        prompts = generate_sam_prompts_from_cam(cam_map)
        sam_res = run_sam_segmentation(rgb_224, prompts, sam_predictor, cam_map)

        # 4. DIP Canopy & Severity Quantification
        lesion_mask = morphological_postprocess(sam_res["selected_mask"], min_area=15)
        canopy_mask = estimate_total_canopy_mask(rgb_224)
        sev_info = compute_disease_severity(lesion_mask, canopy_mask, pred_class)

        # 5. Visual Overlays for Mobile UI
        # A. Composite Pathological Overlay
        composite = rgb_224.copy()
        composite[canopy_mask == 1] = (0.75 * composite[canopy_mask == 1] + 0.25 * np.array([0, 200, 0])).astype(np.uint8)
        if pred_class != "normal":
            composite[lesion_mask == 1] = np.array([255, 30, 30], dtype=np.uint8)

        # B. Grad-CAM++ Jet Heatmap Overlay
        cam_resized = cv2.resize(cam_map, (IMAGE_SIZE, IMAGE_SIZE))
        heatmap_uint8 = np.uint8(255 * cam_resized)
        heatmap_jet = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)
        gradcam_overlay = np.uint8(np.clip(0.55 * heatmap_rgb + 0.45 * rgb_224, 0, 255))

        # C. Binary Lesion Mask
        binary_mask_rgb = np.repeat((lesion_mask * 255)[:, :, None], 3, axis=2)

        return {
            "status": "success",
            "filename": file.filename,
            "prediction": {
                "class_name": pred_class,
                "display_name": DISPLAY_NAMES[pred_class],
                "confidence_percentage": round(conf * 100, 2),
                "severity_percentage": sev_info["severity_percentage"],
                "severity_category": sev_info["severity_category"],
                "lesion_pixels": sev_info["lesion_pixels"],
                "leaf_pixels": sev_info["leaf_pixels"],
                "treatment_recommendation": TREATMENTS[pred_class]
            },
            "visualizations": {
                "composite_overlay_base64": to_base64_png(composite),
                "gradcam_heatmap_base64": to_base64_png(gradcam_overlay),
                "lesion_mask_base64": to_base64_png(binary_mask_rgb)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Run on all network interfaces (0.0.0.0) so phone can connect via Wi-Fi hotspot
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
