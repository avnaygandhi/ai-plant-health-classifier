import io
import os
from pathlib import Path
import urllib.request
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    convnext_tiny,
    efficientnet_b0,
)
import yaml

app = FastAPI(title="Plant AI Inference Service")

# Setup device (Apple Silicon MPS / CUDA / CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Acceleration Enabled: Using Apple Silicon MPS GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 Acceleration Enabled: Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("ℹ️ Using CPU fallback")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Define local target paths (Path objects)
HEALTH_MODEL_PATH = MODELS_DIR / "best_health_model_v3.pth"
SPECIES_MODEL_PATH = MODELS_DIR / "species_efficientnet_b0_full.pth"

# AWS S3 Direct Download URLs
HEALTH_S3_URL = "https://plant-ai-weights-avnay.s3.ap-southeast-1.amazonaws.com/best_health_model_v3.pth"
SPECIES_S3_URL = "https://plant-ai-weights-avnay.s3.ap-southeast-1.amazonaws.com/species_efficientnet_b0_full.pth"


def download_s3_weights():
    # 1. Download Health Model from S3 if missing locally
    if not HEALTH_MODEL_PATH.exists():
        print("☁️ Fetching Health Model from AWS S3...")
        try:
            urllib.request.urlretrieve(HEALTH_S3_URL, HEALTH_MODEL_PATH)
            print("✅ Health Model downloaded from AWS S3 successfully!")
        except Exception as e:
            print(f"⚠️ Failed to download Health Model from S3: {e}")

    # 2. Download Species Model from S3 if missing locally
    if not SPECIES_MODEL_PATH.exists():
        print("☁️ Fetching Species Model from AWS S3...")
        try:
            urllib.request.urlretrieve(SPECIES_S3_URL, SPECIES_MODEL_PATH)
            print("✅ Species Model downloaded from AWS S3 successfully!")
        except Exception as e:
            print(f"ℹ️ Species model not yet available on S3 (will use fallback).")


download_s3_weights()


# --- 1. MODEL ARCHITECTURES ---
class PlantHealthConvNeXt(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        self.backbone = convnext_tiny(weights=weights)

        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.4), nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class PlantSpeciesEfficientNet(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# --- 2. LOAD CLASS LABELS ---
def load_labels():
    possible_yaml_paths = [
        BASE_DIR / "Plant_project_data" / "data.yaml",
        BASE_DIR / "data.yaml",
    ]

    for yaml_path in possible_yaml_paths:
        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
                names = data.get("names", [])
                if isinstance(names, dict):
                    names = [names[i] for i in sorted(names.keys())]
                if len(names) == 38:
                    return names

    return [
        "Apple Scab Leaf", "Apple leaf", "Apple rust leaf", "Bell_pepper leaf spot",
        "Bell_pepper leaf", "Blueberry leaf", "Cherry leaf", "Corn Gray leaf spot",
        "Corn leaf blight", "Corn rust leaf", "Peach leaf", "Potato leaf early blight",
        "Potato leaf late blight", "Potato leaf", "Raspberry leaf", "Soyabean leaf",
        "Soybean leaf", "Squash Powdery mildew leaf", "Strawberry leaf",
        "Tomato Early blight leaf", "Tomato Septoria leaf spot",
        "Tomato leaf bacterial spot", "Tomato leaf late blight",
        "Tomato leaf mosaic virus", "Tomato leaf yellow virus", "Tomato leaf",
        "Tomato mold leaf", "Tomato two spotted spider mites leaf",
        "grape leaf black rot", "grape leaf", "Apple Black Rot", "Apple Healthy",
        "Cedar Apple Rust", "Cherry Healthy", "Corn Healthy", "Peach Healthy",
        "Pepper Bell Healthy", "Potato Healthy",
    ]


CLASS_LABELS = load_labels()
NUM_CLASSES = 38

# --- 3. INITIALIZE MODELS & LOAD CHECKPOINTS ---
health_model = PlantHealthConvNeXt(num_classes=NUM_CLASSES).to(device)
if HEALTH_MODEL_PATH.exists():
    try:
        health_model.load_state_dict(
            torch.load(HEALTH_MODEL_PATH, map_location=device, weights_only=True)
        )
        health_model.eval()
        print(f"✅ Loaded Health Model from {HEALTH_MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Error loading Health Model state_dict: {e}")

species_model = PlantSpeciesEfficientNet(num_classes=NUM_CLASSES).to(device)
if SPECIES_MODEL_PATH.exists():
    try:
        species_model.load_state_dict(
            torch.load(SPECIES_MODEL_PATH, map_location=device, weights_only=True)
        )
        species_model.eval()
        print(f"✅ Loaded Species Model from {SPECIES_MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Error loading Species Model state_dict: {e}")
        species_model = None
else:
    print("ℹ️ Species model checkpoint missing. Using health model fallback for taxonomy.")
    species_model = None

# --- 4. PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
])


# --- 5. INFERENCE ENDPOINT ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            health_logits = health_model(input_tensor)
            health_probs = F.softmax(health_logits, dim=1)
            h_conf, h_idx = torch.max(health_probs, 1)

            if species_model is not None:
                species_logits = species_model(input_tensor)
                species_probs = F.softmax(species_logits, dim=1)
                s_conf, s_idx = torch.max(species_probs, 1)
                species_label = (
                    CLASS_LABELS[s_idx.item()]
                    if s_idx.item() < len(CLASS_LABELS)
                    else "Unknown Plant"
                )
                species_confidence = f"{s_conf.item() * 100:.2f}%"
            else:
                species_label = CLASS_LABELS[h_idx.item()].split()[0]
                species_confidence = f"{h_conf.item() * 100:.2f}%"

        health_label = (
            CLASS_LABELS[h_idx.item()]
            if h_idx.item() < len(CLASS_LABELS)
            else "Unknown Condition"
        )
        health_confidence = f"{h_conf.item() * 100:.2f}%"

        return {
            "species": species_label,
            "species_conf": species_confidence,
            "health_diagnosis": health_label,
            "health_conf": health_confidence,
            "watering_assessment": (
                "Soil is balanced. Maintain regular watering schedule based on moisture levels."
            ),
            "improvement_plan": (
                "No immediate treatment required. Continue routine monitoring."
            ),
            "status": "success",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Inference error: {str(e)}"
        )
