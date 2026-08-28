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
    ResNet50_Weights,
    convnext_tiny,
    efficientnet_b0,
    resnet50,
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
FAKE_REAL_MODEL_PATH = MODELS_DIR / "fake_vs_real_resnet50.pth"

# AWS S3 Direct Download URLs
HEALTH_S3_URL = "https://plant-ai-weights-avnay.s3.ap-southeast-1.amazonaws.com/best_health_model_v3.pth"
SPECIES_S3_URL = "https://plant-ai-weights-avnay.s3.ap-southeast-1.amazonaws.com/species_efficientnet_b0_full.pth"
FAKE_REAL_S3_URL = "https://plant-ai-weights-avnay.s3.ap-southeast-1.amazonaws.com/fake_vs_real_resnet50.pth" 


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

    # 3. Download Fake/Real Model from S3 if missing locally
    if not FAKE_REAL_MODEL_PATH.exists() and FAKE_REAL_S3_URL:
        print("☁️ Fetching Fake/Real Model from AWS S3...")
        try:
            urllib.request.urlretrieve(FAKE_REAL_S3_URL, FAKE_REAL_MODEL_PATH)
            print("✅ Fake/Real Model downloaded from AWS S3 successfully!")
        except Exception as e:
            print(f"⚠️ Failed to download Fake/Real Model from S3: {e}")


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


def build_fake_real_model():
    """ResNet-50 head matching the original fake_vs_real training architecture."""
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2),
    )
    return model


def build_species_model(num_classes=25):
    """Matches the plain (unwrapped) efficientnet_b0 architecture the
    species checkpoint was trained and saved with — no `backbone`/`model`
    submodule nesting, since the checkpoint's keys are top-level
    ("features.*", "classifier.*")."""
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1280, out_features=512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=512, out_features=num_classes),
    )
    return model


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


_LEAFSNAP_25 = [
    "Abies Concolor", "Abies Nordmanniana", "Acer Campestre", "Acer Ginnala",
    "Acer Griseum", "Acer Negundo", "Acer Palmatum", "Acer Pensylvanicum",
    "Acer Platanoides", "Acer Pseudoplatanus", "Acer Rubrum", "Acer Saccharinum",
    "Acer Saccharum", "Aesculus Flava", "Aesculus Glabra", "Aesculus Hippocastamon",
    "Aesculus Pavi", "Ailanthus Altissima", "Albizia Julibrissin",
    "Amelanchier Arborea", "Amelanchier Canadensis", "Amelanchier Laevis",
    "Asimina Triloba", "Betula Alleghaniensis", "Betula Jacqemontii",
]

SPECIES_NOT_FOUND = "Sorry, species not found in our database"
SPECIES_CONF_THRESHOLD = 40.0  # percent; below this we report "not found"


def load_species_labels():
    leafsnap_dir = BASE_DIR / "leafsnap_data" / "train"
    if leafsnap_dir.exists():
        names = sorted(p.name for p in leafsnap_dir.iterdir() if p.is_dir())
        if names:
            return [n.replace("_", " ").title() for n in names[:NUM_SPECIES_CLASSES]]
    return _LEAFSNAP_25


CLASS_LABELS = load_labels()
NUM_CLASSES = 38
NUM_SPECIES_CLASSES = 25
SPECIES_LABELS = load_species_labels()

# Labels that represent a healthy plant.
# Includes explicit "Healthy" labels AND plain-leaf labels for species that
# only have one class in PlantVillage (i.e. they are healthy by definition).
HEALTHY_LABELS = {lbl.lower() for lbl in CLASS_LABELS if "healthy" in lbl.lower()} | {
    "apple leaf",        # healthy apple in this label set
    "bell_pepper leaf",  # healthy pepper (Pepper Bell Healthy also exists; both count)
    "blueberry leaf",    # PlantVillage only has one blueberry class → healthy
    "cherry leaf",       # healthy cherry variant
    "peach leaf",        # healthy peach variant
    "potato leaf",       # healthy potato variant
    "raspberry leaf",    # PlantVillage only has one raspberry class → healthy
    "soyabean leaf",     # healthy soybean variant
    "soybean leaf",      # healthy soybean variant
    "strawberry leaf",   # healthy strawberry variant
    "tomato leaf",       # healthy tomato variant
    "grape leaf",        # healthy grape variant
}

FAKE_REAL_CONF_THRESHOLD = 70.0  # below this, skip the fake/real gate

# --- 3. INITIALIZE MODELS & LOAD CHECKPOINTS ---
fake_real_model = build_fake_real_model().to(device)
if FAKE_REAL_MODEL_PATH.exists():
    try:
        fake_real_model.load_state_dict(
            torch.load(FAKE_REAL_MODEL_PATH, map_location=device, weights_only=True)
        )
        fake_real_model.eval()
        print(f"✅ Loaded Fake/Real Model from {FAKE_REAL_MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Error loading Fake/Real Model state_dict: {e}")
        print("⚠️ Fake/Real gate DISABLED — architecture mismatch or corrupt file.")
        fake_real_model = None
else:
    print("ℹ️ Fake/Real model checkpoint missing — gate disabled.")
    fake_real_model = None

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

species_model = build_species_model(num_classes=NUM_SPECIES_CLASSES).to(device)
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
            # --- Fake / Real gate (runs first) ---
            is_artificial = False
            fake_real_confidence = None
            if fake_real_model is not None:
                fr_logits = fake_real_model(input_tensor)
                fr_probs = F.softmax(fr_logits, dim=1)
                fr_conf, fr_idx = torch.max(fr_probs, 1)
                fr_conf_pct = fr_conf.item() * 100
                # class 0 = fake, class 1 = real
                if fr_idx.item() == 0 and fr_conf_pct >= FAKE_REAL_CONF_THRESHOLD:
                    is_artificial = True
                fake_real_confidence = f"{fr_conf_pct:.2f}%"

            if is_artificial:
                return {
                    "is_artificial": True,
                    "fake_real_conf": fake_real_confidence,
                    "species": "N/A",
                    "species_conf": "N/A",
                    "health_diagnosis": "N/A",
                    "health_conf": "N/A",
                    "is_healthy": None,
                    "watering_assessment": "This appears to be an artificial plant — no care needed!",
                    "improvement_plan": "No health analysis performed (artificial plant detected).",
                    "status": "artificial",
                }

            health_logits = health_model(input_tensor)
            health_probs = F.softmax(health_logits, dim=1)
            h_conf, h_idx = torch.max(health_probs, 1)

            if species_model is not None:
                species_logits = species_model(input_tensor)
                species_probs = F.softmax(species_logits, dim=1)
                s_conf, s_idx = torch.max(species_probs, 1)
                s_conf_pct = s_conf.item() * 100
                if s_conf_pct < SPECIES_CONF_THRESHOLD:
                    species_label = SPECIES_NOT_FOUND
                elif s_idx.item() < len(SPECIES_LABELS):
                    species_label = SPECIES_LABELS[s_idx.item()]
                else:
                    species_label = SPECIES_NOT_FOUND
                species_confidence = f"{s_conf_pct:.2f}%"
            else:
                species_label = SPECIES_NOT_FOUND
                species_confidence = "N/A"

        health_label = (
            CLASS_LABELS[h_idx.item()]
            if h_idx.item() < len(CLASS_LABELS)
            else "Unknown Condition"
        )
        health_confidence = f"{h_conf.item() * 100:.2f}%"
        is_healthy = health_label.lower() in HEALTHY_LABELS

        if is_healthy:
            watering_msg = (
                "Soil moisture is balanced. Maintain your regular watering schedule "
                "and adjust based on seasonal conditions."
            )
            improvement_msg = (
                "Plant appears healthy! Continue routine care: ensure adequate light, "
                "good drainage, and monitor for early signs of stress."
            )
        else:
            watering_msg = (
                "⚠️ Stress detected. Check soil moisture manually before watering — "
                "over-watering can worsen disease symptoms."
            )
            improvement_msg = (
                f"⚠️ Condition detected: {health_label}.\n\n"
                "1. Inspect leaf undersides for pests or fungal growth.\n"
                "2. Remove and dispose of heavily affected leaves.\n"
                "3. Improve air circulation and avoid wetting foliage when watering.\n"
                "4. Consider an appropriate fungicide or pesticide treatment."
            )

        return {
            "is_artificial": False,
            "fake_real_conf": fake_real_confidence,
            "species": species_label,
            "species_conf": species_confidence,
            "health_diagnosis": health_label,
            "health_conf": health_confidence,
            "is_healthy": is_healthy,
            "watering_assessment": watering_msg,
            "improvement_plan": improvement_msg,
            "status": "success",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Inference error: {str(e)}"
        )
