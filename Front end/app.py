import io
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2


# --- 1. MODEL ARCHITECTURE DEFINITION ---
class PlantHealthModel(nn.Module):

  def __init__(self, num_classes=38):
    super().__init__()
    # Instantiate raw EfficientNet-B2
    model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)

    # Attach modules directly to self
    self.features = model.features
    self.avgpool = model.avgpool

    in_features = model.classifier[1].in_features
    self.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features=in_features, out_features=512),
        nn.BatchNorm1d(512),
        nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=512, out_features=num_classes),
    )

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x


# --- 2. FASTAPI APP INITIALIZATION ---
app = FastAPI(title="Plant AI Inference Service")

# Setup device (Apple Silicon MPS / CUDA / CPU)
if torch.backends.mps.is_available():
  device = torch.device("mps")
  print("🚀 Acceleration Enabled: Using Apple Silicon MPS GPU")
else:
  device = torch.device("cpu")
  print("ℹ️ Using CPU fallback")

# --- 3. PATH & MODEL LOADING ---
# Anchors model path relative to Plant_project root folder
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_health_model_v2.pth"
num_classes = 38

health_model = PlantHealthModel(num_classes=num_classes)

if not MODEL_PATH.exists():
  print(f"⚠️ Warning: Could not find model file at {MODEL_PATH}")
  raise FileNotFoundError(
      f"❌ Could not find model weights at expected path: {MODEL_PATH}"
  )

health_model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
health_model.to(device)
health_model.eval()
print(f"✅ Successfully loaded health weights from {MODEL_PATH}")

# --- 4. IMAGE TRANSFORMATIONS ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
])

DISEASE_CLASSES = [
    "Apple - Scab",
    "Apple - Black Rot",
    "Apple - Cedar Rust",
    "Apple - Healthy",
    "Peach - Healthy",
    "Peach - Bacterial Spot",
    # Add remaining class names here matching dataset order
]


# --- 5. ENDPOINTS ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  try:
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
      outputs = health_model(input_tensor)
      probabilities = F.softmax(outputs, dim=1)
      conf, pred_idx = torch.max(probabilities, 1)

    predicted_label = (
        DISEASE_CLASSES[pred_idx.item()]
        if pred_idx.item() < len(DISEASE_CLASSES)
        else "Unknown Condition"
    )
    confidence_pct = f"{conf.item() * 100:.2f}%"

    return {
        "species": "Fagus grandifolia (American Beech)",
        "species_conf": "99.85%",
        "health_diagnosis": predicted_label,
        "health_conf": confidence_pct,
        "watering_assessment": (
            "Soil is balanced. Maintain regular watering schedule based on"
            " moisture levels."
        ),
        "improvement_plan": (
            "No immediate treatment required. Continue routine monitoring."
        ),
    }

  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")