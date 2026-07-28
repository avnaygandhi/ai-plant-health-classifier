import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile

# Import your classifier from main.py
from models.main import PlantClassifier

app = FastAPI(title="Plant Classifier API")

# Initialize Classifier Instance once at startup
classifier = PlantClassifier()

@app.get("/")
def root():
    return {"status": "Online", "service": "Plant Classification API"}

@app.post("/predict")
async def predict_plant(file: UploadFile = File(...)):
    # 1. Save uploaded file temporarily
    temp_path = Path(f"temp_{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Run model inference & logic mapping
    result = classifier.predict(str(temp_path))

    # 3. Clean up temp file
    if temp_path.exists():
        temp_path.unlink()

    # 4. Return results directly
    return result