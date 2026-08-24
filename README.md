# Ai-plant-health-classifier

Plant Care AI App: Master Project Plan
This master plan outlines the step-by-step development strategy for a computer vision-based plant care application. The architecture separates visual recognition from structured plant care data to maximize model accuracy and app performance.

## 📌 Step 1: Data Collection & Preparation
Instead of searching for a single dataset containing all visual and care attributes, the project combines a visual dataset with a structured plant knowledge base.

* **1A. Image Dataset Procurement:** Source established visual datasets such as PlantDoc (for real-world disease detection) and PlantVillage via platforms like Kaggle and Roboflow.
* **1B. Structured Care Knowledge Base:** Source a structured reference metadata (`data.yaml` / JSON) mapping plant species to their specific care rules:
  * Plant Name → Environment Type (Indoor / Outdoor)
  * Plant Name → Watering Frequency & Requirements
* **1C. Annotation Strategy:** Ensure training images are appropriately labeled for a multi-task dual-model system:
  * **Label Set A:** Taxonomic classification (e.g., Apple, Corn, Tomato, Grape).
  * **Label Set B:** Structural and health anomalies (38 classes including Healthy, Early Blight, Rust, Mites, Powdery Mildew).

---

## 🏗️ Step 2: Model Architecture & Pipeline
The vision pipeline processes images to extract two distinct inference vectors: identity and condition.

* **2A. Model 1: Species Classifier**
  * **Objective:** Taxonomic identification of the plant species.
  * **Architecture:** EfficientNet-B0 fine-tuned backbone (`species_efficientnet_b0_full.pth`).
* **2B. Model 2: Health Condition Detector**
  * **Objective:** Analyze leaf tissue condition for physiological stress or disease.
  * **Architecture:** ConvNeXt-Tiny fine-tuned model (`best_health_model_v3.pth`) operating at $256 \times 256$ input resolution with Dropout ($p=0.4$) and custom classification head.

---

## ⚙️ Step 3: Logic Integration & App Features
This layer acts as the software "brain," mapping computer vision outputs to the database to deliver actionable user insights.

| App Feature | Vision Input Source | Backend Logic / Database Mapping |
| :--- | :--- | :--- |
| **1. Environment Identification** | Species Model identifies plant species (e.g., Grape Leaf). | Queries knowledge base (`data.yaml`) to display species name and confidence. |
| **2. Watering Assessment** | Health Model detects physical cues (e.g., Leaf Burn or Blight). | Cross-references visual cues with plant baseline schedule to generate actionable responses. |
| **3. Health & Improvement Plan** | Health Model outputs diagnostic (e.g., Tomato Early Blight). | Triggers targeted recovery protocols and tailored care plans. |

---

## 📊 Project Status
* **Status:** **Completed & Deployed locally via Docker!** Both models fine-tuned, evaluated, and integrated into FastAPI & Streamlit services.

### Next Steps
- [x] Complete model fine-tuning across 38 classes
- [x] Evaluate test set performance on complex real-world leaves
- [x] Build dual-model inference pipeline (Species + Health Diagnosis)
- [x] Resolve state-dict shape mismatch bugs ($38 \times 768$)
- [x] Containerize full stack with FastAPI, Streamlit, and Docker

---

## 📈 Latest Model Results

| Metric | Health Model (ConvNeXt-Tiny) | Species Model (EfficientNet-B0) |
| :--- | :--- | :--- |
| **Backbone Architecture** | `convnext_tiny` | `efficientnet_b0` |
| **Input Resolution** | $256 \times 256$ | $256 \times 256$ |
| **Target Classes** | 38 Classes (PlantDoc / PlantVillage) | 38 Classes |
| **Test Accuracy** | **66.15%** *(Weighted F1: 0.67)* | **80.04%** *(Validation Peak)* |
| **Test Loss** | **1.5185** | 1.2899 |
| **Status** | Checkpoint Saved (`models/best_health_model_v3.pth`) | Checkpoint Saved (`models/species_efficientnet_b0_full.pth`) |

---

## 💡 Active Development & Highlights
* **Model Architecture Upgrades:** Built custom classifier head using `Dropout(p=0.4)` and a Linear projection layer to stabilize feature extraction across noisy backgrounds.
* **Loss & Optimization:** Applied `LabelSmoothing=0.1` inside `nn.CrossEntropyLoss` and trained on Apple Silicon MPS GPU acceleration.
* **Dual Inference Backend:** FastAPI service runs parallel inference across both species and health models in under 3 seconds per image.

---

## 🐳 Containerization & Deployment

This application is fully dockerized using a multi-service setup. Both the **FastAPI backend** (PyTorch inference server) and the **Streamlit frontend** are packaged into an isolated Docker environment for seamless execution across Mac, Windows, and Linux.

