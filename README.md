# ai-plant-health-classifier
Plant Care AI App: Master Project Plan
This master plan outlines the step-by-step development strategy for a computer vision-based plant care application. The architecture separates visual recognition from structured plant care data to maximize model accuracy and app performance.
 Step 1: Data Collection & Preparation
Instead of searching for a single dataset containing all visual and care attributes, the project will combine a visual dataset with a structured plant knowledge base.

●	1A. Image Dataset Procurement: Source established visual datasets such as PlantDoc (for health/disease detection) or Pl@ntNet / Leafsnap (for species identification) via platforms like Kaggle

●	1B. Structured Care Knowledge Base: Source a structured reference dataset (CSV or JSON format) mapping plant species to their specific care rules:
○	Plant Name → Environment Type (Indoor / Outdoor)
○	Plant Name → Watering Frequency & Requirements

●	1C. Annotation Strategy: Ensure training images are appropriately labeled for a multi-task learning system or two distinct models:
○	Label Set A: Taxonomic classification (e.g., Monstera Deliciosa, Snake Plant, Fiddle Leaf Fig).
○	Label Set B: Structural and health anomalies (e.g., Healthy, Scorched Leaves, Chlorosis/Yellowing, Wilting).

Step 2: Model Architecture & Training
The vision pipeline will process the image to extract two distinct vectors: identity and condition.

●	2A. Model 1: Species Classifier
○	Objective: Taxonomic identification of the plant.
○	Architecture: Utilize a Convolutional Neural Network (CNN) such as ResNet

●	2B. Model 2: Health Condition Detector
○	Objective: Analyze tissue condition for physiological stress or disease.
○	Architecture: Deploy a regional object detection model (such as YOLOv8) or a fine-grained classification model trained to isolate and classify surface anomalies like brown spots, powdery mildew, or physical droop.

Step 3: Logic Integration & App Features
This layer acts as the software "brain," mapping computer vision outputs to the database to deliver actionable user insights.

App Feature	Vision Input Source	Backend Logic / Database Mapping

1. Environment Identification	Model 1 identifies species (e.g., Snake Plant).	Queries the knowledge base (Step 1B) to display "Type: Indoor".
   
2. Watering Assessment	Model 2 detects physical cues (e.g., Wilting or Dry soil surface).	Cross-references visual cues with the plant's baseline watering schedule from the database to generate an actionable response (e.g., "Needs water immediately").

3. Health & Improvement Plan	Model 2 outputs health diagnostic (e.g., Leaf Burn).	Triggers a targeted care protocol from the database or leverages an LLM API to output tailored recovery instructions (e.g., "Move away from direct sunlight; prune scorched edges.").

Plan for dataset
1. Plantdoc
2. Leafsmap 

## Project Status
* Exploring confidence intervals and preparing for model fine-tuning.
