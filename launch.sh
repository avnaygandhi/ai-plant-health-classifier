#!/bin/bash

# Navigate to project root
cd /Users/avnay/Desktop/AIML/Plant_project

# Set Environment Paths
export PATH="/opt/anaconda3/bin:$PATH"
export PYTHONPATH="/Users/avnay/Desktop/AIML/Plant_project/Front end:/Users/avnay/Desktop/AIML/Plant_project:$PYTHONPATH"

echo "🚀 Starting FastAPI Backend Server..."
cd "/Users/avnay/Desktop/AIML/Plant_project/Front end"
/opt/anaconda3/bin/python3 -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload &

# Give FastAPI time to initialize PyTorch in memory
sleep 4

echo "🌿 Starting Streamlit Frontend..."
/opt/anaconda3/bin/streamlit run "UI.py"