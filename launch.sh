#!/bin/bash

# 1. Navigate to project directory
cd /Users/avnay/Desktop/AIML/Plant_project

# 2. Use 'python' (points to active Anaconda base environment)
PY_PATH="python"

# 3. Start FastAPI backend in background on port 8000
"$PY_PATH" -m uvicorn "Front end.app:app" --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Wait 5 seconds for model weights to load into RAM
sleep 5

# 4. Start Streamlit UI in background on port 8501
"$PY_PATH" -m streamlit run "Front end/UI.py" --server.headless true &
STREAMLIT_PID=$!

# Wait 2 seconds for Streamlit server to bind to port 8501
sleep 2

# 5. Automatically open browser
open http://localhost:8501

# Keep script running until user closes Streamlit
wait $STREAMLIT_PID

# Clean up FastAPI backend when Streamlit closes
kill $BACKEND_PID 2>/dev/null