#!/bin/bash

# 1. Start FastAPI in the background
uvicorn "Front end.app:app" --host 0.0.0.0 --port 8000 &

# 2. Wait a few seconds for PyTorch models to load
sleep 5

# 3. Start Streamlit in the foreground (main process)
streamlit run "Front end/UI.py" \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --client.toolbarMode "minimal"