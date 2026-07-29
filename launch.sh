#!/bin/zsh

# 1. Load Conda / Zsh environment
source ~/.zshrc 2>/dev/null || source ~/.bash_profile 2>/dev/null

# 2. Navigate to project directory
cd /Users/avnay/Desktop/AIML/Plant_project || exit 1

# 3. Locate Conda's Python specifically
if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/python3" ]; then
    PY_PATH="$CONDA_PREFIX/bin/python3"
elif [ -f "$HOME/opt/anaconda3/bin/python3" ]; then
    PY_PATH="$HOME/opt/anaconda3/bin/python3"
elif [ -f "$HOME/anaconda3/bin/python3" ]; then
    PY_PATH="$HOME/anaconda3/bin/python3"
else
    PY_PATH="$(which python3)"
fi

# 4. Start FastAPI backend
"$PY_PATH" -m uvicorn "Front end.app:app" --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Wait 6 seconds for backend models to load
sleep 6

# 5. Start Streamlit UI
"$PY_PATH" -m streamlit run "Front end/UI.py" --server.headless true &
STREAMLIT_PID=$!

# Wait 2 seconds for Streamlit server
sleep 2

# 6. Open browser
open http://127.0.0.1:8501

# Keep active and clean up processes on exit
wait $STREAMLIT_PID
kill $BACKEND_PID 2>/dev/null
pkill -f uvicorn 2>/dev/null