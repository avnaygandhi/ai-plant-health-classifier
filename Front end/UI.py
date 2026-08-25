import os
import re
import signal
import sys
import threading
import time
from pathlib import Path
import requests
import streamlit as st
import uvicorn

# --- FIX FOR MODULE NOT FOUND ERROR ---
# Add the project root directory to Python's path so Streamlit Cloud can locate the Backend folder
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent.parent

# Add root directory to sys.path so 'Backend' can be imported
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

# Import your FastAPI app
from Backend.main import app
`


# --- BACKGROUND FASTAPI SERVER ---
def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000)


if "server_started" not in st.session_state:
    thread = threading.Thread(target=run_fastapi, daemon=True)
    thread.start()
    st.session_state["server_started"] = True
    time.sleep(2)  # Allow uvicorn time to start

# --- STREAMLIT UI ---
st.set_page_config(
    page_title="AI Plant Health & Species Identifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2.3rem; font-weight: 700; color: #2E7D32; margin-bottom: 0px; }
    .sub-header { font-size: 1.1rem; color: #555555; margin-bottom: 15px; }
    .tip-box { background-color: #E8F5E9; padding: 12px 16px; border-radius: 8px; border: 1px solid #C8E6C9; margin-bottom: 20px; }
    </style>
""",
    unsafe_allow_html=True,
)

# Header Section
st.markdown(
    '<div class="main-header">🌿 AI Plant Health & Species Identifier</div>',
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='sub-header'>Upload a leaf image to instantly identify species and receive health assessments.</div>",
    unsafe_allow_html=True,
)

# Pro-Tip Banner
st.markdown(
    """
    <div class="tip-box">
        💡 <strong>Accuracy Tip:</strong> For best results, upload a close-up photo focused on a single leaf with good lighting rather than a whole plant canopy.
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Controls
with st.sidebar:
    st.header("Application Controls")
    st.info("Backend Status: **Active**\n\nAPI Server: `http://127.0.0.1:8000/predict`")
    st.divider()

    if st.button("🛑 Stop Server & Exit", use_container_width=True):
        st.warning("Shutting down session...")
        st.components.v1.html(
            """
            <script>
                window.close();
                window.location.href = "about:blank";
            </script>
            """,
            height=0,
        )
        os.kill(os.getpid(), signal.SIGTERM)

# Image Uploader
uploaded_file = st.file_uploader(
    "Choose a plant leaf image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(uploaded_file, use_container_width=True)

    with col2:
        st.subheader("🧠 Real-time AI Analysis")
        with st.spinner("Processing image through vision pipeline..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }
                response = requests.post("http://127.0.0.1:8000/predict", files=files)

                if response.status_code == 200:
                    data = response.json()
                    st.success("Analysis Complete!")

                    species_name = data.get("species", "N/A")
                    species_conf = data.get("species_conf", "0%")

                    full_health_diagnosis = data.get("health_diagnosis", "Unknown")
                    health_conf_raw = data.get("health_conf", "0%")

                    health_conf_match = re.search(r"([\d\.]+)", health_conf_raw)
                    health_conf_val = float(health_conf_match.group(1)) if health_conf_match else 0.0

                    raw_status = (
                        full_health_diagnosis.split("-")[-1].strip()
                        if "-" in full_health_diagnosis
                        else full_health_diagnosis
                    )

                    # 50% Confidence Threshold Logic
                    if raw_status.lower() == "healthy" and health_conf_val < 50.0:
                        health_status = "Unhealthy"
                        is_unhealthy = True
                    elif raw_status.lower() == "healthy":
                        health_status = "Healthy"
                        is_unhealthy = False
                    else:
                        health_status = "Unhealthy"
                        is_unhealthy = True

                    if is_unhealthy:
                        status_color = "#D32F2F"
                        bg_badge_color = "#FFEBEE"
                        watering_plan = (
                            "⚠️ **Adjusted Watering Schedule:** Stress detected. Check soil moisture manually before watering."
                        )
                        improvement_plan = (
                            "⚠️ **Targeted Action Required:** Leaf abnormalities or low prediction confidence detected.\n\n"
                            "1. Inspect leaf undersides for active pests.\n"
                            "2. Prune heavily damaged tissue.\n"
                            "3. Ensure indirect sunlight and clean water drainage."
                        )
                    else:
                        status_color = "#2E7D32"
                        bg_badge_color = "#E8F5E9"
                        watering_plan = data.get("watering_assessment", "Maintain regular moisture monitoring.")
                        improvement_plan = data.get("improvement_plan", "No immediate treatment needed.")

                    m1, m2 = st.columns(2)

                    with m1:
                        st.markdown(
                            f"""
                            <div style="background-color: #F8F9FA; padding: 16px 18px; border-radius: 10px; border-left: 5px solid #2E7D32; min-height: 150px; display: flex; flex-direction: column; justify-content: space-between;">
                                <div style="font-size: 0.95rem; color: #555555; font-weight: 600;">🌿 Predicted Species</div>
                                <div style="font-size: 1.25rem; font-weight: 700; color: #111111; margin: 6px 0; word-wrap: break-word;">{species_name}</div>
                                <div style="font-size: 0.85rem; color: #2E7D32; background-color: #E8F5E9; padding: 4px 10px; border-radius: 15px; width: fit-content; font-weight: 600;">↑ Confidence: {species_conf}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with m2:
                        st.markdown(
                            f"""
                            <div style="background-color: #F8F9FA; padding: 16px 18px; border-radius: 10px; border-left: 5px solid {status_color}; min-height: 150px; display: flex; flex-direction: column; justify-content: space-between;">
                                <div style="font-size: 0.95rem; color: #555555; font-weight: 600;">🩺 Health Diagnosis</div>
                                <div style="font-size: 1.25rem; font-weight: 700; color: {status_color}; margin: 6px 0; word-wrap: break-word;">{health_status}</div>
                                <div style="font-size: 0.85rem; color: {status_color}; background-color: {bg_badge_color}; padding: 4px 10px; border-radius: 15px; width: fit-content; font-weight: 600;">↑ Confidence: {health_conf_raw}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.divider()
                    st.markdown("### 💧 Watering Assessment")
                    st.info(watering_plan)

                    st.markdown("### 🛠️ Targeted Improvement Plan")
                    st.warning(improvement_plan)

                else:
                    st.error(f"Server Error {response.status_code}: {response.text}")

            except Exception as e:
                st.error(f"⚠️ Unable to reach FastAPI backend server. Ensure the server is running!\n\nDetails: {e}")
