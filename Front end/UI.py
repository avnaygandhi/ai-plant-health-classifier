import io
import re
import threading
import time
import requests
import streamlit as st
import uvicorn
from PIL import Image

# Import FastAPI app directly from app.py in the same directory
from app import app


# --- BACKGROUND FASTAPI SERVER ---
def run_fastapi():
  uvicorn.run(app, host="127.0.0.1", port=8000)


if "server_started" not in st.session_state:
  thread = threading.Thread(target=run_fastapi, daemon=True)
  thread.start()
  st.session_state["server_started"] = True
  time.sleep(2)

# --- STREAMLIT UI CONFIG ---
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

st.markdown(
    '<div class="main-header">🌿 AI Plant Health & Species Identifier</div>',
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='sub-header'>Upload or take a photo of a leaf to identify species and health.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
  st.header("Application Controls")
  st.info("Backend Status: **Active**\n\nAPI Server: `http://127.0.0.1:8000/predict`")
  st.divider()

input_mode = st.radio(
    "Choose Input Method:",
    ["Upload File", "Take Photo with Camera"],
    horizontal=True,
)

selected_image_bytes = None
image_filename = "captured_leaf.jpg"

if input_mode == "Upload File":
  uploaded_file = st.file_uploader(
      "Choose a plant leaf image...", type=["jpg", "jpeg", "png"]
  )
  if uploaded_file is not None:
    selected_image_bytes = uploaded_file.getvalue()
    image_filename = uploaded_file.name
else:
  camera_file = st.camera_input("Take a photo of the plant leaf")
  if camera_file is not None:
    selected_image_bytes = camera_file.getvalue()
    image_filename = "camera_capture.jpg"

if selected_image_bytes is not None:
  col1, col2 = st.columns([1, 1.2], gap="large")

  with col1:
    st.subheader("📷 Image Preview")
    image_obj = Image.open(io.BytesIO(selected_image_bytes))
    st.image(image_obj, use_container_width=True)

  with col2:
    st.subheader("🧠 Real-time AI Analysis")
    with st.spinner("Processing image through vision pipeline..."):
      try:
        files = {"file": (image_filename, selected_image_bytes, "image/jpeg")}
        response = requests.post("http://127.0.0.1:8000/predict", files=files)

        if response.status_code == 200:
          data = response.json()

          # ── Artificial plant gate ──────────────────────────────────────────
          if data.get("is_artificial"):
            fr_conf = data.get("fake_real_conf", "")
            conf_str = f" ({fr_conf} confidence)" if fr_conf else ""
            st.warning(
                f"🪴 **Artificial plant detected{conf_str}.**\n\n"
                "This looks like a fake or plastic plant. "
                "Health and species analysis is not applicable."
            )
            st.stop()

          st.success("Analysis Complete!")

          # ── Real-plant badge (model available + confident) ─────────────────
          fr_conf = data.get("fake_real_conf")
          if fr_conf:
            st.markdown(
                f'<div style="display:inline-block;background:#E8F5E9;color:#2E7D32;'
                f'padding:4px 12px;border-radius:15px;font-size:0.82rem;font-weight:600;'
                f'margin-bottom:10px;">✅ Real plant confirmed · {fr_conf}</div>',
                unsafe_allow_html=True,
            )

          species_name = data.get("species", "N/A")
          species_conf = data.get("species_conf", "0%")
          full_health_diagnosis = str(data.get("health_diagnosis", "Unknown")).strip()
          health_conf_raw = data.get("health_conf", "0%")

          # Trust the authoritative is_healthy flag from the API.
          # Fall back to string check only if the field is absent (old server).
          api_is_healthy = data.get("is_healthy")
          if api_is_healthy is not None:
            is_healthy = bool(api_is_healthy)
          else:
            is_healthy = "healthy" in full_health_diagnosis.lower()

          if is_healthy:
            health_status = "Healthy"
            health_detail = ""
            status_color = "#2E7D32"
            bg_badge_color = "#E8F5E9"
          else:
            health_status = "Unhealthy"
            health_detail = full_health_diagnosis
            status_color = "#D32F2F"
            bg_badge_color = "#FFEBEE"

          watering_plan = data.get("watering_assessment", "")
          improvement_plan = data.get("improvement_plan", "")

          m1, m2 = st.columns(2)

          with m1:
            species_conf_display = (
                "" if species_name.startswith("Sorry") else
                f'<div style="font-size:0.85rem;color:#2E7D32;background-color:#E8F5E9;padding:4px 10px;border-radius:15px;width:fit-content;font-weight:600;">↑ Confidence: {species_conf}</div>'
            )
            st.html(
                f'<div style="background-color:#F8F9FA;padding:16px 18px;border-radius:10px;border-left:5px solid #2E7D32;min-height:150px;display:flex;flex-direction:column;justify-content:space-between;">'
                f'<div style="font-size:0.95rem;color:#555555;font-weight:600;">🌿 Predicted Species</div>'
                f'<div style="font-size:1.1rem;font-weight:700;color:#111111;margin:6px 0;word-wrap:break-word;">{species_name}</div>'
                f'{species_conf_display}'
                f'</div>'
            )

          with m2:
            detail_line = (
                f'<div style="font-size:0.82rem;color:#888;margin:2px 0 6px 0;">{health_detail}</div>'
                if health_detail else ""
            )
            st.html(
                f'<div style="background-color:#F8F9FA;padding:16px 18px;border-radius:10px;border-left:5px solid {status_color};min-height:150px;display:flex;flex-direction:column;justify-content:space-between;">'
                f'<div style="font-size:0.95rem;color:#555555;font-weight:600;">🩺 Health Diagnosis</div>'
                f'<div style="font-size:1.25rem;font-weight:700;color:{status_color};margin:6px 0 2px 0;">{health_status}</div>'
                f'{detail_line}'
                f'<div style="font-size:0.85rem;color:{status_color};background-color:{bg_badge_color};padding:4px 10px;border-radius:15px;width:fit-content;font-weight:600;">↑ Confidence: {health_conf_raw}</div>'
                f'</div>'
            )

          st.divider()
          st.markdown("### 💧 Watering Assessment")
          st.info(watering_plan)

          st.markdown("### 🛠️ Targeted Improvement Plan")
          st.warning(improvement_plan)

        else:
          st.error(f"Server Error {response.status_code}: {response.text}")

      except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")