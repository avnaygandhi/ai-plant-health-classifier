import requests
import streamlit as st

st.set_page_config(page_title="AI Plant Health & Species Identifier", page_icon="🌿", layout="wide",
                   initial_sidebar_state='expanded')
st.markdown(
    """
    <style>
    .main-header { font-size: 2.3rem; font-weight: 700; color: #2E7D32; margin-bottom: 0px; }
    .sub-header { font-size: 1.1rem; color: #555555; margin-bottom: 20px; }
    .card { background-color: #F8F9FA; padding: 20px; border-radius: 10px; border-left: 5px solid #2E7D32; margin-bottom: 15px; }
    </style>
""",
    unsafe_allow_html=True,
)
st.markdown("<div class=\"main-header\">>🌿 AI Plant Health & Species Identifier</div>",
            unsafe_allow_html=True, )
st.markdown(
    "<div class='sub-header'>Upload a leaf image to instantly identify the species and receive actionable health care plans.</div>",
    unsafe_allow_html=True,
    )
#Sidebar
with st.sidebar:
    st.header("Application Controls")
    st.info("Backend Status:**Active**\n\nAPI Server: `http://127.0.0.1:8000/predict`")
    st.divider()
    # Add Shutdown Button
    if st.button("🛑 Stop Server & Exit", use_container_width=True):
        st.warning("Shutting down session...")

        # 1. Trigger JavaScript to immediately close the browser tab
        st.components.v1.html(
            """
                <script>
                    window.close();
                    // Fallback for browsers blocking window.close()
                    window.location.href = "about:blank";
                </script>
                """,
            height=0,
        )

        # 2. Stop Python processes cleanly
        import os
        import signal

        os.kill(os.getpid(), signal.SIGTERM)
#Image Uploader
uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    col1, col2 = st.columns([1,1.2],gap="large")
    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(uploaded_file,use_container_width=True)
    with col2:
        st.subheader("🧠 Real-time AI Analysis")
        with st.spinner("Processing image through vision pipeline..."):
            try:
                files={"file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )}
                response = requests.post("http://127.0.0.1:8000/predict",files=files)
                if response.status_code == 200:
                    data = response.json()
                    st.success("Analysis Complete!")

                    health_diagnosis = data.get("health_diagnosis", "Unknown")
                    health_status = (
                        health_diagnosis.split("-")[-1].strip()
                        if "-" in health_diagnosis
                        else health_diagnosis
                    )

                    m1, m2 = st.columns(2)

                    m1.metric(
                        label="🌿 Predicted Species",
                        value=data.get("species", "N/A"),
                        delta=data.get("species_conf", "0%"),
                    )

                    m2.metric(
                        label="🩺 Health Status",
                        value=health_status,  # Displays "Healthy"
                        delta=f"Confidence: {data.get('health_conf', '0%')}",  # Displays percentage
                    )
                    st.divider()
                    st.markdown("### 💧 Watering Assessment")
                    st.info(data["watering_assessment"])

                    st.markdown("### 🛠️ Targeted Improvement Plan")
                    st.warning(data["improvement_plan"])

                else:
                    st.error(f"Server Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(
                    "⚠️ Unable to reach FastAPI backend server. Ensure the server is"
                    f" running!\n\nDetails: {e}"
                )