import streamlit as st
from models.main import PlantClassifier

st.title("🌿 AI Plant Health & Species Identifier")
uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    # Save temp file & predict
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    classifier = PlantClassifier()
    results = classifier.predict("temp.jpg")

    st.subheader("Results")
    st.json(results)