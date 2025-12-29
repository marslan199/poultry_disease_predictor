# poultry_app.py
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import os
import zipfile

# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Classes
# -----------------------------
class_names = ['COCCI', 'CRD', 'HEALTHY', 'HEALTHY_EYE', 'NEW_CASTLE', 'SALMONELA']
num_classes = len(class_names)

# -----------------------------
# Model paths
# -----------------------------
zip_path = "diseases_predictor_weights.zip"
weights_path = "resnet18_bird_disease_weights.pt"

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    # Unzip if weights file doesn't exist
    if not os.path.exists(weights_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
    
    # Initialize architecture
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load weights-only checkpoint
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# -----------------------------
# Image Transformation
# -----------------------------
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# -----------------------------
# Streamlit Layout
# -----------------------------
st.set_page_config(
    page_title="Poultry Disease Classifier",
    page_icon="🐔",
    layout="centered"
)

st.title("🐔 Poultry Disease Classifier")
st.markdown("Upload an image of a bird or its eye to detect possible diseases.")

# Upload section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Disease"):
        with st.spinner("Analyzing..."):
            input_tensor = transform_image(image).to(device)
            with torch.inference_mode():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probs, 1)
                disease_name = class_names[pred_idx.item()]
                confidence = conf.item() * 100

        st.success(f"**Prediction:** {disease_name}")
        st.info(f"**Confidence:** {confidence:.2f}%")
        
        # Back button
        if st.button("🔙 Back"):
            st.experimental_rerun()

# Optional footer
st.markdown("---")
st.markdown("Built with ❤️ using **PyTorch** and **Streamlit**")
