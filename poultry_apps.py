import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import os
import zipfile

# -----------------------------
# Paths
# -----------------------------
MODEL_ZIP_LOCAL = "diseases_predictor_weights.zip"
WEIGHTS_DIR = "diseases_predictor_weights"
WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, "diseases_predictor_weights.pt")  # correct filename

# -----------------------------
# Ensure weights are extracted
# -----------------------------
if not os.path.exists(WEIGHTS_DIR):
    if not os.path.exists(MODEL_ZIP_LOCAL):
        st.error(f"Local zip file {MODEL_ZIP_LOCAL} not found!")
    else:
        with zipfile.ZipFile(MODEL_ZIP_LOCAL, "r") as zip_ref:
            zip_ref.extractall(WEIGHTS_DIR)

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_classes = 6  # your dataset classes
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    if not os.path.exists(WEIGHTS_FILE):
        st.error(f"Weights file not found: {WEIGHTS_FILE}")
        return None
    
    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -----------------------------
# Transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Class labels
# -----------------------------
classes = ['COCCI', 'CRD', 'HEALTHY', 'HEALTHY_EYE', 'NEW_CASTLE', 'SALMONELA']

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Poultry Disease Predictor", layout="centered")
st.title("🐔 Poultry Disease Predictor")

# File uploader
uploaded_file = st.file_uploader("Upload a chicken image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        if model is not None:
            input_tensor = transform(image).unsqueeze(0)  # add batch dimension
            with torch.no_grad():
                output = model(input_tensor)
                pred_idx = output.argmax(dim=1).item()
                pred_class = classes[pred_idx]
            st.success(f"Predicted Disease: {pred_class}")
        else:
            st.error("Model not loaded. Cannot predict.")
    
    if st.button("Back"):
        st.experimental_rerun()
