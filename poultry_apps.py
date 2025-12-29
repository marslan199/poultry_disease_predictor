import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import requests
import io

st.set_page_config(page_title="Poultry Disease Predictor", layout="centered")

st.title("🐔 Poultry Disease Predictor")

# ----------- Model download -----------
@st.cache_resource
def load_model():
    url = "https://drive.google.com/file/d/1VK1NuYajH4o35d-8NiiSh78Bx8F7-G8y/view?usp=drive_link"  # replace with Google Drive or Hugging Face link
    response = requests.get(url)
    model_file = io.BytesIO(response.content)
    
    model = torch.load(model_file, map_location="cpu")
    model.eval()
    return model

model = load_model()

# ----------- Upload image -----------
uploaded_file = st.file_uploader("Upload a chicken image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # ----------- Preprocess -----------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)

    # ----------- Predict -----------
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(1).item()

    # ----------- Display result -----------
    classes = ['COCCI', 'CRD', 'HEALTHY', 'HEALTHY_EYE', 'NEW_CASTLE', 'SALMONELA']
    st.success(f"Predicted Disease: {classes[pred_class]}")

    # Back button
    if st.button("Back"):
        st.experimental_rerun()
