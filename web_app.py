import streamlit as st
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from train import Model

st.set_page_config(page_title="AI House Price Predictor", layout="centered")

# Title
st.markdown("<h1 style='text-align:center; color:#2E86C1;'>🏠 AI Real Estate Price Predictor</h1>", unsafe_allow_html=True)
st.write("Upload a property image and fill in the details to predict its price.")

# Load model
model = Model()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Upload Section
st.subheader("📸 Upload Property Image")
image_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])

if image_file:
    st.image(image_file, width=300, caption="Uploaded Image Preview")

# Form Inputs
st.subheader("📍 Property Details")

col1, col2 = st.columns(2)

with col1:
    latitude = st.number_input("Latitude", value=12.95)
    area = st.number_input("Area (sqft)", value=1200)

with col2:
    longitude = st.number_input("Longitude", value=77.60)
    bedrooms = st.number_input("Bedrooms", step=1, value=2)
    bathrooms = st.number_input("Bathrooms", step=1, value=2)

# Prediction Button
if st.button("Predict Price 💰"):
    if image_file is None:
        st.error("Please upload an image first.")
    else:
        img = Image.open(image_file).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        tab = torch.tensor([[latitude, longitude, area, bedrooms, bathrooms]], dtype=torch.float32)

        with torch.no_grad():
            log_price = model(img_tensor, tab).item()

        price = float(np.expm1(log_price))

        st.success(f"🏡 Estimated Property Price: ₹{price:,.2f}")
