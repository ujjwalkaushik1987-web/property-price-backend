from fastapi import FastAPI, File, UploadFile, Form
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from train import Model
import uvicorn

app = FastAPI()

# Load your model
model = Model()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    area: float = Form(...),
    bedrooms: int = Form(...),
    bathrooms: int = Form(...)
):
    img = Image.open(image.file).convert("RGB")
    img = transform(img).unsqueeze(0)

    tab = torch.tensor([[latitude, longitude, area, bedrooms, bathrooms]], dtype=torch.float32)

    with torch.no_grad():
        log_price = model(img, tab).item()

    price = np.expm1(log_price)
    return {"predicted_price": float(price)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
