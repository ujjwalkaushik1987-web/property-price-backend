import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from train import Model   # Load the model class

# Load model
model = Model()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# Image transformer
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict(image_path, latitude, longitude, area, bedrooms, bathrooms):
    # Load image
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    # Tabular input
    tab = torch.tensor([[latitude, longitude, area, bedrooms, bathrooms]], dtype=torch.float32)

    # Predict
    with torch.no_grad():
        pred_log = model(img, tab).item()

    price = np.expm1(pred_log)
    return price

# Example Prediction
result = predict("images/house1.jpg", 12.97, 77.59, 1200, 2, 2)
print("Predicted Price:", result)
