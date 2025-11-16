import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os

class RealEstateDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load image using correct column
        img_path = row["image"]
        
        # If path is relative, load from project folder
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.getcwd(), img_path)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Features from your CSV
        features = torch.tensor([
            row["bedrooms"],
            row["bathrooms"],
            row["area_sqft"],
            row["latitude"],
            row["longitude"]
        ], dtype=torch.float32)

        price = torch.tensor(row["price"], dtype=torch.float32)

        return image, features, price


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(512, 128)

        self.mlp = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.final = nn.Sequential(
            nn.Linear(128+16, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, img, features):
        img_out = self.cnn(img)
        feat_out = self.mlp(features)
        x = torch.cat([img_out, feat_out], dim=1)
        return self.final(x)


# -------- TRAIN ---------

dataset = RealEstateDataset("real_dataset_with_images.csv")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using:", device)

model = Model().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

EPOCHS = 5

for epoch in range(EPOCHS):
    for img, feat, price in loader:
        img, feat, price = img.to(device), feat.to(device), price.to(device)

        pred = model(img, feat).squeeze()
        loss = loss_fn(pred, price)

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {loss.item()}")

torch.save(model.state_dict(), "model_real.pth")
print("TRAINING DONE — model_real.pth saved!")
