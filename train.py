import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np

# Dataset
class PropertyDataset(Dataset):
    def __init__(self, csv):
        self.df = pd.read_csv(csv)
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open("images/" + row["image"]).convert("RGB")
        img = self.transform(img)

        tab = torch.tensor([
            row["latitude"],
            row["longitude"],
            row["area_sqft"],
            row["bedrooms"],
            row["bathrooms"]
        ], dtype=torch.float32)

        price = torch.tensor(np.log1p(row["price"]), dtype=torch.float32)
        return img, tab, price

# Model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet18(weights="IMAGENET1K_V1")
        self.cnn.fc = nn.Linear(512, 128)

        self.tab = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU()
        )

        self.final = nn.Linear(128 + 32, 1)

    def forward(self, img, t):
        img_f = self.cnn(img)
        tab_f = self.tab(t)
        x = torch.cat([img_f, tab_f], dim=1)
        return self.final(x)

# Training
ds = PropertyDataset("data.csv")
loader = DataLoader(ds, batch_size=4, shuffle=True)

model = Model()
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

for epoch in range(5):
    for img, tab, price in loader:
        img = img.to(device)
        tab = tab.to(device)
        price = price.to(device)

        pred = model(img, tab).view(-1)

        loss = loss_fn(pred, price)

        opt.zero_grad()
        loss.backward()
        opt.step()

    print("Epoch", epoch, "Loss:", loss.item())

torch.save(model.state_dict(), "model.pth")
print("Training Completed!")
