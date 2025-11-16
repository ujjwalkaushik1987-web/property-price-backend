import pandas as pd
import numpy as np
import os

df = pd.read_csv("bhp.csv")

# Clean dataset — remove missing values
df = df.dropna()

# Extract numeric area (sqft)
def convert_sqft(x):
    try:
        if "-" in x:
            a, b = x.split("-")
            return (float(a) + float(b)) / 2
        return float(x)
    except:
        return None

df["area_sqft"] = df["total_sqft"].apply(convert_sqft)
df = df.dropna(subset=["area_sqft"])

# Convert bedroom count
df["bedrooms"] = df["size"].str.extract("(\d+)").astype(float)

# Rename columns
df["bathrooms"] = df["bath"]
df["price"] = df["price"] * 100000  # Lakhs → Rupees

# STEP: Create placeholder images (We will replace later)
os.makedirs("real_images", exist_ok=True)
placeholder = "real_images/default.jpg"

# create a placeholder image
from PIL import Image
img = Image.new("RGB", (224, 224), (200, 200, 200))
img.save(placeholder)

df["image"] = placeholder

# STEP: Add fake lat/long for now (real will be added later)
# Just random values inside Bangalore
df["latitude"] = 12.9 + np.random.rand(len(df)) * 0.1
df["longitude"] = 77.5 + np.random.rand(len(df)) * 0.1

# Save final dataset
df_final = df[["price", "latitude", "longitude", "area_sqft",
               "bedrooms", "bathrooms", "image"]]

df_final.to_csv("real_dataset.csv", index=False)

print("Dataset created successfully as real_dataset.csv")
print("Next step: download real house images.")
