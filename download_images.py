import os
import requests
import pandas as pd
from tqdm import tqdm

# Load your dataset created earlier
df = pd.read_csv("real_dataset.csv")

# Folder to save images
os.makedirs("house_images", exist_ok=True)

# Simple function to download from Unsplash
def download_image(query, save_path):
    url = f"https://source.unsplash.com/600x400/?{query}"
    try:
        image_data = requests.get(url, timeout=7)
        if image_data.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(image_data.content)
            return True
        return False
    except:
        return False

image_paths = []

print("Downloading images for each property...")

for i, row in tqdm(df.iterrows(), total=len(df)):
    bhk = int(row["bedrooms"])
    area = row["area_sqft"]

    # Create a search query
    query = f"{bhk}bhk+bengaluru+house+apartment+interior"

    save_path = f"house_images/house_{i}.jpg"

    ok = download_image(query, save_path)

    if ok:
        image_paths.append(save_path)
    else:
        image_paths.append("house_images/default.jpg")

# Update dataset
df["image"] = image_paths

df.to_csv("real_dataset_with_images.csv", index=False)

print("Image downloading completed!")
print("Saved dataset as real_dataset_with_images.csv")
