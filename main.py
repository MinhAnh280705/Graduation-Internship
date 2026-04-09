from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import faiss
import numpy as np
import json

app = FastAPI()

print("Loading ResNet18 model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(base_model.children())[:-1])
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_feature(image: Image.Image) -> np.ndarray:
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = model(image_tensor)

    feature = feature.view(feature.size(0), -1)
    feature = feature.cpu().numpy().astype("float32")
    faiss.normalize_L2(feature)

    return feature

print("Loading FAISS index...")
index = faiss.read_index("product_index.faiss")

with open("image_metadata.json", "r", encoding="utf-8") as f:
    image_metadata = json.load(f)

print("System ready")

@app.post("/search-image")
async def search_image(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        vector = extract_feature(image)

        scores, indices = index.search(vector, 5)

        results = []
        for i, score in zip(indices[0], scores[0]):
            if 0 <= i < len(image_metadata):
                meta = image_metadata[i]
                results.append({
                    "product_id": meta["product_id"],
                    "product_name": meta.get("product_name"),
                    "slug": meta.get("slug"),
                    "image_url": meta["image_url"],
                    "similarity_percent": round(float(score) * 100, 2)
                })

        return {"similar_images": results}

    except Exception as e:
        return {"error": str(e)}