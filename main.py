from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import faiss
import numpy as np
import os
from transformers import CLIPProcessor, CLIPModel

app = FastAPI()

print("Loading model...")

# Load CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Loading FAISS index...")

# Load FAISS index
index = faiss.read_index("product_index.faiss")

# Load danh sách ảnh
image_folder = "product_images"
image_files = os.listdir(image_folder)

print("System ready")

@app.post("/search-image")
async def search_image(file: UploadFile = File(...)):

    # đọc ảnh upload
    image = Image.open(file.file).convert("RGB")

    # preprocess ảnh
    inputs = processor(images=image, return_tensors="pt")

    # tạo vector embedding
    with torch.no_grad():
        outputs = model.vision_model(**inputs)

    vector = outputs.pooler_output[0].cpu().numpy().astype("float32")

    # reshape cho FAISS
    vector = np.expand_dims(vector, axis=0)

    # search
    distances, indices = index.search(vector, 5)

    # convert index -> file name
    results = [image_files[i] for i in indices[0]]

    return {
        "similar_images": results,
        "distance": distances.tolist()
    }