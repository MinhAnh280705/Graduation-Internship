from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import faiss
import numpy as np
import os

print("Loading model...")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Model loaded")

vectors = []

image_folder = "product_images"
files = os.listdir(image_folder)

print("Total images:", len(files))

for file in files:

    print("Processing:", file)

    image = Image.open(f"{image_folder}/{file}").convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model.vision_model(**inputs)

    # vector embedding 768 chiều
    vector = outputs.pooler_output[0].cpu().numpy()

    vectors.append(vector)

vectors = np.array(vectors).astype("float32")

print("Vector matrix shape:", vectors.shape)

dimension = vectors.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(vectors)

faiss.write_index(index, "product_index.faiss")

print("Index created successfully!")