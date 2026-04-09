import requests
from io import BytesIO
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import faiss
import numpy as np
from pymongo import MongoClient
import json

print("Loading ResNet18 model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(base_model.children())[:-1])
model.eval()
model.to(device)

print("Model loaded")

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
    return feature[0]

# =========================
# KẾT NỐI MONGODB
# =========================
MONGO_URI = "mongodb+srv://quangtk205_db_user:uA4N937qK7EaThcm@thuc-tap-tot-nghiep-tha.gb1v5qc.mongodb.net/Graduation-Internship"
DB_NAME = "Graduation-Internship"
COLLECTION_NAME = "products"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

vectors = []
metadata = []

# Nếu ảnh là đường dẫn tương đối /media/... thì cần base domain
BASE_IMAGE_URL = "https://tttn-he-thong-luu-tru.onrender.com/"

products = collection.find({}, {
    "_id": 1,
    "name": 1,
    "slug": 1,
    "images.url": 1
})

for product in products:
    product_id = str(product["_id"])
    product_name = product.get("name", "")
    slug = product.get("slug", "")
    images = product.get("images", [])

    for img in images:
        raw_url = img.get("url")
        if not raw_url:
            continue

        image_url = raw_url
        if raw_url.startswith("/"):
            image_url = BASE_IMAGE_URL + raw_url

        print(f"Processing product_id={product_id} | url={image_url}")

        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()

            image = Image.open(BytesIO(response.content)).convert("RGB")
            vector = extract_feature(image)

            vectors.append(vector)
            metadata.append({
                "product_id": product_id,
                "product_name": product_name,
                "slug": slug,
                "image_url": image_url
            })

        except Exception as e:
            print(f"Skip {image_url}: {e}")

if len(vectors) == 0:
    raise ValueError("Không có ảnh hợp lệ để tạo index.")

vectors = np.array(vectors).astype("float32")
print("Vector matrix shape:", vectors.shape)

dimension = vectors.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(vectors)

faiss.write_index(index, "product_index.faiss")

with open("image_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("Index created successfully!")
print("Saved image_metadata.json successfully!")