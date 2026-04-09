from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import faiss
import numpy as np

app = FastAPI()

print("Loading ResNet18 model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet18 pretrained
base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Bỏ lớp fully connected cuối
model = torch.nn.Sequential(*list(base_model.children())[:-1])
model.eval()
model.to(device)

# Preprocess ảnh
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
        feature = model(image_tensor)   # [1, 512, 1, 1]

    feature = feature.view(feature.size(0), -1)  # [1, 512]
    feature = feature.cpu().numpy().astype("float32")

    faiss.normalize_L2(feature)

    return feature

print("Loading FAISS index...")
index = faiss.read_index("product_index.faiss")

# Load danh sách URL ảnh đã index thành công
image_urls = np.load("image_urls.npy", allow_pickle=True)

print("System ready")

@app.post("/search-image")
async def search_image(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        vector = extract_feature(image)

        scores, indices = index.search(vector, 5)

        results = []
        for i, score in zip(indices[0], scores[0]):
            if 0 <= i < len(image_urls):
                results.append({
                    "image_url": str(image_urls[i]),
                    "similarity_percent": round(float(score) * 100, 2)
                })

        return {
            "similar_images": results
        }

    except Exception as e:
        return {
            "error": str(e)
        }