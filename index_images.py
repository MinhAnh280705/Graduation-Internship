import requests
from io import BytesIO
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import faiss
import numpy as np

print("Loading ResNet18 model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet18 pretrained
base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Bỏ lớp fully connected cuối, chỉ lấy feature vector
model = torch.nn.Sequential(*list(base_model.children())[:-1])
model.eval()
model.to(device)

print("Model loaded")

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
        feature = model(image_tensor)   # shape: [1, 512, 1, 1]

    feature = feature.view(feature.size(0), -1)  # [1, 512]
    feature = feature.cpu().numpy().astype("float32")

    # Normalize để dùng cosine similarity với FAISS IndexFlatIP
    faiss.normalize_L2(feature)

    return feature[0]

vectors = []
valid_urls = []

with open("image_urls.txt", "r", encoding="utf-8") as f:
    urls = [line.strip() for line in f if line.strip()]

print("Total images:", len(urls))

for url in urls:
    print("Processing:", url)

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content)).convert("RGB")
        vector = extract_feature(image)

        vectors.append(vector)
        valid_urls.append(url)

    except Exception as e:
        print(f"Skip {url}: {e}")

if len(vectors) == 0:
    raise ValueError("Không có ảnh hợp lệ để tạo index.")

vectors = np.array(vectors).astype("float32")

print("Vector matrix shape:", vectors.shape)

dimension = vectors.shape[1]

# Dùng Inner Product sau khi normalize -> tương đương cosine similarity
index = faiss.IndexFlatIP(dimension)
index.add(vectors)

faiss.write_index(index, "product_index.faiss")
np.save("image_urls.npy", np.array(valid_urls))

print("Index created successfully!")
print("Saved image_urls.npy successfully!")