from fastapi import FastAPI, File, UploadFile
import torch
import cv2
import numpy as np
from torchvision import transforms
from model import VisionTransformer

from PIL import Image

app = FastAPI()

model = VisionTransformer(96, 4, 7, 128, 4, 4, 360)
model.load_state_dict(torch.load("emotion_classifier.pth", map_location="cpu"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

CLASSES = ["angry","happy","sad","neutral","fear","surprise","disgust"]

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    img = np.frombuffer(await image.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(img)  

    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.argmax(model(x), dim=1).item()

    return {"emotion": CLASSES[pred]}
