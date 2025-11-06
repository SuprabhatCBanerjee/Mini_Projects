import torch
from torchvision import datasets, transforms, models
from torchvision.models import ConvNeXt_Tiny_Weights
from PIL import Image
from io import BytesIO

#mean and standard deviation for normalization
MEAN = (0.35)
STD = (0.48)

#for changing grayscale image to rgb
def to_rgb():
    return transforms.Lambda(lambda img: img.convert("RGB"))

transform = transforms.Compose([
    to_rgb(),
    transforms.Resize((300,300)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

#settng up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")


#loading pretrained weights
weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1


def get_model():
    model = models.convnext_tiny(weights=weights)
    model.load_state_dict(torch.load("brain_tumor_classifier.pth"))
    model.eval()
    return model

def predict(model : torch.nn.Module, image_bytes: bytes):
    result_class = ["No", "Yes"]
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    in_img = transform(img)
    final_input = in_img.unsqueeze(0)
    final_input = final_input.to(device)
    model.to(device)

    with torch.no_grad():
        output = model(final_input)
        _, predict = torch.max(output.data, 1)
        return result_class[predict.item()]
