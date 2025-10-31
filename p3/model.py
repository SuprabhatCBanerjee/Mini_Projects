import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image
from io import BytesIO


#data preprocess
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.49),(0.49))
])

#Net Replica
class RetinographyNN(nn.Module):
     def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        #224->112->56->28 due to pulling layer, 
        #128 filter from Conv2d
        self.classifier = nn.Sequential(
            nn.Flatten(),

            #input layer
            nn.Linear(128*28*28, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            #hidden layer
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            #hidden layer
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            #output layer
            nn.Linear(64,num_classes)            
        )   

     def forward(self,x):
            x = self.features(x)
            x = self.classifier(x)
            return x
     


def get_model():
     model = RetinographyNN(5)
     model.load_state_dict(torch.load("normal_retinography_cnn_model.pth"))
     model.eval()
     return model

def predict(model : nn.Module, image_bytes : bytes):
     result_list = ['Healthy', 'Mild', 'Moderate', 'Proliferate', 'Severe']
     image = Image.open(BytesIO(image_bytes)).convert("RGB")
     input_im = transform(image)
     input_data = input_im.unsqueeze(0)
     #device
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     input_data = input_data.to(device)
     model.to(device)

     with torch.no_grad():
          output = model(input_data)
          _, predict = torch.max(output.data, 1)
          return result_list[predict.item()]


