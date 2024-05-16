import torch
from torchvision import transforms
from PIL import Image
import cv2
import os


# model = torch.load('Tumordensenet201.pt')
# model.eval()

labels = ['Bacterial Pneumonia', 'COVID-19', 'Normal','Viral Pneumonia']
model = torch.load('methods\CovidPneumonia2densenet201.pt')
device = torch.device('cpu')
model.to(device)
transform = transforms.Compose([
        transforms.Resize((400,300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
model.eval()






def predict_CovidPneumonia(filename):
    img = cv2.imread(os.path.join('static',filename)) 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    # img = Image.open(os.path.join('static',filename))
    # print(os.path.join('static',filename))
    transformed_img = transform(img_pil)
    # print(transformed_img.shape)
    transformed_img_4d = transformed_img.unsqueeze(0)
    print(transformed_img_4d.shape)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(transformed_img_4d)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in range(3)}
    return labels[torch.argmax(prediction)]



