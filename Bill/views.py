from django.shortcuts import redirect, render



from django.shortcuts import render
from django.http import JsonResponse
from .models import load_resnet50_model, HAM10000_CLASSES
from PIL import Image
import torch
import torchvision.transforms as transforms

resnet50_model = load_resnet50_model('resnet50_best.pt')

def evaluate_image(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        image = Image.open(image)
        
        transform = transforms.Compose([
        transforms.Resize((224, 280)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        image_tensor = transform(image)
        
        print(f"Transformed image tensor shape: {image_tensor.shape}")

        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = resnet50_model(image_tensor)
            print(f"Output tensor shape from model: {output.shape}")
            
            prediction_idx = output.argmax(dim=1).item()
            prediction_label = HAM10000_CLASSES[prediction_idx]
            prediction_message = f"{prediction_label.capitalize()} detected"

        return render(request, 'upload_image.html', {'prediction': prediction_message})



    return render(request, 'upload_image.html')


    
def home(request):
    
    return render(request, "home.html")