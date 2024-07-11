# myapp/models.py

import torch
import torchvision.models as models
import torch.nn as nn

HAM10000_CLASSES = {
    0: 'Melanoma',  
    1: 'Melanocytic nevi',    
    2: 'Basal cell carcinoma',   
    3: 'Actinic keratoses and intraepithelial carcinoma',   
    4: 'Benign keratosis-like lesions',    
    5: 'Dermatofibroma',     
    6: 'Vascular lesions'   
}

def load_resnet50_model(filepath):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(HAM10000_CLASSES))
    state_dict = torch.load(filepath, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode
    return model
