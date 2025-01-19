import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json


# Define the EfficientNet model structure
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # Modify the classifier for the given number of classes
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.efficientnet(x)


# Load the class indices
def load_class_indices(json_path):
    with open(json_path, 'r') as f:
        class_indices = json.load(f)
    return {i: label for i, label in enumerate(class_indices)}


# Preprocessing pipeline for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Load the trained model
def load_model(model_path, num_classes):
    model = EfficientNetModel(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


# Predict the disease from an input image
def predict_disease(image_path, model, idx_to_class):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = idx_to_class[predicted_idx.item()]

    return predicted_class


# Main function
if __name__ == "__main__":
    # Paths
    model_path = "efficientnet_model.pth"  # Path to your trained model
    class_indices_path = "class_indices.json"  # Path to class indices
    image_path = "03EczemaExcoriated011204.jpg"  # Path to the test image

    # Load class indices and model
    idx_to_class = load_class_indices(class_indices_path)
    model = load_model(model_path, num_classes=len(idx_to_class))

    # Predict the disease
    predicted_disease = predict_disease(image_path, model, idx_to_class)
    print(f"Predicted Disease: {predicted_disease}")
