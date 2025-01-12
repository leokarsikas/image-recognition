import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import os 

# Initialize the model
model = models.densenet121(weights='IMAGENET1K_V1')
model.classifier = nn.Linear(model.classifier.in_features, 7) # 7 classes in our dataset

model_path=os.path.abspath(os.path.join("..", "trained_models" , "utandecay_cuda_10epoch.pth")) #choose model path

model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)

if torch.cuda.is_available():
    device = torch.device("cuda")  # CUDA 
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple GPU
else:
    device = torch.device("cpu")  # CPU

model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class_to_idx =  {
    0: 'akiec',
    1: 'bcc',
    2: 'bkl',
    3: 'df',
    4: 'mel',
    5: 'nv',
    6: 'vasc'
}

def predict_image(model, img_path, transform):
    
    image = Image.open(img_path).convert("RGB")
    image = transform(image)
    image = image.to(device)
    
    model.eval()
    
    with torch.no_grad():
        output = model(image.unsqueeze(0)) # AI prediction
        _, predicted = torch.max(output, 1)  # Get the class with the highest probability
    
    # find predicted class with index to the class label
    predicted_class = class_to_idx[predicted.item()]

    result = ""

    if(predicted_class == "bcc" or predicted_class == "mel"):
        result = "Malignant"

    elif(predicted_class == "akiec"):
        result = "Potentially malignant"    
    else:
        result = "benign"    
    
    return predicted_class, result


image_path = os.path.abspath(os.path.join("test_images" ,"melanoma.jpg")) #choose image path
# Test image
#img_path = "demo\test_images\ISIC_0469776.jpg"  # image path
predicted_class, result = predict_image(model, image_path, transform)
print("\n-------------------- RESULTS --------------------\n")
print(f"The predicted class for the image is: {predicted_class}\nWhich indicates that it is: {result}\n")
print("----------------- END OF RESULTS ----------------\n")