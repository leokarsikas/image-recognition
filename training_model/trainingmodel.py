import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
from tqdm import tqdm  # The progress bar

# Code inspiration: 
# https://dev.to/santoshpremi/fine-tuning-a-pre-trained-model-in-pytorch-a-step-by-step-guide-for-beginners-4p6l

# Dataset Structure 
class SkinLesionDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        self.classes = sorted(os.listdir(img_dir))  # Folder names equals to the classes
        self.class_map = {}
        for id, cls in enumerate(self.classes):
            self.class_map[cls] = id
   
        # Store all image paths and their corresponding labels in an array
        self.image_paths = []
        for cls in self.classes:
            cls_folder = os.path.join(img_dir, cls)
            if os.path.isdir(cls_folder):
                for img_name in os.listdir(cls_folder):
                    img_path = os.path.join(cls_folder, img_name)
                    self.image_paths.append((img_path, self.class_map[cls]))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, id):
        img_path, label = self.image_paths[id]
        image = Image.open(img_path).convert("RGB")  # Open image in RGB mode
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)

# Transformations 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Set the dataset path
train_dir = os.path.abspath(os.path.join("base_dir" , "train_dir"))  # Replace with your training directory path
val_dir = os.path.abspath(os.path.join("base_dir" , "val_dir"))      # Replace with your validation directory path      

# assign the datasets
train_dataset = SkinLesionDataset(train_dir, transform=transform)
val_dataset = SkinLesionDataset(val_dir, transform=transform)
# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model initialization
model = models.densenet121(weights='IMAGENET1K_V1') # pretrained weights
num_classes = len(train_dataset.classes)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

if torch.cuda.is_available():
    device = torch.device("cuda")  # CUDA 
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple GPU
else:
    device = torch.device("cpu")  # CPU

model = model.to(device)

# Loss function and Optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training time 
for epoch in range(2):  # Number of epochs
    model.train()
    running_loss = 0.0
    
    # show a progress bar, UI - pleasure
    with tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch") as progress_bar:
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad() # resets the gradients of model parameters
            outputs = model(inputs)  # AI prediction
            loss = loss_function(outputs, labels)  # Compute loss
            loss.backward()   # backpropogate prediction loss
            optimizer.step()  # update the models paramters
            
            running_loss += loss.item()  # running loss

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

# Evaluation time
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy after evaluation: {100 * correct / total:.2f}%")

# Saving the model...
torch.save(model.state_dict(), 'utandecay_cuda_test.pth')