
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F

class CellDetectionCNN(nn.Module):
    def __init__(self):
        super(CellDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) # 16, 256, 256
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1) # 16, 256, 256
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2) # 16, 128, 128
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1) # 32, 128, 128
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1) # 32, 128, 128
        self.relu4 = nn.ReLU()
        self.fc1 = nn.Linear(32 * 128 * 128, 1024)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 2)
        self.sigmoid = nn.Sigmoid()
        
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        
        return x


# Update the custom dataset class
class CustomCellDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_files = [f for f in os.listdir(root) if f.endswith('_img.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root, img_name)
        mask_name = img_name.replace('_img.png', '_masks.png')
        mask_path = os.path.join(self.root, mask_name)

        raw_image = Image.open(img_path).convert("RGB") # Convert to RGB
        mask = Image.open(mask_path).convert("RGB")

        if self.transform:
            raw_image = self.transform(raw_image)
            mask = self.transform(mask)

        return raw_image, mask

# Sample data loading and preprocessing
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Use the custom dataset class
dataset = CustomCellDataset(root='Resources/train', transform=data_transform)

# Split the dataset into training, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Instantiate the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CellDetectionCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.float().to(device)
            outputs = model(inputs)
            targets = model(inputs) 

            val_loss += criterion(outputs, targets).item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {val_loss}")

# Save the trained model
torch.save(model.state_dict(), "cell_detection_cnn_model.pth")

