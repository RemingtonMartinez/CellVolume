import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from DataSet import CustomCellDataset  # Replace with your actual dataset module
from UnetModule import Unet  # Replace with your actual U-Net module

def train_model(num_epochs=20, learning_rate=0.001, batch_size=4, target_size=(256, 256)):
    # Initialize U-Net model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss for binary segmentation
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load your custom dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = CustomCellDataset(root='Resources/train', transform=transform, target_size=target_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, masks in train_dataloader:
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)

            ## Resize masks if necessary (assuming 1 channel for each class)
            #if outputs.shape != masks.shape:
            #    outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=True)

            # Calculate loss
            loss = criterion(outputs, masks)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

    # Save the trained model in the current directory as cell_detection_unet_model.pth
    torch.save(model.state_dict(), "cell_detection_unet_model.pth")

if __name__ == "__main__":
    train_model(num_epochs=10, learning_rate=0.0001, batch_size=8, target_size=(128, 128))
