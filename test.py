import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from DataSet import CustomCellDataset  # Replace with your actual dataset module
from UnetModule import Unet  # Replace with your actual U-Net module

def test_model():
    # Initialize U-Net model and load the trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet().to(device)
    model.load_state_dict(torch.load("cell_detection_unet_model.pth"))
    model.eval()

    # Load your custom test dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = CustomCellDataset(root='Resources/test', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Use batch_size=1 for testing

    # Define a binary cross-entropy loss function
    criterion = nn.BCEWithLogitsLoss()

    # Testing loop
    total_loss = 0.0
    with torch.no_grad():
        for images, masks in test_dataloader:
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)

            ## Resize masks if necessary (assuming 1 channel for each class)
            #if outputs.shape != masks.shape:
            #    outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=True)

            # Calculate loss
            loss = criterion(outputs, masks)
            total_loss += loss.item()

        average_loss = total_loss / len(test_dataloader)
        print(f"Average Test Loss: {average_loss:.4f}")

if __name__ == "__main__":
    test_model()
