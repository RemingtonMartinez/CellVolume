import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from UnetModule import Unet  # Replace with your actual U-Net module

class DataManager:
    @staticmethod
    def resize_image(image_path, size=(256, 256)):
        image = Image.open(image_path).convert("L")
        image = image.resize(size, Image.BICUBIC)
        return np.array(image)

def test_model(image_path):
    # Initialize U-Net model and load the trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet().to(device)
    model.load_state_dict(torch.load("cell_detection_unet_model.pth"))
    model.eval()

    # Load and resize the input image
    original_image = DataManager.resize_image(image_path)
    input_image = transforms.ToTensor()(original_image).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        output_mask = model(input_image)

    
    # Resize the output mask to the original image dimensions
    output_mask = output_mask.squeeze().cpu().numpy()
    output_mask = Image.fromarray((output_mask * 255).astype(np.uint8))
    output_mask = output_mask.resize(original_image.shape[:2][::-1], Image.BICUBIC)

    # Display the original image and the outputted mask side by side
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(output_mask, cmap="gray")
    plt.title("Output Mask")

    plt.show()

    #output_mask.save("output.png")

if __name__ == "__main__":
    image_path = "path/to/your/single/image.jpg"  # Replace with the path to your test image
    test_model(image_path)
