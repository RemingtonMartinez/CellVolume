from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class CustomCellDataset(Dataset):
    def __init__(self, root, transform=None, target_size=(256, 256)):
        self.root = root
        self.transform = transform
        self.target_size = target_size

        # Identify both TIFF and PNG files
        self.image_files = [f for f in os.listdir(root) if f.endswith('_img.png') or f.endswith('_img.tif')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        # Handle different file extensions
        if img_name.endswith('_img.png'):
            img_path = os.path.join(self.root, img_name)
            mask_name = img_name.replace('_img.png', '_masks.png')
            mask_path = os.path.join(self.root, mask_name)
        elif img_name.endswith('_img.tif'):
            img_path = os.path.join(self.root, img_name)
            mask_name = img_name.replace('_img.tif', '_masks.tif')
            mask_path = os.path.join(self.root, mask_name)
        else:
            raise ValueError("Unsupported file format")

        raw_image = Image.open(img_path).convert("L")  # Convert to L to prevent adherence to cytoplasm color
        mask = Image.open(mask_path).convert("L")

        # Resize images
        raw_image = raw_image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.BILINEAR)

        # Convert PIL Image to NumPy array for manipulation
        mask_np = np.array(mask)

        # Set all non-black pixels to white (255)
        mask_np[mask_np > 0] = 255

        # Convert NumPy array back to PIL Image
        mask = Image.fromarray(mask_np)

        if self.transform:
            raw_image = self.transform(raw_image)
            mask = self.transform(mask)

        return raw_image, mask
