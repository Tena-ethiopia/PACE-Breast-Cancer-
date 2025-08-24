import torchvision.transforms as transforms
from PIL import Image
import torch

def preprocess_segmentation(image_path, device):
    """
    Load and preprocess an image for segmentation.
    Returns: tensor of shape [1, C, H, W]
    """
    transform_img = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    img_orig = Image.open(image_path).convert("L")
    img_tensor = transform_img(img_orig).unsqueeze(0).to(device)
    return img_tensor, img_orig.size   # keep original size for resizing mask


def preprocess_classification(image_path, device):
    """
    Load and preprocess an image for classification.
    Returns: tensor of shape [1, C, H, W]
    """
    transform_img = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    img_orig = Image.open(image_path).convert("L")
    img_tensor = transform_img(img_orig).unsqueeze(0).to(device)
    return img_tensor
