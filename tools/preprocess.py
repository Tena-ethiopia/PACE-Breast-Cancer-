
    

import torchvision.transforms as transforms

def preprocess_segmentation(image):
    """
    Preprocess image for segmentation task.
    
    """
    
# Preprocessing
    transform_img = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    pass
    
def preprocess_classification(image):
    """
    Preprocess image for classification task.
    
    """
    
# Preprocessing
    transform_img = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    pass
