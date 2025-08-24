import argparse
import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
from model import MultiTaskModel
from tools.Postprocess import postprocess_segmentation, postprocess_classification
from tools.preprocess import preprocess_classification, preprocess_segmentation


def load_model(checkpoint_path, device):
    """Load the trained model from checkpoint."""
    model = MultiTaskModel()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)   # no ['model_state_dict'] here
    model.to(device)
    model.eval()
    return model



def get_image_files(input_dir):
    """Get all PNG image files from input directory."""
    image_extensions = ['.png', '.jpg', '.jpeg']
    image_files = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    return image_files



def run_segmentation(model, input_dir, output_dir, device):
    seg_output_dir = os.path.join(output_dir, 'segmentation')
    os.makedirs(seg_output_dir, exist_ok=True)
    
    image_files = get_image_files(input_dir)
    print(f"Processing {len(image_files)} images for segmentation...")
    
    with torch.no_grad():
        for image_path in tqdm(image_files, desc="Segmentation"):
            # ✅ Load and preprocess image
            img_tensor, orig_size = preprocess_segmentation(image_path, device)

            seg_output, _ = model(img_tensor)

            mask = postprocess_segmentation(seg_output)

            # Resize back to original size
            mask = Image.fromarray((mask * 255).astype(np.uint8))
            mask = mask.resize(orig_size)
            # Save mask
            image_name = os.path.basename(image_path)
            mask_name = image_name.replace('.png', '_mask.png')
            mask.save(os.path.join(seg_output_dir, mask_name))
    
    print(f"Segmentation masks saved to: {seg_output_dir}")



def run_classification(model, input_dir, output_dir, device):
    """Run classification task and save results to CSV."""
    cls_output_dir = os.path.join(output_dir, 'classification')
    os.makedirs(cls_output_dir, exist_ok=True)
    
    image_files = get_image_files(input_dir)
    print(f"Processing {len(image_files)} images for classification...")
    
    results = []
    
    with torch.no_grad():
        for image_path in tqdm(image_files, desc="Classification"):
            # ✅ Load and preprocess image
            img_tensor = preprocess_classification(image_path, device)

            _, cls_output = model(img_tensor)

            
            # Postprocess classification
            predicted_label = postprocess_classification(cls_output)
            
            # Get image ID (filename without extension)
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            
            results.append({
                'image_id': image_id,
                'label': predicted_label
            })
    
    # Save results to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(cls_output_dir, 'predictions.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Classification results saved to: {csv_path}")


def main(input_dir, output_dir, task, device_type):
    """Main inference function."""
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Task: {task}")
    print(f"Device: {device_type}")
    
    # Set device
    if device_type == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load model
    checkpoint_path = os.path.join('checkpoints', 'best_multitask_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    model = load_model(checkpoint_path, device)
    print("Model loaded successfully")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run inference based on task
    if task == 'seg':
        run_segmentation(model, input_dir, output_dir, device)
    elif task == 'cls':
        run_classification(model, input_dir, output_dir, device)
    else:
        raise ValueError(f"Invalid task: {task}. Must be 'seg' or 'cls'")
    
    print("Inference completed successfully!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Ultrasound Multi-task Model Inference")
    
    parser.add_argument("-i", "--input", required=True, 
                       help="Path to input directory containing ultrasound images")
    parser.add_argument("-o", "--output", required=True, 
                       help="Path to output directory")
    parser.add_argument("-t", "--task", choices=['seg', 'cls'], required=True,
                       help="Task to perform: 'seg' for segmentation, 'cls' for classification")
    parser.add_argument("-d", "--device", choices=['cpu', 'gpu'], default='gpu',
                       help="Device to use for inference: 'cpu' or 'gpu'")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        input_dir=args.input,
        output_dir=args.output,
        task=args.task,
        device_type=args.device
    )