import torch
import numpy as np


def postprocess_segmentation(seg_output):
    """
    Postprocess segmentation output to create clean binary masks.
    
    """
    
    seg_mask = torch.sigmoid(seg_output).cpu().numpy().squeeze()
    
    return seg_mask


def postprocess_classification(cls_output):
    """
    Postprocess classification output to get predicted class.
    
    """
    # Classification mapping
    cls_map = {0: "Normal", 1: "Benign", 2: "Malignant"}
    cls_idx = torch.argmax(cls_output, dim=1).item()
    cls_label = cls_map[cls_idx]

   
    return cls_label



