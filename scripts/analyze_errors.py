import torch
import cv2
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import matplotlib.pyplot as plt
from dataloader import create_dataloader
from deepfake_classifier import get_model
import os

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else 
    "mps" if torch.backends.mps.is_available() else 
    "cpu"
)

def analyze_misclassified(model_name="custom", dataset="val", num_images=8):
    """
    Analyzes a batch of misclassified images and generates Grad-CAM visualizations
    """
    # Load model
    model = get_model(model_name).to(DEVICE)
    model.load_state_dict(torch.load(f"models/{model_name}_deepfake.pth", map_location=DEVICE))
    model.eval()

    # Get the last convolutional layer for Grad-CAM
    target_layer = model.conv3 if model_name == "custom" else None  # Adjust for other models
    
    # Create dataloader
    test_loader = create_dataloader(f"processed_data/{dataset}_cropped", batch_size=32, shuffle=True)
    
    misclassified_images = []
    misclassified_labels = []
    predictions = []
    seen_images = set()  # Track unique images
    
    # Find misclassified images
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Get indices of misclassified images
            misclassified_idx = (preds != labels).nonzero(as_tuple=True)[0]
            
            for idx in misclassified_idx:
                # Create a unique identifier for the image
                img_hash = hash(images[idx].cpu().numpy().tobytes())
                
                if img_hash not in seen_images:
                    seen_images.add(img_hash)
                    misclassified_images.append(images[idx].cpu())
                    misclassified_labels.append(labels[idx].cpu().item())
                    predictions.append(preds[idx].cpu().item())
                    
                    if len(misclassified_images) >= num_images:
                        break
            if len(misclassified_images) >= num_images:
                break
    
    # Create Grad-CAM visualizations
    fig, axes = plt.subplots(2, 8, figsize=(30, 10))  # Changed to 2x8 for side-by-side views
    
    for idx, (img, true_label, pred_label) in enumerate(zip(misclassified_images[:num_images], 
                                                           misclassified_labels[:num_images],
                                                           predictions[:num_images])):
        # Convert tensor to numpy and normalize to [0, 1]
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        img_np = img_np.astype(np.float32)
        
        # Calculate Grad-CAM
        cam = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=img.unsqueeze(0).to(DEVICE), targets=None)
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay Grad-CAM on image
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        # Plot original with overlay
        axes[0, idx].imshow(visualization)
        axes[0, idx].set_title(f'True: {"Real" if true_label == 0 else "Fake"}\nPred: {"Real" if pred_label == 0 else "Fake"}')
        axes[0, idx].axis('off')
        
        # Plot heatmap
        heatmap = axes[1, idx].imshow(grayscale_cam, cmap='hot')
        axes[1, idx].set_title('Attention Heatmap')
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs("utils/analysis", exist_ok=True)
    plt.savefig("utils/analysis/misclassified_gradcam.png")
    plt.close()
    
    print(f"✅ Analysis saved to utils/analysis/misclassified_gradcam.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze misclassified images with Grad-CAM")
    parser.add_argument("--model", type=str, choices=["mobilenet", "xception", "custom"], 
                        default="custom", help="Model to analyze")
    parser.add_argument("--dataset", type=str, choices=["train", "val", "test-dev"], 
                        default="val", help="Dataset to analyze")
    parser.add_argument("--num_images", type=int, default=8,  # Changed default from 6 to 8
                        help="Number of misclassified images to analyze")
    
    args = parser.parse_args()
    analyze_misclassified(args.model, args.dataset, args.num_images)