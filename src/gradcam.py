"""
Grad-CAM: Gradient-weighted Class Activation Mapping
======================================================
Produces interpretable heatmaps showing which image regions
the CNN focused on when making its forensics decision.

Author: Amit Mishra
Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms


class GradCAM:
    """
    Grad-CAM implementation for CNN forensics models.

    Usage:
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam.generate(image_tensor, target_class)
        overlay = gradcam.overlay_heatmap(original_image, heatmap)
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch CNN model
            target_layer: Target convolutional layer for gradient extraction
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.feature_maps = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""

        def forward_hook(module, input, output):
            self.feature_maps = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, image_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for a given image.

        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor (1, C, H, W)
            target_class (int): Target class index (None = predicted class)

        Returns:
            heatmap (np.ndarray): Normalised heatmap (H, W) in range [0, 1]
        """
        self.model.eval()
        image_tensor.requires_grad = True

        # Forward pass
        output = self.model(image_tensor)

        # Use predicted class if target_class not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients and backpropagate
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Compute Grad-CAM
        # Global Average Pooling of gradients
        weights = torch.mean(self.gradients, dim=[2, 3])  # (1, C)

        # Weighted combination of feature maps
        cam = torch.zeros(self.feature_maps.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * self.feature_maps[0, i]

        # Apply ReLU — only positive influences matter
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam.numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, target_class

    def overlay_heatmap(self, original_image, heatmap, alpha=0.5,
                        colormap=cv2.COLORMAP_JET):
        """
        Overlay Grad-CAM heatmap on original image.

        Args:
            original_image (np.ndarray): Original RGB image
            heatmap (np.ndarray): Grad-CAM heatmap (H, W)
            alpha (float): Transparency of heatmap overlay
            colormap: OpenCV colormap

        Returns:
            overlay (np.ndarray): RGB image with heatmap overlay
        """
        # Resize heatmap to match image size
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Convert to uint8 and apply colormap
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Overlay on original image
        if original_image.dtype != np.uint8:
            original_image = np.uint8(original_image)

        overlay = cv2.addWeighted(original_image, 1 - alpha,
                                  heatmap_colored, alpha, 0)
        return overlay

    def visualise(self, image_path, target_class=None, save_path=None,
                  class_names=None):
        """
        Full Grad-CAM visualisation pipeline.

        Args:
            image_path (str): Path to input image
            target_class (int): Target class (None = predicted)
            save_path (str): Optional path to save visualisation
            class_names (list): Class name labels

        Returns:
            fig: Matplotlib figure
        """
        if class_names is None:
            class_names = ['Authentic', 'Manipulated']

        # Load and preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        pil_image = Image.open(image_path).convert('RGB')
        original_np = np.array(pil_image.resize((224, 224)))
        image_tensor = transform(pil_image).unsqueeze(0)

        # Generate Grad-CAM
        heatmap, predicted_class = self.generate(image_tensor, target_class)
        overlay = self.overlay_heatmap(original_np, heatmap)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_np)
        axes[0].set_title('Original Image', fontsize=13, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=13, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(axes[1].imshow(heatmap, cmap='jet'), ax=axes[1],
                     fraction=0.046, pad=0.04)

        axes[2].imshow(overlay)
        pred_label = class_names[predicted_class] if predicted_class < len(class_names) else str(predicted_class)
        axes[2].set_title(f'Overlay — Predicted: {pred_label}',
                          fontsize=13, fontweight='bold')
        axes[2].axis('off')

        plt.suptitle('Grad-CAM Forensic Explainability', fontsize=15,
                     fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"[GradCAM] Saved to: {save_path}")

        plt.show()
        return fig


def get_target_layer(model, model_name='resnet50'):
    """
    Get the appropriate target layer for Grad-CAM.

    Args:
        model: PyTorch model
        model_name (str): Model architecture name

    Returns:
        target_layer: PyTorch layer
    """
    model_name = model_name.lower()

    if model_name == 'resnet50':
        return model.backbone.layer4[-1].conv3
    elif model_name == 'vgg16':
        return model.backbone.features[-1]
    elif model_name == 'custom':
        return model.conv_block4[-3]  # Last conv before dropout
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    import sys
    from model import get_model

    print("[GradCAM] Running demo...")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model('resnet50', num_classes=2, pretrained=False).to(device)
    model.eval()

    # Setup Grad-CAM
    target_layer = get_target_layer(model, 'resnet50')
    gradcam = GradCAM(model, target_layer)

    # Test with dummy tensor
    dummy = torch.randn(1, 3, 224, 224)
    heatmap, pred_class = gradcam.generate(dummy)

    print(f"  Heatmap shape: {heatmap.shape}")
    print(f"  Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    print(f"  Predicted class: {pred_class}")
    print("[GradCAM] Demo complete!")

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        gradcam.visualise(image_path, save_path="results/gradcam_output.png")
