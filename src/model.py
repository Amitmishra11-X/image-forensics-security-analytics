"""
CNN Models for Image Forensics
================================
Fine-tuned ResNet-50 and VGG-16 for detecting image tampering.
Includes frozen-feature and full fine-tuning strategies.

Author: Amit Mishra
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


class ForensicsResNet50(nn.Module):
    """
    Fine-tuned ResNet-50 for image forensics classification.
    Detects: Authentic vs Manipulated (Spliced / Copy-Move / Morphed).
    """

    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes (int): Number of output classes (default: 2)
            pretrained (bool): Use ImageNet pretrained weights
            freeze_backbone (bool): Freeze all layers except classifier
        """
        super(ForensicsResNet50, self).__init__()

        # Load pretrained ResNet-50
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Optionally freeze backbone for feature extraction mode
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace final classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x):
        return self.backbone(x)

    def get_feature_maps(self, x):
        """Extract feature maps from the last convolutional layer (for Grad-CAM)."""
        feature_maps = None
        gradients = None

        def hook_forward(module, input, output):
            nonlocal feature_maps
            feature_maps = output

        def hook_backward(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0]

        # Register hooks on last conv layer
        handle_f = self.backbone.layer4[-1].conv3.register_forward_hook(hook_forward)
        handle_b = self.backbone.layer4[-1].conv3.register_backward_hook(hook_backward)

        output = self.forward(x)

        handle_f.remove()
        handle_b.remove()

        return output, feature_maps, gradients


class ForensicsVGG16(nn.Module):
    """
    Fine-tuned VGG-16 for image forensics classification.
    """

    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Use ImageNet pretrained weights
            freeze_backbone (bool): Freeze convolutional layers
        """
        super(ForensicsVGG16, self).__init__()

        # Load pretrained VGG-16
        self.backbone = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Freeze feature layers if required
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x):
        return self.backbone(x)


class CustomCNN(nn.Module):
    """
    Custom CNN baseline for image forensics.
    Lightweight architecture for comparison with transfer learning models.
    """

    def __init__(self, num_classes=2, input_channels=3):
        super(CustomCNN, self).__init__()

        # Convolutional blocks
        self.conv_block1 = self._make_conv_block(input_channels, 32)
        self.conv_block2 = self._make_conv_block(32, 64)
        self.conv_block3 = self._make_conv_block(64, 128)
        self.conv_block4 = self._make_conv_block(128, 256)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )

    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_model(model_name='resnet50', num_classes=2, pretrained=True,
              freeze_backbone=False):
    """
    Factory function to get model by name.

    Args:
        model_name (str): 'resnet50', 'vgg16', or 'custom'
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained weights
        freeze_backbone (bool): Freeze backbone layers

    Returns:
        model: PyTorch model
    """
    model_name = model_name.lower()

    if model_name == 'resnet50':
        model = ForensicsResNet50(num_classes, pretrained, freeze_backbone)
    elif model_name == 'vgg16':
        model = ForensicsVGG16(num_classes, pretrained, freeze_backbone)
    elif model_name == 'custom':
        model = CustomCNN(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: resnet50, vgg16, custom")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[Model] {model_name.upper()} loaded")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Pretrained:           {pretrained}")
    print(f"  Frozen backbone:      {freeze_backbone}")

    return model


def load_checkpoint(model, checkpoint_path, device='cpu'):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[Model] Loaded checkpoint from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")
    return model, checkpoint


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Test all models
    dummy_input = torch.randn(2, 3, 224, 224).to(device)

    for model_name in ['resnet50', 'vgg16', 'custom']:
        print(f"\n{'='*50}")
        model = get_model(model_name, num_classes=2, pretrained=False).to(device)
        output = model(dummy_input)
        print(f"  Output shape: {output.shape}")
        assert output.shape == (2, 2), "Output shape mismatch!"
        print(f"  ✓ Forward pass successful")

    print("\n[Model] All model tests passed!")
