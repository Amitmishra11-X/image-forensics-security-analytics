"""
CNN Models for Gesture Recognition
=====================================
Fine-tuned ResNet-50 and VGG-16 for real-time hand gesture classification.
Supports ASL (26 classes) and HaGRID (18 classes) datasets.

Author: Amit Mishra
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ── Gesture Class Labels ─────────────────────────────────────────────────────

ASL_CLASSES = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')  # 26 classes

HAGRID_CLASSES = [
    'call', 'dislike', 'fist', 'four', 'like',
    'mute', 'ok', 'one', 'palm', 'peace',
    'peace_inverted', 'rock', 'stop', 'stop_inverted',
    'three', 'three2', 'two_up', 'two_up_inverted'
]  # 18 classes


# ── ResNet-50 Gesture Model ───────────────────────────────────────────────────

class GestureResNet50(nn.Module):
    """
    Fine-tuned ResNet-50 for real-time gesture recognition.
    """

    def __init__(self, num_classes=26, pretrained=True, freeze_backbone=False,
                 dropout_rate=0.5):
        super(GestureResNet50, self).__init__()

        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'layer4' not in name:  # Keep layer4 trainable
                    param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.6),
            nn.Linear(512, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x):
        return self.backbone(x)

    def predict_with_confidence(self, x):
        """Returns predicted class and confidence scores."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
        return predicted, confidence, probs


# ── VGG-16 Gesture Model ──────────────────────────────────────────────────────

class GestureVGG16(nn.Module):
    """
    Fine-tuned VGG-16 for gesture recognition.
    """

    def __init__(self, num_classes=26, pretrained=True, freeze_backbone=False):
        super(GestureVGG16, self).__init__()

        self.backbone = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        )

        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        self.backbone.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x):
        return self.backbone(x)


# ── MobileNetV3 (Lightweight for Real-time) ───────────────────────────────────

class GestureMobileNet(nn.Module):
    """
    Lightweight MobileNetV3 for fast real-time inference.
    Best for deployment on edge devices or webcam applications.
    """

    def __init__(self, num_classes=26, pretrained=True):
        super(GestureMobileNet, self).__init__()

        self.backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )

        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        return self.backbone(x)


# ── Factory Function ──────────────────────────────────────────────────────────

def get_gesture_model(model_name='resnet50', dataset='asl',
                      pretrained=True, freeze_backbone=False):
    """
    Factory to get gesture recognition model.

    Args:
        model_name (str): 'resnet50', 'vgg16', 'mobilenet'
        dataset (str): 'asl' (26 classes) or 'hagrid' (18 classes)
        pretrained (bool): Use ImageNet pretrained weights
        freeze_backbone (bool): Freeze backbone layers

    Returns:
        model, class_names, num_classes
    """
    # Set number of classes based on dataset
    if dataset.lower() == 'asl':
        num_classes = len(ASL_CLASSES)
        class_names = ASL_CLASSES
    elif dataset.lower() == 'hagrid':
        num_classes = len(HAGRID_CLASSES)
        class_names = HAGRID_CLASSES
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose 'asl' or 'hagrid'")

    model_name = model_name.lower()

    if model_name == 'resnet50':
        model = GestureResNet50(num_classes, pretrained, freeze_backbone)
    elif model_name == 'vgg16':
        model = GestureVGG16(num_classes, pretrained, freeze_backbone)
    elif model_name == 'mobilenet':
        model = GestureMobileNet(num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[GestureModel] Configuration:")
    print(f"  Architecture:  {model_name.upper()}")
    print(f"  Dataset:       {dataset.upper()} ({num_classes} classes)")
    print(f"  Pretrained:    {pretrained}")
    print(f"  Frozen:        {freeze_backbone}")
    print(f"  Parameters:    {total:,} total | {trainable:,} trainable")

    return model, class_names, num_classes


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dummy = torch.randn(4, 3, 224, 224).to(device)

    for model_name in ['resnet50', 'vgg16', 'mobilenet']:
        for dataset in ['asl', 'hagrid']:
            model, class_names, num_classes = get_gesture_model(
                model_name, dataset, pretrained=False
            )
            model = model.to(device)
            out = model(dummy)
            assert out.shape == (4, num_classes)
            print(f"  ✓ {model_name} + {dataset}: output {out.shape}")

    print("\n[GestureModel] All tests passed!")
