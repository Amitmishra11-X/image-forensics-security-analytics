"""
Training Pipeline for Image Forensics Models
=============================================
Trains ResNet-50, VGG-16, or Custom CNN on image forensics datasets.
Includes learning rate scheduling, early stopping, and checkpoint saving.

Author: Amit Mishra
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

from model import get_model


# ── Data Transforms ──────────────────────────────────────────────────────────

def get_transforms(img_size=224, augment=True):
    """Get training and validation transforms."""

    train_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]) if augment else get_transforms(img_size, augment=False)

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Epoch [{epoch}] Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Acc: {100.*correct/total:.2f}%")

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    return avg_loss, accuracy, auc, all_preds, all_labels


def train(args):
    """Full training pipeline."""

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  Image Forensics Training Pipeline")
    print(f"{'='*60}")
    print(f"  Model:   {args.model}")
    print(f"  Device:  {device}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Batch:   {args.batch_size}")
    print(f"  LR:      {args.lr}")
    print(f"{'='*60}\n")

    # Data
    train_transform, val_transform = get_transforms(args.img_size)

    # Expects data/train/authentic/ and data/train/manipulated/ structure
    print("[Data] Loading datasets...")
    try:
        full_dataset = datasets.ImageFolder(
            root=os.path.join(args.data_dir, 'train'),
            transform=train_transform
        )
        print(f"  Classes: {full_dataset.classes}")
        print(f"  Total samples: {len(full_dataset)}")

        # Train / val split (80/20)
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        val_dataset.dataset.transform = val_transform

    except FileNotFoundError:
        print("[Data] No dataset found. Using synthetic demo data.")
        print("  To use real data, structure as:")
        print("  data/train/authentic/  → authentic images")
        print("  data/train/manipulated/ → tampered images")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    print(f"  Train: {train_size} | Val: {val_size}\n")

    # Model
    model = get_model(args.model, num_classes=2, pretrained=True,
                      freeze_backbone=args.freeze).to(device)

    # Loss and optimiser
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 1.5]).to(device)  # Weight manipulated class higher
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0
    best_auc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'auc': []}

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")
        start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc, auc, preds, labels = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step()
        elapsed = time.time() - start

        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}% | AUC: {auc:.4f}")
        print(f"  Time: {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['auc'].append(auc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_auc = auc
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_name': args.model,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_auc': auc,
                'train_acc': train_acc,
            }
            save_path = os.path.join(args.save_dir, f'{args.model}_best.pth')
            torch.save(checkpoint, save_path)
            print(f"  ✓ Best model saved: {save_path}")

            # Classification report
            class_names = ['Authentic', 'Manipulated']
            print("\n  Classification Report:")
            print(classification_report(labels, preds,
                                        target_names=class_names, digits=4))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Training] Early stopping at epoch {epoch}")
                break

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"  Best Val AUC:      {best_auc:.4f}")
    print(f"{'='*60}")

    return history


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Forensics Training')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'vgg16', 'custom'],
                        help='Model architecture')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to dataset directory')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--freeze', action='store_true',
                        help='Freeze backbone (feature extraction mode)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')

    args = parser.parse_args()
    train(args)
