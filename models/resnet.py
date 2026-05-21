import time
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace the final layer: ResNet18's fc is Linear(512, 1000) → we need Linear(512, 1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def get_dataloaders(path, batch_size=32):
    train_transform, eval_transform = get_transforms()

    train_dataset = datasets.ImageFolder(f"{path}/train", transform=train_transform)
    eval_dataset  = datasets.ImageFolder(f"{path}/eval",  transform=eval_transform)
    test_dataset  = datasets.ImageFolder(f"{path}/test",  transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    eval_loader  = DataLoader(eval_dataset,  batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2)

    train_subset = Subset(train_dataset, range(min(10000, len(train_dataset))))
    train_subset_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, eval_loader, test_loader, train_subset_loader


def binary_accuracy(y_prob, y_true, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return (y_pred == y_true.astype(int)).mean()


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    num_batches = 0
    device = next(model.parameters()).device

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0


def compute_eval_loss(model, loader, criterion):
    model.eval()
    total_loss = 0
    num_batches = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            total_loss += criterion(model(images), labels).item()
            num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0


def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = torch.sigmoid(model(images))
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)


def run(path, device, num_epochs=10, lr=1e-4):
    train_loader, eval_loader, test_loader, train_subset_loader = get_dataloaders(path)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    train_losses = []
    eval_losses = []

    print(f"-- ResNet18 | Epochs: {num_epochs} | LR: {lr}")
    t0 = time.time()
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        eval_loss = compute_eval_loss(model, eval_loader, criterion)
        scheduler.step(eval_loss)
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")

    print(f"Training time: {time.time() - t0:.3f}s")

    train_preds, train_labels = evaluate(model, train_subset_loader)
    test_preds, test_labels = evaluate(model, test_loader)

    score_train = binary_accuracy(train_preds, train_labels)
    score_test = binary_accuracy(test_preds, test_labels)
    print(f"Score Train: {score_train:.3f} | Score Test: {score_test:.3f}")

    return model, train_losses, eval_losses, test_preds, test_labels
