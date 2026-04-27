# mlp.py
import torch
import torch.nn as nn
import numpy as np


"""
A simple MLP for binary classification of chest X-ray images.
The MLP consists of:
- Input layer: 150528 neurons (for 224x224 RGB images)
- Hidden layer 1: 512 neurons, ReLU activation, Dropout(0.4)
- Hidden layer 2: 128 neurons, ReLU activation, Dropout(0.3)
- Output layer: 1 neuron, Sigmoid activation (for binary classification)
"""
class MLP(nn.Module):
    def __init__(self, input_size=224*224*3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            # No Sigmoid here — use BCEWithLogitsLoss during training for numerical stability.
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        return self.net(x)
    
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    num_batches = 0

    for images, labels in loader:
        images, labels = images.to(model.net[0].weight.device), labels.to(model.net[0].weight.device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(model.net[0].weight.device), labels.to(model.net[0].weight.device).float().unsqueeze(1)
            outputs = torch.sigmoid(model(images))
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)