import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_IF import get_image_dataloaders       # For IF models
from data_IFOF_updated import get_dataloaders, get_class_weights  # For IFOF models
from model_compare import (ResNetModel_IF, EfficientNetModel_IF, DenseNetModel_IF, MobileNetModel_IF,
                          ResNetModel_IFOF, EfficientNetModel_IFOF, DenseNetModel_IFOF, MobileNetModel_IFOF)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set directories and batch size
image_dir = os.path.join(BASE_DIR, 'WACV data')
feature_dir = os.path.join(BASE_DIR, 'WACV data')
batch_size = 32  # Change as needed

# Load validation and test dataloaders (504 validation images, 252 test images)
_, val_loader_IF, test_loader_IF = get_image_dataloaders(image_dir, batch_size)
_, val_loader_IFOF, test_loader_IFOF = get_dataloaders(image_dir, feature_dir, batch_size)

# Define evaluation functions for IF models
def evaluate_IF(data_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    return avg_loss, all_labels, all_probs

# Define evaluation functions for IFOF models
def evaluate_IFOF(data_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, features, labels in data_loader:
            images, features, labels = images.to(device), features.to(device), labels.to(device)
            outputs = model(images.unsqueeze(1), features.unsqueeze(1))
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    return avg_loss, all_labels, all_probs

# Helper functions to compute metrics for IF and IFOF models
def compute_metrics_IF(data_loader, model, criterion, device):
    loss, true_labels, probs = evaluate_IF(data_loader, model, criterion, device)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(true_labels, preds)
    prec = precision_score(true_labels, preds, average='weighted', zero_division=0)
    rec = recall_score(true_labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, preds, average='weighted')
    print(f"Evaluated {len(true_labels)} samples -> Accuracy: {acc*100:.2f}%, F1-score: {f1:.4f}")
    return {
        "loss": loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

def compute_metrics_IFOF(data_loader, model, criterion, device):
    loss, true_labels, probs = evaluate_IFOF(data_loader, model, criterion, device)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(true_labels, preds)
    prec = precision_score(true_labels, preds, average='weighted', zero_division=0)
    rec = recall_score(true_labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, preds, average='weighted')
    print(f"Evaluated {len(true_labels)} samples -> Accuracy: {acc*100:.2f}%, F1-score: {f1:.4f}")
    return {
        "loss": loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

# For simplicity, we use dummy class weights (ones) here.
num_classes = 3
dummy_weights = torch.ones(num_classes).float().to(device)
criterion_IF = nn.CrossEntropyLoss(weight=dummy_weights)
criterion_IFOF = nn.CrossEntropyLoss(weight=dummy_weights)

# Define paths for saved models
paths = {
    "ResNet_IF": "hyper-1/best_model_ResNet_IF.pth",
    "EfficientNet_IF": "hyper-1/best_model_EfficientNet_IF.pth",
    "DenseNet_IF": "hyper-1/best_model_DenseNet_IF.pth",
    "MobileNet_IF": "hyper-1/best_model_MobileNet_IF.pth",
    "ResNet_IFOF": "hyper-1/best_model_ResNet_IFOF.pth",
    "EfficientNet_IFOF": "hyper-1/best_model_EfficientNet_IFOF.pth",
    "DenseNet_IFOF": "hyper-1/best_model_DenseNet_IFOF.pth",
    "MobileNet_IFOF": "hyper-1/best_model_MobileNet_IFOF.pth"
}

import model as base_models
import model_compare as compare_models

# Models dictionary mapping name -> (compute_func, data_loader, criterion)
models_to_eval = {
    "ResNet_IF": (compute_metrics_IF, val_loader_IF, criterion_IF),
    "EfficientNet_IF": (compute_metrics_IF, val_loader_IF, criterion_IF),
    "DenseNet_IF": (compute_metrics_IF, val_loader_IF, criterion_IF),
    "MobileNet_IF": (compute_metrics_IF, val_loader_IF, criterion_IF),
    "ResNet_IFOF": (compute_metrics_IFOF, val_loader_IFOF, criterion_IFOF),
    "EfficientNet_IFOF": (compute_metrics_IFOF, val_loader_IFOF, criterion_IFOF),
    "DenseNet_IFOF": (compute_metrics_IFOF, val_loader_IFOF, criterion_IFOF),
    "MobileNet_IFOF": (compute_metrics_IFOF, val_loader_IFOF, criterion_IFOF)
}

# Resolve candidate path locations (local or under DL_models)
metrics = {}
for name, (compute_fn, loader, crit) in models_to_eval.items():
    rel_path = paths[name]
    candidate_paths = [
        os.path.join(SCRIPT_DIR, rel_path),
        os.path.abspath(rel_path)
    ]
    found_path = None
    for cp in candidate_paths:
        if os.path.exists(cp):
            found_path = cp
            break
            
    if found_path:
        print(f"Loading and evaluating {name} from {found_path}...")
        checkpoint_data = torch.load(found_path, map_location=device)
        m = None
        cls_name = f"{name.split('_')[0]}Model_{name.split('_')[1]}"
        
        # Try loading from base_models (model.py) first, then compare_models (model_compare.py)
        for mod_pkg in [base_models, compare_models]:
            if hasattr(mod_pkg, cls_name):
                try:
                    model_cls = getattr(mod_pkg, cls_name)
                    temp_m = model_cls()
                    temp_m.load_state_dict(checkpoint_data)
                    m = temp_m
                    break
                except Exception:
                    continue
                    
        if m is None:
            # Fallback attempt with default parameterizations
            for mod_pkg in [compare_models, base_models]:
                if hasattr(mod_pkg, cls_name):
                    try:
                        model_cls = getattr(mod_pkg, cls_name)
                        temp_m = model_cls(dropout_rate=0.5) if 'dropout_rate' in model_cls.__init__.__code__.co_varnames else model_cls()
                        temp_m.load_state_dict(checkpoint_data)
                        m = temp_m
                        break
                    except Exception:
                        continue

        if m is not None:
            print(f"Loaded checkpoint successfully: Architecture matched for {name}!")
            m.to(device)
            metrics[name] = compute_fn(loader, m, crit, device)
        else:
            print(f"Warning: Could not match checkpoint architecture for {name}.")
    else:
        print(f"Skipping {name}: Checkpoint not found at {rel_path} (train it first to evaluate).")

if not metrics:
    print("No trained checkpoints found to evaluate. Train a model with train_hyper_*.py first.")
    exit(0)

# Convert metrics dictionary to a pandas DataFrame
df_metrics = pd.DataFrame(metrics).T  # Transpose so each row is a model
df_metrics.index.name = "Model"
df_metrics.reset_index(inplace=True)
outputpath = os.path.join(BASE_DIR, "Results", "DL")
os.makedirs(outputpath, exist_ok=True)
# Save the metrics to CSV
csv_filename = os.path.join(outputpath, "evaluation_metrics_valset.csv")
df_metrics.to_csv(csv_filename, index=False)
print(f"\nSaved evaluation metrics to {csv_filename}")
print(df_metrics.to_string(index=False))
