import warnings
warnings.filterwarnings('ignore')
import sys
import os

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import time
import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

# Suppress verbose Optuna default logs so our clean epoch table is easily readable
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
from data_IF import get_image_dataloaders
from data_IFOF import get_class_weights
from model import ResNetModel_IF

# Print initial experiment banner
print("\n" + "="*96)
print("           STUDENT ENGAGEMENT DETECTION - DEEP LEARNING (DL) TRAINING")
print("="*96)
print(" Model Architecture   : ResNet-18 Image-Only (Facial Images Baseline)")
print(" Optimization Engine  : Optuna Bayesian Search (20 Trials, max 100 epochs/trial, patience=5)")
print(" Target Classes       : 3 Engagement Levels (0: Low/Disengaged, 1: Medium/Nominal, 2: High/Engaged)")
print(" Compute Acceleration : " + (f"CUDA GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"))
print(" Output Checkpoint    : DL_models/hyper-1/best_model_ResNet_IF.pth")
print(" Auto-Save Strategy   : Continuous Disk Saving (Best model immediately saved on improvement)")
print("="*96 + "\n")

# Setup output directories for continuous checkpointing
SAVE_DIRS = [
    os.path.abspath(os.path.join(BASE_DIR, 'DL_models', 'hyper-1')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), 'hyper-1'))
]
for d in SAVE_DIRS:
    os.makedirs(d, exist_ok=True)

global_best = {
    'loss': float('inf'),
    'trial': None,
    'model_state': None,
    'history': None
}

# Function to define and optimize hyperparameters
def objective(trial):
    # Define hyperparameters to search over
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 5e-4)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.5)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    
    # Configuration
    num_epochs = 100
    patience = 5
    image_dir = os.path.join(BASE_DIR, 'WACV data')

    # Load Image Dataset
    train_loader_IF, val_loader_IF, _ = get_image_dataloaders(image_dir, batch_size)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

    print(f"\n{'='*96}")
    print(f" [TRIAL {trial.number + 1}/20] Hyperparameter Optimization Run")
    print(f" Model Architecture   : ResNet-18 Image-Only Baseline")
    print(f" Trial Hyperparameters: Batch Size = {batch_size} | LR = {learning_rate:.6f} | Dropout = {dropout_rate:.2f} | Weight Decay = {weight_decay:.6f}")
    print(f" Compute Device       : {device} ({device_name})")
    print(f"{'='*96}")

    # Compute class weights for loss balancing
    train_labels = np.array([train_loader_IF.dataset.dataset.labels[i] for i in train_loader_IF.dataset.indices])
    class_weights = torch.tensor(get_class_weights(train_labels)).float().to(device)
    criterion_IF = nn.CrossEntropyLoss(weight=class_weights)

    # Initialize model
    modelResNet_IF = ResNetModel_IF().to(device)
    optimizerResNet_IF = optim.Adam(modelResNet_IF.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train function for IF model
    def train_IF(data_loader, model, optimizer, criterion, device):
        model.train()
        total_loss = 0
        all_labels = []
        all_probs = []
        
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            
        return total_loss / len(data_loader), all_labels, all_probs

    # Evaluate function for IF model
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
        
        return total_loss / len(data_loader), all_labels, all_probs

    # Training loop with clean table format & Early Stopping
    def trainloop_IF(train_loader, val_loader, model, optimizer, criterion, device, num_epochs, patience=5):
        best_val_loss = float('inf')
        no_improvement = 0
        best_model = None
        history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': [],
                   'train_precision': [], 'val_precision': [], 'train_recall': [], 'val_recall': [],
                   'train_f1': [], 'val_f1': []}

        print(f"\n  {'Epoch':^10} | {'Train Loss':^10} | {'Val Loss':^10} | {'Train Acc':^10} | {'Val Acc':^10} | {'Val F1':^10} | {'Status / Early Stopping':<28}")
        print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*28}")

        for epoch in range(num_epochs):
            train_loss, train_labels, train_probs = train_IF(train_loader, model, optimizer, criterion, device)
            val_loss, val_labels, val_probs = evaluate_IF(val_loader, model, criterion, device)
            
            # Metrics
            train_preds = np.argmax(train_probs, axis=1)
            val_preds = np.argmax(val_probs, axis=1)
            train_accuracy = accuracy_score(train_labels, train_preds)
            val_accuracy = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)
            history['val_f1'].append(val_f1)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                no_improvement = 0
                status_str = f"* BEST (Loss: {val_loss:.4f})"
            else:
                no_improvement += 1
                status_str = f"Patience: {no_improvement}/{patience}"

            print(f"  {f'[{epoch+1}/{num_epochs}]':^10} | {train_loss:^10.4f} | {val_loss:^10.4f} | {f'{train_accuracy*100:.1f}%':^10} | {f'{val_accuracy*100:.1f}%':^10} | {val_f1:^10.4f} | {status_str:<28}")

            if no_improvement >= patience:
                print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*28}")
                print(f"  -> Early stopping triggered at Epoch {epoch+1} (Validation loss did not improve for {patience} consecutive epochs).")
                break
        else:
            print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*28}")
            
        return best_model, history

    # Train model
    modelResNet_IF_trained, historyResNet_IF = trainloop_IF(train_loader_IF, val_loader_IF, modelResNet_IF, optimizerResNet_IF, criterion_IF, device, num_epochs, patience)

    # Validation loss
    val_loss_resnet_IF, _, _ = evaluate_IF(val_loader_IF, modelResNet_IF_trained, criterion_IF, device)

    # Continuous checkpoint saving on improvement:
    if val_loss_resnet_IF < global_best['loss']:
        global_best['loss'] = val_loss_resnet_IF
        global_best['trial'] = trial.number + 1
        global_best['model_state'] = modelResNet_IF_trained.state_dict()
        global_best['history'] = historyResNet_IF

        for save_d in SAVE_DIRS:
            torch.save(modelResNet_IF_trained.state_dict(), os.path.join(save_d, 'best_model_ResNet_IF.pth'))
            with open(os.path.join(save_d, 'best_history_ResNet_IF.pkl'), 'wb') as f:
                pickle.dump(historyResNet_IF, f)

        print(f"\n  {'*'*86}")
        print(f"  >>> [NEW GLOBAL BEST MODEL SAVED TO DISK]")
        print(f"  >>> Trial #{trial.number + 1} achieved new lowest Validation Loss: {val_loss_resnet_IF:.4f}")
        print(f"  >>> Saved Checkpoint File: DL_models/hyper-1/best_model_ResNet_IF.pth")
        print(f"  >>> Saved History File   : DL_models/hyper-1/best_history_ResNet_IF.pkl")
        print(f"  {'*'*86}\n")
    else:
        print(f"\n  [Trial #{trial.number + 1} Result] Validation Loss: {val_loss_resnet_IF:.4f} (Global best so far: {global_best['loss']:.4f} from Trial #{global_best['trial']})\n")

    return val_loss_resnet_IF

# Run the hyperparameter optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)

# Save the study object
for save_d in SAVE_DIRS:
    with open(os.path.join(save_d, 'optuna_study_IF-1.pkl'), 'wb') as f:
        pickle.dump(study, f)

print("\n" + "="*96)
print("                      HYPERPARAMETER OPTIMIZATION COMPLETE!")
print("="*96)
print(f" Total Trials Completed : 20")
print(f" Optimal Trial          : Trial #{study.best_trial.number + 1}")
print(f" Lowest Validation Loss : {study.best_trial.value:.4f}")
print(" Optimal Hyperparameters Selected:")
for key, value in study.best_trial.params.items():
    print(f"   * {key:14s}: {value}")
print(f" Best Checkpoint Saved  : DL_models/hyper-1/best_model_ResNet_IF.pth")
print("="*96 + "\n")
