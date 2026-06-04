import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from app.ml.models.diagnostic_model import MedicalDiagnosticModel
from app.ml.models.model_registry import ModelRegistry
from app.ml.data.preprocessor import MedicalDataPreprocessor

class MedicalModelTrainer:
    """Trainer for medical diagnostic models"""
    
    def __init__(self, model: MedicalDiagnosticModel, device: str = 'cuda'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100.0 * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_accuracy = 100.0 * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        epochs: int = 50,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """Complete training loop"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training on {self.device}...")
        print(f"{'Epoch':^6} | {'Train Loss':^12} | {'Val Loss':^12} | {'Train Acc':^10} | {'Val Acc':^10}")
        print("-" * 70)
        
        for epoch in range(epochs):
            
            train_loss, train_acc = self.train_epoch(train_loader)
                        
            val_loss, val_acc = self.validate_epoch(val_loader)
                        
            self.scheduler.step()
                        
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
                        
            print(f"{epoch+1:^6} | {train_loss:^12.4f} | {val_loss:^12.4f} | {train_acc:^10.2f} | {val_acc:^10.2f}")
            
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                self.save_checkpoint(epoch, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        os.makedirs('ml/models/checkpoints', exist_ok=True)
        
        if is_best:
            torch.save(checkpoint, 'ml/models/checkpoints/best_model.pth')
        else:
            torch.save(checkpoint, f'ml/models/checkpoints/checkpoint_epoch_{epoch}.pth')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        
        return checkpoint['epoch']
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
                
        ax2.plot(self.train_accuracies, label='Training Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def train_medical_model(
    model_name: str = "diagnostic_v1",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> MedicalDiagnosticModel:
    """Complete training pipeline for medical diagnostic model"""
        
    preprocessor = MedicalDataPreprocessor()
        
    train_loader, val_loader = preprocessor.get_data_loaders(batch_size=batch_size)
    
    
    model = MedicalDiagnosticModel(
        input_size=preprocessor.feature_size,
        hidden_size=128,
        num_classes=len(preprocessor.condition_mapping)
    )
    
    
    model.symptom_mapping = preprocessor.symptom_mapping
    model.condition_mapping = {v: k for k, v in preprocessor.condition_mapping.items()}
        
    trainer = MedicalModelTrainer(model)
    trainer.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    metrics = trainer.train(train_loader, val_loader, epochs=epochs)
        
    plots_dir = "ml/models/training_plots"
    os.makedirs(plots_dir, exist_ok=True)
    trainer.plot_training_history(f"{plots_dir}/{model_name}_training.png")
    
    
    model_registry = ModelRegistry()
    model_path = model_registry.save_model(model_name, model)
    preprocessor.save_mappings(f"ml/models/{model_name}_mappings.json")
        
    save_training_report(model_name, metrics, model_path)
    
    print(f"Training completed! Model saved to: {model_path}")
    return model

def save_training_report(model_name: str, metrics: Dict, model_path: str):
    """Save training report"""
    report = {
        'model_name': model_name,
        'training_date': datetime.now().isoformat(),
        'final_metrics': {
            'final_train_loss': metrics['train_losses'][-1],
            'final_val_loss': metrics['val_losses'][-1],
            'final_train_accuracy': metrics['train_accuracies'][-1],
            'final_val_accuracy': metrics['val_accuracies'][-1],
            'best_val_accuracy': max(metrics['val_accuracies'])
        },
        'model_path': model_path,
        'total_epochs': len(metrics['train_losses'])
    }
    
    os.makedirs('ml/models/reports', exist_ok=True)
    with open(f'ml/models/reports/{model_name}_report.json', 'w') as f:
        json.dump(report, f, indent=2)