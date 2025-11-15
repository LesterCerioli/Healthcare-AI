import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import gc
import psutil
import logging
import os
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class MemoryOptimizedTrainer:
    """
    Memory-optimized trainer for low-resource environments
    Designed to work within 500MB RAM constraints
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        
        torch.set_grad_enabled(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []
        
        
        self.memory_warnings = 0
        self.max_memory_warnings = 5
        
        logger.info(f"MemoryOptimizedTrainer initialized on device: {self.device}")
    
    def memory_check(self) -> float:
        """
        Check available memory in MB
        
        Returns:
            Available memory in MB
        """
        memory = psutil.virtual_memory()
        available_mb = memory.available / (1024 * 1024)
        return available_mb
    
    def memory_safe_operation(self, min_memory_mb: float = 50) -> bool:
        """
        Check if there's enough memory for safe operation
        
        Args:
            min_memory_mb: Minimum required memory in MB
            
        Returns:
            True if safe to proceed
        """
        available_mb = self.memory_check()
        
        if available_mb < min_memory_mb:
            logger.warning(f"Low memory: {available_mb:.1f}MB available, {min_memory_mb}MB required")
            self.memory_warnings += 1
            
            if self.memory_warnings >= self.max_memory_warnings:
                logger.error("Too many memory warnings, stopping training")
                return False
            
            
            self._force_memory_cleanup()
            available_mb = self.memory_check()
            
            if available_mb < min_memory_mb:
                logger.warning("Insufficient memory even after cleanup")
                return False
        
        return True
    
    def _force_memory_cleanup(self):
        """Aggressive memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Memory cleanup performed")
    
    def train_epoch_memory_safe(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch with memory safety checks
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        processed_batches = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            
            if not self.memory_safe_operation(min_memory_mb=100):
                logger.warning("Skipping batch due to memory constraints")
                continue
            
            data, targets = data.to(self.device), targets.to(self.device)
            
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            processed_batches += 1
            
            
            if batch_idx % 5 == 0:  # Clean every 5 batches
                self._force_memory_cleanup()
        
        if processed_batches == 0:
            return 0.0, 0.0
            
        epoch_loss = running_loss / processed_batches
        epoch_accuracy = 100.0 * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def validate_light(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Lightweight validation with minimal memory usage
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        if not self.memory_safe_operation(min_memory_mb=150):
            logger.warning("Skipping validation due to memory constraints")
            return 0.0, 0.0
        
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
        
        self.model.train()
        
        val_loss = running_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        return val_loss, val_accuracy
    
    def train_with_memory_limits(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        epochs: int = 10,
        early_stopping_patience: int = 5,
        min_memory_mb: float = 100
    ) -> Dict[str, List[float]]:
        """
        Complete training loop with memory constraints
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            early_stopping_patience: Early stopping patience
            min_memory_mb: Minimum memory required
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Starting memory-optimized training for {epochs} epochs")
        
        best_val_loss = float('inf')
        patience_counter = 0
        skipped_epochs = 0
        
        print(f"{'Epoch':^6} | {'Train Loss':^10} | {'Val Loss':^10} | {'Train Acc':^9} | {'Val Acc':^9} | {'Memory':^8}")
        print("-" * 70)
        
        for epoch in range(epochs):
            
            available_mb = self.memory_check()
            
            if available_mb < min_memory_mb:
                skipped_epochs += 1
                logger.warning(f"Epoch {epoch+1} skipped due to low memory: {available_mb:.1f}MB")
                
                if skipped_epochs >= 3:
                    logger.error("Too many skipped epochs, stopping training")
                    break
                continue
            
            
            skipped_epochs = 0
            
            
            train_loss, train_acc = self.train_epoch_memory_safe(train_loader)
            
            
            val_loss, val_acc = 0.0, 0.0
            if self.memory_safe_operation(min_memory_mb=150):
                val_loss, val_acc = self.validate_light(val_loader)
            else:
                logger.warning("Skipping validation due to memory constraints")
                val_loss, val_acc = train_loss, train_acc  # Use train metrics as proxy
            
            
            self.scheduler.step()
            
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
                        
            current_memory = self.memory_check()
            print(f"{epoch+1:^6} | {train_loss:^10.4f} | {val_loss:^10.4f} | {train_acc:^9.2f} | {val_acc:^9.2f} | {current_memory:^8.1f}")
            
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model checkpoint
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"New best model saved at epoch {epoch+1}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            
            self._force_memory_cleanup()
        
        logger.info("Training completed")
        return self._get_training_metrics()
    
    def _get_training_metrics(self) -> Dict[str, List[float]]:
        """Get all training metrics"""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint with memory optimization
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        if not self.memory_safe_operation(min_memory_mb=200):
            logger.warning("Low memory, skipping checkpoint save")
            return
        
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies,
                'timestamp': datetime.now().isoformat()
            }
            
            os.makedirs('app/ml/models/checkpoints', exist_ok=True)
            
            if is_best:
                checkpoint_path = 'app/ml/models/checkpoints/best_model.pth'
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Best model saved to {checkpoint_path}")
            else:
                checkpoint_path = f'app/ml/models/checkpoints/checkpoint_epoch_{epoch}.pth'
                torch.save(checkpoint, checkpoint_path)
                logger.debug(f"Checkpoint saved to {checkpoint_path}")
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Epoch number from checkpoint
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.train_accuracies = checkpoint.get('train_accuracies', [])
            self.val_accuracies = checkpoint.get('val_accuracies', [])
            
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            return checkpoint['epoch']
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        if not self.train_losses:
            return {"status": "No training data available"}
        
        return {
            "status": "completed",
            "total_epochs": len(self.train_losses),
            "final_train_loss": self.train_losses[-1],
            "final_val_loss": self.val_losses[-1] if self.val_losses else None,
            "final_train_accuracy": self.train_accuracies[-1],
            "final_val_accuracy": self.val_accuracies[-1] if self.val_accuracies else None,
            "best_val_accuracy": max(self.val_accuracies) if self.val_accuracies else None,
            "memory_warnings": self.memory_warnings,
            "device": str(self.device)
        }


def create_optimized_trainer(
    model: nn.Module, 
    learning_rate: float = 0.001,
    device: str = 'auto'
) -> MemoryOptimizedTrainer:
    """
    Factory function to create optimized trainer
    
    Args:
        model: PyTorch model to train
        learning_rate: Learning rate for optimizer
        device: Device to use ('auto', 'cpu', 'cuda')
    
    Returns:
        Configured MemoryOptimizedTrainer instance
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = MemoryOptimizedTrainer(model, device)
    trainer.optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-5
    )
    
    return trainer


if __name__ == "__main__":
    
    from app.ml.models.ultra_light_model import UltraLightMedicalModel
    from app.ml.data.lightweight_dataset import get_lightweight_dataloader
    
    
    model = UltraLightMedicalModel(input_size=30, hidden_size=32, num_classes=5)
        
    trainer = create_optimized_trainer(model, learning_rate=0.001)
    
    
    train_loader, val_loader = get_lightweight_dataloader(batch_size=8)
        
    metrics = trainer.train_with_memory_limits(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        min_memory_mb=100
    )
    
    print("Training completed!")
    print(f"Final metrics: {trainer.get_training_summary()}")