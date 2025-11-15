"""
Medical AI Training Module

This module provides training utilities and trainers for medical AI models.
Includes both full-featured trainers for WSL and memory-optimized trainers for cloud.

Produced by Lucas Technology Service
"""

from .trainer import MedicalModelTrainer, train_medical_model, save_training_report
from .optimized_trainer import MemoryOptimizedTrainer, create_optimized_trainer

# Version info
__version__ = "1.0.0"
__author__ = "Medical AI Team"
__description__ = "Training utilities for medical diagnostic models"

# Available trainers for easy import
__all__ = [
    "MedicalModelTrainer",
    "MemoryOptimizedTrainer", 
    "create_optimized_trainer",
    "train_medical_model",
    "save_training_report",
    "get_trainer_for_environment",
    "TrainingConfig",
    "TrainingResult"
]

# Import training components
import torch
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class TrainingEnvironment(Enum):
    """Available training environments"""
    WSL_FULL = "wsl_full"           # Full power (16GB RAM, i5)
    CLOUD_LIGHT = "cloud_light"     # Limited resources (500MB RAM)
    HYBRID = "hybrid"               # Mixed approach

@dataclass
class TrainingConfig:
    """Configuration for medical model training"""
    
    # Model parameters
    model_name: str = "medical_model"
    model_type: str = "diagnostic"  # "diagnostic" or "ultra_light"
    
    # Training parameters
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    
    # Environment settings
    environment: TrainingEnvironment = TrainingEnvironment.WSL_FULL
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    # Memory constraints (for cloud training)
    min_memory_mb: float = 100
    max_memory_usage: float = 0.8  # 80% of available memory
    
    # Output settings
    save_checkpoints: bool = True
    save_best_only: bool = True
    create_training_plots: bool = True
    
    def validate(self) -> bool:
        """Validate training configuration"""
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.model_type not in ["diagnostic", "ultra_light"]:
            raise ValueError("Model type must be 'diagnostic' or 'ultra_light'")
        
        return True
    
    def get_optimized_settings(self) -> Dict[str, Any]:
        """Get settings optimized for current environment"""
        if self.environment == TrainingEnvironment.CLOUD_LIGHT:
            return {
                "batch_size": min(self.batch_size, 8),
                "epochs": min(self.epochs, 20),
                "early_stopping_patience": 5,
                "save_checkpoints": True,
                "save_best_only": True
            }
        else:  # WSL_FULL or HYBRID
            return {
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "early_stopping_patience": self.early_stopping_patience,
                "save_checkpoints": True,
                "save_best_only": False
            }

@dataclass
class TrainingResult:
    """Result of training operation"""
    
    success: bool
    model_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    training_time: Optional[float] = None  # in seconds
    final_accuracy: Optional[float] = None
    error_message: Optional[str] = None
    environment_used: Optional[TrainingEnvironment] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "success": self.success,
            "model_path": self.model_path,
            "training_time": self.training_time,
            "final_accuracy": self.final_accuracy,
            "environment_used": self.environment_used.value if self.environment_used else None,
            "error_message": self.error_message
        }

def get_trainer_for_environment(
    model: torch.nn.Module,
    config: TrainingConfig
) -> Union[MedicalModelTrainer, MemoryOptimizedTrainer]:
    """
    Factory function to get appropriate trainer for environment
    
    Args:
        model: PyTorch model to train
        config: Training configuration
        
    Returns:
        Appropriate trainer instance
    """
    config.validate()
    
    if config.environment == TrainingEnvironment.CLOUD_LIGHT:
        logger.info("Creating memory-optimized trainer for cloud environment")
        trainer = create_optimized_trainer(
            model=model,
            learning_rate=config.learning_rate,
            device=config.device
        )
    else:
        logger.info("Creating full-featured trainer for WSL environment")
        trainer = MedicalModelTrainer(model=model, device=config.device)
        trainer.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=1e-5
        )
    
    return trainer

def train_medical_model_with_config(
    config: TrainingConfig,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None
) -> TrainingResult:
    """
    High-level training function with configuration
    
    Args:
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        
    Returns:
        Training result with metrics and model path
    """
    import time
    from app.ml.models import create_model
    
    start_time = time.time()
    
    try:
        # Create model based on configuration
        model = create_model(
            model_type=config.model_type,
            input_size=50 if config.model_type == "diagnostic" else 30,
            hidden_size=128 if config.model_type == "diagnostic" else 32,
            num_classes=10 if config.model_type == "diagnostic" else 5
        )
        
        # Get appropriate trainer
        trainer = get_trainer_for_environment(model, config)
        
        # Get optimized settings for environment
        optimized_settings = config.get_optimized_settings()
        
        logger.info(f"Starting training with config: {config}")
        
        # Perform training
        if config.environment == TrainingEnvironment.CLOUD_LIGHT:
            metrics = trainer.train_with_memory_limits(
                train_loader=train_loader,
                val_loader=val_loader or train_loader,  # Fallback to train loader
                epochs=optimized_settings["epochs"],
                early_stopping_patience=optimized_settings["early_stopping_patience"],
                min_memory_mb=config.min_memory_mb
            )
        else:
            metrics = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader or train_loader,
                epochs=optimized_settings["epochs"],
                early_stopping_patience=optimized_settings["early_stopping_patience"]
            )
        
        # Save model
        from app.ml.models.model_registry import ModelRegistry
        registry = ModelRegistry()
        model_path = registry.save_model(config.model_name, model)
        
        # Save training report
        if config.environment != TrainingEnvironment.CLOUD_LIGHT and config.create_training_plots:
            trainer.plot_training_history(f"app/ml/models/training_plots/{config.model_name}_training.png")
        
        training_time = time.time() - start_time
        
        return TrainingResult(
            success=True,
            model_path=model_path,
            metrics=metrics,
            training_time=training_time,
            final_accuracy=metrics.get('val_accuracies', [0])[-1] if metrics.get('val_accuracies') else metrics.get('train_accuracies', [0])[-1],
            environment_used=config.environment
        )
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return TrainingResult(
            success=False,
            error_message=str(e),
            environment_used=config.environment
        )

def get_recommended_config(environment: TrainingEnvironment) -> TrainingConfig:
    """
    Get recommended configuration for specific environment
    
    Args:
        environment: Target training environment
        
    Returns:
        Recommended training configuration
    """
    if environment == TrainingEnvironment.WSL_FULL:
        return TrainingConfig(
            model_name="medical_model_wsl",
            model_type="diagnostic",
            epochs=100,
            batch_size=64,
            learning_rate=0.001,
            environment=environment,
            device="auto"
        )
    elif environment == TrainingEnvironment.CLOUD_LIGHT:
        return TrainingConfig(
            model_name="medical_model_cloud",
            model_type="ultra_light", 
            epochs=20,
            batch_size=8,
            learning_rate=0.001,
            environment=environment,
            device="cpu",
            min_memory_mb=100
        )
    else:  # HYBRID
        return TrainingConfig(
            model_name="medical_model_hybrid",
            model_type="diagnostic",
            epochs=50,
            batch_size=32,
            learning_rate=0.001,
            environment=environment,
            device="auto"
        )

# Utility functions for training management
def list_available_checkpoints() -> list:
    """List all available training checkpoints"""
    import os
    checkpoint_dir = "app/ml/models/checkpoints"
    
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pth'):
            checkpoints.append(os.path.join(checkpoint_dir, file))
    
    return sorted(checkpoints)

def get_training_status(model_name: str) -> Dict[str, Any]:
    """Get training status for a model"""
    import os
    import json
    
    report_path = f"app/ml/models/reports/{model_name}_report.json"
    
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            return json.load(f)
    else:
        return {"status": "no_training_data", "model_name": model_name}

def cleanup_old_checkpoints(keep_last_n: int = 3):
    """Clean up old checkpoints, keeping only the most recent ones"""
    checkpoints = list_available_checkpoints()
    
    if len(checkpoints) > keep_last_n:
        # Keep the most recent checkpoints
        checkpoints_to_delete = checkpoints[:-keep_last_n]
        
        for checkpoint in checkpoints_to_delete:
            try:
                os.remove(checkpoint)
                logger.info(f"Deleted old checkpoint: {checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint {checkpoint}: {e}")

# Training presets for common scenarios
TRAINING_PRESETS = {
    "full_training": {
        "description": "Full training on WSL with maximum resources",
        "config": get_recommended_config(TrainingEnvironment.WSL_FULL)
    },
    "quick_training": {
        "description": "Quick training for testing", 
        "config": TrainingConfig(
            model_name="quick_test",
            epochs=5,
            batch_size=16,
            save_checkpoints=False,
            create_training_plots=False
        )
    },
    "cloud_optimized": {
        "description": "Optimized for cloud deployment",
        "config": get_recommended_config(TrainingEnvironment.CLOUD_LIGHT)
    }
}

def get_training_preset(preset_name: str) -> TrainingConfig:
    """
    Get predefined training configuration preset
    
    Args:
        preset_name: Name of the preset
        
    Returns:
        Training configuration
    """
    if preset_name not in TRAINING_PRESETS:
        available_presets = list(TRAINING_PRESETS.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available_presets}")
    
    return TRAINING_PRESETS[preset_name]["config"]

# Package initialization
logger.info(f"Medical AI Training Module v{__version__} initialized")
logger.info(f"Available trainers: MedicalModelTrainer, MemoryOptimizedTrainer")
logger.info(f"Available presets: {list(TRAINING_PRESETS.keys())}")

# Export training presets
__all__.extend([
    "TRAINING_PRESETS",
    "get_training_preset",
    "TrainingEnvironment"
])