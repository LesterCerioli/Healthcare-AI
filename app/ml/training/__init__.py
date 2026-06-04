"""
Medical AI Training Module

This module provides training utilities and trainers for medical AI models.
Includes both full-featured trainers for WSL and memory-optimized trainers
for cloud deployment.

Produced by Lucas Technology Service
"""

from .trainer import MedicalModelTrainer, train_medical_model, save_training_report
from .optimized_trainer import MemoryOptimizedTrainer, create_optimized_trainer

import torch
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

__version__ = "1.0.0"
__author__ = "Medical AI Team"
__description__ = "Training utilities for medical diagnostic models"

logger = logging.getLogger(__name__)


class TrainingEnvironment(Enum):
    """Available training environments."""
    WSL_FULL = "wsl_full"
    CLOUD_LIGHT = "cloud_light"
    HYBRID = "hybrid"


@dataclass
class TrainingConfig:
    """Configuration for medical model training."""

    model_name: str = "medical_model"
    model_type: str = "diagnostic"

    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10

    environment: TrainingEnvironment = TrainingEnvironment.WSL_FULL
    device: str = "auto"

    min_memory_mb: float = 100
    max_memory_usage: float = 0.8

    save_checkpoints: bool = True
    save_best_only: bool = True
    create_training_plots: bool = True

    def validate(self) -> bool:
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.model_type not in ["diagnostic", "ultra_light"]:
            raise ValueError("model_type must be 'diagnostic' or 'ultra_light'")
        return True

    def get_optimized_settings(self) -> Dict[str, Any]:
        if self.environment == TrainingEnvironment.CLOUD_LIGHT:
            return {
                "batch_size": min(self.batch_size, 8),
                "epochs": min(self.epochs, 20),
                "early_stopping_patience": 5,
                "save_checkpoints": True,
                "save_best_only": True,
            }
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "save_checkpoints": True,
            "save_best_only": False,
        }


@dataclass
class TrainingResult:
    """Result of a training operation."""

    success: bool
    model_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    training_time: Optional[float] = None
    final_accuracy: Optional[float] = None
    error_message: Optional[str] = None
    environment_used: Optional[TrainingEnvironment] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "model_path": self.model_path,
            "training_time": self.training_time,
            "final_accuracy": self.final_accuracy,
            "environment_used": self.environment_used.value if self.environment_used else None,
            "error_message": self.error_message,
        }


def get_trainer_for_environment(
    model: torch.nn.Module,
    config: TrainingConfig,
) -> Union[MedicalModelTrainer, MemoryOptimizedTrainer]:
    """Return the appropriate trainer instance for the given environment."""
    config.validate()

    if config.environment == TrainingEnvironment.CLOUD_LIGHT:
        logger.info("Creating memory-optimized trainer for cloud environment")
        return create_optimized_trainer(
            model=model,
            learning_rate=config.learning_rate,
            device=config.device,
        )

    logger.info("Creating full-featured trainer for WSL/hybrid environment")
    trainer = MedicalModelTrainer(model=model, device=config.device)
    trainer.optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-5,
    )
    return trainer


def get_recommended_config(environment: TrainingEnvironment) -> TrainingConfig:
    """Return a sensible default TrainingConfig for the given environment."""
    if environment == TrainingEnvironment.WSL_FULL:
        return TrainingConfig(
            model_name="medical_model_wsl",
            model_type="diagnostic",
            epochs=100,
            batch_size=64,
            learning_rate=0.001,
            environment=environment,
            device="auto",
        )
    if environment == TrainingEnvironment.CLOUD_LIGHT:
        return TrainingConfig(
            model_name="medical_model_cloud",
            model_type="ultra_light",
            epochs=20,
            batch_size=8,
            learning_rate=0.001,
            environment=environment,
            device="cpu",
            min_memory_mb=100,
        )
    # HYBRID
    return TrainingConfig(
        model_name="medical_model_hybrid",
        model_type="diagnostic",
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        environment=environment,
        device="auto",
    )


TRAINING_PRESETS: Dict[str, Dict[str, Any]] = {
    "full_training": {
        "description": "Full training on WSL with maximum resources",
        "config": get_recommended_config(TrainingEnvironment.WSL_FULL),
    },
    "quick_training": {
        "description": "Quick training for testing",
        "config": TrainingConfig(
            model_name="quick_test",
            epochs=5,
            batch_size=16,
            save_checkpoints=False,
            create_training_plots=False,
        ),
    },
    "cloud_optimized": {
        "description": "Optimized for cloud deployment",
        "config": get_recommended_config(TrainingEnvironment.CLOUD_LIGHT),
    },
}


def get_training_preset(preset_name: str) -> TrainingConfig:
    """Return a predefined TrainingConfig preset by name."""
    if preset_name not in TRAINING_PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. Available: {list(TRAINING_PRESETS.keys())}"
        )
    return TRAINING_PRESETS[preset_name]["config"]


__all__ = [
    "MedicalModelTrainer",
    "MemoryOptimizedTrainer",
    "create_optimized_trainer",
    "train_medical_model",
    "save_training_report",
    "get_trainer_for_environment",
    "TrainingConfig",
    "TrainingResult",
    "TrainingEnvironment",
    "TRAINING_PRESETS",
    "get_training_preset",
    "get_recommended_config",
]
