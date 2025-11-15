"""
Medical AI Models Package

This package contains all machine learning models for medical diagnosis.
Following SOLID principles and factory pattern for model management.

# Produced by Lucas Technology Service
"""

from .base_model import BaseMedicalModel
from .diagnostic_model import MedicalDiagnosticModel
from .ultra_light_model import UltraLightMedicalModel, create_ultra_light_model
from .model_registry import ModelRegistry


__version__ = "1.0.0"
__author__ = "Medical AI Team"
__description__ = "Medical diagnostic AI models for healthcare applications"


__all__ = [
    "BaseMedicalModel",
    "MedicalDiagnosticModel", 
    "UltraLightMedicalModel",
    "create_ultra_light_model",
    "ModelRegistry",
]


MODEL_FACTORY = {
    "diagnostic": MedicalDiagnosticModel,
    "ultra_light": UltraLightMedicalModel,
    "default": MedicalDiagnosticModel,
}

def create_model(model_type: str = "default", **kwargs):
    """
    Factory function to create models dynamically
    
    Args:
        model_type: Type of model to create ('diagnostic', 'ultra_light', 'default')
        **kwargs: Model-specific parameters
    
    Returns:
        Instance of the requested model
        
    Raises:
        ValueError: If model_type is not supported
    """
    if model_type not in MODEL_FACTORY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_FACTORY.keys())}")
    
    model_class = MODEL_FACTORY[model_type]
    
    
    if model_type == "ultra_light":
        default_params = {
            'input_size': 30,
            'hidden_size': 32, 
            'num_classes': 5
        }
    else:  # diagnostic/default
        default_params = {
            'input_size': 50,
            'hidden_size': 128,
            'num_classes': 10
        }
    
    
    model_params = {**default_params, **kwargs}
    
    return model_class(**model_params)

def get_available_models() -> dict:
    """
    Get information about all available models
    
    Returns:
        Dictionary with model information
    """
    return {
        "diagnostic": {
            "class": MedicalDiagnosticModel,
            "description": "Standard medical diagnostic model with balanced performance",
            "recommended_use": "General medical diagnosis",
            "resource_requirements": "Medium (1-2GB RAM)"
        },
        "ultra_light": {
            "class": UltraLightMedicalModel, 
            "description": "Ultra-lightweight model for low-resource environments",
            "recommended_use": "Cloud deployment with limited resources",
            "resource_requirements": "Low (500MB RAM)"
        }
    }

def load_model_from_checkpoint(model_type: str, checkpoint_path: str, **kwargs):
    """
    Load a model from saved checkpoint
    
    Args:
        model_type: Type of model to load
        checkpoint_path: Path to model checkpoint file
        **kwargs: Additional model parameters
    
    Returns:
        Loaded and initialized model instance
    """
    import torch
    
    model = create_model(model_type, **kwargs)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
        
    except Exception as e:
        raise ValueError(f"Failed to load model from {checkpoint_path}: {str(e)}")


_model_registry = ModelRegistry()

def get_model_registry() -> ModelRegistry:
    """
    Get the global model registry instance
    
    Returns:
        ModelRegistry instance for managing model instances
    """
    return _model_registry


def get_all_supported_symptoms() -> list:
    """
    Get union of all symptoms supported by available models
    
    Returns:
        List of all unique symptoms supported by any model
    """
    all_symptoms = set()
    
    for model_info in get_available_models().values():
        model_class = model_info["class"]
        try:
            
            temp_model = model_class(input_size=1, hidden_size=1, num_classes=1)
            symptoms = temp_model.get_supported_symptoms()
            all_symptoms.update(symptoms)
        except:
            continue
    
    return sorted(list(all_symptoms))

print(f"Medical AI Models v{__version__} initialized")
print(f"Available models: {list(get_available_models().keys())}")