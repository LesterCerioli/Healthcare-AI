"""
Medical AI Models Package

Contains all machine learning models for medical diagnosis, including:
  - Legacy MLP-based models (BaseMedicalModel, MedicalDiagnosticModel, UltraLightMedicalModel)
  - Attention-based specialized LLMs (DiabetesLLM, CardiovascularLLM, SymptomAnalysisLLM)

Follows SOLID principles and uses the factory pattern for model management.
"""

from .base_model import BaseMedicalModel
from .llm_base import BaseMedicalLLM
from .diagnostic_model import MedicalDiagnosticModel
from .ultra_light_model import UltraLightMedicalModel, create_ultra_light_model
from .model_registry import ModelRegistry

# Specialized LLMs
from .diabetes_llm import DiabetesLLM, DiabetesFeatureNormalizer, create_diabetes_llm
from .cardiovascular_llm import (
    CardiovascularLLM,
    CardiovascularFeatureNormalizer,
    create_cardiovascular_llm,
)
from .symptom_analysis_llm import SymptomAnalysisLLM, create_symptom_analysis_llm


__version__ = "2.0.0"
__author__ = "Medical AI Team"
__description__ = (
    "Medical diagnostic AI models: MLP-based classifiers and "
    "attention-based specialized LLMs for healthcare applications"
)

__all__ = [
    # Base classes
    "BaseMedicalModel",
    "BaseMedicalLLM",
    # Legacy MLP models
    "MedicalDiagnosticModel",
    "UltraLightMedicalModel",
    "create_ultra_light_model",
    # Specialized LLMs
    "DiabetesLLM",
    "DiabetesFeatureNormalizer",
    "create_diabetes_llm",
    "CardiovascularLLM",
    "CardiovascularFeatureNormalizer",
    "create_cardiovascular_llm",
    "SymptomAnalysisLLM",
    "create_symptom_analysis_llm",
    # Registry and factory helpers
    "ModelRegistry",
    "MODEL_FACTORY",
    "create_model",
    "get_available_models",
    "load_model_from_checkpoint",
    "get_model_registry",
    "get_all_supported_symptoms",
]


_LLM_TYPES = {"diabetes", "cardiovascular", "symptom_analysis"}

MODEL_FACTORY = {
    "diagnostic": MedicalDiagnosticModel,
    "ultra_light": UltraLightMedicalModel,
    "diabetes": DiabetesLLM,
    "cardiovascular": CardiovascularLLM,
    "symptom_analysis": SymptomAnalysisLLM,
    "default": MedicalDiagnosticModel,
}

_DEFAULT_PARAMS = {
    "diagnostic": {"input_size": 50, "hidden_size": 128, "num_classes": 10},
    "ultra_light": {"input_size": 30, "hidden_size": 32, "num_classes": 5},
    "default": {"input_size": 50, "hidden_size": 128, "num_classes": 10},
}


def create_model(model_type: str = "default", **kwargs):
    """
    Factory function to create any registered model type.

    LLM models (diabetes, cardiovascular, symptom_analysis) are self-configuring
    and ignore **kwargs. MLP models accept input_size, hidden_size, num_classes.

    Args:
        model_type: One of 'diagnostic', 'ultra_light', 'diabetes',
                    'cardiovascular', 'symptom_analysis', or 'default'.
        **kwargs:   Forwarded to MLP-based constructors only.

    Returns:
        Instantiated model ready for inference or training.
    """
    if model_type not in MODEL_FACTORY:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available: {list(MODEL_FACTORY.keys())}"
        )

    model_class = MODEL_FACTORY[model_type]

    if model_type in _LLM_TYPES:
        return model_class()

    params = {**_DEFAULT_PARAMS.get(model_type, {}), **kwargs}
    return model_class(**params)


def get_available_models() -> dict:
    """Return metadata for all available model types."""
    return {
        "diagnostic": {
            "class": MedicalDiagnosticModel,
            "description": "Standard MLP diagnostic model",
            "recommended_use": "General medical diagnosis",
            "resource_requirements": "Medium (1-2GB RAM)",
            "architecture": "Multi-layer Perceptron",
        },
        "ultra_light": {
            "class": UltraLightMedicalModel,
            "description": "Lightweight MLP for low-resource cloud deployment",
            "recommended_use": "Cloud deployment with limited resources",
            "resource_requirements": "Low (500MB RAM)",
            "architecture": "Multi-layer Perceptron",
        },
        "diabetes": {
            "class": DiabetesLLM,
            "description": (
                "Attention-based LLM for Diabetes Mellitus: "
                "Type 1, Type 2, Pre-diabetes, GDM, MODY"
            ),
            "recommended_use": "Diabetes Mellitus diagnosis and clinical support",
            "resource_requirements": "Low (<100MB RAM)",
            "architecture": "Clinical Transformer (multi-head self-attention)",
        },
        "cardiovascular": {
            "class": CardiovascularLLM,
            "description": (
                "Attention-based LLM for cardiovascular diseases: "
                "Hypertension, CAD, CHF, AF, AMI risk, Angina"
            ),
            "recommended_use": "Cardiovascular risk stratification and workup",
            "resource_requirements": "Low (<100MB RAM)",
            "architecture": "Clinical Transformer (multi-head self-attention)",
        },
        "symptom_analysis": {
            "class": SymptomAnalysisLLM,
            "description": (
                "Broad-spectrum symptom triage LLM across 15 disease categories "
                "with routing to specialist models"
            ),
            "recommended_use": "Initial symptom triage and specialist routing",
            "resource_requirements": "Low-Medium (<200MB RAM)",
            "architecture": "Clinical Transformer (multi-head self-attention)",
        },
    }


def load_model_from_checkpoint(model_type: str, checkpoint_path: str, **kwargs):
    """
    Load any registered model type from a saved checkpoint.

    Args:
        model_type:      Model type identifier (see MODEL_FACTORY keys).
        checkpoint_path: Path to the .pth checkpoint file.
        **kwargs:        Forwarded to MLP constructors; ignored for LLM types.

    Returns:
        Loaded and eval()-set model instance.
    """
    import torch

    model = create_model(model_type, **kwargs)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as exc:
        raise ValueError(
            f"Failed to load {model_type} model from '{checkpoint_path}': {exc}"
        ) from exc


_model_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """Return the global ModelRegistry singleton."""
    return _model_registry


def get_all_supported_symptoms() -> list:
    """Return sorted union of all symptoms supported across all model types."""
    all_symptoms: set = set()
    for model_info in get_available_models().values():
        model_class = model_info["class"]
        try:
            if model_class in (DiabetesLLM, CardiovascularLLM, SymptomAnalysisLLM):
                temp = model_class()
            else:
                temp = model_class(input_size=1, hidden_size=1, num_classes=1)
            all_symptoms.update(temp.get_supported_symptoms())
        except Exception:
            continue
    return sorted(all_symptoms)

