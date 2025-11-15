from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, List

class BaseMedicalModel(ABC):
    """Abstract base class for all medical AI models following SOLID principles"""
    
    @abstractmethod
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction based on input symptoms"""
        pass
    
    @abstractmethod
    def preprocess(self, input_data: Dict[str, Any]) -> torch.Tensor:
        """Convert input data to tensor format"""
        pass
    
    @abstractmethod
    def postprocess(self, predictions: torch.Tensor) -> Dict[str, Any]:
        """Convert model output to medical diagnosis format"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata and version information"""
        pass
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data structure"""
        required_keys = ['symptoms']
        return all(key in input_data for key in required_keys)
    
    def get_supported_symptoms(self) -> List[str]:
        """Return list of symptoms the model can process"""
        return list(getattr(self, 'symptom_mapping', {}).keys())
    
    def get_supported_conditions(self) -> List[str]:
        """Return list of medical conditions the model can diagnose"""
        return list(getattr(self, 'condition_mapping', {}).values())