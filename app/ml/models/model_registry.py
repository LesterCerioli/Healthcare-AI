from typing import Dict, Any, Optional
from app.ml.models.diagnostic_model import MedicalDiagnosticModel
import torch
import os

class ModelRegistry:
    """Registry for managing AI model instances and versions"""
    
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self.model_directory = "app/ml/models/saved_models"  # Corrigido o caminho
        os.makedirs(self.model_directory, exist_ok=True)
    
    def register_model(self, model_name: str, model: MedicalDiagnosticModel):
        self._models[model_name] = model
    
    def get_model(self, model_name: str) -> Optional[MedicalDiagnosticModel]:
        return self._models.get(model_name)
    
    def load_model(self, model_name: str, model_path: str) -> MedicalDiagnosticModel:
        """Load a trained model from disk"""
        model = self.get_model(model_name)
        if model is None:
            # Create model instance - precisamos saber a arquitetura
            model = MedicalDiagnosticModel(input_size=50, hidden_size=128, num_classes=10)
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        self.register_model(model_name, model)
        return model
    
    def save_model(self, model_name: str, model: MedicalDiagnosticModel):
        """Save model to disk"""
        model_path = os.path.join(self.model_directory, f"{model_name}.pth")
        torch.save(model.state_dict(), model_path)
        return model_path
    
    def list_models(self) -> list:
        """List all available models"""
        return list(self._models.keys())