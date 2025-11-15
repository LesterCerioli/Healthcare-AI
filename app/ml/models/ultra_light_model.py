from app.ml.models.base_model import BaseMedicalModel
import torch
import torch.nn as nn
from typing import Dict, Any, List
import json
import os


class UltraLightMedicalModel(nn.Module, BaseMedicalModel):
    """
    Ultra-lightweight medical diagnostic model optimized for low-resource environments
    Suitable for 500MB RAM cloud deployment
    """
    
    def __init__(self, input_size: int = 30, hidden_size: int = 32, num_classes: int = 5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Reduced dropout for smaller model
                
        self.symptom_mapping = {}
        self.condition_mapping = {}
        self._initialize_default_mappings()
    
    def _initialize_default_mappings(self):
        """Initialize with default medical mappings"""
        self.symptom_mapping = {
            "fever": 0, "headache": 1, "cough": 2, "fatigue": 3, "nausea": 4,
            "chest_pain": 5, "shortness_breath": 6, "sore_throat": 7, "runny_nose": 8,
            "muscle_pain": 9, "chills": 10, "loss_of_taste": 11, "loss_of_smell": 12
        }
        
        self.condition_mapping = {
            0: "Common Cold",
            1: "Influenza", 
            2: "COVID-19",
            3: "Pneumonia",
            4: "Bronchitis"
        }
        
        
        self.recommendations_db = {
            "Common Cold": [
                "Rest and hydration",
                "Over-the-counter cold medicine",
                "Consult doctor if symptoms worsen"
            ],
            "Influenza": [
                "Antiviral medication if early",
                "Rest and fluids", 
                "Medical consultation recommended"
            ],
            "COVID-19": [
                "Isolation recommended",
                "Medical consultation",
                "Symptom monitoring",
                "Follow local health guidelines"
            ],
            "Pneumonia": [
                "Antibiotics if bacterial",
                "Hospitalization if severe",
                "Immediate medical attention"
            ],
            "Bronchitis": [
                "Bronchodilators if needed",
                "Cough medicine",
                "Rest and avoid irritants"
            ]
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with minimal operations"""
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.layer2(x)
        return x
    
    def preprocess(self, input_data: Dict[str, Any]) -> torch.Tensor:
        """Convert symptoms dictionary to feature tensor"""
        if not self.validate_input(input_data):
            raise ValueError("Invalid input data structure")
        
        symptoms = input_data.get("symptoms", {})
        feature_vector = [0.0] * self.input_size
        
        for symptom, severity in symptoms.items():
            if symptom in self.symptom_mapping:
                idx = self.symptom_mapping[symptom]
                if idx < self.input_size:  # Safety check
                    if isinstance(severity, (int, float)):
                        
                        feature_vector[idx] = min(max(float(severity), 0.0), 1.0)
                    else:
                        
                        feature_vector[idx] = 1.0 if severity else 0.0
        
        return torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
    
    def postprocess(self, predictions: torch.Tensor) -> Dict[str, Any]:
        """Convert model output to medical diagnosis format"""
        probabilities = torch.softmax(predictions, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        
        diagnosis = self.condition_mapping.get(predicted_class.item(), "Unknown Condition")
                
        top_probs, top_classes = torch.topk(probabilities, min(3, self.num_classes))
        
        possible_conditions = []
        for i in range(top_classes.shape[1]):
            condition_idx = top_classes[0][i].item()
            prob = top_probs[0][i].item()
            condition_name = self.condition_mapping.get(condition_idx, "Unknown")
            possible_conditions.append({
                "condition": condition_name,
                "probability": round(prob, 4)
            })
        
        return {
            "diagnosis": diagnosis,
            "confidence": round(confidence.item(), 4),
            "possible_conditions": possible_conditions,
            "recommendations": self._get_recommendations(diagnosis),
            "urgency_level": self._assess_urgency(diagnosis, confidence.item())
        }
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Complete prediction pipeline"""
        self.eval()
        
        if not self.validate_input(input_data):
            return {
                "error": "Invalid input format",
                "required_format": {"symptoms": {"symptom_name": "severity"}}
            }
        
        try:
            with torch.no_grad():
                processed_input = self.preprocess(input_data)
                predictions = self.forward(processed_input)
                result = self.postprocess(predictions)
                result["success"] = True
                return result
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Prediction failed: {str(e)}"
            }
    
    def _get_recommendations(self, diagnosis: str) -> List[str]:
        """Get medical recommendations for diagnosis"""
        return self.recommendations_db.get(diagnosis, ["Consult a healthcare professional for proper diagnosis"])
    
    def _assess_urgency(self, diagnosis: str, confidence: float) -> str:
        """Assess urgency level based on diagnosis and confidence"""
        urgent_conditions = ["Pneumonia", "COVID-19"]
        semi_urgent = ["Influenza"]
        
        if diagnosis in urgent_conditions and confidence > 0.7:
            return "High - Seek immediate medical attention"
        elif diagnosis in semi_urgent and confidence > 0.6:
            return "Medium - Consult doctor within 24 hours"
        else:
            return "Low - Self-care and monitor symptoms"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata"""
        return {
            "model_type": "UltraLightMedicalModel",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_classes": self.num_classes,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "supported_symptoms": self.get_supported_symptoms(),
            "supported_conditions": self.get_supported_conditions(),
            "optimized_for": "Low-memory environments (500MB RAM)",
            "version": "1.0.0"
        }
    
    def load_mappings(self, filepath: str):
        """Load symptom and condition mappings from file"""
        try:
            with open(filepath, 'r') as f:
                mappings = json.load(f)
                self.symptom_mapping = mappings.get('symptom_mapping', {})
                self.condition_mapping = mappings.get('condition_mapping', {})
        except Exception as e:
            print(f"Warning: Could not load mappings from {filepath}: {e}")
    
    def save_mappings(self, filepath: str):
        """Save current mappings to file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump({
                    'symptom_mapping': self.symptom_mapping,
                    'condition_mapping': self.condition_mapping
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save mappings to {filepath}: {e}")


def create_ultra_light_model(input_size: int = 30, hidden_size: int = 32, num_classes: int = 5) -> UltraLightMedicalModel:
    """Factory function to create ultra-light medical model"""
    return UltraLightMedicalModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes
    )