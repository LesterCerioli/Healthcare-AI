import torch
import torch.nn as nn
from typing import Dict, Any, List
import json
import os
from .base_model import BaseMedicalModel

class MedicalDiagnosticModel(nn.Module, BaseMedicalModel):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(MedicalDiagnosticModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
                
        self.symptom_mapping = {}
        self.condition_mapping = {}
        self.load_mappings()
    
    def load_mappings(self):
        """Load mappings from file or use defaults"""
        mappings_path = "app/ml/models/diagnostic_mappings.json"
        if os.path.exists(mappings_path):
            with open(mappings_path, 'r') as f:
                mappings = json.load(f)
                self.symptom_mapping = mappings.get('symptom_mapping', {})
                self.condition_mapping = mappings.get('condition_mapping', {})
        else:
            
            self.symptom_mapping = {
                "fever": 0, "headache": 1, "cough": 2, "fatigue": 3,
                "nausea": 4, "chest_pain": 5, "shortness_breath": 6
            }
            self.condition_mapping = {
                0: "Common Cold", 1: "Influenza", 2: "COVID-19", 
                3: "Pneumonia", 4: "Bronchitis"
            }
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x
    
    def preprocess(self, input_data: Dict[str, Any]) -> torch.Tensor:
        symptoms = input_data.get("symptoms", {})
        feature_vector = [0] * self.input_size
        
        for symptom, severity in symptoms.items():
            if symptom in self.symptom_mapping:
                idx = self.symptom_mapping[symptom]
                if idx < self.input_size:  
                    if isinstance(severity, (int, float)):
                        feature_vector[idx] = min(max(severity, 0), 1)  
                    else:
                        feature_vector[idx] = 1.0
        
        return torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
    
    def postprocess(self, predictions: torch.Tensor) -> Dict[str, Any]:
        probabilities = torch.softmax(predictions, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        
        diagnosis = self.condition_mapping.get(predicted_class.item(), "Unknown")
                
        top_probs, top_classes = torch.topk(probabilities, 3)
        possible_conditions = [
            self.condition_mapping.get(cls.item(), "Unknown")
            for cls in top_classes[0]
        ]
        
        return {
            "diagnosis": diagnosis,
            "confidence": confidence.item(),
            "possible_conditions": possible_conditions,
            "recommendations": self._generate_recommendations(diagnosis)
        }
    
    def _generate_recommendations(self, diagnosis: str) -> List[str]:
        recommendations = {
            "Common Cold": ["Rest and hydration", "Over-the-counter cold medicine", "Consult doctor if symptoms worsen"],
            "Influenza": ["Antiviral medication", "Rest", "Fluids", "Medical consultation"],
            "COVID-19": ["Isolation", "Medical consultation", "Symptom monitoring", "Follow local health guidelines"],
            "Pneumonia": ["Antibiotics", "Hospitalization if severe", "Rest", "Immediate medical attention"],
            "Bronchitis": ["Bronchodilators", "Cough medicine", "Rest", "Avoid irritants"]
        }
        return recommendations.get(diagnosis, ["Consult a healthcare professional for proper diagnosis"])
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.eval()
        with torch.no_grad():
            processed_input = self.preprocess(input_data)
            predictions = self.forward(processed_input)
            return self.postprocess(predictions)