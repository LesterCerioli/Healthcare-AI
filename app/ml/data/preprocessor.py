import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any
import json

class MedicalDataset(Dataset):
    """PyTorch Dataset for medical data"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, transform=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            feature = self.transform(feature)
        
        return feature, label

class MedicalDataPreprocessor:
    """Preprocess medical data for training"""
    
    def __init__(self):
        self.symptom_mapping = {}
        self.condition_mapping = {}
        self.feature_size = 0
    
    def load_sample_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sample medical data for training
        In production, this would load from database or files
        """
        
        symptoms = [
            "fever", "headache", "cough", "fatigue", "nausea",
            "chest_pain", "shortness_breath", "sore_throat", "runny_nose",
            "muscle_pain", "chills", "loss_of_taste", "loss_of_smell"
        ]
        
        conditions = [
            "Common Cold", "Influenza", "COVID-19", "Pneumonia", "Bronchitis",
            "Strep Throat", "Migraine", "Allergies", "Asthma", "Gastroenteritis"
        ]
                
        self.symptom_mapping = {symptom: idx for idx, symptom in enumerate(symptoms)}
        self.condition_mapping = {condition: idx for idx, condition in enumerate(conditions)}
        self.feature_size = len(symptoms)
        
        
        num_samples = 1000
        features = np.random.randint(0, 2, (num_samples, len(symptoms))).astype(np.float32)
                
        for i in range(num_samples):
            
            if np.random.random() < 0.2:
                features[i, [0, 2, 8, 9]] = 1  # fever, cough, runny_nose, muscle_pain
                        
            elif np.random.random() < 0.2:
                features[i, [0, 1, 3, 9, 10]] = 1  # fever, headache, fatigue, muscle_pain, chills
            
            
            elif np.random.random() < 0.2:
                features[i, [0, 2, 3, 11, 12]] = 1  # fever, cough, fatigue, loss_of_taste, loss_of_smell
        
        
        labels = np.random.randint(0, len(conditions), num_samples)
        
        return features, labels
    
    def preprocess_symptoms(self, symptoms: Dict[str, Any]) -> np.ndarray:
        """Convert symptoms dictionary to feature vector"""
        feature_vector = np.zeros(len(self.symptom_mapping), dtype=np.float32)
        
        for symptom, severity in symptoms.items():
            if symptom in self.symptom_mapping:
                idx = self.symptom_mapping[symptom]
                if isinstance(severity, (int, float)):
                    feature_vector[idx] = min(max(severity, 0), 1)  # Normalize to 0-1
                else:
                    feature_vector[idx] = 1.0  # Binary presence
        
        return feature_vector
    
    def get_data_loaders(self, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders"""
        features, labels = self.load_sample_data()
                
        split_idx = int(0.8 * len(features))
        train_features, val_features = features[:split_idx], features[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]
        
        
        train_dataset = MedicalDataset(train_features, train_labels)
        val_dataset = MedicalDataset(val_features, val_labels)
                
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def save_mappings(self, filepath: str):
        """Save symptom and condition mappings to file"""
        mappings = {
            'symptom_mapping': self.symptom_mapping,
            'condition_mapping': self.condition_mapping,
            'feature_size': self.feature_size
        }
        with open(filepath, 'w') as f:
            json.dump(mappings, f, indent=2)
    
    def load_mappings(self, filepath: str):
        """Load symptom and condition mappings from file"""
        with open(filepath, 'r') as f:
            mappings = json.load(f)
        
        self.symptom_mapping = mappings['symptom_mapping']
        self.condition_mapping = mappings['condition_mapping']
        self.feature_size = mappings['feature_size']