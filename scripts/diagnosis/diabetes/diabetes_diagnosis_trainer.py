#!/usr/bin/env python3
"""
Enhanced Diabetes Mellitus Diagnosis & Management System
Robust ML model for Type 1/Type 2 diabetes diagnosis with age-specific treatment recommendations
EDUCATIONAL USE ONLY - NOT FOR CLINICAL DECISIONS
"""

import os
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('diabetes_training.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -----------------------------
# Configuration
# -----------------------------
@dataclass
class DiabetesConfig:
    """Configuration for diabetes diagnosis model"""
    seed: int = 42
    train_batch_size: int = 64
    val_batch_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 100
    weight_decay: float = 1e-4
    hidden_dims: tuple = (256, 128, 64)
    num_diagnosis_classes: int = 5
    num_treatment_classes: int = 4
    dropout: float = 0.3
    early_stop_patience: int = 15
    grad_clip_norm: float = 1.0
    
    
    age_bins: tuple = (
        (0, 12),    # Pediatric: Children
        (13, 18),   # Pediatric: Adolescents  
        (19, 30),   # Young Adults
        (31, 45),   # Adults
        (46, 60),   # Middle-aged
        (61, 75),   # Elderly
        (76, 200)   # Geriatric
    )
    
    
    label_col: str = "diagnosis"
    treatment_col: str = "treatment"
    sex_col: str = "sex"
    age_col: str = "age"
    
    
    output_dir: str = "ml_models/diabetes"
    model_name: str = "diabetes_diagnosis_enhanced"
    
    
    clinical_thresholds: dict = None
    
    def __post_init__(self):
        if self.clinical_thresholds is None:
            self.clinical_thresholds = {
                "fasting_glucose_diabetes": 7.0,      # mmol/L
                "fasting_glucose_prediabetes": 5.6,   # mmol/L
                "hba1c_diabetes": 6.5,               # %
                "hba1c_prediabetes": 5.7,            # %
                "bmi_obese": 30.0,
                "bmi_overweight": 25.0,
                "c_peptide_low": 0.6,                # ng/mL
                "random_glucose_diabetes": 11.1      # mmol/L
            }

CFG = DiabetesConfig()

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = CFG.seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# Medical Feature Engineering
# -----------------------------
class MedicalFeatureEngineer:
    """Enhanced medical feature engineering with clinical validation"""
    
    def __init__(self, config: DiabetesConfig):
        self.config = config
        self.age_bins = config.age_bins
        self.clinical_thresholds = config.clinical_thresholds
        
    def age_to_band(self, age: float) -> int:
        """Convert age to appropriate age band with validation"""
        if pd.isna(age) or age < 0 or age > 120:
            return len(self.age_bins) - 1  # Default to last band
        
        for i, (low, high) in enumerate(self.age_bins):
            if low <= age <= high:
                return i
        return len(self.age_bins) - 1
    
    def sex_to_onehot(self, sex: str) -> np.ndarray:
        """Convert sex to one-hot encoding with validation"""
        if pd.isna(sex):
            return np.array([0.5, 0.5], dtype=np.float32)  # Unknown sex
        
        sex_str = str(sex).strip().lower()
        if sex_str in ("m", "male", "masculino", "1"):
            return np.array([1.0, 0.0], dtype=np.float32)
        elif sex_str in ("f", "female", "feminino", "2"):
            return np.array([0.0, 1.0], dtype=np.float32)
        else:
            return np.array([0.5, 0.5], dtype=np.float32)
    
    def calculate_derived_features(self, row: pd.Series) -> Dict[str, float]:
        """Calculate clinically relevant derived features"""
        features = {}
        
        
        if not pd.isna(row.get('weight')) and not pd.isna(row.get('height')):
            height_m = float(row['height']) / 100.0
            if height_m > 0:
                bmi = float(row['weight']) / (height_m ** 2)
                features['bmi_category'] = (
                    2 if bmi >= self.clinical_thresholds["bmi_obese"] else
                    1 if bmi >= self.clinical_thresholds["bmi_overweight"] else 0
                )
                features['bmi_raw'] = bmi
        
        
        fasting_glucose = row.get('fasting_glucose')
        if not pd.isna(fasting_glucose):
            features['glucose_status'] = (
                2 if fasting_glucose >= self.clinical_thresholds["fasting_glucose_diabetes"] else
                1 if fasting_glucose >= self.clinical_thresholds["fasting_glucose_prediabetes"] else 0
            )
        
        
        hba1c = row.get('hba1c')
        if not pd.isna(hba1c):
            features['hba1c_status'] = (
                2 if hba1c >= self.clinical_thresholds["hba1c_diabetes"] else
                1 if hba1c >= self.clinical_thresholds["hba1c_prediabetes"] else 0
            )
        
        
        c_peptide = row.get('c_peptide')
        if not pd.isna(c_peptide):
            features['insulin_deficiency'] = 1 if c_peptide < self.clinical_thresholds["c_peptide_low"] else 0
        
        
        autoantibodies = [
            row.get('gad65_ab', 0), row.get('ia2_ab', 0), 
            row.get('insulin_ab', 0), row.get('zeta8_ab', 0)
        ]
        features['autoimmune_markers'] = sum(1 for ab in autoantibodies if not pd.isna(ab) and ab > 0)
        
        return features
    
    def one_hot_encode(self, index: int, size: int) -> np.ndarray:
        """One-hot encoding with bounds checking"""
        if index < 0 or index >= size:
            index = size - 1  # Default to last category
        vec = np.zeros(size, dtype=np.float32)
        vec[index] = 1.0
        return vec

# -----------------------------
# Enhanced Dataset
# -----------------------------
class EnhancedDiabetesDataset(Dataset):
    """Enhanced diabetes dataset with comprehensive medical features"""
    
    def __init__(self, df: pd.DataFrame, feature_stats: Dict, config: DiabetesConfig):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.feature_engineer = MedicalFeatureEngineer(config)
        self.feature_stats = feature_stats
        
        
        self.symptom_features = [
            "polyuria", "polydipsia", "polyphagia", "weight_loss", 
            "fatigue", "blurred_vision", "slow_healing", "numbness_tingling",
            "recurrent_infections", "dry_skin", "irritability", "ketosis"
        ]
        
        self.lab_features = [
            "fasting_glucose", "hba1c", "random_glucose", "post_prandial_glucose",
            "c_peptide", "gad65_ab", "ia2_ab", "insulin_ab", "zeta8_ab",
            "cholesterol_total", "cholesterol_ldl", "cholesterol_hdl", "triglycerides",
            "creatinine", "egfr", "microalbuminuria", "proteinuria", "ketones_blood", "ketones_urine"
        ]
        
        self.vital_features = [
            "weight", "height", "bmi", "systolic_bp", "diastolic_bp", "heart_rate"
        ]
        
        self.history_features = [
            "family_history_diabetes", "hypertension", "cardiovascular_disease",
            "gestational_diabetes_history", "pcos", "autoimmune_history"
        ]
        
        self._prepare_features()
    
    def _prepare_features(self):
        """Prepare all features for training"""
        self.features = []
        self.diagnosis_labels = []
        self.treatment_labels = []
        
        for _, row in self.df.iterrows():
            feature_parts = []
            
            
            age = row.get(self.config.age_col, 45)  # Default middle age
            age_band = self.feature_engineer.age_to_band(age)
            feature_parts.append(self.feature_engineer.one_hot_encode(age_band, len(self.config.age_bins)))
            
            sex_encoding = self.feature_engineer.sex_to_onehot(row.get(self.config.sex_col, ""))
            feature_parts.append(sex_encoding)
            
            
            derived_features = self.feature_engineer.calculate_derived_features(row)
            derived_vec = np.array([
                derived_features.get('bmi_category', 0),
                derived_features.get('glucose_status', 0),
                derived_features.get('hba1c_status', 0),
                derived_features.get('insulin_deficiency', 0),
                derived_features.get('autoimmune_markers', 0)
            ], dtype=np.float32)
            feature_parts.append(derived_vec)
            
            
            symptom_vec = np.array([
                1.0 if row.get(f"symptom_{symptom}", 0) > 0 else 0.0 
                for symptom in self.symptom_features
            ], dtype=np.float32)
            feature_parts.append(symptom_vec)
            
            
            lab_vec = []
            for lab in self.lab_features:
                value = row.get(lab, np.nan)
                if pd.isna(value):
                    # Impute with mean
                    value = self.feature_stats['means'].get(lab, 0.0)
                else:
                    # Normalize
                    mean = self.feature_stats['means'].get(lab, 0.0)
                    std = max(self.feature_stats['stds'].get(lab, 1.0), 1e-6)
                    value = (float(value) - mean) / std
                lab_vec.append(value)
            feature_parts.append(np.array(lab_vec, dtype=np.float32))
            
            
            history_vec = np.array([
                1.0 if row.get(history, 0) > 0 else 0.0 
                for history in self.history_features
            ], dtype=np.float32)
            feature_parts.append(history_vec)
            
            
            combined_features = np.concatenate(feature_parts).astype(np.float32)
            self.features.append(combined_features)
            
            
            diagnosis = int(row.get(self.config.label_col, 0))
            treatment = int(row.get(self.config.treatment_col, 0))
            
            self.diagnosis_labels.append(diagnosis)
            self.treatment_labels.append(treatment)
        
        self.features = torch.from_numpy(np.stack(self.features))
        self.diagnosis_labels = torch.tensor(self.diagnosis_labels, dtype=torch.long)
        self.treatment_labels = torch.tensor(self.treatment_labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.features[idx], self.diagnosis_labels[idx], self.treatment_labels[idx]
    
    @property
    def input_dim(self):
        return self.features.shape[1]

# -----------------------------
# Enhanced Model Architecture
# -----------------------------
class DiabetesDiagnosisModel(nn.Module):
    """Enhanced neural network for diabetes diagnosis and treatment recommendation"""
    
    def __init__(self, input_dim: int, config: DiabetesConfig):
        super().__init__()
        self.config = config
        h1, h2, h3 = config.hidden_dims
        
        
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        
        
        self.diagnosis_head = nn.Sequential(
            nn.Linear(h3, h3 // 2),
            nn.ReLU(),
            nn.Linear(h3 // 2, config.num_diagnosis_classes)
        )
        
        
        self.treatment_head = nn.Sequential(
            nn.Linear(h3, h3 // 2),
            nn.ReLU(),
            nn.Linear(h3 // 2, config.num_treatment_classes)
        )
        
        
        self.risk_head = nn.Sequential(
            nn.Linear(h3, h3 // 4),
            nn.ReLU(),
            nn.Linear(h3 // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.feature_net(x)
        
        diagnosis_logits = self.diagnosis_head(features)
        treatment_logits = self.treatment_head(features)
        risk_score = self.risk_head(features)
        
        return diagnosis_logits, treatment_logits, risk_score

# -----------------------------
# Advanced Data Generator
# -----------------------------
class RealisticDiabetesDataGenerator:
    """Generates realistic synthetic diabetes data based on medical literature"""
    
    def __init__(self, config: DiabetesConfig):
        self.config = config
        self.diagnosis_mapping = {
            0: "No Diabetes",
            1: "Type 1 Diabetes",
            2: "Type 2 Diabetes", 
            3: "Prediabetes",
            4: "Gestational Diabetes"
        }
        self.treatment_mapping = {
            0: "Lifestyle Modification",
            1: "Oral Hypoglycemics",
            2: "Insulin Therapy",
            3: "Combined Therapy"
        }
        
    def generate_realistic_dataset(self, n_samples: int = 20000) -> pd.DataFrame:
        """Generate realistic synthetic diabetes dataset"""
        np.random.seed(self.config.seed)
        
        data = []
        
        for i in range(n_samples):
            patient = {}
            
            
            age = self._sample_age()
            sex = np.random.choice(['M', 'F'], p=[0.49, 0.51])
            
            patient['age'] = age
            patient['sex'] = sex
            
            
            diagnosis, treatment = self._assign_diagnosis_and_treatment(age, sex)
            patient['diagnosis'] = diagnosis
            patient['treatment'] = treatment
            
            
            self._generate_symptoms(patient, diagnosis, age)
            
            
            self._generate_labs(patient, diagnosis, age)
            
            
            self._generate_history(patient, diagnosis, age, sex)
            
            data.append(patient)
        
        return pd.DataFrame(data)
    
    def _sample_age(self) -> float:
        """Sample age from realistic distribution"""
        
        if np.random.random() < 0.3:  # 30% younger population
            return np.random.uniform(1, 30)
        else:  
            return np.random.uniform(31, 85)
    
    def _assign_diagnosis_and_treatment(self, age: float, sex: str) -> Tuple[int, int]:
        """Assign diagnosis and treatment based on age and sex"""
        rand_val = np.random.random()
        age_group = MedicalFeatureEngineer(self.config).age_to_band(age)
        
        
        if age < 30 and rand_val < 0.08:  # 8% prevalence in young
            diagnosis = 1  
            treatment = 2  
            
        
        elif age >= 30 and rand_val < 0.15:  # 15% prevalence in adults
            diagnosis = 2  # Type 2
            
            if age < 50:
                treatment = np.random.choice([0, 1], p=[0.3, 0.7])  # Mostly oral meds
            else:
                treatment = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])  # More insulin
                
        
        elif rand_val < 0.25:  # 25% prediabetes prevalence
            diagnosis = 3  # Prediabetes
            treatment = 0  # Lifestyle modification
            
        
        elif sex == 'F' and 18 <= age <= 45 and rand_val < 0.1:
            diagnosis = 4  # Gestational diabetes
            treatment = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
            
        else:
            diagnosis = 0  # No diabetes
            treatment = 0  # No treatment
            
        return diagnosis, treatment
    
    def _generate_symptoms(self, patient: Dict, diagnosis: int, age: float):
        """Generate realistic symptoms based on diagnosis"""
        
        if diagnosis in [1, 2, 4]:  # Diabetes cases
            patient['symptom_polyuria'] = np.random.choice([0, 1], p=[0.2, 0.8])
            patient['symptom_polydipsia'] = np.random.choice([0, 1], p=[0.3, 0.7])
            patient['symptom_fatigue'] = np.random.choice([0, 1], p=[0.4, 0.6])
            
            
            if diagnosis == 1:
                patient['symptom_weight_loss'] = np.random.choice([0, 1], p=[0.1, 0.9])
                patient['symptom_ketosis'] = np.random.choice([0, 1], p=[0.5, 0.5])
                
            
            if diagnosis == 2:
                patient['symptom_blurred_vision'] = np.random.choice([0, 1], p=[0.6, 0.4])
                patient['symptom_slow_healing'] = np.random.choice([0, 1], p=[0.7, 0.3])
                patient['symptom_numbness_tingling'] = np.random.choice([0, 1], p=[0.8, 0.2])
        else:
            # Non-diabetic patients may have some symptoms occasionally
            for symptom in ['polyuria', 'polydipsia', 'fatigue', 'weight_loss']:
                patient[f'symptom_{symptom}'] = np.random.choice([0, 1], p=[0.9, 0.1])
    
    def _generate_labs(self, patient: Dict, diagnosis: int, age: float):
        """Generate realistic lab results"""
        
        if diagnosis == 1:
            patient['fasting_glucose'] = np.random.uniform(8.0, 20.0)
            patient['hba1c'] = np.random.uniform(8.0, 14.0)
            patient['c_peptide'] = np.random.uniform(0.1, 0.5)  # Low
            patient['gad65_ab'] = np.random.uniform(5.0, 50.0)  # Positive
            patient['ketones_blood'] = np.random.uniform(0.5, 3.0)
            
        
        elif diagnosis == 2:
            patient['fasting_glucose'] = np.random.uniform(7.0, 12.0)
            patient['hba1c'] = np.random.uniform(6.5, 10.0)
            patient['c_peptide'] = np.random.uniform(1.0, 4.0)  # Normal/high
            patient['bmi'] = np.random.uniform(28.0, 40.0)  # Higher BMI
            
        
        elif diagnosis == 3:
            patient['fasting_glucose'] = np.random.uniform(5.6, 6.9)
            patient['hba1c'] = np.random.uniform(5.7, 6.4)
            patient['bmi'] = np.random.uniform(25.0, 32.0)
            
        
        elif diagnosis == 4:
            patient['fasting_glucose'] = np.random.uniform(5.1, 7.0)
            patient['post_prandial_glucose'] = np.random.uniform(8.0, 11.0)
            
        
        else:
            patient['fasting_glucose'] = np.random.uniform(4.0, 5.5)
            patient['hba1c'] = np.random.uniform(4.0, 5.6)
            patient['bmi'] = np.random.uniform(18.5, 25.0)
    
    def _generate_history(self, patient: Dict, diagnosis: int, age: float, sex: str):
        """Generate medical history"""
        
        if diagnosis in [1, 2]:
            patient['family_history_diabetes'] = np.random.choice([0, 1], p=[0.3, 0.7])
        else:
            patient['family_history_diabetes'] = np.random.choice([0, 1], p=[0.8, 0.2])
        
        
        if diagnosis == 2 or age > 50:
            patient['hypertension'] = np.random.choice([0, 1], p=[0.4, 0.6])
        else:
            patient['hypertension'] = np.random.choice([0, 1], p=[0.8, 0.2])
            
        
        if sex == 'F' and diagnosis == 2:
            patient['pcos'] = np.random.choice([0, 1], p=[0.6, 0.4])
        else:
            patient['pcos'] = 0

# -----------------------------
# Enhanced Training System
# -----------------------------
class DiabetesTrainer:
    """Enhanced trainer for diabetes diagnosis model"""
    
    def __init__(self, model, config: DiabetesConfig, device: str = 'cuda'):
        self.model = model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        
        self.diagnosis_criterion = nn.CrossEntropyLoss()
        self.treatment_criterion = nn.CrossEntropyLoss()
        self.risk_criterion = nn.MSELoss()
        
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        self.train_history = {
            'diagnosis_loss': [], 'treatment_loss': [], 'total_loss': [],
            'diagnosis_acc': [], 'treatment_acc': [], 'val_loss': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_diagnosis_loss = 0.0
        total_treatment_loss = 0.0
        total_risk_loss = 0.0
        diagnosis_correct = 0
        treatment_correct = 0
        total_samples = 0
        
        for features, diagnosis_labels, treatment_labels in train_loader:
            features = features.to(self.device)
            diagnosis_labels = diagnosis_labels.to(self.device)
            treatment_labels = treatment_labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            
            diagnosis_logits, treatment_logits, risk_scores = self.model(features)
            
            
            diagnosis_loss = self.diagnosis_criterion(diagnosis_logits, diagnosis_labels)
            treatment_loss = self.treatment_criterion(treatment_logits, treatment_labels)
            
            
            risk_targets = (diagnosis_labels > 0).float().unsqueeze(1)
            risk_loss = self.risk_criterion(risk_scores, risk_targets)
            
            
            total_loss = diagnosis_loss + 0.7 * treatment_loss + 0.3 * risk_loss
            
            
            total_loss.backward()
            
            if self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            
            self.optimizer.step()
            
            
            total_diagnosis_loss += diagnosis_loss.item() * features.size(0)
            total_treatment_loss += treatment_loss.item() * features.size(0)
            total_risk_loss += risk_loss.item() * features.size(0)
            
            diagnosis_preds = diagnosis_logits.argmax(dim=1)
            treatment_preds = treatment_logits.argmax(dim=1)
            
            diagnosis_correct += (diagnosis_preds == diagnosis_labels).sum().item()
            treatment_correct += (treatment_preds == treatment_labels).sum().item()
            total_samples += features.size(0)
        
        epoch_metrics = {
            'diagnosis_loss': total_diagnosis_loss / total_samples,
            'treatment_loss': total_treatment_loss / total_samples,
            'risk_loss': total_risk_loss / total_samples,
            'diagnosis_acc': diagnosis_correct / total_samples,
            'treatment_acc': treatment_correct / total_samples,
            'total_loss': (total_diagnosis_loss + total_treatment_loss) / total_samples
        }
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model performance"""
        self.model.eval()
        total_diagnosis_loss = 0.0
        total_treatment_loss = 0.0
        diagnosis_correct = 0
        treatment_correct = 0
        total_samples = 0
        
        all_diagnosis_preds = []
        all_diagnosis_labels = []
        all_treatment_preds = []
        all_treatment_labels = []
        
        for features, diagnosis_labels, treatment_labels in val_loader:
            features = features.to(self.device)
            diagnosis_labels = diagnosis_labels.to(self.device)
            treatment_labels = treatment_labels.to(self.device)
            
            diagnosis_logits, treatment_logits, _ = self.model(features)
            
            diagnosis_loss = self.diagnosis_criterion(diagnosis_logits, diagnosis_labels)
            treatment_loss = self.treatment_criterion(treatment_logits, treatment_labels)
            
            total_diagnosis_loss += diagnosis_loss.item() * features.size(0)
            total_treatment_loss += treatment_loss.item() * features.size(0)
            
            diagnosis_preds = diagnosis_logits.argmax(dim=1)
            treatment_preds = treatment_logits.argmax(dim=1)
            
            diagnosis_correct += (diagnosis_preds == diagnosis_labels).sum().item()
            treatment_correct += (treatment_preds == treatment_labels).sum().item()
            total_samples += features.size(0)
            
            all_diagnosis_preds.extend(diagnosis_preds.cpu().numpy())
            all_diagnosis_labels.extend(diagnosis_labels.cpu().numpy())
            all_treatment_preds.extend(treatment_preds.cpu().numpy())
            all_treatment_labels.extend(treatment_labels.cpu().numpy())
        
        val_metrics = {
            'val_diagnosis_loss': total_diagnosis_loss / total_samples,
            'val_treatment_loss': total_treatment_loss / total_samples,
            'val_diagnosis_acc': diagnosis_correct / total_samples,
            'val_treatment_acc': treatment_correct / total_samples,
            'val_total_loss': (total_diagnosis_loss + total_treatment_loss) / total_samples
        }
        
        return val_metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Complete training loop"""
        logger.info(f"🚀 Starting training on {self.device}")
        logger.info(f"📊 Training samples: {len(train_loader.dataset)}")
        logger.info(f"📊 Validation samples: {len(val_loader.dataset)}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            
            train_metrics = self.train_epoch(train_loader)
            
            
            val_metrics = self.validate(val_loader)
            
            
            self.scheduler.step(val_metrics['val_total_loss'])
            
            
            for key, value in train_metrics.items():
                self.train_history[key].append(value)
            self.train_history['val_loss'].append(val_metrics['val_total_loss'])
            
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"📈 Epoch {epoch+1:03d}/{self.config.epochs} | "
                    f"Train Loss: {train_metrics['total_loss']:.4f} | "
                    f"Val Loss: {val_metrics['val_total_loss']:.4f} | "
                    f"Diag Acc: {train_metrics['diagnosis_acc']:.3f} | "
                    f"Treat Acc: {train_metrics['treatment_acc']:.3f}"
                )
            
            
            if val_metrics['val_total_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_total_loss']
                patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stop_patience:
                    logger.info(f"🛑 Early stopping at epoch {epoch+1}")
                    break
        
        logger.info(f"✅ Training completed! Best validation loss: {best_val_loss:.4f}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'config': asdict(self.config)
        }
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        if is_best:
            torch.save(checkpoint, f'{self.config.output_dir}/best_model.pth')
            logger.info("💾 Best model saved!")

# -----------------------------
# Main Training Pipeline
# -----------------------------
def main():
    """Main training pipeline"""
    set_seed(CFG.seed)
    
    logger.info("🏥 Starting Enhanced Diabetes Diagnosis Training Pipeline")
    
    
    os.makedirs(CFG.output_dir, exist_ok=True)
    
    
    logger.info("📊 Generating realistic diabetes dataset...")
    data_generator = RealisticDiabetesDataGenerator(CFG)
    df = data_generator.generate_realistic_dataset(n_samples=25000)
    
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=CFG.seed, stratify=df['diagnosis'])
    
    
    feature_columns = []
    for col in df.columns:
        if col not in ['diagnosis', 'treatment', 'age', 'sex'] and df[col].dtype in [np.float64, np.int64]:
            feature_columns.append(col)
    
    feature_stats = {
        'means': {col: train_df[col].mean() for col in feature_columns},
        'stds': {col: train_df[col].std() for col in feature_columns}
    }
    
    
    train_dataset = EnhancedDiabetesDataset(train_df, feature_stats, CFG)
    val_dataset = EnhancedDiabetesDataset(val_df, feature_stats, CFG)
        
    train_loader = DataLoader(train_dataset, batch_size=CFG.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.val_batch_size, shuffle=False)
        
    model = DiabetesDiagnosisModel(train_dataset.input_dim, CFG)
    logger.info(f"🧠 Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"📐 Input dimension: {train_dataset.input_dim}")
    
    
    trainer = DiabetesTrainer(model, CFG)
    
    
    logger.info("⏳ Starting model training...")
    trainer.train(train_loader, val_loader)
        
    artifacts = {
        'feature_stats': feature_stats,
        'diagnosis_mapping': data_generator.diagnosis_mapping,
        'treatment_mapping': data_generator.treatment_mapping,
        'config': asdict(CFG),
        'training_date': datetime.now().isoformat(),
        'dataset_info': {
            'total_samples': len(df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'diagnosis_distribution': dict(df['diagnosis'].value_counts()),
            'treatment_distribution': dict(df['treatment'].value_counts())
        }
    }
    
    with open(f'{CFG.output_dir}/training_artifacts.json', 'w') as f:
        json.dump(artifacts, f, indent=2)
    
    logger.info("✅ Diabetes diagnosis training pipeline completed successfully!")
    logger.info(f"📁 Model saved to: {CFG.output_dir}/")
    logger.info("🎯 REMINDER: This model is for educational purposes only - NOT FOR CLINICAL USE")

if __name__ == "__main__":
    main()