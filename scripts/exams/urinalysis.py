import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import json
warnings.filterwarnings('ignore')

class UrinalysisDataset(Dataset):
    """Dataset for urinalysis parameters with comprehensive patient context"""
    
    def __init__(self, features, labels, patient_info, clinical_context):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.patient_info = patient_info
        self.clinical_context = clinical_context
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'labels': self.labels[idx],
            'age': torch.FloatTensor([self.patient_info['age'][idx]]),
            'sex': torch.LongTensor([self.patient_info['sex'][idx]]),
            'bmi': torch.FloatTensor([self.patient_info['bmi'][idx]]),
            'diabetes_status': torch.LongTensor([self.clinical_context['diabetes'][idx]]),
            'hypertension_status': torch.LongTensor([self.clinical_context['hypertension'][idx]]),
            'kidney_disease_history': torch.LongTensor([self.clinical_context['kidney_disease'][idx]])
        }

class UrinalysisReferenceValidator:
    """Comprehensive reference value validator for urinalysis parameters"""
    
    def __init__(self):
        
        self.reference_ranges = {
            
            'protein_creatinine_ratio': {
                'normal': {'male': (0, 150), 'female': (0, 150)},
                'microalbuminuria': {'male': (150, 500), 'female': (150, 500)},
                'macroalbuminuria': {'male': (500, 3500), 'female': (500, 3500)},
                'nephrotic_range': {'male': (3500, 10000), 'female': (3500, 10000)},
                'critical': {'male': (10000, 50000), 'female': (10000, 50000)}
            },
            
            
            'albumin_creatinine_ratio': {
                'normal': {'male': (0, 30), 'female': (0, 30)},
                'microalbuminuria': {'male': (30, 300), 'female': (30, 300)},
                'macroalbuminuria': {'male': (300, 2000), 'female': (300, 2000)},
                'severe': {'male': (2000, 5000), 'female': (2000, 5000)},
                'critical': {'male': (5000, 10000), 'female': (5000, 10000)}
            },
            
            
            'serum_creatinine': {
                'neonate': {'male': (0.3, 1.0), 'female': (0.3, 1.0)},
                'infant': {'male': (0.2, 0.4), 'female': (0.2, 0.4)},
                'child_1y': {'male': (0.3, 0.7), 'female': (0.3, 0.7)},
                'child_6y': {'male': (0.5, 0.8), 'female': (0.5, 0.8)},
                'adolescent_12y': {'male': (0.6, 1.0), 'female': (0.6, 1.0)},
                'adult_18y': {'male': (0.7, 1.3), 'female': (0.5, 1.1)},
                'adult_40y': {'male': (0.8, 1.3), 'female': (0.6, 1.1)},
                'elderly_60y': {'male': (0.9, 1.3), 'female': (0.7, 1.2)},
                'elderly_80y': {'male': (0.8, 1.3), 'female': (0.7, 1.2)},
                'muscular_adult': {'male': (0.9, 1.5), 'female': (0.7, 1.3)}
            },
            
            
            'egfr': {
                'stage1_normal': {'male': (90, 120), 'female': (90, 120)},
                'stage2_mild': {'male': (60, 89), 'female': (60, 89)},
                'stage3a_moderate': {'male': (45, 59), 'female': (45, 59)},
                'stage3b_moderate': {'male': (30, 44), 'female': (30, 44)},
                'stage4_severe': {'male': (15, 29), 'female': (15, 29)},
                'stage5_eskd': {'male': (0, 14), 'female': (0, 14)}
            },
            
            
            'urine_creatinine': {
                'adult_male': (20, 25),  # mg/kg/24h
                'adult_female': (15, 20), # mg/kg/24h
                'elderly_male': (15, 20),
                'elderly_female': (10, 15)
            },
            
            
            'urine_specific_gravity': {
                'all': {'male': (1.003, 1.035), 'female': (1.003, 1.035)},
                'concentrated': {'male': (1.025, 1.035), 'female': (1.025, 1.035)},
                'dilute': {'male': (1.003, 1.010), 'female': (1.003, 1.010)}
            },
            
            
            'urine_ph': {
                'all': {'male': (4.5, 8.0), 'female': (4.5, 8.0)},
                'normal': {'male': (5.0, 7.0), 'female': (5.0, 7.0)},
                'acidic': {'male': (4.5, 5.5), 'female': (4.5, 5.5)},
                'alkaline': {'male': (7.0, 8.0), 'female': (7.0, 8.0)}
            }
        }
        
        
        self.critical_values = {
            'serum_creatinine': {'low': 0.3, 'high': 5.0, 'immediate_action': True},
            'egfr': {'low': 15, 'high': 120, 'immediate_action': True},
            'albumin_creatinine_ratio': {'low': 0, 'high': 5000, 'immediate_action': True},
            'protein_creatinine_ratio': {'low': 0, 'high': 10000, 'immediate_action': True}
        }
        
        
        self.ckd_stages = {
            'stage1': {'egfr': (90, 120), 'acr': (0, 30), 'risk': 'Low'},
            'stage2': {'egfr': (60, 89), 'acr': (0, 30), 'risk': 'Moderate'},
            'stage3a': {'egfr': (45, 59), 'acr': (30, 300), 'risk': 'High'},
            'stage3b': {'egfr': (30, 44), 'acr': (30, 300), 'risk': 'High'},
            'stage4': {'egfr': (15, 29), 'acr': (300, 2000), 'risk': 'Very High'},
            'stage5': {'egfr': (0, 14), 'acr': (300, 2000), 'risk': 'Very High'}
        }

    def calculate_egfr(self, creatinine: float, age: float, sex: str, ethnicity: str = 'non-black') -> float:
        """Calculate eGFR using CKD-EPI 2021 formula"""
        if creatinine <= 0:
            return 0.0
        
        k = 0.7 if sex.lower() == 'female' else 0.9
        alpha = -0.329 if sex.lower() == 'female' else -0.411
        min_cr = min(creatinine / k, 1)
        max_cr = max(creatinine / k, 1)
        
        
        egfr = 142 * (min_cr ** alpha) * (max_cr ** -1.2) * (0.9938 ** age)
        
        if sex.lower() == 'female':
            egfr *= 1.012
        
        
        if ethnicity.lower() == 'black':
            egfr *= 1.159
        
        return max(egfr, 0.0)

    def validate_proteinuria(self, upcr: float, clinical_context: Dict) -> Dict:
        """Comprehensive proteinuria validation with clinical context"""
        validation = self._validate_parameter('protein_creatinine_ratio', upcr, 
                                            clinical_context['age'], clinical_context['sex'])
        
        
        validation['kdigo_classification'] = self._classify_proteinuria_kdigo(upcr)
        
        
        validation['clinical_significance'] = self._get_proteinuria_significance(
            upcr, clinical_context
        )
        
        return validation

    def validate_microalbuminuria(self, uacr: float, clinical_context: Dict) -> Dict:
        """Comprehensive microalbuminuria validation"""
        validation = self._validate_parameter('albumin_creatinine_ratio', uacr,
                                            clinical_context['age'], clinical_context['sex'])
        
        
        if clinical_context.get('diabetes', False):
            validation['ada_classification'] = self._classify_albuminuria_ada(uacr)
        
        
        validation['cardiovascular_risk'] = self._assess_cardiovascular_risk(uacr)
        
        return validation

    def validate_serum_creatinine(self, creatinine: float, age: float, sex: str, 
                                muscle_mass: str = 'normal') -> Dict:
        """Validate serum creatinine with muscle mass consideration"""
        
        adjusted_ranges = self._adjust_creatinine_ranges(age, sex, muscle_mass)
        
        validation = self._validate_with_custom_ranges(
            'serum_creatinine', creatinine, age, sex, adjusted_ranges
        )
        
        
        egfr = self.calculate_egfr(creatinine, age, sex)
        validation['egfr'] = egfr
        validation['egfr_validation'] = self.validate_egfr(egfr, age, sex)
        
        
        validation['ckd_stage'] = self._classify_ckd_stage(egfr, 0)  # ACR would be added separately
        
        return validation

    def _validate_parameter(self, param_name: str, value: float, age: float, 
                          sex: str, pregnant: bool = False) -> Dict:
        """Core parameter validation against reference ranges"""
        if param_name not in self.reference_ranges:
            return {'status': 'unknown_parameter', 'value': value}
        
        param_ranges = self.reference_ranges[param_name]
        age_group = self._get_age_group(age, pregnant)
        
        
        ref_range = None
        for group_key in [age_group, 'adult_18y', 'adult', 'all']:
            if group_key in param_ranges:
                ref_range = param_ranges[group_key].get(sex.lower(), 
                                                       param_ranges[group_key].get('male'))
                if ref_range:
                    break
        
        if not ref_range:
            return {'status': 'no_reference_range', 'value': value}
        
        
        critical_info = self._check_critical_values(param_name, value)
        if critical_info['is_critical']:
            return critical_info
        
        
        low, high = ref_range
        deviation = self._calculate_deviation(value, low, high)
        
        if value < low:
            severity = self._assess_severity(value, low, high, 'low')
            return {
                'status': 'low',
                'severity': severity,
                'value': value,
                'reference_range': ref_range,
                'deviation_percentage': deviation,
                'clinical_significance': self._get_urinalysis_significance(param_name, 'low', value)
            }
        elif value > high:
            severity = self._assess_severity(value, low, high, 'high')
            return {
                'status': 'high',
                'severity': severity,
                'value': value,
                'reference_range': ref_range,
                'deviation_percentage': deviation,
                'clinical_significance': self._get_urinalysis_significance(param_name, 'high', value)
            }
        else:
            return {
                'status': 'normal',
                'value': value,
                'reference_range': ref_range,
                'deviation_percentage': deviation
            }

    def _classify_proteinuria_kdigo(self, upcr: float) -> Dict:
        """Classify proteinuria according to KDIGO guidelines"""
        if upcr < 150:
            return {'category': 'A1', 'description': 'Normal to mildly increased', 'risk': 'Low'}
        elif upcr < 500:
            return {'category': 'A2', 'description': 'Moderately increased', 'risk': 'Moderate'}
        else:
            return {'category': 'A3', 'description': 'Severely increased', 'risk': 'High'}

    def _classify_albuminuria_ada(self, uacr: float) -> Dict:
        """Classify albuminuria according to ADA guidelines"""
        if uacr < 30:
            return {'category': 'Normal', 'action': 'Annual monitoring'}
        elif uacr < 300:
            return {'category': 'Microalbuminuria', 'action': 'Optimize glycemic control, ACE-I/ARB consideration'}
        else:
            return {'category': 'Macroalbuminuria', 'action': 'ACE-I/ARB therapy, nephrology referral'}

    def _classify_ckd_stage(self, egfr: float, uacr: float) -> Dict:
        """Classify CKD stage based on KDIGO guidelines"""
        if egfr >= 90 and uacr < 30:
            return {'stage': 'G1A1', 'description': 'Normal kidney function'}
        elif egfr >= 90 and uacr >= 30:
            return {'stage': 'G1A2', 'description': 'Kidney damage with normal GFR'}
        elif egfr >= 60 and uacr < 30:
            return {'stage': 'G2A1', 'description': 'Mildly decreased GFR'}
        elif egfr >= 60 and uacr >= 30:
            return {'stage': 'G2A2', 'description': 'Mildly decreased GFR with kidney damage'}
        elif egfr >= 45 and uacr < 30:
            return {'stage': 'G3aA1', 'description': 'Mild-moderately decreased GFR'}
        elif egfr >= 45 and uacr >= 30:
            return {'stage': 'G3aA2', 'description': 'Mild-moderately decreased GFR with kidney damage'}
        elif egfr >= 30 and uacr < 30:
            return {'stage': 'G3bA1', 'description': 'Moderate-severely decreased GFR'}
        elif egfr >= 30 and uacr >= 30:
            return {'stage': 'G3bA2', 'description': 'Moderate-severely decreased GFR with kidney damage'}
        elif egfr >= 15 and uacr < 30:
            return {'stage': 'G4A1', 'description': 'Severely decreased GFR'}
        elif egfr >= 15 and uacr >= 30:
            return {'stage': 'G4A2', 'description': 'Severely decreased GFR with kidney damage'}
        else:
            return {'stage': 'G5A3', 'description': 'Kidney failure'}

    def _get_urinalysis_significance(self, param: str, direction: str, value: float) -> str:
        """Provide clinical significance for urinalysis findings"""
        significance_map = {
            'protein_creatinine_ratio_high': "Proteinuria - Consider glomerular disease, diabetic nephropathy, hypertension, multiple myeloma",
            'albumin_creatinine_ratio_high': "Albuminuria - Early marker of diabetic nephropathy, cardiovascular risk factor, glomerular damage",
            'serum_creatinine_high': "Impaired kidney function - Consider AKI, CKD, dehydration, rhabdomyolysis, drug toxicity",
            'serum_creatinine_low': "Low muscle mass, malnutrition, liver disease, or advanced age",
            'egfr_low': "Decreased kidney function - Requires staging and etiology evaluation"
        }
        
        return significance_map.get(f"{param}_{direction}", "Clinical correlation required")

    def _check_critical_values(self, param_name: str, value: float) -> Dict:
        """Check for critical values requiring immediate action"""
        if param_name in self.critical_values:
            crit = self.critical_values[param_name]
            
            if value <= crit['low']:
                return {
                    'is_critical': True,
                    'status': 'critical_low',
                    'urgency': 'immediate' if crit['immediate_action'] else 'urgent',
                    'message': f'CRITICALLY LOW {param_name.upper()} - Requires immediate evaluation!',
                    'actions': self._get_critical_actions(param_name, 'low')
                }
            elif value >= crit['high']:
                return {
                    'is_critical': True,
                    'status': 'critical_high',
                    'urgency': 'immediate' if crit['immediate_action'] else 'urgent',
                    'message': f'CRITICALLY HIGH {param_name.upper()} - Requires immediate evaluation!',
                    'actions': self._get_critical_actions(param_name, 'high')
                }
        
        return {'is_critical': False}

    def _get_critical_actions(self, param: str, direction: str) -> List[str]:
        """Get recommended actions for critical values"""
        actions = {
            'serum_creatinine_high': [
                "Immediate nephrology consultation",
                "Evaluate for acute kidney injury",
                "Check urine output and volume status",
                "Review nephrotoxic medications",
                "Consider renal ultrasound"
            ],
            'egfr_low': [
                "Immediate nephrology referral",
                "Evaluate for CKD complications",
                "Adjust medication doses for renal function",
                "Monitor electrolytes and acid-base status"
            ],
            'albumin_creatinine_ratio_high': [
                "Optimize blood pressure control",
                "ACE-I or ARB therapy consideration",
                "Cardiovascular risk assessment",
                "Diabetes management optimization"
            ]
        }
        
        return actions.get(f"{param}_{direction}", ["Immediate physician notification required"])

class UrinalysisNeuralNetwork(nn.Module):
    """Advanced neural network for comprehensive urinalysis prediction"""
    
    def __init__(self, input_dim: int, num_parameters: int, hidden_dims: List[int] = [256, 128, 64]):
        super(UrinalysisNeuralNetwork, self).__init__()
        
        self.num_parameters = num_parameters
        
        
        self.age_embedding = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        self.sex_embedding = nn.Embedding(2, 8)
        self.bmi_embedding = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        
        
        self.diabetes_embedding = nn.Embedding(3, 8)  # 0: no, 1: type1, 2: type2
        self.hypertension_embedding = nn.Embedding(2, 4)
        self.kidney_disease_embedding = nn.Embedding(3, 8)  # 0: no, 1: history, 2: current
        
        
        layers = []
        prev_dim = input_dim + 40  # Features + all embeddings
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.feature_processor = nn.Sequential(*layers)
        
        
        self.proteinuria_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # Normal, microalbuminuria, macroalbuminuria
        )
        
        self.creatinine_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # Serum creatinine, eGFR
        )
        
        self.risk_assessment_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),  # CKD risk, CVD risk, progression risk, overall risk
            nn.Sigmoid()
        )
        
        
        self.anomaly_detector = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_parameters * 3),  # Normal, abnormal, critical
            nn.Softmax(dim=1)
        )

    def forward(self, features, age, sex, bmi, diabetes, hypertension, kidney_disease):
        
        age_embedded = self.age_embedding(age.unsqueeze(1))
        sex_embedded = self.sex_embedding(sex.squeeze())
        bmi_embedded = self.bmi_embedding(bmi.unsqueeze(1))
        diabetes_embedded = self.diabetes_embedding(diabetes.squeeze())
        hypertension_embedded = self.hypertension_embedding(hypertension.squeeze())
        kidney_embedded = self.kidney_disease_embedding(kidney_disease.squeeze())
        
        
        combined = torch.cat([
            features, age_embedded, sex_embedded, bmi_embedded,
            diabetes_embedded, hypertension_embedded, kidney_embedded
        ], dim=1)
        
        
        processed = self.feature_processor(combined)
        
        
        proteinuria_pred = self.proteinuria_head(processed)
        creatinine_pred = self.creatinine_head(processed)
        risk_assessment = self.risk_assessment_head(processed)
        anomaly_pred = self.anomaly_detector(processed)
        anomaly_pred = anomaly_pred.view(-1, self.num_parameters, 3)
        
        return {
            'proteinuria': proteinuria_pred,
            'creatinine': creatinine_pred,
            'risk_assessment': risk_assessment,
            'anomaly': anomaly_pred
        }

class ComprehensiveUrinalysisAnalyzer:
    """Main class for comprehensive urinalysis analysis"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.validator = UrinalysisReferenceValidator()
        self.parameter_names = [
            'protein_creatinine_ratio', 'albumin_creatinine_ratio', 
            'serum_creatinine', 'egfr', 'urine_creatinine',
            'urine_specific_gravity', 'urine_ph'
        ]
        
        self.sex_encoder = LabelEncoder()
        self.diabetes_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_sample_data(self, num_samples: int = 2000) -> pd.DataFrame:
        """Create realistic urinalysis sample data"""
        np.random.seed(42)
        
        data = {
            'age': np.concatenate([
                np.random.uniform(18, 40, num_samples // 3),
                np.random.uniform(40, 65, num_samples // 3),
                np.random.uniform(65, 90, num_samples // 3)
            ]),
            'sex': np.random.choice(['male', 'female'], num_samples),
            'bmi': np.random.normal(26, 4, num_samples),
            'diabetes': np.random.choice(['none', 'type1', 'type2'], num_samples, p=[0.7, 0.1, 0.2]),
            'hypertension': np.random.choice([0, 1], num_samples, p=[0.6, 0.4]),
            'kidney_disease': np.random.choice(['none', 'history', 'current'], num_samples, p=[0.8, 0.1, 0.1])
        }
        
        
        for i in range(num_samples):
            age = data['age'][i]
            sex = data['sex'][i]
            diabetes = data['diabetes'][i]
            hypertension = data['hypertension'][i]
            kidney_disease = data['kidney_disease'][i]
            
            
            base_creatinine = np.random.normal(0.9, 0.2) if sex == 'male' else np.random.normal(0.7, 0.15)
            
            
            if age > 60:
                base_creatinine *= 0.9
            
            
            if diabetes != 'none' or hypertension or kidney_disease != 'none':
                albuminuria_factor = np.random.lognormal(0, 0.5)
                proteinuria_factor = np.random.lognormal(0, 0.3)
            else:
                albuminuria_factor = np.random.lognormal(-1, 0.2)
                proteinuria_factor = np.random.lognormal(-1.5, 0.2)
            
            data.setdefault('serum_creatinine', []).append(base_creatinine)
            data.setdefault('albumin_creatinine_ratio', []).append(max(1, 10 * albuminuria_factor))
            data.setdefault('protein_creatinine_ratio', []).append(max(5, 50 * proteinuria_factor))
        
        return pd.DataFrame(data)

    def analyze_urinalysis(self, patient_data: Dict) -> Dict:
        """Comprehensive urinalysis analysis with clinical context"""
        print(f"\n🧪 COMPREHENSIVE URINALYSIS ANALYSIS")
        print("=" * 50)
        
        analysis_results = {
            'patient_info': patient_data,
            'proteinuria_analysis': {},
            'microalbuminuria_analysis': {},
            'creatinine_analysis': {},
            'kidney_function_assessment': {},
            'risk_stratification': {},
            'clinical_recommendations': []
        }
        
        clinical_context = {
            'age': patient_data['age'],
            'sex': patient_data['sex'],
            'diabetes': patient_data.get('diabetes', False),
            'hypertension': patient_data.get('hypertension', False),
            'kidney_disease': patient_data.get('kidney_disease_history', False)
        }
        
        
        if 'protein_creatinine_ratio' in patient_data:
            analysis_results['proteinuria_analysis'] = self.validator.validate_proteinuria(
                patient_data['protein_creatinine_ratio'], clinical_context
            )
        
        
        if 'albumin_creatinine_ratio' in patient_data:
            analysis_results['microalbuminuria_analysis'] = self.validator.validate_microalbuminuria(
                patient_data['albumin_creatinine_ratio'], clinical_context
            )
        
        
        if 'serum_creatinine' in patient_data:
            analysis_results['creatinine_analysis'] = self.validator.validate_serum_creatinine(
                patient_data['serum_creatinine'], patient_data['age'], patient_data['sex']
            )
        
        
        analysis_results['kidney_function_assessment'] = self._assess_kidney_function(analysis_results)
        
        
        analysis_results['risk_stratification'] = self._stratify_risks(analysis_results, clinical_context)
        
        
        analysis_results['clinical_recommendations'] = self._generate_recommendations(analysis_results)
        
        return analysis_results

    def _assess_kidney_function(self, analysis_results: Dict) -> Dict:
        """Comprehensive kidney function assessment"""
        assessment = {}
        
        
        proteinuria = analysis_results.get('proteinuria_analysis', {})
        albuminuria = analysis_results.get('microalbuminuria_analysis', {})
        creatinine = analysis_results.get('creatinine_analysis', {})
        
        
        egfr = creatinine.get('egfr', 0)
        uacr = albuminuria.get('value', 0)
        assessment['ckd_stage'] = self.validator._classify_ckd_stage(egfr, uacr)
        
        
        if proteinuria:
            assessment['proteinuria_class'] = self.validator._classify_proteinuria_kdigo(
                proteinuria.get('value', 0)
            )
        
        
        assessment['kidney_health_score'] = self._calculate_kidney_health_score(egfr, uacr)
        
        return assessment

    def _calculate_kidney_health_score(self, egfr: float, uacr: float) -> float:
        """Calculate composite kidney health score (0-100)"""
        egfr_score = min(egfr / 120 * 50, 50)  # Max 50 points for eGFR
        acr_score = max(50 - (uacr / 100), 0)  # Max 50 points for ACR, decreases with albuminuria
        
        return egfr_score + acr_score

    def _stratify_risks(self, analysis_results: Dict, clinical_context: Dict) -> Dict:
        """Stratify cardiovascular and kidney disease risks"""
        risks = {}
        
        egfr = analysis_results['creatinine_analysis'].get('egfr', 90)
        uacr = analysis_results['microalbuminuria_analysis'].get('value', 0)
        
        
        if egfr < 60 or uacr >= 30:
            risks['ckd_progression_risk'] = 'High'
        elif egfr < 90 or uacr >= 10:
            risks['ckd_progression_risk'] = 'Moderate'
        else:
            risks['ckd_progression_risk'] = 'Low'
        
        
        if uacr >= 30:
            risks['cardiovascular_risk'] = 'High'
        elif uacr >= 10:
            risks['cardiovascular_risk'] = 'Moderate'
        else:
            risks['cardiovascular_risk'] = 'Low'
        
        
        if risks.get('ckd_progression_risk') == 'High' or risks.get('cardiovascular_risk') == 'High':
            risks['overall_risk'] = 'High'
        elif risks.get('ckd_progression_risk') == 'Moderate' or risks.get('cardiovascular_risk') == 'Moderate':
            risks['overall_risk'] = 'Moderate'
        else:
            risks['overall_risk'] = 'Low'
        
        return risks

    def _generate_recommendations(self, analysis_results: Dict) -> List[str]:
        """Generate evidence-based clinical recommendations"""
        recommendations = []
        
        proteinuria = analysis_results.get('proteinuria_analysis', {})
        albuminuria = analysis_results.get('microalbuminuria_analysis', {})
        creatinine = analysis_results.get('creatinine_analysis', {})
        risks = analysis_results.get('risk_stratification', {})
        
        
        if proteinuria.get('status', '').startswith('critical'):
            recommendations.extend(proteinuria.get('actions', []))
        
        if albuminuria.get('status', '').startswith('critical'):
            recommendations.extend(albuminuria.get('actions', []))
        
        if creatinine.get('status', '').startswith('critical'):
            recommendations.extend(creatinine.get('actions', []))
        
        
        if risks.get('ckd_progression_risk') == 'High':
            recommendations.extend([
                "Nephrology consultation recommended",
                "Monitor blood pressure strictly (<130/80 mmHg)",
                "ACE-I or ARB therapy consideration",
                "Avoid nephrotoxic medications"
            ])
        
        
        if albuminuria.get('value', 0) >= 30:
            recommendations.extend([
                "Optimize glycemic control (HbA1c <7%)",
                "Annual monitoring of urinary albumin excretion",
                "Cardiovascular risk factor management"
            ])
        
        
        if not recommendations:
            recommendations.append("Routine follow-up as per standard care guidelines")
        
        return recommendations

    def generate_comprehensive_report(self, analysis_results: Dict) -> str:
        """Generate comprehensive urinalysis report"""
        report = f"""
COMPREHENSIVE URINALYSIS AND KIDNEY FUNCTION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=================================================================

PATIENT INFORMATION:
-------------------
Name: {analysis_results['patient_info'].get('name', 'N/A')}
Age: {analysis_results['patient_info']['age']} years
Sex: {analysis_results['patient_info']['sex']}
BMI: {analysis_results['patient_info'].get('bmi', 'N/A')}

KEY FINDINGS:
-------------
"""
        
        
        parameters = [
            ('Protein/Creatinine Ratio', analysis_results.get('proteinuria_analysis', {})),
            ('Albumin/Creatinine Ratio', analysis_results.get('microalbuminuria_analysis', {})),
            ('Serum Creatinine', analysis_results.get('creatinine_analysis', {}))
        ]
        
        for param_name, validation in parameters:
            if validation:
                status_icon = "✓" if validation.get('status') == 'normal' else "⚠️"
                if 'critical' in str(validation.get('status')):
                    status_icon = "🚨"
                
                report += f"{status_icon} {param_name:<25} {validation['value']:>8.2f} "
                
                if validation.get('status') == 'normal':
                    ref_range = validation.get('reference_range', (0, 0))
                    report += f"(Ref: {ref_range[0]}-{ref_range[1]})\n"
                else:
                    report += f"**{validation['status'].upper()}**\n"
        
        
        kidney_assessment = analysis_results.get('kidney_function_assessment', {})
        if kidney_assessment:
            report += f"\nKIDNEY FUNCTION ASSESSMENT:\n"
            report += f"CKD Stage: {kidney_assessment.get('ckd_stage', {}).get('stage', 'N/A')}\n"
            report += f"Kidney Health Score: {kidney_assessment.get('kidney_health_score', 0):.1f}/100\n"
        
        
        risks = analysis_results.get('risk_stratification', {})
        if risks:
            report += f"\nRISK STRATIFICATION:\n"
            report += f"CKD Progression Risk: {risks.get('ckd_progression_risk', 'N/A')}\n"
            report += f"Cardiovascular Risk: {risks.get('cardiovascular_risk', 'N/A')}\n"
            report += f"Overall Risk: {risks.get('overall_risk', 'N/A')}\n"
        
        
        report += f"\nCLINICAL RECOMMENDATIONS:\n"
        for i, recommendation in enumerate(analysis_results['clinical_recommendations'], 1):
            report += f"{i}. {recommendation}\n"
        
        report += "\n" + "="*80
        report += "\nThis report follows KDIGO and ADA clinical practice guidelines"
        report += "\nResults must be interpreted by qualified healthcare professionals"
        report += "\nCritical findings require immediate physician notification"
        
        return report


def demonstrate_urinalysis_analysis():
    """Demonstrate the comprehensive urinalysis analysis system"""
    analyzer = ComprehensiveUrinalysisAnalyzer()
    
    
    patient = {
        'name': 'John Smith',
        'age': 58,
        'sex': 'male',
        'bmi': 28.5,
        'diabetes': True,
        'hypertension': True,
        'kidney_disease_history': False,
        'protein_creatinine_ratio': 280,
        'albumin_creatinine_ratio': 85,
        'serum_creatinine': 1.4,
        'urine_specific_gravity': 1.015,
        'urine_ph': 6.0
    }
    
    print("🧪 COMPREHENSIVE URINALYSIS ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    results = analyzer.analyze_urinalysis(patient)
    report = analyzer.generate_comprehensive_report(results)
    
    print(report)
    
    
    print("\n" + "="*80)
    print("ADDITIONAL TEST CASE: SEVERE NEPHROPATHY")
    print("=" * 80)
    
    patient2 = {
        'name': 'Maria Garcia',
        'age': 72,
        'sex': 'female', 
        'bmi': 31.2,
        'diabetes': True,
        'hypertension': True,
        'kidney_disease_history': True,
        'protein_creatinine_ratio': 3200,
        'albumin_creatinine_ratio': 1800,
        'serum_creatinine': 2.8,
        'urine_specific_gravity': 1.010,
        'urine_ph': 5.5
    }
    
    results2 = analyzer.analyze_urinalysis(patient2)
    report2 = analyzer.generate_comprehensive_report(results2)
    print(report2)

if __name__ == "__main__":
    demonstrate_urinalysis_analysis()