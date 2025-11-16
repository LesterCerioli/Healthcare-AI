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
import json
from datetime import datetime
warnings.filterwarnings('ignore')

class StrictBloodCountDataset(Dataset):
    """Strict validation dataset for complete blood count with quality control"""
    
    def __init__(self, features, labels, patient_info, quality_flags):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.patient_info = patient_info
        self.quality_flags = torch.FloatTensor(quality_flags)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'labels': self.labels[idx],
            'age': torch.FloatTensor([self.patient_info['age'][idx]]),
            'sex': torch.LongTensor([self.patient_info['sex'][idx]]),
            'ethnicity': torch.LongTensor([self.patient_info['ethnicity'][idx]]),
            'quality_flags': self.quality_flags[idx]
        }

class LaboratoryStandardValidator:
    """Strict laboratory standard validator based on CLSI guidelines"""
    
    def __init__(self):
        
        self.reference_ranges = {
            
            'hemoglobin': {
                'cord_blood': {'male': (13.5, 20.0), 'female': (13.5, 20.0)},
                'newborn_1d': {'male': (14.0, 24.0), 'female': (14.0, 24.0)},
                'newborn_1w': {'male': (13.0, 22.0), 'female': (13.0, 22.0)},
                'infant_1m': {'male': (10.0, 18.0), 'female': (10.0, 18.0)},
                'infant_2m': {'male': (9.0, 14.0), 'female': (9.0, 14.0)},
                'child_6m': {'male': (10.5, 13.5), 'female': (10.5, 13.5)},
                'child_1y': {'male': (11.0, 13.5), 'female': (11.0, 13.5)},
                'child_2y': {'male': (11.5, 13.5), 'female': (11.5, 13.5)},
                'child_6y': {'male': (11.5, 15.5), 'female': (11.5, 15.5)},
                'child_12y': {'male': (12.0, 16.0), 'female': (12.0, 16.0)},
                'adult_18y': {'male': (13.5, 17.5), 'female': (12.0, 15.5)},
                'adult_50y': {'male': (13.0, 17.0), 'female': (11.5, 15.0)},
                'elderly_65y': {'male': (12.0, 16.0), 'female': (11.5, 15.0)},
                'elderly_80y': {'male': (11.5, 15.5), 'female': (11.0, 14.5)},
                'pregnancy_1t': {'female': (11.0, 14.0)},
                'pregnancy_2t': {'female': (10.5, 13.5)},
                'pregnancy_3t': {'female': (10.0, 13.0)}
            },
            
            'hematocrit': {
                'cord_blood': {'male': (42, 65), 'female': (42, 65)},
                'newborn_1d': {'male': (44, 72), 'female': (44, 72)},
                'infant_1m': {'male': (31, 55), 'female': (31, 55)},
                'child_1y': {'male': (33, 41), 'female': (33, 41)},
                'child_6y': {'male': (35, 45), 'female': (35, 45)},
                'adolescent_12y': {'male': (36, 46), 'female': (36, 46)},
                'adult': {'male': (41, 53), 'female': (36, 46)},
                'elderly_65y': {'male': (38, 50), 'female': (35, 46)},
                'pregnancy': {'female': (33, 44)}
            },
            
            
            'rbc': {
                'cord_blood': {'male': (3.9, 5.5), 'female': (3.9, 5.5)},
                'newborn_1d': {'male': (4.1, 6.1), 'female': (4.1, 6.1)},
                'infant_1m': {'male': (3.0, 5.4), 'female': (3.0, 5.4)},
                'child_1y': {'male': (3.8, 5.2), 'female': (3.8, 5.2)},
                'child_6y': {'male': (4.0, 5.4), 'female': (4.0, 5.4)},
                'adolescent_12y': {'male': (4.1, 5.5), 'female': (4.1, 5.5)},
                'adult': {'male': (4.5, 6.0), 'female': (4.0, 5.2)},
                'elderly_65y': {'male': (4.0, 5.5), 'female': (3.8, 5.0)},
                'pregnancy': {'female': (3.5, 4.8)}
            },
            
            
            'wbc': {
                'cord_blood': {'male': (9.0, 30.0), 'female': (9.0, 30.0)},
                'newborn_1d': {'male': (9.4, 34.0), 'female': (9.4, 34.0)},
                'newborn_1w': {'male': (5.0, 21.0), 'female': (5.0, 21.0)},
                'infant_1m': {'male': (5.0, 19.5), 'female': (5.0, 19.5)},
                'child_6m': {'male': (6.0, 17.5), 'female': (6.0, 17.5)},
                'child_1y': {'male': (6.0, 17.5), 'female': (6.0, 17.5)},
                'child_2y': {'male': (6.0, 17.0), 'female': (6.0, 17.0)},
                'child_6y': {'male': (5.0, 14.5), 'female': (5.0, 14.5)},
                'child_12y': {'male': (4.5, 13.5), 'female': (4.5, 13.5)},
                'adult': {'male': (4.5, 11.0), 'female': (4.5, 11.0)},
                'elderly_70y': {'male': (3.0, 9.0), 'female': (3.0, 9.0)}
            },
            
            
            'platelets': {
                'all': {'male': (150, 450), 'female': (150, 450)},
                'newborn': {'male': (150, 450), 'female': (150, 450)},
                'pregnancy': {'female': (150, 400)}
            },
            
            
            'mcv': {
                'cord_blood': {'male': (95, 121), 'female': (95, 121)},
                'newborn_1d': {'male': (100, 125), 'female': (100, 125)},
                'infant_1m': {'male': (85, 110), 'female': (85, 110)},
                'child_1y': {'male': (70, 86), 'female': (70, 86)},
                'child_6y': {'male': (75, 87), 'female': (75, 87)},
                'adult': {'male': (80, 100), 'female': (80, 100)},
                'elderly': {'male': (80, 100), 'female': (80, 100)}
            },
            
            'mch': {
                'all': {'male': (27, 33), 'female': (27, 33)},
                'newborn': {'male': (31, 37), 'female': (31, 37)}
            },
            
            'mchc': {
                'all': {'male': (32, 36), 'female': (32, 36)}
            },
            
            'rdw': {
                'all': {'male': (11.5, 14.5), 'female': (11.5, 14.5)}
            },
            
            
            'neutrophils': {
                'absolute': {
                    'newborn_1d': {'male': (2.0, 11.0), 'female': (2.0, 11.0)},
                    'child_1y': {'male': (1.0, 8.5), 'female': (1.0, 8.5)},
                    'adult': {'male': (1.8, 7.7), 'female': (1.8, 7.7)},
                    'elderly': {'male': (1.5, 7.0), 'female': (1.5, 7.0)}
                },
                'percentage': {
                    'all': {'male': (40, 75), 'female': (40, 75)}
                }
            },
            
            'lymphocytes': {
                'absolute': {
                    'newborn_1d': {'male': (2.0, 11.0), 'female': (2.0, 11.0)},
                    'child_1y': {'male': (4.0, 10.5), 'female': (4.0, 10.5)},
                    'adult': {'male': (1.0, 4.8), 'female': (1.0, 4.8)},
                    'elderly': {'male': (1.0, 4.0), 'female': (1.0, 4.0)}
                },
                'percentage': {
                    'all': {'male': (20, 50), 'female': (20, 50)}
                }
            },
            
            'monocytes': {
                'absolute': {
                    'all': {'male': (0.1, 0.8), 'female': (0.1, 0.8)}
                },
                'percentage': {
                    'all': {'male': (2, 10), 'female': (2, 10)}
                }
            },
            
            'eosinophils': {
                'absolute': {
                    'all': {'male': (0.0, 0.5), 'female': (0.0, 0.5)}
                },
                'percentage': {
                    'all': {'male': (0, 6), 'female': (0, 6)}
                }
            },
            
            'basophils': {
                'absolute': {
                    'all': {'male': (0.0, 0.2), 'female': (0.0, 0.2)}
                },
                'percentage': {
                    'all': {'male': (0, 2), 'female': (0, 2)}
                }
            }
        }
        
        
        self.critical_values = {
            'hemoglobin': {'low': 7.0, 'high': 22.0, 'immediate_action': True},
            'hematocrit': {'low': 21.0, 'high': 60.0, 'immediate_action': True},
            'wbc': {'low': 2.0, 'high': 30.0, 'immediate_action': True},
            'platelets': {'low': 50.0, 'high': 1000.0, 'immediate_action': True},
            'neutrophils': {'low': 0.5, 'high': 20.0, 'immediate_action': True},
            'rbc': {'low': 2.5, 'high': 7.0, 'immediate_action': False}
        }
        
        
        self.quality_checks = {
            'hgb_hct_ratio': (0.30, 0.36),  # Hgb/Hct ratio
            'mch_mcv_consistency': (0.28, 0.36),  # MCH/MCV ratio
            'rbc_hgb_consistency': (2.8, 3.5)  # Hgb/RBC ratio (pg)
        }
        
        
        self.allowable_error = {
            'hemoglobin': 0.03,  # 3%
            'hematocrit': 0.04,  # 4%
            'wbc': 0.10,        # 10%
            'platelets': 0.15,   # 15%
            'rbc': 0.05,        # 5%
            'mcv': 0.03,        # 3%
            'mch': 0.04,        # 4%
            'mchc': 0.03        # 3%
        }

    def get_precise_age_group(self, age: float, pregnant: bool = False, trimester: int = 0) -> str:
        """Precise age group classification for strict reference ranges"""
        if pregnant:
            if trimester == 1:
                return 'pregnancy_1t'
            elif trimester == 2:
                return 'pregnancy_2t'
            else:
                return 'pregnancy_3t'
        elif age <= 0.004:  # 1 day
            return 'newborn_1d'
        elif age <= 0.019:  # 1 week
            return 'newborn_1w'
        elif age <= 0.083:  # 1 month
            return 'infant_1m'
        elif age <= 0.167:  # 2 months
            return 'infant_2m'
        elif age <= 0.5:    # 6 months
            return 'child_6m'
        elif age <= 1:
            return 'child_1y'
        elif age <= 2:
            return 'child_2y'
        elif age <= 6:
            return 'child_6y'
        elif age <= 12:
            return 'child_12y'
        elif age <= 18:
            return 'adolescent_12y'
        elif age <= 50:
            return 'adult_18y'
        elif age <= 65:
            return 'adult_50y'
        elif age <= 80:
            return 'elderly_65y'
        else:
            return 'elderly_80y'

    def validate_erythrocyte_parameters(self, parameters: Dict, age: float, sex: str) -> Dict:
        """Strict validation of erythrocyte parameters with internal consistency checks"""
        validations = {}
        
        
        for param in ['hemoglobin', 'hematocrit', 'rbc', 'mcv', 'mch', 'mchc', 'rdw']:
            if param in parameters:
                validations[param] = self._validate_single_parameter(
                    param, parameters[param], age, sex, False, 0
                )
        
        
        consistency_checks = self._check_erythrocyte_consistency(parameters)
        validations['consistency_checks'] = consistency_checks
        
        
        derived_checks = self._validate_derived_parameters(parameters)
        validations['derived_parameters'] = derived_checks
        
        return validations

    def validate_leukocyte_parameters(self, parameters: Dict, age: float, sex: str) -> Dict:
        """Strict validation of leukocyte parameters with differential analysis"""
        validations = {}
        
        
        if 'wbc' in parameters:
            validations['wbc'] = self._validate_single_parameter(
                'wbc', parameters['wbc'], age, sex, False, 0
            )
        
        
        differential_params = ['neutrophils', 'lymphocytes', 'monocytes', 'eosinophils', 'basophils']
        total_percentage = 0
        
        for param in differential_params:
            if f"{param}_absolute" in parameters or f"{param}_percentage" in parameters:
                abs_value = parameters.get(f"{param}_absolute")
                perc_value = parameters.get(f"{param}_percentage")
                
                param_validation = self._validate_differential_parameter(
                    param, abs_value, perc_value, age, sex
                )
                validations[param] = param_validation
                
                if perc_value:
                    total_percentage += perc_value
        
        
        if total_percentage > 0:
            percentage_validation = self._validate_total_differential(total_percentage)
            validations['differential_total'] = percentage_validation
        
        return validations

    def _validate_differential_parameter(self, param: str, abs_value: float, 
                                      perc_value: float, age: float, sex: str) -> Dict:
        """Validate differential parameters with both absolute and percentage values"""
        validation = {}
        
        
        if abs_value is not None:
            abs_validation = self._validate_single_parameter(
                param, abs_value, age, sex, False, 0, value_type='absolute'
            )
            validation['absolute'] = abs_validation
        
        
        if perc_value is not None:
            perc_validation = self._validate_single_parameter(
                param, perc_value, age, sex, False, 0, value_type='percentage'
            )
            validation['percentage'] = perc_validation
        
        
        if abs_value is not None and perc_value is not None:
            cross_validation = self._cross_validate_differential(abs_value, perc_value, param)
            validation['cross_validation'] = cross_validation
        
        return validation

    def _validate_single_parameter(self, param_name: str, value: float, age: float, 
                                 sex: str, pregnant: bool, trimester: int, 
                                 value_type: str = 'standard') -> Dict:
        """Validate single parameter against strict reference ranges"""
        
        age_group = self.get_precise_age_group(age, pregnant, trimester)
        
        if param_name in self.reference_ranges:
            param_ranges = self.reference_ranges[param_name]
            
            
            if isinstance(param_ranges, dict) and 'absolute' in param_ranges:
                if value_type == 'absolute':
                    ranges_dict = param_ranges['absolute']
                else:
                    ranges_dict = param_ranges['percentage']
            else:
                ranges_dict = param_ranges
            
            
            ref_range = None
            for group in [age_group, 'adult', 'all']:
                if group in ranges_dict:
                    ref_range = ranges_dict[group].get(sex.lower(), ranges_dict[group]['male'])
                    break
            
            if ref_range is None:
                return {'status': 'unknown_range', 'message': f'No reference range for {param_name}'}
            
            
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
                    'clinical_significance': self._get_clinical_significance(param_name, 'low', value),
                    'suggested_actions': self._get_suggested_actions(param_name, 'low', severity)
                }
            elif value > high:
                severity = self._assess_severity(value, low, high, 'high')
                return {
                    'status': 'high',
                    'severity': severity,
                    'value': value,
                    'reference_range': ref_range,
                    'deviation_percentage': deviation,
                    'clinical_significance': self._get_clinical_significance(param_name, 'high', value),
                    'suggested_actions': self._get_suggested_actions(param_name, 'high', severity)
                }
            else:
                return {
                    'status': 'normal',
                    'value': value,
                    'reference_range': ref_range,
                    'deviation_percentage': deviation
                }
        
        return {'status': 'parameter_not_found'}

    def _check_erythrocyte_consistency(self, parameters: Dict) -> Dict:
        """Check internal consistency of erythrocyte parameters"""
        checks = {}
        
        
        if 'hemoglobin' in parameters and 'hematocrit' in parameters:
            hgb_hct_ratio = parameters['hemoglobin'] / parameters['hematocrit']
            expected_range = self.quality_checks['hgb_hct_ratio']
            checks['hgb_hct_ratio'] = {
                'value': hgb_hct_ratio,
                'expected_range': expected_range,
                'status': 'normal' if expected_range[0] <= hgb_hct_ratio <= expected_range[1] else 'abnormal'
            }
        
        
        if 'mch' in parameters and 'mcv' in parameters and parameters['mcv'] > 0:
            mch_mcv_ratio = parameters['mch'] / parameters['mcv']
            expected_range = self.quality_checks['mch_mcv_consistency']
            checks['mch_mcv_ratio'] = {
                'value': mch_mcv_ratio,
                'expected_range': expected_range,
                'status': 'normal' if expected_range[0] <= mch_mcv_ratio <= expected_range[1] else 'abnormal'
            }
        
        
        if 'hemoglobin' in parameters and 'rbc' in parameters and parameters['rbc'] > 0:
            calculated_mch = (parameters['hemoglobin'] * 10) / parameters['rbc']
            if 'mch' in parameters:
                mch_difference = abs(calculated_mch - parameters['mch'])
                checks['mch_consistency'] = {
                    'calculated_mch': calculated_mch,
                    'reported_mch': parameters['mch'],
                    'difference': mch_difference,
                    'status': 'consistent' if mch_difference <= 1.0 else 'inconsistent'
                }
        
        return checks

    def _calculate_deviation(self, value: float, low: float, high: float) -> float:
        """Calculate percentage deviation from reference range"""
        if value < low:
            return ((low - value) / low) * 100
        elif value > high:
            return ((value - high) / high) * 100
        else:
            return 0.0

    def _assess_severity(self, value: float, low: float, high: float, direction: str) -> str:
        """Assess severity of abnormality based on deviation"""
        if direction == 'low':
            deviation = ((low - value) / low) * 100
        else:
            deviation = ((value - high) / high) * 100
        
        if deviation <= 10:
            return 'mild'
        elif deviation <= 25:
            return 'moderate'
        elif deviation <= 50:
            return 'severe'
        else:
            return 'critical'

    def _get_clinical_significance(self, param: str, direction: str, value: float) -> str:
        """Provide detailed clinical significance"""
        significance = {
            'rbc_low': "Erythrocytopenia - Consider anemia, hemorrhage, bone marrow failure, nutritional deficiencies",
            'rbc_high': "Erythrocytosis - Consider polycythemia, dehydration, hypoxia, renal disease",
            'wbc_low': "Leukopenia - Consider bone marrow suppression, viral infection, autoimmune disorders, drug effects",
            'wbc_high': "Leukocytosis - Consider infection, inflammation, leukemia, stress response, steroid use",
            'neutrophils_low': "Neutropenia - Increased infection risk, consider bone marrow disorders, viral infections",
            'neutrophils_high': "Neutrophilia - Suggestive of bacterial infection, inflammation, stress, myeloproliferative disorders"
        }
        
        return significance.get(f"{param}_{direction}", "Clinical correlation required")

    def _get_suggested_actions(self, param: str, direction: str, severity: str) -> List[str]:
        """Provide specific suggested actions based on findings"""
        actions = {
            ('rbc', 'low', 'severe'): [
                "Immediate hematology consultation",
                "Consider blood transfusion evaluation",
                "Check reticulocyte count",
                "Evaluate for occult bleeding",
                "Iron studies, B12, folate levels"
            ],
            ('wbc', 'low', 'critical'): [
                "IMMEDIATE physician notification",
                "Reverse isolation precautions",
                "Infection workup",
                "Bone marrow evaluation",
                "Avoid live vaccines"
            ],
            ('neutrophils', 'low', 'severe'): [
                "Infection risk assessment",
                "Consider G-CSF therapy",
                "Monitor for febrile neutropenia",
                "Hematology consultation"
            ]
        }
        
        return actions.get((param, direction, severity), ["Clinical correlation and follow-up as indicated"])


class StrictCBCNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, num_parameters: int, hidden_dims: List[int] = [512, 256, 128]):
        super(StrictCBCNeuralNetwork, self).__init__()
        
        self.num_parameters = num_parameters
        
        
        self.age_embedding = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        self.sex_embedding = nn.Embedding(2, 8)
        self.ethnicity_embedding = nn.Embedding(5, 8)
        
        
        layers = []
        prev_dim = input_dim + 24  # Features + embeddings
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(hidden_dim, hidden_dim),  # Residual block
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.feature_processor = nn.Sequential(*layers)
        
        
        self.erythrocyte_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7)  # Hgb, Hct, RBC, MCV, MCH, MCHC, RDW
        )
        
        self.leukocyte_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)  # WBC + 5 differentials
        )
        
        
        self.quality_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5),  # Quality flags
            nn.Sigmoid()
        )
        
        
        self.anomaly_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_parameters * 4),  # 4 classes: normal, mild, moderate, severe/critical
            nn.Softmax(dim=1)
        )

    def forward(self, features, age, sex, ethnicity):
        
        age_embedded = self.age_embedding(age.unsqueeze(1))
        sex_embedded = self.sex_embedding(sex.squeeze())
        ethnicity_embedded = self.ethnicity_embedding(ethnicity.squeeze())
        
        
        combined = torch.cat([features, age_embedded, sex_embedded, ethnicity_embedded], dim=1)
        
        
        processed = self.feature_processor(combined)
        
        
        erythrocyte_pred = self.erythrocyte_head(processed)
        leukocyte_pred = self.leukocyte_head(processed)
        quality_flags = self.quality_head(processed)
        anomaly_pred = self.anomaly_head(processed)
        anomaly_pred = anomaly_pred.view(-1, self.num_parameters, 4)
        
        return {
            'erythrocyte': erythrocyte_pred,
            'leukocyte': leukocyte_pred,
            'quality_flags': quality_flags,
            'anomaly': anomaly_pred
        }


class EnhancedCBCAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.validator = LaboratoryStandardValidator()
        self.parameter_names = [
            'hemoglobin', 'hematocrit', 'rbc', 'wbc', 'platelets',
            'mcv', 'mch', 'mchc', 'rdw', 'neutrophils_absolute',
            'lymphocytes_absolute', 'monocytes_absolute', 
            'eosinophils_absolute', 'basophils_absolute'
        ]
        
        self.sex_encoder = LabelEncoder()
        self.ethnicity_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def analyze_complete_blood_count(self, patient_data: Dict) -> Dict:
        """Comprehensive CBC analysis with strict validation"""
        print(f"\n🔬 STRICT CBC ANALYSIS for {patient_data.get('name', 'patient')}")
        print("=" * 60)
        
        analysis_results = {
            'patient_info': patient_data,
            'erythrocyte_analysis': {},
            'leukocyte_analysis': {},
            'quality_assessment': {},
            'critical_findings': [],
            'comprehensive_interpretation': '',
            'recommendations': []
        }
        
        
        erythrocyte_params = {k: v for k, v in patient_data.items() 
                            if k in ['hemoglobin', 'hematocrit', 'rbc', 'mcv', 'mch', 'mchc', 'rdw']}
        
        if erythrocyte_params:
            analysis_results['erythrocyte_analysis'] = self.validator.validate_erythrocyte_parameters(
                erythrocyte_params, patient_data['age'], patient_data['sex']
            )
        
        
        leukocyte_params = {k: v for k, v in patient_data.items() 
                          if k in ['wbc', 'neutrophils_absolute', 'lymphocytes_absolute', 
                                 'monocytes_absolute', 'eosinophils_absolute', 'basophils_absolute']}
        
        if leukocyte_params:
            analysis_results['leukocyte_analysis'] = self.validator.validate_leukocyte_parameters(
                leukocyte_params, patient_data['age'], patient_data['sex']
            )
        
        
        analysis_results.update(self._generate_comprehensive_report(analysis_results))
        
        return analysis_results

    def _generate_comprehensive_report(self, analysis_results: Dict) -> Dict:
        """Generate detailed clinical report"""
        report_sections = []
        
        
        erythrocyte_findings = self._summarize_erythrocyte_findings(analysis_results['erythrocyte_analysis'])
        if erythrocyte_findings:
            report_sections.append("ERYTHROCYTE SERIES ANALYSIS:\n" + erythrocyte_findings)
        
        
        leukocyte_findings = self._summarize_leukocyte_findings(analysis_results['leukocyte_analysis'])
        if leukocyte_findings:
            report_sections.append("LEUKOCYTE SERIES ANALYSIS:\n" + leukocyte_findings)
        
        
        quality_assessment = self._assess_quality(analysis_results)
        if quality_assessment:
            report_sections.append("QUALITY ASSESSMENT:\n" + quality_assessment)
        
        comprehensive_interpretation = "\n".join(report_sections)
        recommendations = self._generate_strict_recommendations(analysis_results)
        
        return {
            'comprehensive_interpretation': comprehensive_interpretation,
            'recommendations': recommendations
        }

    def _summarize_erythrocyte_findings(self, erythrocyte_analysis: Dict) -> str:
        """Summarize erythrocyte series findings"""
        if not erythrocyte_analysis:
            return "No erythrocyte data available."
        
        summary = []
        abnormal_count = 0
        
        for param, validation in erythrocyte_analysis.items():
            if param == 'consistency_checks':
                continue
                
            if validation.get('status') != 'normal':
                abnormal_count += 1
                summary.append(f"⚠️ {param.upper()}: {validation['value']} ({validation['status'].upper()})")
                summary.append(f"   Deviation: {validation.get('deviation_percentage', 0):.1f}%")
                summary.append(f"   Significance: {validation.get('clinical_significance', '')}")
        
        if abnormal_count == 0:
            return "✓ All erythrocyte parameters within strict reference ranges."
        
        return "\n".join(summary)

    def generate_strict_report(self, analysis_results: Dict) -> str:
        """Generate laboratory-style strict report"""
        report = f"""
COMPLETE BLOOD COUNT - STRICT LABORATORY ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=================================================================

PATIENT DEMOGRAPHICS:
--------------------
Name: {analysis_results['patient_info'].get('name', 'N/A')}
Age: {analysis_results['patient_info']['age']} years
Sex: {analysis_results['patient_info']['sex']}
Ethnicity: {analysis_results['patient_info'].get('ethnicity', 'N/A')}

RESULTS:
--------
"""
        
        
        all_params = {**analysis_results.get('erythrocyte_analysis', {}), 
                     **analysis_results.get('leukocyte_analysis', {})}
        
        for param, validation in all_params.items():
            if param in ['consistency_checks', 'derived_parameters']:
                continue
                
            status_icon = "✓" if validation.get('status') == 'normal' else "⚠️" 
            if 'critical' in str(validation.get('status')):
                status_icon = "🚨"
                
            report += f"{status_icon} {param.upper():<20} {validation['value']:>8.2f} "
            
            if validation.get('status') == 'normal':
                ref_range = validation.get('reference_range', (0, 0))
                report += f"(Ref: {ref_range[0]}-{ref_range[1]})\n"
            else:
                report += f"**{validation['status'].upper()}**\n"
        
        report += f"\nINTERPRETATION:\n{analysis_results['comprehensive_interpretation']}"
        
        report += "\n\nCRITICAL ACTION ITEMS:\n"
        for i, recommendation in enumerate(analysis_results['recommendations'], 1):
            report += f"{i}. {recommendation}\n"
        
        report += "\n" + "="*80
        report += "\nThis report follows strict laboratory standards (CLSI guidelines)"
        report += "\nAll critical values require immediate physician notification"
        report += "\nResults must be interpreted by qualified medical personnel"
        
        return report


def demonstrate_strict_analysis():
    """Demonstrate the strict CBC analysis system"""
    analyzer = EnhancedCBCAnalyzer()
    
    
    patient = {
        'name': 'TEST PATIENT',
        'age': 45,
        'sex': 'female',
        'ethnicity': 'caucasian',
        'hemoglobin': 6.8,      # Critically low
        'hematocrit': 21.0,     # Critically low
        'rbc': 3.1,            # Low
        'wbc': 2.8,            # Leukopenia
        'platelets': 185,
        'mcv': 68,             # Microcytic
        'mch': 20,             # Hypochromic
        'mchc': 30,
        'rdw': 18.5,           # Anisocytosis
        'neutrophils_absolute': 1.2,  # Neutropenia
        'lymphocytes_absolute': 1.4,
        'monocytes_absolute': 0.1,
        'eosinophils_absolute': 0.05,
        'basophils_absolute': 0.02
    }
    
    results = analyzer.analyze_complete_blood_count(patient)
    report = analyzer.generate_strict_report(results)
    
    print(report)

if __name__ == "__main__":
    demonstrate_strict_analysis()