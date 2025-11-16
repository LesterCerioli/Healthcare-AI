import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class GlycemiaDataset(Dataset):
    """Custom Dataset for glycated hemoglobin data with age and sex"""
    
    def __init__(self, features, labels, age_groups):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.age_groups = torch.LongTensor(age_groups)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'label': self.labels[idx],
            'age_group': self.age_groups[idx]
        }

class GlycemiaEvaluator(nn.Module):
    """Neural network for glycated hemoglobin evaluation with age and sex considerations"""
    
    def __init__(self, input_dim, num_age_groups, hidden_dim=64):
        super(GlycemiaEvaluator, self).__init__()
        
        self.age_embedding = nn.Embedding(num_age_groups, 8)
        
        
        self.feature_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        
        self.combined_branch = nn.Sequential(
            nn.Linear((hidden_dim // 2) + 8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Output: risk score or HbA1c prediction
        )
        
        
        self.classifier = nn.Sequential(
            nn.Linear((hidden_dim // 2) + 8, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # 4 risk categories
            nn.Softmax(dim=1)
        )
    
    def forward(self, features, age_group):
        age_embedded = self.age_embedding(age_group)
        features_processed = self.feature_branch(features)
        
        combined = torch.cat([features_processed, age_embedded], dim=1)
        
        regression_output = self.combined_branch(combined)
        classification_output = self.classifier(combined)
        
        return regression_output, classification_output

class GlycemiaAnalysis:
    """Comprehensive glycated hemoglobin analysis system"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder_sex = LabelEncoder()
        self.age_group_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        self.age_groups = {
            'young_adult': (18, 35),
            'middle_age': (36, 50),
            'older_adult': (51, 65),
            'senior': (66, 100)
        }
        
        
        self.reference_ranges = {
            'young_adult': {'normal': (4.0, 5.6), 'prediabetes': (5.7, 6.4), 'diabetes': (6.5, 15.0)},
            'middle_age': {'normal': (4.2, 5.7), 'prediabetes': (5.8, 6.4), 'diabetes': (6.5, 15.0)},
            'older_adult': {'normal': (4.3, 5.8), 'prediabetes': (5.9, 6.5), 'diabetes': (6.6, 15.0)},
            'senior': {'normal': (4.4, 5.9), 'prediabetes': (6.0, 6.5), 'diabetes': (6.6, 15.0)}
        }
    
    def create_sample_data(self, num_samples=1000):
        """Create synthetic glycated hemoglobin data for demonstration"""
        np.random.seed(42)
        
        data = {
            'age': np.random.randint(18, 80, num_samples),
            'sex': np.random.choice(['Male', 'Female'], num_samples),
            'hb_a1c': np.random.normal(5.8, 1.2, num_samples),
            'fasting_glucose': np.random.normal(110, 25, num_samples),
            'postprandial_glucose': np.random.normal(140, 35, num_samples),
            'bmi': np.random.normal(26.5, 4.5, num_samples),
            'family_history': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
            'blood_pressure_systolic': np.random.normal(125, 15, num_samples)
        }
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, df):
        """Preprocess the glycated hemoglobin data"""
        
        df['sex_encoded'] = self.label_encoder_sex.fit_transform(df['sex'])
        
        
        df['age_group'] = pd.cut(df['age'], 
                               bins=[18, 35, 50, 65, 100],
                               labels=['young_adult', 'middle_age', 'older_adult', 'senior'])
        
        df['age_group_encoded'] = self.age_group_encoder.fit_transform(df['age_group'])
        
        
        feature_columns = ['age', 'sex_encoded', 'hb_a1c', 'fasting_glucose', 
                          'postprandial_glucose', 'bmi', 'family_history', 
                          'blood_pressure_systolic']
        
        features = df[feature_columns].values
        labels = df['hb_a1c'].values  # Using HbA1c as target for regression
        age_groups = df['age_group_encoded'].values
        
        
        features_scaled = self.scaler.fit_transform(features)
        
        return features_scaled, labels, age_groups, feature_columns
    
    def train_model(self, df, epochs=100, batch_size=32):
        """Train the glycated hemoglobin evaluation model"""
        features, labels, age_groups, feature_columns = self.preprocess_data(df)
        
        
        X_train, X_test, y_train, y_test, age_train, age_test = train_test_split(
            features, labels, age_groups, test_size=0.2, random_state=42
        )
        
        
        train_dataset = GlycemiaDataset(X_train, y_train, age_train)
        test_dataset = GlycemiaDataset(X_test, y_test, age_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        
        input_dim = X_train.shape[1]
        num_age_groups = len(self.age_groups)
        
        self.model = GlycemiaEvaluator(input_dim, num_age_groups).to(self.device)
        
        
        regression_criterion = nn.MSELoss()
        classification_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        
        train_losses = []
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch in train_loader:
                features_batch = batch['features'].to(self.device)
                labels_batch = batch['label'].to(self.device)
                age_batch = batch['age_group'].to(self.device)
                
                optimizer.zero_grad()
                
                regression_output, classification_output = self.model(features_batch, age_batch)
                                
                reg_loss = regression_criterion(regression_output.squeeze(), labels_batch)
                
                
                risk_categories = self._get_risk_category(labels_batch.cpu().numpy(), 
                                                         age_batch.cpu().numpy())
                risk_categories = torch.LongTensor(risk_categories).to(self.device)
                class_loss = classification_criterion(classification_output, risk_categories)
                                
                total_loss = reg_loss + class_loss
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            train_losses.append(epoch_loss / len(train_loader))
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}')
        
        return train_losses
    
    def _get_risk_category(self, hba1c_values, age_groups):
        """Convert HbA1c values to risk categories based on age-specific ranges"""
        risk_categories = []
        
        for hba1c, age_group_idx in zip(hba1c_values, age_groups):
            age_group = self.age_group_encoder.inverse_transform([age_group_idx])[0]
            ranges = self.reference_ranges[age_group]
            
            if hba1c <= ranges['normal'][1]:
                risk_categories.append(0)  # Normal
            elif hba1c <= ranges['prediabetes'][1]:
                risk_categories.append(1)  # Prediabetes
            elif hba1c <= ranges['diabetes'][1]:
                risk_categories.append(2)  # Diabetes
            else:
                risk_categories.append(3)  # Critical
        
        return risk_categories
    
    def evaluate_patient(self, patient_data):
        """Evaluate a single patient's glycated hemoglobin results"""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        self.model.eval()
        
        
        patient_df = pd.DataFrame([patient_data])
        patient_df['sex_encoded'] = self.label_encoder_sex.transform(patient_df['sex'])
        patient_df['age_group'] = pd.cut(patient_df['age'], 
                                       bins=[18, 35, 50, 65, 100],
                                       labels=['young_adult', 'middle_age', 'older_adult', 'senior'])
        patient_df['age_group_encoded'] = self.age_group_encoder.transform(patient_df['age_group'])
        
        feature_columns = ['age', 'sex_encoded', 'hb_a1c', 'fasting_glucose', 
                          'postprandial_glucose', 'bmi', 'family_history', 
                          'blood_pressure_systolic']
        
        features = patient_df[feature_columns].values
        features_scaled = self.scaler.transform(features)
        age_group = patient_df['age_group_encoded'].values[0]
        
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            age_tensor = torch.LongTensor([age_group]).to(self.device)
            
            regression_output, classification_output = self.model(features_tensor, age_tensor)
            
            predicted_hba1c = regression_output.cpu().numpy()[0][0]
            risk_probabilities = classification_output.cpu().numpy()[0]
            risk_category = np.argmax(risk_probabilities)
        
        
        interpretation = self._interpret_results(patient_data, predicted_hba1c, 
                                               risk_category, age_group)
        
        return {
            'predicted_hba1c': predicted_hba1c,
            'risk_probabilities': risk_probabilities,
            'risk_category': risk_category,
            'interpretation': interpretation
        }
    
    def _interpret_results(self, patient_data, predicted_hba1c, risk_category, age_group):
        """Provide detailed interpretation of results"""
        age = patient_data['age']
        sex = patient_data['sex']
        actual_hba1c = patient_data['hb_a1c']
        
        age_group_name = self.age_group_encoder.inverse_transform([age_group])[0]
        reference_range = self.reference_ranges[age_group_name]
        
        risk_labels = {
            0: 'Normal',
            1: 'Prediabetes',
            2: 'Diabetes',
            3: 'Critical - Immediate medical attention required'
        }
        
        interpretation = f"""
GLYCATED HEMOGLOBIN (HbA1c) EVALUATION REPORT
============================================
Patient Details:
- Age: {age} years ({age_group_name.replace('_', ' ').title()})
- Sex: {sex}
- Measured HbA1c: {actual_hba1c:.1f}%

Age-Specific Reference Ranges for {age_group_name.replace('_', ' ').title()}:
- Normal: {reference_range['normal'][0]:.1f}% - {reference_range['normal'][1]:.1f}%
- Prediabetes: {reference_range['prediabetes'][0]:.1f}% - {reference_range['prediabetes'][1]:.1f}%
- Diabetes: ≥{reference_range['diabetes'][0]:.1f}%

EVALUATION RESULTS:
- Risk Category: {risk_labels[risk_category]}
- Predicted HbA1c: {predicted_hba1c:.1f}%

CLINICAL INTERPRETATION:
"""
        
        if risk_category == 0:
            interpretation += "✓ Glycemic control within normal limits for age group.\n✓ Continue healthy lifestyle maintenance."
        elif risk_category == 1:
            interpretation += "⚠️ Prediabetes range detected.\n✓ Recommend lifestyle modifications.\n✓ Consider repeat testing in 3-6 months.\n✓ Monitor for diabetes prevention."
        elif risk_category == 2:
            interpretation += "🚨 Diabetes range confirmed.\n✓ Requires medical consultation.\n✓ Implement comprehensive diabetes management.\n✓ Regular monitoring essential."
        else:
            interpretation += "🚨 CRITICAL - Poor glycemic control detected.\n✓ Immediate medical attention required.\n✓ Intensive management needed.\n✓ High risk of complications."
        
        
        if age > 65 and risk_category >= 1:
            interpretation += "\n\nSPECIAL CONSIDERATIONS FOR OLDER ADULTS:\n✓ Consider functional status and comorbidities\n✓ Individualize glycemic targets\n✓ Monitor for hypoglycemia"
        
        if sex == 'Female' and age < 50:
            interpretation += "\n\nSPECIAL CONSIDERATIONS FOR WOMEN:\n✓ Consider pregnancy planning if applicable\n✓ Monitor during hormonal changes"
        
        return interpretation

def main():
    """Main function to demonstrate the glycated hemoglobin evaluation system"""
    print("Glycated Hemoglobin (HbA1c) Evaluation System")
    print("=" * 50)
    
    
    analyzer = GlycemiaAnalysis()
    
    
    print("\n1. Generating sample data...")
    df = analyzer.create_sample_data(1000)
    print(f"Generated {len(df)} sample records")
    
    
    print("\n2. Training evaluation model...")
    train_losses = analyzer.train_model(df, epochs=100, batch_size=32)
    print("Model training completed!")
    
    
    print("\n3. Evaluating sample patients...")
    
    
    patient1 = {
        'age': 28,
        'sex': 'Female',
        'hb_a1c': 5.2,
        'fasting_glucose': 95,
        'postprandial_glucose': 120,
        'bmi': 22.5,
        'family_history': 0,
        'blood_pressure_systolic': 118
    }
    
    result1 = analyzer.evaluate_patient(patient1)
    print("\n" + "="*60)
    print("PATIENT 1 EVALUATION:")
    print(result1['interpretation'])
    
    
    patient2 = {
        'age': 45,
        'sex': 'Male',
        'hb_a1c': 7.8,
        'fasting_glucose': 145,
        'postprandial_glucose': 210,
        'bmi': 29.8,
        'family_history': 1,
        'blood_pressure_systolic': 138
    }
    
    result2 = analyzer.evaluate_patient(patient2)
    print("\n" + "="*60)
    print("PATIENT 2 EVALUATION:")
    print(result2['interpretation'])
    
    
    patient3 = {
        'age': 72,
        'sex': 'Female',
        'hb_a1c': 9.2,
        'fasting_glucose': 180,
        'postprandial_glucose': 280,
        'bmi': 31.2,
        'family_history': 1,
        'blood_pressure_systolic': 148
    }
    
    result3 = analyzer.evaluate_patient(patient3)
    print("\n" + "="*60)
    print("PATIENT 3 EVALUATION:")
    print(result3['interpretation'])

if __name__ == "__main__":
    main()