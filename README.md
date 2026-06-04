
# Medical App AI 🏥🤖

## Overview

AI-powered clinical decision support system built with FastAPI and PyTorch.
The platform provides data-driven diagnostic insights to assist attending physicians — all outputs are intended as clinical decision support only. The physician retains full responsibility for the final diagnosis and treatment plan.

### Key Features

- **🧠 AI Diagnostic Engine**: Machine learning models for symptom analysis and condition prediction
- **⚡ Real-time API**: Fast, scalable REST API for instant medical insights
- **🏥 Multi-Context**: Supports both clinical and corporate health environments
- **🔒 Secure & Compliant**: Built with healthcare data security in mind
- **🔄 CI/CD Ready**: Complete development pipeline from training to deployment

### Technology Stack

- **Backend**: FastAPI, Python 3.12
- **AI/ML**: PyTorch, scikit-learn, pandas
- **Database**: PostgreSQL with async support
- **Deployment**: Docker, cloud-ready architecture
- **Monitoring**: Built-in health checks and metrics

### Use Cases

- Clinical decision support systems
- Corporate employee health screening
- Telemedicine platforms
- Medical education and training
- Health insurance risk assessment

**Empowering healthcare with artificial intelligence - making medical insights accessible, accurate, and actionable.** 🚀

### 🚀 Features

* AI-Powered Diagnostics: Machine learning models for medical symptom analysis
* RESTful API: FastAPI-based endpoints for easy integration
* Multi-Context Support: Medical-hospital and corporate health environments
* Real-time Inference: Low-latency prediction serving
* Model Management: Versioning and registry system
* Training Pipeline: Complete ML lifecycle management
* PostgreSQL Integration: Robust data persistence
* Docker Ready: Containerized deployment

### 🏗️ Architecture

Medical App AI follows a clean architecture with:
- FastAPI for web layer
- PyTorch for AI/ML models  
- PostgreSQL for data storage
- Factory pattern for model management
- Repository pattern for data access
- SOLID principles throughout

### 📁 Project Structure


<img width="314" height="436" alt="image" src="https://github.com/user-attachments/assets/0c8e4e5a-64a0-46e6-ba0b-4377ff50b268" />

<img width="306" height="500" alt="image" src="https://github.com/user-attachments/assets/28a95422-7cb3-40bc-9c77-33af21d2fd10" />


<img width="404" height="294" alt="image" src="https://github.com/user-attachments/assets/710a67da-74ce-451b-8432-876db4b1e59b" />


<img width="339" height="486" alt="image" src="https://github.com/user-attachments/assets/aed1f352-5311-49e5-9f6d-e9b0305d0693" />

---

## 🤖 AI Models — Issue #6: Medical Diagnostic LLM Enhancement

> Resolves [#6 Medical Diagnostic LLM Enhancement](https://github.com/LesterCerioli/Healthcare-AI/issues/6)

### New Specialized LLMs (PyTorch — Attention-based)

Three new specialized transformer-style models have been added under `app/ml/models/`.
Each model uses multi-head self-attention to capture correlations between clinical
features that simple feed-forward networks miss.

---

#### `BaseMedicalLLM` — `app/ml/models/llm_base.py`

Abstract base class for all specialized LLMs. Provides:

| Component | Description |
|---|---|
| `ClinicalAttentionBlock` | Multi-head self-attention for feature correlation |
| `ClinicalFeedForward` | Position-wise FFN with GELU activation |
| `ClinicalTransformerLayer` | Attention + FFN with residuals and LayerNorm |
| `BaseMedicalLLM` | Base class: embedding → N transformer layers → classifier + urgency heads |

All subclasses output:
- `primary_diagnosis` — most probable condition
- `confidence` — model confidence score
- `differential_diagnoses` — top-3 conditions with probabilities
- `recommended_examinations` — disease-specific lab and imaging workup
- `treatment_suggestions` — physician-directed treatment options
- `urgency_level` — Low / Medium / High urgency signal
- `clinical_disclaimer` — reinforces physician decision-making authority

---

#### `DiabetesLLM` — `app/ml/models/diabetes_llm.py`

Specialization: **Diabetes Mellitus** (Type 1, Type 2, Pre-diabetes, Gestational DM, MODY)

**Input features (26):**
- *Symptoms (13)*: polyuria, polydipsia, polyphagia, unexplained weight loss, fatigue, blurred vision, slow healing wounds, frequent infections, numbness/tingling, acanthosis nigricans, fruity breath, abdominal pain, nausea/vomiting
- *Lab markers (8)*: blood glucose, HbA1c, fasting glucose, BMI category, insulin level, C-peptide, triglycerides, HDL cholesterol
- *Patient context (5)*: age group, family history, hypertension, gestational history, physical activity level

**Diagnostic classes (6):**
1. Normal Glucose Regulation
2. Pre-diabetes (Impaired Glucose Tolerance)
3. Type 2 Diabetes Mellitus
4. Type 1 Diabetes Mellitus
5. Gestational Diabetes Mellitus
6. Maturity-Onset Diabetes of the Young (MODY)

**Clinical outputs per class include:**
- HbA1c and OGTT ordering guidance
- Nephrology / ophthalmology / podiatry referral triggers
- First-line pharmacotherapy suggestions (Metformin, SGLT-2 inhibitors, GLP-1 agonists, insulin regimens)
- Lifestyle and diabetes education recommendations

**Helper:** `DiabetesFeatureNormalizer` — converts raw clinical units (mg/dL, %, BMI) to the [0, 1] range.

---

#### `CardiovascularLLM` — `app/ml/models/cardiovascular_llm.py`

Specialization: **Cardiovascular Disease** (Hypertension, CAD, CHF, AF, AMI risk, Angina)

**Input features (27):**
- *Symptoms (11)*: chest pain/pressure, shortness of breath, palpitations, fatigue, swollen legs/ankles, irregular heartbeat, dizziness/fainting, arm-jaw-neck pain, cold sweats, nausea, rapid heartbeat
- *Vitals / Lab markers (10)*: systolic BP, diastolic BP, heart rate, LDL, HDL, triglycerides, troponin, BNP, ECG abnormality, ejection fraction
- *Patient context (6)*: age group, sex, smoker, family history CVD, diabetes present, obesity

**Diagnostic classes (7):**
1. Normal Cardiac Profile
2. Hypertension
3. Coronary Artery Disease (CAD)
4. Congestive Heart Failure (CHF)
5. Atrial Fibrillation (AF)
6. Acute Myocardial Infarction (AMI) Risk
7. Angina Pectoris

**Clinical outputs per class include:**
- Cardiac workup: ECG, echocardiogram, stress test, coronary angiography, CT angiography
- GRACE / CHA₂DS₂-VASc / HAS-BLED score triggers
- Emergency AMI protocol activation guidance
- First-line pharmacotherapy (antiplatelets, statins, beta-blockers, ACE inhibitors, DOACs)

**Helper:** `CardiovascularFeatureNormalizer` — converts raw clinical measurements to [0, 1].

---

#### `SymptomAnalysisLLM` — `app/ml/models/symptom_analysis_llm.py`

Specialization: **General symptom triage** across 15 disease categories with specialist routing.

**Input features (40):**
- *General symptoms (25)*: constitutional, respiratory, neurological, GI, musculoskeletal, metabolic/endocrine
- *Patient context (15)*: demographic and risk-factor features

**Disease categories (15):**
1. Common Infectious Disease (Cold / Flu / COVID-19)
2. Respiratory Condition (Asthma / COPD / Pneumonia)
3. Cardiovascular Disease → routes to `CardiovascularLLM`
4. Diabetes Mellitus → routes to `DiabetesLLM`
5. Thyroid Disorder
6. Neurological Condition (Migraine / Neuropathy)
7. Gastrointestinal Disorder
8. Musculoskeletal Condition (Arthritis / Fibromyalgia)
9. Autoimmune / Rheumatological Disease
10. Mental Health Condition (Anxiety / Depression)
11. Oncological Concern (Requires specialist evaluation)
12. Renal / Urological Disorder
13. Anaemia / Haematological Disorder
14. Dermatological Condition
15. Non-specific / Multisystem — Further Evaluation Required

When the triage LLM identifies Diabetes Mellitus or Cardiovascular Disease as the primary suspicion, it includes a `specialist_referral` block in the response, directing the caller to pass the patient data to the appropriate specialized LLM for deeper analysis.

---

### Dataset — `app/ml/data/diabetes_dataset.py`

Synthetic patient record generator for `DiabetesLLM` training:

- `generate_synthetic_records(n_per_class, seed)` — produces clinically plausible feature vectors for all 6 diabetic condition classes
- `get_diabetes_dataloaders(n_per_class, batch_size, val_split)` — returns `(train_loader, val_loader)` ready for training
- In production: replace the synthetic generator with the PostgreSQL data loader

---

### Model Registry — updated `app/ml/models/model_registry.py`

Three new convenience loaders:

```python
registry = get_model_registry()
diabetes_model     = registry.load_diabetes_llm()
cardio_model       = registry.load_cardiovascular_llm()
symptom_model      = registry.load_symptom_analysis_llm()
```

Optionally pass a `checkpoint_path` to restore trained weights.

---

### Usage Examples

**General symptom triage:**
```python
from app.ml.models import create_model

triage = create_model("symptom_analysis")
result = triage.predict({
    "symptoms": {
        "excessive_thirst": 0.8,
        "excessive_urination": 0.8,
        "fatigue": 0.7,
        "blurred_vision": 0.4,
    }
})
# result["primary_diagnosis"] may be "Diabetes Mellitus"
# result["specialist_referral"]["recommended_model"] == "DiabetesLLM"
```

**Diabetes Mellitus deep analysis:**
```python
from app.ml.models import create_model, DiabetesFeatureNormalizer

normalizer = DiabetesFeatureNormalizer()
raw_data = {
    "symptoms": {"polyuria": 1.0, "polydipsia": 1.0, "fatigue": 0.8},
    "lab_markers": {"blood_glucose": 240.0, "hba1c": 8.5, "fasting_glucose": 160.0},
    "patient_context": {"family_history": 1.0, "bmi_category": 0.5}
}
normalized = normalizer.normalize(raw_data)

model = create_model("diabetes")
result = model.predict(normalized)
# result["primary_diagnosis"]        → "Type 2 Diabetes Mellitus"
# result["recommended_examinations"] → list of specific lab and imaging orders
# result["treatment_suggestions"]    → physician-directed treatment options
# result["clinical_disclaimer"]      → reinforces physician decision authority
```

**Cardiovascular risk assessment:**
```python
from app.ml.models import create_model, CardiovascularFeatureNormalizer

normalizer = CardiovascularFeatureNormalizer()
raw_data = {
    "symptoms": {"chest_pain_pressure": 0.9, "shortness_of_breath": 0.7},
    "lab_markers": {"systolic_bp": 165.0, "troponin": 2.5, "ldl_cholesterol": 180.0},
    "patient_context": {"smoker": 1.0, "family_history_cvd": 1.0}
}
normalized = normalizer.normalize(raw_data)

model = create_model("cardiovascular")
result = model.predict(normalized)
# result["primary_diagnosis"] → e.g. "Acute Myocardial Infarction (AMI) Risk"
# result["urgency_level"]     → "High"
# result["recommended_examinations"] → includes serial troponin, urgent ECG, GRACE score
```

---

### Clinical Disclaimer

> All model outputs are **clinical decision support only**. The attending physician is solely responsible for the final diagnosis and treatment plan. These models are not a substitute for professional medical judgement.

---

### Model Architecture Summary

| Model | Input Size | d_model | Heads | Layers | Parameters | Conditions |
|---|---|---|---|---|---|---|
| `DiabetesLLM` | 26 | 64 | 4 | 3 | ~156 K | 6 |
| `CardiovascularLLM` | 27 | 64 | 4 | 3 | ~156 K | 7 |
| `SymptomAnalysisLLM` | 40 | 128 | 8 | 4 | ~812 K | 15 |
| `MedicalDiagnosticModel` (legacy) | 50 | — | — | 3 | variable | 10 |
| `UltraLightMedicalModel` (legacy) | 30 | — | — | 2 | ~1.5 K | 5 |
