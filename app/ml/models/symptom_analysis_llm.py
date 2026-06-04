"""
Enhanced general symptom analysis LLM.

Provides a broad first-pass symptom triage across 15 disease categories
and routes to specialized LLMs (DiabetesLLM, CardiovascularLLM) when the
primary suspicion warrants a deeper specialist workup.

Input features (40 total):
  General symptoms [0–24]: 25 common symptoms across organ systems
  Patient context [25–39]: 15 demographic and risk-factor features

All values normalized to [0.0, 1.0].
"""

from typing import Dict, Any, List, Optional
from .llm_base import BaseMedicalLLM


class SymptomAnalysisLLM(BaseMedicalLLM):
    """
    Broad-spectrum symptom analysis LLM for initial clinical triage.

    Identifies the most likely disease category from 15 possibilities and
    signals when referral to a specialized LLM is warranted.
    """

    VERSION = "1.0.0"

    # Conditions that should trigger specialized-model referral
    SPECIALIST_REFERRAL_MAP = {
        "Diabetes Mellitus": "DiabetesLLM",
        "Cardiovascular Disease": "CardiovascularLLM",
    }

    def __init__(self):
        super().__init__(
            input_size=40,
            d_model=128,
            num_heads=8,
            num_layers=4,
            num_classes=15,
            dropout=0.15,
        )

    def _initialize_clinical_knowledge(self) -> None:
        # ── General symptoms (index 0–24) ────────────────────────────────────
        self.symptom_mapping = {
            # Constitutional
            "fever": 0,
            "fatigue": 1,
            "unexplained_weight_loss": 2,
            "unexplained_weight_gain": 3,
            "night_sweats": 4,
            # Respiratory
            "cough": 5,
            "shortness_of_breath": 6,
            "wheezing": 7,
            "chest_pain": 8,
            # Neurological
            "headache": 9,
            "dizziness": 10,
            "numbness_tingling": 11,
            "memory_confusion": 12,
            "vision_changes": 13,
            # Gastrointestinal
            "nausea_vomiting": 14,
            "abdominal_pain": 15,
            "diarrhoea": 16,
            "constipation": 17,
            # Musculoskeletal
            "joint_pain": 18,
            "muscle_pain": 19,
            "back_pain": 20,
            # Metabolic / Endocrine
            "excessive_thirst": 21,
            "excessive_urination": 22,
            "increased_appetite": 23,
            "heat_cold_intolerance": 24,
        }

        # ── Patient context features (index 25–39) ───────────────────────────
        # These are set via the patient_context field of the input dict.
        # They are appended after symptom + lab features during preprocessing.
        self.lab_marker_mapping = {}  # General model has no mandatory labs

        # ── Disease categories (15 classes) ─────────────────────────────────
        self.condition_mapping = {
            0: "Common Infectious Disease (Cold / Flu / COVID-19)",
            1: "Respiratory Condition (Asthma / COPD / Pneumonia)",
            2: "Cardiovascular Disease",
            3: "Diabetes Mellitus",
            4: "Thyroid Disorder",
            5: "Neurological Condition (Migraine / Neuropathy)",
            6: "Gastrointestinal Disorder",
            7: "Musculoskeletal Condition (Arthritis / Fibromyalgia)",
            8: "Autoimmune / Rheumatological Disease",
            9: "Mental Health Condition (Anxiety / Depression)",
            10: "Oncological Concern (Requires specialist evaluation)",
            11: "Renal / Urological Disorder",
            12: "Anaemia / Haematological Disorder",
            13: "Dermatological Condition",
            14: "Non-specific / Multisystem — Further Evaluation Required",
        }

        # ── Recommended first-line workup per category ───────────────────────
        self.recommended_exams = {
            "Common Infectious Disease (Cold / Flu / COVID-19)": [
                "COVID-19 rapid antigen test or PCR",
                "Influenza rapid antigen test",
                "Complete blood count with differential (leucocytosis pattern)",
                "C-reactive protein (CRP) and ESR",
                "Throat swab culture if bacterial infection suspected",
                "Chest X-ray if lower respiratory involvement",
            ],
            "Respiratory Condition (Asthma / COPD / Pneumonia)": [
                "Spirometry (FEV1/FVC ratio for obstruction pattern)",
                "Peak expiratory flow rate (PEFR)",
                "Chest X-ray",
                "High-resolution CT chest if indicated",
                "Pulse oximetry",
                "Sputum culture and sensitivity",
                "Allergy panel (IgE, specific allergens) for asthma",
                "ABG (arterial blood gas) if severe dyspnoea",
            ],
            "Cardiovascular Disease": [
                "12-lead ECG",
                "Echocardiogram",
                "Lipid panel (LDL, HDL, total cholesterol, triglycerides)",
                "High-sensitivity troponin",
                "BNP / NT-proBNP",
                "Blood pressure monitoring (24-hour ABPM)",
                "→ Refer to CardiovascularLLM for detailed workup",
            ],
            "Diabetes Mellitus": [
                "Fasting plasma glucose",
                "HbA1c",
                "Oral Glucose Tolerance Test (OGTT)",
                "Urine albumin-to-creatinine ratio",
                "→ Refer to DiabetesLLM for comprehensive assessment",
            ],
            "Thyroid Disorder": [
                "TSH (thyroid-stimulating hormone)",
                "Free T4 (fT4) and Free T3 (fT3)",
                "Thyroid peroxidase antibodies (TPOAb)",
                "Thyroglobulin antibodies (TgAb)",
                "Thyroid ultrasound if nodule suspected",
                "Radioiodine uptake scan if hyperthyroidism",
            ],
            "Neurological Condition (Migraine / Neuropathy)": [
                "Neurological examination",
                "CT head (to exclude intracranial pathology)",
                "MRI brain (preferred if focal signs or non-migraine headache)",
                "Nerve conduction study (NCS) and EMG for neuropathy",
                "Lumbar puncture if meningitis / SAH suspected",
                "Blood glucose, HbA1c (diabetic neuropathy screen)",
                "Vitamin B12 and folate",
            ],
            "Gastrointestinal Disorder": [
                "Full blood count, liver function tests",
                "Helicobacter pylori stool antigen or urea breath test",
                "Upper GI endoscopy (OGD) if peptic symptoms",
                "Colonoscopy if lower GI bleeding or change in bowel habit",
                "Abdominal ultrasound",
                "CT abdomen/pelvis if mass or organ pathology suspected",
                "Faecal calprotectin (IBD screening)",
                "Coeliac screen: anti-tTG IgA",
            ],
            "Musculoskeletal Condition (Arthritis / Fibromyalgia)": [
                "X-ray of affected joints",
                "CRP and ESR (inflammatory markers)",
                "Rheumatoid factor (RF) and anti-CCP antibodies",
                "Uric acid (gout screen)",
                "ANA, ANCA, anti-dsDNA (autoimmune screen if indicated)",
                "MRI of joints if soft tissue pathology suspected",
                "Bone density scan (DEXA) if osteoporosis risk",
            ],
            "Autoimmune / Rheumatological Disease": [
                "ANA, anti-dsDNA, anti-Sm (SLE)",
                "ANCA (vasculitis)",
                "Complement levels (C3, C4)",
                "ESR, CRP",
                "Complete blood count with differential",
                "Urinalysis with microscopy",
                "Skin or kidney biopsy if clinically indicated",
            ],
            "Mental Health Condition (Anxiety / Depression)": [
                "PHQ-9 (depression screening)",
                "GAD-7 (anxiety screening)",
                "TSH (thyroid cause exclusion)",
                "Full blood count (anaemia exclusion)",
                "Cortisol level if Cushing's suspected",
                "Substance use screening",
                "Referral to psychiatry or psychology as appropriate",
            ],
            "Oncological Concern (Requires specialist evaluation)": [
                "URGENT: Refer to oncology specialist",
                "Full blood count with differential",
                "Comprehensive metabolic panel",
                "CT chest/abdomen/pelvis (staging)",
                "PET-CT if indicated",
                "Tumour markers as guided by suspected primary (PSA, CEA, CA-125, etc.)",
                "Tissue biopsy for definitive diagnosis",
                "Bone marrow biopsy if haematological malignancy suspected",
            ],
            "Renal / Urological Disorder": [
                "Urinalysis with microscopy and culture",
                "Serum creatinine and eGFR",
                "Urine albumin-to-creatinine ratio",
                "Renal ultrasound",
                "24-hour urine protein collection",
                "Electrolyte panel",
                "Cystoscopy if haematuria or bladder pathology suspected",
            ],
            "Anaemia / Haematological Disorder": [
                "Complete blood count with reticulocyte count",
                "Iron studies: serum iron, TIBC, ferritin",
                "Vitamin B12 and folate",
                "Peripheral blood film",
                "Bone marrow biopsy if aplastic anaemia or malignancy suspected",
                "Haemoglobin electrophoresis (haemoglobinopathy screen)",
                "LDH and haptoglobin (haemolysis screen)",
            ],
            "Dermatological Condition": [
                "Dermatological clinical examination",
                "Skin biopsy if malignancy or unclear diagnosis",
                "Patch testing (allergic contact dermatitis)",
                "KOH preparation (fungal infection)",
                "Autoimmune panel if bullous disease suspected",
            ],
            "Non-specific / Multisystem — Further Evaluation Required": [
                "Comprehensive metabolic panel",
                "Complete blood count with differential",
                "CRP, ESR, ferritin (systemic inflammation)",
                "Urinalysis",
                "TSH",
                "Clinical reassessment in 2–4 weeks",
                "Refer to appropriate specialist based on dominant symptoms",
            ],
        }

        # ── Treatment guidelines per category ────────────────────────────────
        self.treatment_guidelines = {
            "Common Infectious Disease (Cold / Flu / COVID-19)": [
                "Supportive care: rest, adequate hydration, antipyretics (paracetamol/ibuprofen)",
                "Antivirals: oseltamivir (Tamiflu) for confirmed influenza within 48h",
                "COVID-19 antivirals (nirmatrelvir/ritonavir) for high-risk patients",
                "Antibiotics only for confirmed bacterial co-infection",
                "Isolation precautions per local public health guidelines",
            ],
            "Respiratory Condition (Asthma / COPD / Pneumonia)": [
                "Asthma: ICS (budesonide/fluticasone) + SABA (salbutamol) reliever",
                "Severe asthma: add LABA (salmeterol/formoterol) or LTRA",
                "COPD: LAMA (tiotropium) or LABA + ICS if exacerbations",
                "Pneumonia: antibiotics per local guidelines (amoxicillin or broader spectrum)",
                "Bronchodilators for acute dyspnoea",
                "Oxygen therapy if SpO2 <94%",
                "Pulmonary rehabilitation for chronic disease",
            ],
            "Cardiovascular Disease": [
                "→ See CardiovascularLLM for detailed treatment suggestions",
                "Lifestyle: Mediterranean diet, regular exercise, smoking cessation",
                "Blood pressure and lipid management",
            ],
            "Diabetes Mellitus": [
                "→ See DiabetesLLM for detailed treatment suggestions",
                "Blood glucose monitoring and dietary counselling",
            ],
            "Thyroid Disorder": [
                "Hypothyroidism: levothyroxine, titrate to TSH target 0.5–2.5 mIU/L",
                "Hyperthyroidism: carbimazole or propylthiouracil (anti-thyroid drugs)",
                "Radioiodine therapy for Graves' disease if indicated",
                "Thyroid surgery for goitre, malignancy, or failed medical therapy",
                "Beta-blocker (propranolol) for symptom relief in thyrotoxicosis",
            ],
            "Neurological Condition (Migraine / Neuropathy)": [
                "Migraine acute: NSAIDs + triptan (sumatriptan) combination",
                "Migraine prophylaxis: topiramate, propranolol, amitriptyline, CGRP antagonists",
                "Neuropathic pain: pregabalin, gabapentin, duloxetine, amitriptyline",
                "Vitamin B12 supplementation for deficiency neuropathy",
                "Tight glycaemic control for diabetic neuropathy",
            ],
            "Gastrointestinal Disorder": [
                "H. pylori eradication: triple therapy (PPI + clarithromycin + amoxicillin)",
                "GORD: PPI (omeprazole/pantoprazole), lifestyle modifications",
                "IBD: mesalazine (mild UC), biologics (infliximab/adalimumab) for moderate-severe",
                "IBS: dietary modification, antispasmodics, low-FODMAP diet",
                "Constipation: dietary fibre, osmotic laxatives",
            ],
            "Musculoskeletal Condition (Arthritis / Fibromyalgia)": [
                "OA: paracetamol, topical NSAIDs, physiotherapy, weight loss",
                "RA: DMARDs (methotrexate first-line), biologics if refractory",
                "Gout: allopurinol (urate-lowering), colchicine for acute attacks",
                "Fibromyalgia: low-dose amitriptyline, pregabalin, aerobic exercise",
                "Physiotherapy and occupational therapy for functional rehabilitation",
            ],
            "Autoimmune / Rheumatological Disease": [
                "SLE: hydroxychloroquine as baseline; corticosteroids for flares",
                "Vasculitis: high-dose corticosteroids + rituximab or cyclophosphamide",
                "Referral to rheumatology for specialist-guided immunosuppression",
                "Sun protection and lifestyle modification for photosensitive conditions",
            ],
            "Mental Health Condition (Anxiety / Depression)": [
                "Depression: SSRI (sertraline/escitalopram) as first-line",
                "Anxiety: SSRI/SNRI; consider CBT psychotherapy",
                "SNRI (venlafaxine/duloxetine) for comorbid anxiety and depression",
                "Referral to psychiatry for moderate-severe or treatment-resistant cases",
                "Psychotherapy (CBT, ACT) as adjunct or monotherapy for mild-moderate",
                "Safety assessment and crisis plan for suicidal ideation",
            ],
            "Oncological Concern (Requires specialist evaluation)": [
                "URGENT: Oncology referral — do not delay",
                "Multidisciplinary team (MDT) review for treatment planning",
                "Treatment depends on tumour type, stage, and patient fitness",
                "Supportive care and palliative input as appropriate",
            ],
            "Renal / Urological Disorder": [
                "CKD: ACE inhibitor/ARB, blood pressure control <130/80, "
                "dietary protein/potassium restriction",
                "UTI: trimethoprim or nitrofurantoin (culture-guided)",
                "Nephrolithiasis: hydration, analgesia; urology if obstruction",
                "Referral to nephrology for eGFR <30 or rapid decline",
            ],
            "Anaemia / Haematological Disorder": [
                "Iron deficiency: oral ferrous sulphate 200 mg 2–3 times daily; "
                "IV iron if malabsorption or intolerance",
                "B12 deficiency: intramuscular hydroxocobalamin",
                "Haemolytic anaemia: corticosteroids for autoimmune type",
                "Referral to haematology for aplastic anaemia or malignancy",
            ],
            "Dermatological Condition": [
                "Eczema: emollients, topical corticosteroids; biologics (dupilumab) for severe",
                "Psoriasis: topical agents (calcipotriol), phototherapy, biologics for severe",
                "Fungal infection: topical antifungal (clotrimazole) or systemic if resistant",
                "Referral to dermatology for biopsy or specialist management",
            ],
            "Non-specific / Multisystem — Further Evaluation Required": [
                "Watchful waiting with safety-netting advice",
                "Treat identified symptoms supportively",
                "Expedited follow-up in 2–4 weeks",
                "Systematic workup to exclude serious pathology",
            ],
        }

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        result = super().predict(input_data)
        if result.get("success"):
            diagnosis = result.get("primary_diagnosis", "")
            referral_model = self.SPECIALIST_REFERRAL_MAP.get(diagnosis)
            if referral_model:
                result["specialist_referral"] = {
                    "recommended_model": referral_model,
                    "reason": (
                        f"Primary suspicion is {diagnosis}. "
                        f"Pass patient data to {referral_model} for a detailed "
                        f"disease-specific workup and treatment plan."
                    ),
                }
        return result

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "specialization": "General symptom triage across 15 disease categories",
            "version": self.VERSION,
            "clinical_scope": (
                "First-pass symptom analysis, disease category identification, "
                "and routing to specialist LLMs for detailed workup"
            ),
            "specialist_routing": self.SPECIALIST_REFERRAL_MAP,
        })
        return info

    def get_supported_symptoms(self):
        return list(self.symptom_mapping.keys())

    def get_supported_conditions(self):
        return list(self.condition_mapping.values())


def create_symptom_analysis_llm() -> SymptomAnalysisLLM:
    """Factory function — returns a ready-to-use SymptomAnalysisLLM instance."""
    model = SymptomAnalysisLLM()
    model.eval()
    return model
