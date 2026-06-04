"""
Cardiovascular disease specialized LLM diagnostic model.

Covers hypertension, coronary artery disease, heart failure,
atrial fibrillation, angina pectoris, and acute MI risk assessment.

Input features (27 total):
  Symptoms [0–10]: chest_pain, shortness_of_breath, palpitations, fatigue,
                   swollen_legs, irregular_heartbeat, dizziness_fainting,
                   arm_jaw_neck_pain, cold_sweats, nausea, rapid_heartbeat
  Vitals/Labs [11–20]: systolic_bp, diastolic_bp, heart_rate, ldl_cholesterol,
                        hdl_cholesterol, triglycerides, troponin, bnp,
                        ecg_abnormality, ejection_fraction
  Patient context [21–26]: age_group, sex_male, smoker, family_history_cvd,
                             diabetes_present, obesity

All continuous values must be normalized to [0.0, 1.0].
See CardiovascularFeatureNormalizer for reference ranges.
"""

from typing import Dict, Any, List
from .llm_base import BaseMedicalLLM


class CardiovascularLLM(BaseMedicalLLM):
    """
    Attention-based LLM for cardiovascular disease diagnosis and clinical support.

    Outputs: primary diagnosis, differential diagnoses, recommended cardiac
    workup examinations, and physician-directed treatment suggestions.
    """

    VERSION = "1.0.0"

    def __init__(self):
        super().__init__(
            input_size=27,
            d_model=64,
            num_heads=4,
            num_layers=3,
            num_classes=7,
            dropout=0.15,
        )

    def _initialize_clinical_knowledge(self) -> None:
        # ── Symptom features (index 0–10) ───────────────────────────────────
        self.symptom_mapping = {
            "chest_pain_pressure": 0,
            "shortness_of_breath": 1,
            "palpitations": 2,
            "fatigue_weakness": 3,
            "swollen_legs_ankles": 4,       # oedema → heart failure indicator
            "irregular_heartbeat": 5,
            "dizziness_fainting": 6,        # syncope
            "arm_jaw_neck_pain": 7,         # referred pain → AMI/angina
            "cold_sweats": 8,
            "nausea": 9,
            "rapid_heartbeat": 10,          # tachycardia
        }

        # ── Vitals and lab marker features (index 11–20) ────────────────────
        self.lab_marker_mapping = {
            "systolic_bp": 11,              # 0→<120, 1→≥180 mmHg
            "diastolic_bp": 12,             # 0→<80, 1→≥120 mmHg
            "heart_rate": 13,               # 0→<60 bpm, 1→≥150 bpm
            "ldl_cholesterol": 14,          # 0→<100, 1→≥200 mg/dL
            "hdl_cholesterol": 15,          # inverted: 0→<40, 1→≥60 mg/dL
            "triglycerides": 16,            # 0→<150, 1→≥500 mg/dL
            "troponin": 17,                 # 0→normal, 1→severely elevated
            "bnp": 18,                      # BNP/NT-proBNP; 0→<100, 1→≥5000 pg/mL
            "ecg_abnormality": 19,          # binary: 0→normal, 1→abnormal
            "ejection_fraction": 20,        # inverted: 0→preserved (≥55%), 1→severely reduced (<30%)
        }

        # ── Diagnostic classes ───────────────────────────────────────────────
        self.condition_mapping = {
            0: "Normal Cardiac Profile",
            1: "Hypertension",
            2: "Coronary Artery Disease (CAD)",
            3: "Congestive Heart Failure (CHF)",
            4: "Atrial Fibrillation (AF)",
            5: "Acute Myocardial Infarction (AMI) Risk",
            6: "Angina Pectoris",
        }

        # ── Recommended examinations per diagnosis ───────────────────────────
        self.recommended_exams = {
            "Normal Cardiac Profile": [
                "Annual blood pressure measurement",
                "Lipid panel every 5 years (or annually if risk factors)",
                "Fasting blood glucose (cardiovascular risk assessment)",
                "BMI and waist circumference",
                "Lifestyle and smoking counselling",
            ],
            "Hypertension": [
                "Ambulatory blood pressure monitoring (ABPM) — 24-hour",
                "Electrocardiogram (ECG) — left ventricular hypertrophy",
                "Echocardiogram if LVH suspected",
                "Comprehensive metabolic panel (renal function, electrolytes)",
                "Urine albumin-to-creatinine ratio",
                "Fundoscopy (hypertensive retinopathy)",
                "Lipid panel",
                "Thyroid function (TSH) to exclude secondary hypertension",
                "Renal artery Doppler ultrasound if renovascular cause suspected",
            ],
            "Coronary Artery Disease (CAD)": [
                "Resting ECG",
                "Exercise stress test (treadmill ECG)",
                "Stress echocardiography or nuclear stress test",
                "Coronary CT Angiography (CCTA) — non-invasive",
                "Invasive coronary angiography (if revascularisation considered)",
                "Echocardiogram (assess LV function, wall motion abnormalities)",
                "Lipid panel (LDL-C, HDL-C, triglycerides, Lp(a))",
                "Cardiac biomarkers: high-sensitivity troponin, CRP",
                "HbA1c and fasting glucose (diabetes as CAD risk factor)",
                "Ankle-brachial index (peripheral artery disease co-assessment)",
            ],
            "Congestive Heart Failure (CHF)": [
                "Echocardiogram (ejection fraction, wall motion, valvular disease)",
                "BNP or NT-proBNP (diagnosis and monitoring)",
                "Chest X-ray (cardiomegaly, pulmonary oedema)",
                "ECG (arrhythmia, LVH, bundle branch block)",
                "Comprehensive metabolic panel (renal function, electrolytes)",
                "Complete blood count (anaemia as precipitating factor)",
                "Thyroid function tests",
                "Cardiac MRI if echo inadequate",
                "Cardiopulmonary exercise test (6-minute walk test)",
                "Coronary angiography if ischaemic aetiology suspected",
            ],
            "Atrial Fibrillation (AF)": [
                "12-lead ECG (diagnostic)",
                "24–72h Holter monitor (paroxysmal AF detection)",
                "Echocardiogram (left atrial size, thrombus, valvular disease)",
                "Thyroid function tests (hyperthyroidism trigger)",
                "Comprehensive metabolic panel",
                "CHA₂DS₂-VASc score calculation (stroke risk)",
                "HAS-BLED score (bleeding risk for anticoagulation)",
                "Sleep apnoea screening (polysomnography if indicated)",
                "Pulmonary function tests (if dyspnoea present)",
            ],
            "Acute Myocardial Infarction (AMI) Risk": [
                "URGENT: Serial high-sensitivity troponin (0h/1h/2h protocol)",
                "12-lead ECG immediately and at 6 hours",
                "Chest X-ray",
                "Complete blood count, coagulation profile",
                "Comprehensive metabolic panel",
                "GRACE risk score calculation",
                "Echocardiogram (wall motion abnormalities, LVEF)",
                "Urgent coronary angiography if STEMI or high-risk NSTEMI",
                "Lipid panel (target LDL-C for statin therapy)",
            ],
            "Angina Pectoris": [
                "Resting ECG",
                "Exercise treadmill test (ETT)",
                "Stress myocardial perfusion imaging (MPI)",
                "Coronary CT Angiography (CCTA)",
                "Echocardiogram",
                "Lipid panel",
                "Fasting glucose and HbA1c",
                "Ankle-brachial index",
            ],
        }

        # ── Treatment guidelines (clinical decision support for physician) ───
        self.treatment_guidelines = {
            "Normal Cardiac Profile": [
                "Lifestyle optimization: Mediterranean diet, physical activity",
                "Smoking cessation and alcohol moderation",
                "Maintain BMI 18.5–24.9",
                "Annual preventive cardiovascular assessment",
            ],
            "Hypertension": [
                "Lifestyle modifications first: DASH diet, sodium restriction (<2g/day), "
                "weight loss, alcohol reduction, 150 min/week aerobic exercise",
                "Stage 1 (130–139/80–89): lifestyle alone for low-risk; "
                "add pharmacotherapy for high-risk",
                "First-line medications: ACE inhibitor (ramipril/enalapril) or ARB, "
                "calcium channel blocker (amlodipine), thiazide diuretic",
                "Add beta-blocker if HF, CAD, or rate control needed",
                "Blood pressure target: <130/80 mmHg (general); "
                "<140/90 mmHg (elderly ≥65 years)",
                "Renal protection: ACE inhibitor/ARB preferred if CKD or diabetes",
                "Monitor renal function and electrolytes at follow-up",
            ],
            "Coronary Artery Disease (CAD)": [
                "Antiplatelet: aspirin 75–100 mg/day indefinitely",
                "High-intensity statin: atorvastatin 40–80 mg or rosuvastatin 20–40 mg "
                "(LDL-C target <55 mg/dL for high-risk CAD)",
                "Beta-blocker (metoprolol/bisoprolol) if LV dysfunction or post-MI",
                "ACE inhibitor/ARB if LV dysfunction, hypertension, or diabetes",
                "Nitrates for symptom relief (sublingual glyceryl trinitrate)",
                "Revascularisation (PCI or CABG) based on angiographic findings "
                "and Heart Team decision",
                "Cardiac rehabilitation program",
                "Lifestyle: Mediterranean diet, ≥150 min/week exercise",
                "Smoking cessation — mandatory",
            ],
            "Congestive Heart Failure (CHF)": [
                "HFrEF (EF <40%): ACE inhibitor/ARB/ARNI (sacubitril/valsartan), "
                "beta-blocker (carvedilol/bisoprolol/metoprolol), MRA (spironolactone), "
                "SGLT-2 inhibitor (dapagliflozin/empagliflozin) — Foundational 4",
                "Diuresis: loop diuretic (furosemide) for volume overload relief",
                "HFpEF (EF ≥50%): SGLT-2 inhibitor; treat underlying hypertension, AF",
                "Fluid restriction (<1.5–2 L/day) and daily weight monitoring",
                "ICD/CRT-D if EF ≤35% despite optimal medical therapy",
                "Heart transplantation evaluation for advanced refractory HF",
                "Sodium restriction <2g/day",
                "Multidisciplinary HF clinic follow-up",
            ],
            "Atrial Fibrillation (AF)": [
                "Rate control: beta-blocker (bisoprolol/metoprolol) or "
                "calcium channel blocker (diltiazem/verapamil); "
                "digoxin in HF with reduced EF",
                "Rhythm control: flecainide, propafenone, amiodarone, sotalol "
                "(physician selection based on structural heart disease)",
                "Anticoagulation: DOAC preferred (apixaban, rivaroxaban, edoxaban, "
                "dabigatran) for CHA₂DS₂-VASc ≥2 men / ≥3 women; "
                "warfarin if mechanical valve or severe CKD",
                "Catheter ablation (pulmonary vein isolation) for rhythm control "
                "in symptomatic paroxysmal/persistent AF",
                "Left atrial appendage occlusion (LAAO) if anticoagulation "
                "contraindicated",
                "Treat underlying causes: hypertension, sleep apnoea, thyrotoxicosis",
            ],
            "Acute Myocardial Infarction (AMI) Risk": [
                "EMERGENCY PROTOCOL — Activate STEMI pathway if ST-elevation",
                "Aspirin 300 mg loading dose immediately",
                "P2Y12 inhibitor loading: ticagrelor 180 mg or prasugrel 60 mg "
                "(avoid if prior stroke/TIA; physician decision)",
                "Anticoagulation: heparin or enoxaparin per protocol",
                "Primary PCI target: door-to-balloon <90 min (STEMI)",
                "Thrombolysis if PCI centre unavailable within 120 min (STEMI)",
                "NSTEMI: early invasive strategy (<24h) for high-risk patients",
                "Beta-blocker, ACE inhibitor, high-intensity statin post-MI",
                "ICU/CCU admission for haemodynamic monitoring",
                "DISCLAIMER: Treat as medical emergency — do not delay care",
            ],
            "Angina Pectoris": [
                "Short-acting nitrate: sublingual glyceryl trinitrate PRN",
                "Beta-blocker as first-line anti-anginal (atenolol/bisoprolol)",
                "Calcium channel blocker (amlodipine) as alternative or add-on",
                "Long-acting nitrate (isosorbide mononitrate) for refractory angina",
                "Antiplatelet: aspirin 75–100 mg/day",
                "Statin therapy for LDL-C reduction",
                "ACE inhibitor if co-existing hypertension, diabetes, or LV dysfunction",
                "Revascularisation (PCI/CABG) for refractory symptoms or "
                "anatomically significant coronary stenosis",
                "Lifestyle: aerobic exercise, smoking cessation, cardiac diet",
            ],
        }

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "specialization": "Cardiovascular diseases (Hypertension, CAD, CHF, AF, AMI, Angina)",
            "version": self.VERSION,
            "clinical_scope": (
                "Cardiac risk stratification, workup recommendation, "
                "and physician-directed treatment suggestions"
            ),
        })
        return info

    def get_supported_symptoms(self):
        return list(self.symptom_mapping.keys())

    def get_supported_conditions(self):
        return list(self.condition_mapping.values())


class CardiovascularFeatureNormalizer:
    """
    Converts raw clinical measurements to [0, 1] range for CardiovascularLLM.

    Usage:
        norm = CardiovascularFeatureNormalizer()
        normalized = norm.normalize(raw_patient_data)
    """

    LAB_RANGES = {
        "systolic_bp": (90.0, 220.0),
        "diastolic_bp": (60.0, 140.0),
        "heart_rate": (40.0, 200.0),
        "ldl_cholesterol": (50.0, 250.0),
        "hdl_cholesterol": (20.0, 100.0),
        "triglycerides": (50.0, 600.0),
        "troponin": (0.0, 10.0),           # relative units
        "bnp": (0.0, 5000.0),              # pg/mL
        "ecg_abnormality": (0.0, 1.0),
        "ejection_fraction": (10.0, 70.0), # % (inverted: low EF → worse)
    }

    @classmethod
    def normalize(cls, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of raw_data with lab_markers normalized to [0, 1]."""
        normalized = {k: v for k, v in raw_data.items()}
        raw_labs = raw_data.get("lab_markers", {})
        norm_labs: Dict[str, float] = {}

        for marker, value in raw_labs.items():
            if marker in cls.LAB_RANGES:
                lo, hi = cls.LAB_RANGES[marker]
                norm_value = float(min(max((value - lo) / (hi - lo), 0.0), 1.0))
                # Invert ejection_fraction: higher EF = better = lower risk score
                if marker == "ejection_fraction":
                    norm_value = 1.0 - norm_value
                norm_labs[marker] = norm_value
            else:
                norm_labs[marker] = float(min(max(value, 0.0), 1.0))

        normalized["lab_markers"] = norm_labs
        return normalized


def create_cardiovascular_llm() -> CardiovascularLLM:
    """Factory function — returns a ready-to-use CardiovascularLLM instance."""
    model = CardiovascularLLM()
    model.eval()
    return model
