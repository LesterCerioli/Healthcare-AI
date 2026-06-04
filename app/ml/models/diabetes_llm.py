"""
Diabetes Mellitus specialized LLM diagnostic model.

Covers Type 1, Type 2, Pre-diabetes, and Gestational Diabetes.
Outputs clinical decision support: diagnosis, recommended lab exams,
and treatment suggestions for the attending physician.

Input features (26 total):
  Symptoms [0–12]: polyuria, polydipsia, polyphagia, weight_loss,
                   fatigue, blurred_vision, slow_healing, frequent_infections,
                   numbness_tingling, acanthosis_nigricans, fruity_breath,
                   abdominal_pain, nausea
  Lab markers [13–20]: blood_glucose, hba1c, fasting_glucose, bmi_category,
                        insulin_level, c_peptide, triglycerides, hdl_cholesterol
  Patient context [21–25]: age_group, family_history, hypertension,
                            gestational_history, physical_activity

All values must be normalized to [0.0, 1.0] before passing to the model.
See DiabetesFeatureNormalizer for reference ranges.
"""

from typing import Dict, Any, List
from .llm_base import BaseMedicalLLM


class DiabetesLLM(BaseMedicalLLM):
    """
    Attention-based LLM for Diabetes Mellitus diagnosis and clinical support.

    Distinguishes between normal glucose regulation, pre-diabetes,
    Type 2 DM, Type 1 DM, and Gestational DM.
    """

    VERSION = "1.0.0"

    def __init__(self):
        super().__init__(
            input_size=26,
            d_model=64,
            num_heads=4,
            num_layers=3,
            num_classes=6,
            dropout=0.15,
        )

    def _initialize_clinical_knowledge(self) -> None:
        # ── Symptom features (index 0–12) ──────────────────────────────────
        self.symptom_mapping = {
            "polyuria": 0,               # excessive urination
            "polydipsia": 1,             # excessive thirst
            "polyphagia": 2,             # excessive hunger / appetite
            "unexplained_weight_loss": 3,
            "fatigue": 4,
            "blurred_vision": 5,
            "slow_healing_wounds": 6,
            "frequent_infections": 7,
            "numbness_tingling": 8,      # peripheral neuropathy indicator
            "acanthosis_nigricans": 9,   # darkened skin patches → insulin resistance
            "fruity_breath_odor": 10,    # DKA indicator (Type 1)
            "abdominal_pain": 11,
            "nausea_vomiting": 12,
        }

        # ── Lab marker features (index 13–20) ──────────────────────────────
        # Normalization reference: 0.0 = low/normal, 1.0 = severely elevated
        self.lab_marker_mapping = {
            "blood_glucose": 13,         # random; 0→<140 mg/dL, 1→≥200 mg/dL
            "hba1c": 14,                 # 0→<5.7%, 1→≥10%
            "fasting_glucose": 15,       # 0→<100 mg/dL, 1→≥200 mg/dL
            "bmi_category": 16,          # 0→normal, 0.5→overweight, 1→obese
            "insulin_level": 17,         # 0→normal, 1→severely low (Type 1 indicator)
            "c_peptide": 18,             # 0→absent (Type 1), 1→elevated (Type 2)
            "triglycerides": 19,         # 0→normal, 1→very high
            "hdl_cholesterol": 20,       # 0→low (bad), 1→normal/high
        }

        # ── Diagnostic classes ──────────────────────────────────────────────
        self.condition_mapping = {
            0: "Normal Glucose Regulation",
            1: "Pre-diabetes (Impaired Glucose Tolerance)",
            2: "Type 2 Diabetes Mellitus",
            3: "Type 1 Diabetes Mellitus",
            4: "Gestational Diabetes Mellitus",
            5: "Maturity-Onset Diabetes of the Young (MODY)",
        }

        # ── Recommended examinations per diagnosis ──────────────────────────
        self.recommended_exams = {
            "Normal Glucose Regulation": [
                "Annual fasting blood glucose screening",
                "HbA1c every 3 years if risk factors present",
                "BMI and blood pressure monitoring",
                "Lipid panel annually",
            ],
            "Pre-diabetes (Impaired Glucose Tolerance)": [
                "HbA1c test (target < 5.7%)",
                "Oral Glucose Tolerance Test (OGTT) — 2-hour 75g glucose",
                "Fasting plasma glucose",
                "Lipid panel (LDL, HDL, triglycerides)",
                "Blood pressure measurement",
                "Microalbuminuria urine test",
                "BMI and waist circumference",
            ],
            "Type 2 Diabetes Mellitus": [
                "HbA1c every 3 months until stable, then every 6 months",
                "Fasting plasma glucose",
                "Comprehensive metabolic panel (CMP)",
                "Urine albumin-to-creatinine ratio (kidney function)",
                "Estimated GFR (eGFR)",
                "Dilated eye examination (diabetic retinopathy screening)",
                "Foot examination (neuropathy and vascular assessment)",
                "Lipid panel",
                "Blood pressure and cardiovascular risk assessment",
                "Thyroid function tests (TSH)",
                "Dental examination",
            ],
            "Type 1 Diabetes Mellitus": [
                "HbA1c every 3 months",
                "C-peptide level (confirms absent endogenous insulin)",
                "Autoantibody panel: GAD65, IA-2, ZnT8, IAA",
                "Comprehensive metabolic panel",
                "Urine ketones (DKA screening)",
                "Thyroid peroxidase antibodies (associated autoimmunity)",
                "Celiac disease screening (tissue transglutaminase IgA)",
                "Urine albumin-to-creatinine ratio",
                "Dilated eye examination",
                "Continuous glucose monitoring (CGM) evaluation",
                "Foot neuropathy assessment",
            ],
            "Gestational Diabetes Mellitus": [
                "24–28 week OGTT (100g, 3-hour test)",
                "Fasting plasma glucose",
                "HbA1c at first prenatal visit",
                "Weekly non-stress test after 32 weeks",
                "Fetal biophysical profile",
                "Postpartum 75g OGTT at 4–12 weeks",
                "Annual T2DM screening for 3 years postpartum",
            ],
            "Maturity-Onset Diabetes of the Young (MODY)": [
                "Genetic testing panel (GCK, HNF1A, HNF4A mutations)",
                "C-peptide level",
                "Autoantibody panel (to exclude Type 1)",
                "HbA1c",
                "Liver function tests (HNF1A/HNF4A subtypes)",
                "Family pedigree diabetes history review",
            ],
        }

        # ── Treatment guidelines (clinical decision support for physician) ──
        self.treatment_guidelines = {
            "Normal Glucose Regulation": [
                "Maintain healthy weight (BMI 18.5–24.9)",
                "150 min/week moderate-intensity aerobic exercise",
                "Mediterranean or DASH dietary pattern",
                "Smoking cessation if applicable",
                "Annual screening if risk factors are present",
            ],
            "Pre-diabetes (Impaired Glucose Tolerance)": [
                "Intensive lifestyle intervention: ≥7% body weight loss",
                "150 min/week physical activity",
                "Low glycaemic index diet with reduced saturated fat",
                "Consider Metformin 500–1000 mg/day for high-risk patients "
                "(BMI ≥35, age <60, prior GDM) — physician decision",
                "Follow-up HbA1c every 6 months",
                "Diabetes prevention program referral",
            ],
            "Type 2 Diabetes Mellitus": [
                "First-line: Metformin 500 mg twice daily, titrate to 1000 mg twice daily",
                "Add SGLT-2 inhibitor (empagliflozin/dapagliflozin) if HF or CKD present",
                "Add GLP-1 receptor agonist (semaglutide/liraglutide) if cardiovascular "
                "disease or obesity-driven T2DM",
                "DPP-4 inhibitor (sitagliptin) as alternative low-hypoglycaemia option",
                "Insulin therapy if HbA1c >10% or symptomatic hyperglycaemia",
                "Blood pressure target <130/80 mmHg — ACE inhibitor or ARB preferred",
                "Statin therapy for LDL-lowering (cardiovascular risk reduction)",
                "Aspirin 75–100 mg/day for secondary cardiovascular prevention",
                "Self-monitoring blood glucose (SMBG) education",
                "Dietary counselling: carbohydrate counting, glycaemic index",
                "Diabetes education program enrolment",
                "Smoking cessation support",
            ],
            "Type 1 Diabetes Mellitus": [
                "Basal-bolus insulin regimen: basal insulin (glargine/detemir) + "
                "rapid-acting bolus (lispro/aspart/glulisine)",
                "Insulin pump therapy (CSII) evaluation for selected patients",
                "Continuous glucose monitoring (CGM) — strongly recommended",
                "Target HbA1c <7.0% (individualised by patient characteristics)",
                "Carbohydrate counting and insulin-to-carb ratio education",
                "DKA prevention and sick-day management plan",
                "Glucagon emergency kit prescription",
                "Psychological support for diabetes distress and burnout",
                "Cardiovascular risk factor management",
                "Annual nephrology, ophthalmology, and podiatry referrals",
            ],
            "Gestational Diabetes Mellitus": [
                "Medical nutrition therapy (MNT) as first-line management",
                "Blood glucose targets: fasting <95 mg/dL, 1h post-meal <140 mg/dL",
                "Daily SMBG (fasting + 1–2h after each meal)",
                "Insulin therapy if glucose targets not met within 1–2 weeks of MNT",
                "Metformin or glyburide may be considered (physician preference)",
                "Fetal growth and amniotic fluid monitoring",
                "Postpartum lifestyle intervention to prevent T2DM",
                "Breastfeeding encouraged for metabolic benefits",
            ],
            "Maturity-Onset Diabetes of the Young (MODY)": [
                "MODY 2 (GCK): often no pharmacological treatment needed; "
                "lifestyle management sufficient in most cases",
                "MODY 3 (HNF1A): sulfonylureas as first-line — highly sensitive",
                "MODY 1 (HNF4A): sulfonylureas; monitor for hypoglycaemia risk",
                "Genetic counselling for affected family members",
                "Discontinue insulin if re-classified from Type 1 to MODY",
                "Refer to specialist endocrinologist / diabetes genetics service",
            ],
        }

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "specialization": "Diabetes Mellitus (Type 1, Type 2, Pre-diabetes, GDM, MODY)",
            "version": self.VERSION,
            "clinical_scope": (
                "Diagnosis classification, lab examination recommendation, "
                "and physician-directed treatment suggestions for diabetic conditions"
            ),
        })
        return info

    def get_supported_symptoms(self):
        return list(self.symptom_mapping.keys())

    def get_supported_conditions(self):
        return list(self.condition_mapping.values())


class DiabetesFeatureNormalizer:
    """
    Converts raw clinical measurements to the [0, 1] range expected by DiabetesLLM.

    Usage:
        norm = DiabetesFeatureNormalizer()
        normalized = norm.normalize(raw_patient_data)
    """

    # (min_value, max_value) for linear normalization
    LAB_RANGES = {
        "blood_glucose": (70.0, 400.0),      # mg/dL random
        "hba1c": (4.0, 14.0),                # %
        "fasting_glucose": (70.0, 300.0),    # mg/dL
        "bmi_category": (0.0, 1.0),          # already categorical 0/0.5/1
        "insulin_level": (0.0, 25.0),        # µU/mL (inverted: low = Type 1 risk)
        "c_peptide": (0.0, 4.0),             # ng/mL
        "triglycerides": (50.0, 500.0),      # mg/dL
        "hdl_cholesterol": (20.0, 100.0),    # mg/dL
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
                norm_labs[marker] = float(min(max((value - lo) / (hi - lo), 0.0), 1.0))
            else:
                norm_labs[marker] = float(min(max(value, 0.0), 1.0))

        normalized["lab_markers"] = norm_labs
        return normalized


def create_diabetes_llm() -> DiabetesLLM:
    """Factory function — returns a ready-to-use DiabetesLLM instance."""
    model = DiabetesLLM()
    model.eval()
    return model
