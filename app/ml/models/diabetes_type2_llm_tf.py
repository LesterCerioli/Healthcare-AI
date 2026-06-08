"""
Diabetes Mellitus Type 2 — TensorFlow Diagnostic LLM.

Specialized for insulin-resistance-driven (Type 2) diabetes and pre-diabetes:
  - Normal Glucose Regulation
  - Pre-diabetes: Impaired Fasting Glucose (IFG)
  - Pre-diabetes: Impaired Glucose Tolerance (IGT)
  - Type 2 DM — Controlled (HbA1c < 7.0%)
  - Type 2 DM — Uncontrolled (HbA1c 7.0–9.9%)
  - Type 2 DM — Severe / Decompensated (HbA1c ≥ 10% or acute hyperglycemia)

Input features (34 total):
  Symptoms   [0–11]:  12 features — metabolic syndrome indicators
  Lab markers[12–23]: 12 features — insulin resistance panel critical
  Context    [24–33]: 10 features — modifiable and non-modifiable risk factors

Diagnostic criteria follow ADA 2024 Standards and IDF Diabetes Atlas 2023.
All values must be normalized to [0.0, 1.0].
Use DiabetesType2FeatureNormalizer for raw clinical measurements.
"""

from __future__ import annotations

from typing import Dict, Any

from .tf_diabetes_base import BaseDiabetesTFModel


class DiabetesType2LLM(BaseDiabetesTFModel):
    """
    Attention-based TF LLM for Type 2 Diabetes Mellitus diagnosis.

    34 input features → 4-layer Feature-Token Transformer → 6-class output.

    Captures metabolic syndrome patterns: the model weighs insulin resistance
    indices (HOMA-IR, fasting insulin, C-peptide elevation), dyslipidaemia,
    and lifestyle factors together to distinguish pre-diabetes stages
    from overt T2DM and to grade control level.
    """

    VERSION = "1.0.0"

    def __init__(self):
        super().__init__(
            input_size=34,
            d_model=80,
            num_heads=4,
            num_layers=4,
            num_classes=6,
            dropout=0.15,
            name="DiabetesType2LLM",
        )

    def _initialize_clinical_knowledge(self) -> None:

        # ── Symptom features [0–11] ────────────────────────────────────────
        self.symptom_mapping = {
            "polyuria": 0,
            "polydipsia": 1,
            "polyphagia": 2,
            "fatigue_weakness": 3,
            "blurred_vision": 4,
            "slow_healing_wounds": 5,       # chronic hyperglycemia → impaired immunity
            "frequent_infections": 6,       # UTI, skin, yeast
            "numbness_tingling_feet": 7,    # peripheral neuropathy
            "acanthosis_nigricans": 8,      # insulin resistance skin marker (neck, axilla)
            "recurrent_yeast_infections": 9,
            "intermittent_claudication": 10, # peripheral arterial disease sign
            "polyuria_nocturia": 11,
        }

        # ── Lab marker features [12–23] ────────────────────────────────────
        # 0.0 = normal/low; 1.0 = severely pathological
        self.lab_marker_mapping = {
            "blood_glucose": 12,             # random plasma; 0→<140 mg/dL, 1→≥400 mg/dL
            "hba1c": 13,                     # 0→<5.7%, 1→≥14%
            "fasting_glucose": 14,           # 0→<100 mg/dL, 1→≥300 mg/dL
            "ogtt_2h_glucose": 15,           # OGTT 2h; 0→<140 mg/dL, 1→≥400 mg/dL
            "fasting_insulin": 16,           # µU/mL; 0→normal(2-25), 1→severely elevated(>100)
            "c_peptide": 17,                 # ng/mL; 0→absent, 1→severely elevated(>6)
            "homa_ir": 18,                   # insulin resistance index; 0→≤1.0, 1→≥10.0
            "triglycerides": 19,             # mg/dL; 0→<150, 1→≥600
            "hdl_cholesterol": 20,           # mg/dL; INVERTED: 0→low(≤20, bad), 1→high(≥100, good)
            "ldl_cholesterol": 21,           # mg/dL; 0→<100 (optimal), 1→≥250 (very high)
            "uric_acid": 22,                 # mg/dL; 0→normal(<6), 1→severe(≥12)
            "fasting_c_reactive_protein": 23, # hs-CRP mg/L; 0→<1, 1→≥10 (high CV risk)
        }

        # ── Patient context features [24–33] ──────────────────────────────
        self.context_mapping = {
            "age_group": 24,                 # 0=<30, 0.25=30-44, 0.5=45-59, 0.75=60-74, 1=≥75
            "bmi_category": 25,              # 0=normal, 0.33=overweight, 0.67=obese, 1=morbidly_obese
            "waist_circumference_risk": 26,  # 0=normal, 1=elevated (>102cm M / >88cm F)
            "family_history_t2dm": 27,       # 0=no, 1=yes (1st-degree relative)
            "hypertension": 28,              # 0=no, 1=yes (BP ≥130/80)
            "physical_activity": 29,         # 0=very_active, 0.5=moderate, 1=sedentary
            "smoking_status": 30,            # 0=never, 0.5=former, 1=current
            "sleep_apnea": 31,               # 0=no, 1=yes (independent IR risk factor)
            "prior_gestational_diabetes": 32, # 0=no, 1=yes (strong T2DM risk)
            "metabolic_syndrome_criteria": 33, # 0=0 criteria, 0.25=1, 0.5=2, 0.75=3, 1=4-5
        }

        # ── Diagnostic classes ─────────────────────────────────────────────
        # Criteria (ADA 2024):
        # Pre-DM IFG: fasting 100–125 mg/dL
        # Pre-DM IGT: 2h OGTT 140–199 mg/dL
        # T2DM: HbA1c ≥6.5%, OR fasting ≥126 mg/dL, OR 2h OGTT ≥200, OR random ≥200+symptoms
        self.condition_mapping = {
            0: "Normal Glucose Regulation",
            1: "Pre-diabetes — Impaired Fasting Glucose (IFG)",
            2: "Pre-diabetes — Impaired Glucose Tolerance (IGT)",
            3: "Type 2 Diabetes Mellitus — Controlled (HbA1c <7.0%)",
            4: "Type 2 Diabetes Mellitus — Uncontrolled (HbA1c 7.0–9.9%)",
            5: "Type 2 Diabetes Mellitus — Descompensado / Complicações (HbA1c ≥10%)",
        }

        # ── Recommended examinations per diagnosis ─────────────────────────
        self.recommended_exams = {
            "Normal Glucose Regulation": [
                "Fasting plasma glucose annually if ≥1 risk factor present",
                "HbA1c every 3 years if high-risk (BMI ≥25, family history, GDM history)",
                "Lipid panel: LDL, HDL, triglycerides, total cholesterol",
                "Blood pressure measurement",
                "BMI and waist circumference",
                "FINDRISC or ADA Risk Test screening tool",
            ],
            "Pre-diabetes — Impaired Fasting Glucose (IFG)": [
                "Confirmatory fasting plasma glucose (repeat on separate day)",
                "OGTT 75g 2-hour (to rule out concurrent IGT)",
                "HbA1c measurement",
                "Fasting insulin and HOMA-IR calculation",
                "Complete lipid panel with non-HDL cholesterol",
                "Blood pressure and cardiovascular risk assessment",
                "Urine albumin-to-creatinine ratio (UACR) — early nephropathy",
                "Waist circumference and BMI",
                "Thyroid function (TSH) — if symptoms",
                "Re-test: HbA1c + fasting glucose every 6–12 months",
            ],
            "Pre-diabetes — Impaired Glucose Tolerance (IGT)": [
                "OGTT 75g 2-hour (confirmatory)",
                "Fasting plasma glucose",
                "HbA1c",
                "Fasting insulin and HOMA-IR",
                "Lipid panel (focus: triglycerides and HDL — metabolic syndrome)",
                "Blood pressure measurement",
                "UACR",
                "ECG (cardiovascular risk baseline)",
                "Hepatic ultrasound (NAFLD/MASLD screening — common in pre-DM)",
                "Re-test every 6–12 months",
            ],
            "Type 2 Diabetes Mellitus — Controlled (HbA1c <7.0%)": [
                "HbA1c every 6 months (stable control)",
                "Fasting plasma glucose and self-monitoring blood glucose (SMBG) log review",
                "UACR annually — diabetic nephropathy surveillance",
                "Estimated GFR (eGFR) annually",
                "Dilated fundoscopy — diabetic retinopathy annually",
                "Comprehensive foot examination (neuropathy + pedal pulses) annually",
                "Complete lipid panel annually",
                "Blood pressure at every visit",
                "Thyroid function (TSH) annually",
                "Dental examination annually",
                "Microbiome and nutrition reassessment",
                "CGM evaluation for optimization",
            ],
            "Type 2 Diabetes Mellitus — Uncontrolled (HbA1c 7.0–9.9%)": [
                "HbA1c every 3 months until target achieved",
                "Fasting plasma glucose and post-prandial glucose assessment",
                "UACR every 6 months",
                "eGFR every 6 months",
                "Dilated fundoscopy every 6–12 months",
                "Foot examination at every visit (neuropathy surveillance)",
                "ECG and cardiovascular risk stratification (SCORE2/FRS)",
                "Lipid panel with non-HDL cholesterol",
                "Liver function tests (ALT/AST — MASLD monitoring)",
                "Sleep study if OSA suspected (STOP-BANG questionnaire)",
                "Vitamin B12 if on long-term metformin",
                "Medication adherence and side-effect review",
            ],
            "Type 2 Diabetes Mellitus — Descompensado / Complicações (HbA1c ≥10%)": [
                "HbA1c every 3 months — urgent intensification required",
                "Fasting and post-prandial plasma glucose — daily SMBG minimum",
                "Complete metabolic panel (CMP): electrolytes, BUN, creatinine, LFTs",
                "UACR and eGFR — urgently assess renal function",
                "Urine ketones (exclude HHS/DKA overlap)",
                "Dilated fundoscopy — urgent (proliferative retinopathy risk)",
                "Foot examination and Doppler ABI (peripheral arterial disease)",
                "ECG and cardiac enzymes (silent MI risk in T2DM)",
                "Carotid intima-media thickness (CIMT) if stroke risk high",
                "Lipid panel: LDL-C target <55 mg/dL (ASCVD high-risk)",
                "Coagulation studies if acute complications suspected",
                "Referral to endocrinologist and diabetes nurse educator — urgent",
                "Ophthalmology, nephrology, cardiology referrals",
            ],
        }

        # ── Treatment guidelines ────────────────────────────────────────────
        self.treatment_guidelines = {
            "Normal Glucose Regulation": [
                "Intensive lifestyle intervention: ≥150 min/week moderate aerobic activity",
                "Mediterranean or DASH dietary pattern — reduce refined carbohydrates",
                "Maintain healthy body weight (BMI 18.5–24.9 kg/m²)",
                "Smoking cessation if applicable",
                "Annual screening if ≥2 risk factors present",
            ],
            "Pre-diabetes — Impaired Fasting Glucose (IFG)": [
                "National Diabetes Prevention Program (NDPP) — lifestyle intervention",
                "Target: ≥7% total body weight loss within 6 months",
                "150–300 min/week moderate-intensity aerobic exercise",
                "Strength training ≥2×/week",
                "Low glycaemic index diet, reduced saturated fat and refined sugars",
                "Consider Metformin 500–1000 mg/day for high-risk patients "
                "(BMI ≥35 + age <60, or prior GDM) — physician decision",
                "Follow-up HbA1c + fasting glucose every 6–12 months",
                "Cardiovascular risk factor control (BP, lipids, smoking)",
            ],
            "Pre-diabetes — Impaired Glucose Tolerance (IGT)": [
                "Lifestyle intervention identical to IFG — highest priority",
                "IGT carries higher cardiovascular risk than IFG — address urgently",
                "Weight loss ≥5–7% body weight significantly reduces progression",
                "Consider Metformin if lifestyle intervention insufficient after 3–6 months",
                "Acarbose 50 mg TID may reduce 2h post-load glucose",
                "Pioglitazone as second-line only in selected cases (hepatic steatosis)",
                "Annual diabetes re-evaluation",
            ],
            "Type 2 Diabetes Mellitus — Controlled (HbA1c <7.0%)": [
                "Continue current regimen — reassess at each visit",
                "Metformin remains first-line if tolerated (eGFR ≥30 mL/min/1.73m²)",
                "If ASCVD or HF present: add SGLT-2 inhibitor (empagliflozin 10mg/dapagliflozin 10mg)",
                "If obesity + cardiovascular disease: add GLP-1 RA (semaglutide/liraglutide)",
                "Lifestyle maintenance: structured exercise + dietary adherence",
                "BP target <130/80 mmHg — ACEi/ARB first-line",
                "Statin therapy if ASCVD ≥10% risk (LDL target <70 mg/dL)",
                "Ongoing patient education and self-management support",
            ],
            "Type 2 Diabetes Mellitus — Uncontrolled (HbA1c 7.0–9.9%)": [
                "Intensify pharmacological regimen — dual or triple therapy:",
                "  → Metformin + SGLT-2 inhibitor (empagliflozin 10–25mg or dapagliflozin 10mg)",
                "  → Add GLP-1 RA (semaglutide SC 0.5–2mg/week or liraglutide 1.2–1.8mg/day)",
                "  → DPP-4 inhibitor (sitagliptin 100mg) if GLP-1 not tolerated",
                "  → Consider basal insulin (glargine/detemir) if HbA1c >9.0%",
                "Review medication adherence — common barrier to control",
                "Dietary and physical activity intensification",
                "BP target <130/80 — dual RAS blockade not recommended",
                "Statins: LDL-C target <55 mg/dL (high ASCVD risk in uncontrolled T2DM)",
                "Referral to endocrinologist and diabetes care and education specialist",
            ],
            "Type 2 Diabetes Mellitus — Descompensado / Complicações (HbA1c ≥10%)": [
                "URGENTE: considerar hospitalização se glicemia ≥350 mg/dL, cetose, ou HHNS",
                "Início de insulinoterapia basal ou basal-bolus:",
                "  → Insulina basal: glargina 10 UI SC ao deitar — titular 2 UI a cada 3 dias",
                "     até glicemia de jejum 80–130 mg/dL",
                "  → Adicionar análogo rápido pré-prandial se glicemia pós-prandial >180 mg/dL",
                "Manter Metformin (se eGFR ≥30) + SGLT-2i + GLP-1 RA com insulina",
                "Monitorização de glicemia: SMBG mínimo 4×/dia ou CGM",
                "Tratar complicações identificadas: nefropatia, retinopatia, neuropatia",
                "Anti-hipertensivo: ACEi/ARB se UACR elevada (proteção renal)",
                "Estatina de alta intensidade (atorvastatina 40–80mg) — LDL <55 mg/dL",
                "Aspirin 100mg/dia — prevenção secundária cardiovascular",
                "Equipe multidisciplinar: endocrinologista, nefrologista, oftalmologista, podólogo",
            ],
        }


class DiabetesType2FeatureNormalizer:
    """
    Converts raw clinical measurements to [0, 1] for DiabetesType2LLM.

    HDL cholesterol is inverted: higher raw HDL → lower normalized value
    (so that 1.0 always corresponds to the most pathological state).
    """

    LAB_RANGES: Dict[str, tuple] = {
        "blood_glucose": (70.0, 400.0),          # mg/dL random
        "hba1c": (4.0, 14.0),                    # %
        "fasting_glucose": (70.0, 300.0),        # mg/dL
        "ogtt_2h_glucose": (80.0, 400.0),        # mg/dL
        "fasting_insulin": (2.0, 100.0),         # µU/mL
        "c_peptide": (0.0, 6.0),                 # ng/mL
        "homa_ir": (0.0, 10.0),                  # HOMA-IR index
        "triglycerides": (50.0, 600.0),          # mg/dL
        "ldl_cholesterol": (50.0, 250.0),        # mg/dL
        "uric_acid": (2.0, 12.0),                # mg/dL
        "fasting_c_reactive_protein": (0.0, 10.0),  # hs-CRP mg/L
    }

    @classmethod
    def normalize(cls, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return copy of raw_data with lab_markers normalized to [0, 1]."""
        normalized = {k: v for k, v in raw_data.items()}
        raw_labs = raw_data.get("lab_markers", {})
        norm_labs: Dict[str, float] = {}

        for marker, value in raw_labs.items():
            if marker == "hdl_cholesterol":
                # Invert: low HDL is pathological → normalized value approaches 1
                # HDL range 20 (dangerously low) to 100 mg/dL (optimal)
                norm_labs[marker] = float(
                    min(max((100.0 - float(value)) / (100.0 - 20.0), 0.0), 1.0)
                )
            elif marker in cls.LAB_RANGES:
                lo, hi = cls.LAB_RANGES[marker]
                norm_labs[marker] = float(
                    min(max((float(value) - lo) / (hi - lo), 0.0), 1.0)
                )
            else:
                norm_labs[marker] = float(min(max(float(value), 0.0), 1.0))

        normalized["lab_markers"] = norm_labs
        return normalized


def create_diabetes_type2_llm() -> DiabetesType2LLM:
    """Factory — returns a ready-to-use DiabetesType2LLM instance."""
    return DiabetesType2LLM()
