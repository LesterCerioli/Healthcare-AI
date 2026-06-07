"""
Diabetes Mellitus Type 1 — TensorFlow Diagnostic LLM.

Specialized for autoimmune (Type 1) diabetes detection, including:
  - Pre-clinical Stage 1 (autoantibodies present, normoglycemia)
  - Pre-symptomatic Stage 2 (autoantibodies + dysglycemia)
  - Clinical Stage 3 (symptomatic T1DM without DKA)
  - T1DM with Diabetic Ketoacidosis (DKA) — medical emergency
  - LADA (Latent Autoimmune Diabetes in Adults)

Input features (30 total):
  Symptoms   [0–12]: 13 features — DKA indicators weighted heavily
  Lab markers[13–23]: 11 features — autoantibodies, C-peptide, ketones critical
  Context    [24–29]: 6 features  — age, BMI, onset speed, comorbidities

Diagnostic criteria follow ADA 2024 Standards of Medical Care
and Insel et al. (2021) T1DM Staging Framework.

All values must be normalized to [0.0, 1.0].
Use DiabetesType1FeatureNormalizer for raw clinical measurements.
"""

from __future__ import annotations

from typing import Dict, Any

from .tf_diabetes_base import BaseDiabetesTFModel


class DiabetesType1LLM(BaseDiabetesTFModel):
    """
    Attention-based TF LLM for Type 1 Diabetes Mellitus diagnosis.

    30 input features → 4-layer Feature-Token Transformer → 6-class output.

    The model captures the characteristic triad of T1DM:
    absent C-peptide + positive autoantibodies + hyperglycemia.
    DKA is flagged as a Critical urgency output.
    """

    VERSION = "1.0.0"

    def __init__(self):
        super().__init__(
            input_size=30,
            d_model=80,
            num_heads=4,
            num_layers=4,
            num_classes=6,
            dropout=0.15,
            name="DiabetesType1LLM",
        )

    def _initialize_clinical_knowledge(self) -> None:

        # ── Symptom features [0–12] ────────────────────────────────────────
        self.symptom_mapping = {
            "polyuria": 0,
            "polydipsia": 1,
            "rapid_weight_loss": 2,          # hallmark T1DM presentation
            "fatigue_weakness": 3,
            "blurred_vision": 4,
            "fruity_breath_odor": 5,          # DKA — acetone breath
            "abdominal_pain": 6,              # DKA
            "nausea_vomiting": 7,             # DKA
            "deep_rapid_breathing": 8,        # Kussmaul respiration — DKA
            "altered_consciousness": 9,       # severe DKA / hyperosmolar state
            "frequent_infections": 10,
            "muscle_cramps": 11,
            "mood_changes_irritability": 12,
        }

        # ── Lab marker features [13–23] ────────────────────────────────────
        # 0.0 = normal/low; 1.0 = severely abnormal
        self.lab_marker_mapping = {
            "blood_glucose": 13,       # random plasma; 0→<140 mg/dL, 1→≥400 mg/dL
            "hba1c": 14,               # 0→<5.7%, 1→≥13%
            "fasting_glucose": 15,     # 0→<100 mg/dL, 1→≥300 mg/dL
            "urine_ketones": 16,       # 0=negative, 0.33=trace, 0.67=moderate, 1=large
            "blood_ketones": 17,       # 0→0 mmol/L, 1→≥8 mmol/L (>3.0=DKA)
            "c_peptide": 18,           # 0→absent (T1DM), 1→4 ng/mL (normal/T2DM)
            "insulin_level": 19,       # 0→absent (<2 µU/mL), 1→25 µU/mL
            "gad65_antibody": 20,      # 0=negative, 1=strongly positive
            "ia2_antibody": 21,        # 0=negative, 1=strongly positive
            "znt8_antibody": 22,       # 0=negative, 1=strongly positive
            "blood_ph": 23,            # 0→pH ≤6.8 (severe acidosis), 1→pH ≥7.45 (normal)
        }

        # ── Patient context features [24–29] ──────────────────────────────
        self.context_mapping = {
            "age_group": 24,           # 0=child(0-12), 0.25=teen(13-18), 0.5=young_adult(19-35),
                                       # 0.75=adult(36-60), 1.0=elderly(>60)
            "bmi_category": 25,        # 0=underweight, 0.25=normal, 0.5=overweight,
                                       # 0.75=obese, 1.0=morbidly_obese
            "family_history_t1dm": 26, # 0=no, 1=yes (first-degree relative with T1DM)
            "onset_speed": 27,         # 0=gradual(months-years), 1=abrupt(days-weeks)
            "thyroid_disease": 28,     # 0=no, 1=yes (Hashimoto / Graves — autoimmune cluster)
            "celiac_disease": 29,      # 0=no, 1=yes (autoimmune association)
        }

        # ── Diagnostic classes ─────────────────────────────────────────────
        self.condition_mapping = {
            0: "No Type 1 DM Indicators",
            1: "T1DM Stage 1 — Autoimmune (Autoantibodies+, Normoglycemia)",
            2: "T1DM Stage 2 — Pre-symptomatic (Autoantibodies+ + Dysglycemia)",
            3: "T1DM Stage 3 — Symptomatic without DKA",
            4: "T1DM with Diabetic Ketoacidosis (DKA) — EMERGÊNCIA MÉDICA",
            5: "LADA (Latent Autoimmune Diabetes in Adults)",
        }

        # ── Recommended examinations per diagnosis ─────────────────────────
        self.recommended_exams = {
            "No Type 1 DM Indicators": [
                "Annual fasting glucose if risk factors present",
                "HbA1c baseline measurement",
                "BMI and blood pressure monitoring",
            ],
            "T1DM Stage 1 — Autoimmune (Autoantibodies+, Normoglycemia)": [
                "Autoantibody confirmation panel: GAD65, IA-2, ZnT8, IAA, ICA",
                "Fasting plasma glucose and HbA1c every 6 months",
                "Oral Glucose Tolerance Test (OGTT — 75g, 2-hour)",
                "C-peptide level (fasting and stimulated)",
                "Thyroid peroxidase antibodies (TPO-Ab)",
                "Celiac disease screening: tissue transglutaminase IgA (tTG-IgA)",
                "Consider clinical trial enrolment (immune intervention studies)",
            ],
            "T1DM Stage 2 — Pre-symptomatic (Autoantibodies+ + Dysglycemia)": [
                "Confirmatory OGTT (75g, 2-hour)",
                "HbA1c — target monitoring every 3 months",
                "C-peptide (fasting + stimulated with mixed meal tolerance test)",
                "Full autoantibody panel: GAD65, IA-2, ZnT8, IAA",
                "Continuous glucose monitoring (CGM) evaluation",
                "Urine ketones baseline",
                "Thyroid function panel (TSH, Free T4)",
                "Celiac screening (tTG-IgA)",
                "Ophthalmology baseline examination",
                "Endocrinology specialist referral",
            ],
            "T1DM Stage 3 — Symptomatic without DKA": [
                "HbA1c every 3 months",
                "C-peptide level — confirms absent endogenous insulin secretion",
                "Full autoantibody panel: GAD65, IA-2, ZnT8, IAA",
                "Comprehensive metabolic panel (CMP): electrolytes, BUN, creatinine",
                "Urine ketones and blood ketones (β-hydroxybutyrate)",
                "Thyroid peroxidase antibodies and TSH",
                "Celiac disease: tTG-IgA and total IgA",
                "Urine albumin-to-creatinine ratio (UACR) — nephropathy baseline",
                "Dilated fundoscopy — diabetic retinopathy baseline",
                "Foot examination — neuropathy and vascular assessment",
                "Lipid panel (LDL, HDL, triglycerides, total cholesterol)",
                "Blood pressure measurement",
                "Continuous glucose monitoring (CGM) system evaluation",
                "Diabetes education program enrolment",
            ],
            "T1DM with Diabetic Ketoacidosis (DKA) — EMERGÊNCIA MÉDICA": [
                "EMERGÊNCIA — Admissão hospitalar imediata",
                "Arterial blood gas (ABG): pH, pCO2, bicarbonate",
                "Blood ketones (β-hydroxybutyrate) — target <0.5 mmol/L for resolution",
                "Serum electrolytes: Na+, K+, Cl-, HCO3- (serial measurements)",
                "BUN and creatinine (renal function in dehydration)",
                "Complete blood count (CBC)",
                "Blood and urine cultures if infection suspected (precipitant)",
                "Urine ketones and urinalysis",
                "12-lead ECG (hyperkalaemia screening)",
                "HbA1c",
                "C-peptide level",
                "Autoantibody panel (if new-onset diagnosis)",
                "Blood glucose hourly monitoring during treatment",
                "Phosphate and magnesium levels",
            ],
            "LADA (Latent Autoimmune Diabetes in Adults)": [
                "GAD65 antibody titre (high sensitivity for LADA)",
                "IA-2 antibody",
                "Stimulated C-peptide (mixed meal tolerance test or glucagon stim)",
                "Fasting C-peptide level",
                "HbA1c every 3 months",
                "OGTT if diagnosis uncertain",
                "Thyroid function (TSH, Free T4) and TPO-Ab",
                "Celiac screening (tTG-IgA)",
                "Dilated fundoscopy",
                "UACR (nephropathy screening)",
                "Lipid panel",
                "Endocrinology referral",
            ],
        }

        # ── Treatment guidelines (clinical decision support for physician) ──
        self.treatment_guidelines = {
            "No Type 1 DM Indicators": [
                "No pharmacological intervention required",
                "Healthy lifestyle: balanced diet, regular physical activity",
                "Annual diabetes risk screening if first-degree relatives affected",
                "Educate on symptoms warranting urgent evaluation (polyuria, DKA signs)",
            ],
            "T1DM Stage 1 — Autoimmune (Autoantibodies+, Normoglycemia)": [
                "No insulin therapy required at this stage",
                "Consider teplizumab (anti-CD3 mAb) in high-risk patients ≥8 years "
                "— approved by FDA (2022) to delay progression to Stage 3 by ~2 years",
                "Enrol in diabetes prevention clinical trials if eligible",
                "Close metabolic monitoring every 3–6 months",
                "Psychological support: address anxiety around future diagnosis",
                "Nutritional counselling and physical activity guidance",
            ],
            "T1DM Stage 2 — Pre-symptomatic (Autoantibodies+ + Dysglycemia)": [
                "Strong recommendation: teplizumab therapy evaluation (FDA-approved)",
                "No exogenous insulin indicated unless symptomatic hyperglycemia develops",
                "Intensive monitoring: CGM or SMBG 4–6×/day",
                "Prepare patient and family for insulin initiation",
                "Diabetes education: carbohydrate counting, hypoglycaemia management",
                "Glucagon prescription — emergency preparedness",
                "Mental health support and diabetes distress screening",
                "Endocrinology specialist management",
            ],
            "T1DM Stage 3 — Symptomatic without DKA": [
                "Basal-bolus insulin regimen (intensified insulin therapy):",
                "  → Basal: insulin glargine (U-100/U-300) or detemir — once or twice daily",
                "  → Bolus: rapid-acting analogue (lispro/aspart/glulisine) — before each meal",
                "  → Insulin-to-carb ratio (ICR) and correction factor individualized",
                "Continuous glucose monitoring (CGM) — strongly recommended for all T1DM",
                "Insulin pump therapy (CSII) evaluation for selected patients",
                "Target HbA1c <7.0% (individualized: <7.5% for children, <8.0% for elderly)",
                "Carbohydrate counting education — mandatory",
                "DKA prevention plan: sick-day rules, ketone monitoring protocol",
                "Glucagon kit (1 mg IM) or nasal glucagon — prescription required",
                "Automated insulin delivery (AID/closed-loop) systems evaluation",
                "Annual: nephrology, ophthalmology, podiatry, and cardiology screening",
                "Psychological support for diabetes distress and burnout",
                "Associated conditions: annual thyroid and celiac screening",
            ],
            "T1DM with Diabetic Ketoacidosis (DKA) — EMERGÊNCIA MÉDICA": [
                "PROTOCOLO DE DKA — Tratamento hospitalar urgente:",
                "1. Ressuscitação volêmica: NaCl 0.9% 1L/hora na primeira hora, "
                "   depois 250–500 mL/hora conforme estado hemodinâmico",
                "2. Reposição de potássio: se K+ ≥3.5 mEq/L, iniciar insulina; "
                "   se K+ <3.5 mEq/L, repor potássio ANTES da insulina",
                "3. Insulinoterapia IV: insulina regular 0.1 UI/kg/hora (sem bolus inicial "
                "   se K+ <3.5 mEq/L)",
                "4. Monitorar glicemia/hora; reduzir para 0.05 UI/kg/h quando glicemia ≤250 mg/dL",
                "5. Adicionar SG 5% quando glicemia ≤200 mg/dL para prevenir hipoglicemia",
                "6. Bicarbonato: apenas se pH <6.9 — 100 mEq NaHCO3 em 2h",
                "7. Critérios de resolução da DKA: pH >7.3 + bicarbonato >15 mEq/L + "
                "   cetonemia <0.5 mmol/L",
                "8. Transição para insulina subcutânea: sobrepor SC por 1–2h antes de "
                "   descontinuar infusão IV",
                "9. Identificar e tratar fator precipitante (infecção, omissão de insulina, etc.)",
                "10. Endocrinologia e UTI — supervisão mandatória",
            ],
            "LADA (Latent Autoimmune Diabetes in Adults)": [
                "Avoid sulfonylureas — accelerate beta-cell loss in LADA",
                "Avoid thiazolidinediones as monotherapy",
                "GLP-1 receptor agonists (semaglutide/liraglutide) — may preserve beta cells",
                "Early insulin therapy when C-peptide levels begin to decline",
                "Basal insulin initiation before C-peptide becomes undetectable",
                "CGM or frequent SMBG — monitor progression",
                "Metformin may be used adjunctively while C-peptide is still present",
                "Target HbA1c <7.0%; reassess as beta-cell function declines",
                "Annual C-peptide to track progression to complete insulin dependence",
                "Genetic counselling for relatives",
                "Associated autoimmune disease monitoring: thyroid, celiac",
            ],
        }


class DiabetesType1FeatureNormalizer:
    """
    Converts raw clinical measurements to [0, 1] for DiabetesType1LLM.

    Normalization formula: (value - min) / (max - min), clamped to [0, 1].
    For blood_ph: inverted so that 1.0 = most severe acidosis (lowest pH).

    Clinical reference ranges based on ADA 2024 and standard clinical practice.
    """

    # (min_value, max_value) for linear normalization
    LAB_RANGES: Dict[str, tuple] = {
        "blood_glucose": (70.0, 600.0),      # mg/dL — covers DKA range
        "hba1c": (4.0, 13.0),                # %
        "fasting_glucose": (60.0, 400.0),    # mg/dL
        "urine_ketones": (0.0, 1.0),         # already scaled: 0=neg, 0.33=trace, 0.67=mod, 1=large
        "blood_ketones": (0.0, 8.0),         # mmol/L; >3.0 = DKA criterion
        "c_peptide": (0.0, 4.0),             # ng/mL; T1DM: typically <0.2 ng/mL
        "insulin_level": (0.0, 25.0),        # µU/mL; near-zero in T1DM
        "gad65_antibody": (0.0, 1.0),        # 0=negative, 1=strongly positive
        "ia2_antibody": (0.0, 1.0),
        "znt8_antibody": (0.0, 1.0),
    }

    @classmethod
    def normalize(cls, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return copy of raw_data with lab_markers normalized to [0, 1]."""
        normalized = {k: v for k, v in raw_data.items()}
        raw_labs = raw_data.get("lab_markers", {})
        norm_labs: Dict[str, float] = {}

        for marker, value in raw_labs.items():
            if marker == "blood_ph":
                # Invert: lower pH → higher (more pathological) normalized value
                # pH range 6.80 (severe DKA) to 7.45 (normal)
                norm_labs[marker] = float(
                    min(max((7.45 - float(value)) / (7.45 - 6.80), 0.0), 1.0)
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


def create_diabetes_type1_llm() -> DiabetesType1LLM:
    """Factory — returns a ready-to-use DiabetesType1LLM instance."""
    return DiabetesType1LLM()
