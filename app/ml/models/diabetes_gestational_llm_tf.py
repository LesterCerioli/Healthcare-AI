"""
Diabetes Mellitus Gestacional (DMG / GDM) — TensorFlow Diagnostic LLM.

Especializado no diagnóstico de Diabetes Mellitus em gestantes:
  - Normal Glucose Tolerance in Pregnancy
  - GDM — Diet and Lifestyle Controlled
  - GDM — Pharmacological Treatment Required
  - Pre-existing Type 2 DM discovered in Pregnancy (Overt DM)
  - Pre-existing Type 1 DM in Pregnancy

Input features (28 total):
  Symptoms   [0–9]:   10 features — sintomas específicos da gravidez
  Lab markers[10–19]: 10 features — TOTG/OGTT, glicemia de jejum críticos
  Context    [20–27]:  8 features — fatores de risco obstétricos e maternos

Critérios diagnósticos:
  - IADPSG 2010 / ADA 2024: TOTG 75g — GJ ≥92 mg/dL, 1h ≥180 mg/dL, 2h ≥153 mg/dL
  - DM Manifesto: GJ ≥126 mg/dL ou HbA1c ≥6.5% na 1ª consulta pré-natal
  - Rastreio universal entre 24–28 semanas de gestação

All values must be normalized to [0.0, 1.0].
Use DiabetesGestationalFeatureNormalizer for raw clinical measurements.
"""

from __future__ import annotations

from typing import Dict, Any

from .tf_diabetes_base import BaseDiabetesTFModel


class DiabetesGestationalLLM(BaseDiabetesTFModel):
    """
    Attention-based TF LLM for Gestational Diabetes Mellitus diagnosis.

    28 input features → 4-layer Feature-Token Transformer → 5-class output.

    Weighs obstetric-specific features: OGTT values at 1h and 2h post-load,
    gestational age at screening, prior GDM, and maternal BMI are the
    strongest discriminative signals captured by the attention mechanism.

    Only applicable to pregnant patients. Provide gestational age in context.
    """

    VERSION = "1.0.0"

    def __init__(self):
        super().__init__(
            input_size=28,
            d_model=80,
            num_heads=4,
            num_layers=4,
            num_classes=5,
            dropout=0.15,
            name="DiabetesGestationalLLM",
        )

    def _initialize_clinical_knowledge(self) -> None:

        # ── Symptom features [0–9] ─────────────────────────────────────────
        self.symptom_mapping = {
            "polyuria": 0,
            "polydipsia": 1,
            "excessive_fatigue": 2,           # além do normal da gravidez
            "blurred_vision": 3,
            "recurrent_vaginal_infections": 4, # candidíase recorrente — hiperglicemia
            "lower_limb_edema": 5,             # além do edema fisiológico esperado
            "persistent_headaches": 6,
            "nausea_beyond_first_trimester": 7,
            "excessive_thirst_late_pregnancy": 8,
            "reduced_fetal_movement": 9,       # pode indicar macrossomia / sofrimento fetal
        }

        # ── Lab marker features [10–19] ────────────────────────────────────
        # Valores de referência baseados nos critérios IADPSG/ADA 2024
        self.lab_marker_mapping = {
            "fasting_glucose": 10,         # mg/dL; GDM: ≥92; Overt DM: ≥126
            "ogtt_1h_glucose": 11,         # TOTG 75g, 1h; GDM: ≥180 mg/dL
            "ogtt_2h_glucose": 12,         # TOTG 75g, 2h; GDM: ≥153 mg/dL
            "hba1c_first_trimester": 13,   # %; Overt DM: ≥6.5% na 1ª consulta
            "fasting_insulin": 14,         # µU/mL; resistência insulínica gestacional
            "thyroid_tsh": 15,             # mIU/L; hipotireoidismo → risco DMG
            "urine_glucose": 16,           # 0=negativo, 0.5=traço, 1=positivo (glicosúria)
            "urine_protein": 17,           # 0=negativo, 1=positivo (pré-eclâmpsia/nefropatia)
            "hemoglobin": 18,              # g/dL; INVERTIDO: anemia → risco aumentado
            "ldl_cholesterol": 19,         # mg/dL; dislipidemia gestacional
        }

        # ── Patient context features [20–27] ──────────────────────────────
        self.context_mapping = {
            "gestational_age_weeks": 20,   # normalizado: 0=0 sem, 1=42 sem
                                           # Rastreio padrão: 24–28 sem (≈0.57–0.67)
            "maternal_age_group": 21,      # 0=<25, 0.33=25-34, 0.67=35-39, 1=≥40
            "prepregnancy_bmi": 22,        # 0=normal, 0.33=sobrepeso, 0.67=obeso, 1=morbidamente_obeso
            "gestational_weight_gain": 23, # 0=abaixo_recomendado, 0.5=adequado, 1=excessivo
            "prior_gdm": 24,               # 0=não, 1=sim (maior fator de risco individual)
            "pcos_history": 25,            # 0=não, 1=sim (SOP → resistência insulínica)
            "family_history_diabetes": 26, # 0=não, 1=sim (parente 1º grau DM1 ou DM2)
            "parity": 27,                  # 0=nulípara, 0.33=1 parto, 0.67=2 partos, 1=≥3 partos
        }

        # ── Diagnostic classes ─────────────────────────────────────────────
        # Critério IADPSG (basta 1 valor alterado no TOTG 75g):
        #   GJ ≥92 mg/dL OU 1h ≥180 mg/dL OU 2h ≥153 mg/dL
        # DM Manifesto (Overt DM):
        #   GJ ≥126 mg/dL OU HbA1c ≥6.5% OU glicemia aleatória ≥200 mg/dL + sintomas
        self.condition_mapping = {
            0: "Tolerância Normal à Glicose na Gravidez",
            1: "Diabetes Gestacional — Controlado (Dieta e Atividade Física)",
            2: "Diabetes Gestacional — Requer Tratamento Farmacológico (Insulina/Metformina)",
            3: "DM Tipo 2 Pré-existente Diagnosticado na Gravidez (DM Manifesto)",
            4: "DM Tipo 1 Pré-existente na Gravidez",
        }

        # ── Recommended examinations per diagnosis ─────────────────────────
        self.recommended_exams = {
            "Tolerância Normal à Glicose na Gravidez": [
                "Repetir rastreio de DMG entre 32–34 semanas se alto risco",
                "Glicemia de jejum na 1ª consulta pré-natal",
                "HbA1c na 1ª consulta pré-natal se fatores de risco",
                "TOTG 75g entre 24–28 semanas (rastreio universal — SBD/ADA 2024)",
                "Perfil lipídico gestacional",
                "TSH e T4L (rastreio de hipotireoidismo gestacional)",
                "Urina I + urocultura trimestralmente",
                "Ultrassonografia obstétrica morfológica 2º trimestre",
                "TOTG pós-parto 75g 2h entre 6–12 semanas para rastreio de T2DM",
            ],
            "Diabetes Gestacional — Controlado (Dieta e Atividade Física)": [
                "Auto-monitorização da glicemia capilar (AMGC): GJ + 1h ou 2h pós-prandial",
                "TOTG confirmatório 75g se diagnóstico baseado em apenas 1 valor",
                "HbA1c mensal (meta: ≤6.5% no DMG)",
                "Ultrassonografia obstétrica morfológica e crescimento fetal (28–32–36 sem)",
                "Cardiotocografia (CTG) após 34 semanas",
                "Índice de líquido amniótico (ILA) — polidrâmnio associado ao DMG",
                "Perfil biofísico fetal se controle inadequado",
                "TSH e T4L (hipotireoidismo associado ao DMG)",
                "Urina I + urocultura mensal (maior risco de ITU no DMG)",
                "TOTG pós-parto 75g 2h entre 6–12 semanas (40–60% risco de T2DM em 10 anos)",
                "Acompanhamento endocrinológico e obstétrico conjunto",
            ],
            "Diabetes Gestacional — Requer Tratamento Farmacológico (Insulina/Metformina)": [
                "AMGC ≥6×/dia: jejum + pré-prandial + 1h pós-prandial (café, almoço, jantar)",
                "HbA1c a cada 4–6 semanas",
                "Perfil glicêmico semanal revisado pela equipe",
                "USG crescimento fetal a cada 2–4 semanas (macrossomia)",
                "Doppler de artéria umbilical se restrição de crescimento suspeita",
                "CTG 2×/semana após 34 semanas",
                "Perfil biofísico fetal semanal se controle insatisfatório",
                "Cetonúria em jejum (dieta hipoglicídica inadequada)",
                "Avaliação cardiológica neonatal ao nascimento",
                "Internação eletiva para resolução entre 38–39 semanas",
                "TOTG pós-parto obrigatório entre 6–12 semanas",
            ],
            "DM Tipo 2 Pré-existente Diagnosticado na Gravidez (DM Manifesto)": [
                "HbA1c imediata + perfil glicêmico completo",
                "Função renal: creatinina, eGFR, UACR",
                "Fundoscopia — retinopatia diabética pode agravar na gravidez",
                "ECG e avaliação cardíaca",
                "TSH e T4L",
                "Urina I + urocultura",
                "Perfil lipídico (estatinas contraindicadas na gravidez — suspender)",
                "AMGC ≥6-8×/dia + CGM se disponível",
                "Nefrologista e endocrinologista — co-gestão obrigatória",
                "USG morfológica detalhada (DM pré-gestacional → malformações 1º trimestre)",
                "Ecocardiograma fetal (cardiopatias fetais em DM pré-gestacional)",
                "TOTG pós-parto + seguimento endocrinológico contínuo",
            ],
            "DM Tipo 1 Pré-existente na Gravidez": [
                "HbA1c: meta <6.5% durante a gravidez (evitar hipoglicemia severa)",
                "Glicemia capilar ≥7-8×/dia ou CGM (altamente recomendado)",
                "Ajuste de doses de insulina frequente — aumenta resistência ao longo gestação",
                "Fundoscopia a cada trimestre (retinopatia pode progredir na gravidez)",
                "Função renal: creatinina, UACR, eGFR trimestralmente",
                "TSH e T4L (tireoidite autoimune associada)",
                "Função tireoidiana no pós-parto (tireoidite pós-parto em T1DM)",
                "USG morfológica detalhada + ecocardiograma fetal",
                "Doppler de artéria umbilical trimestralmente",
                "CTG semanal após 34 semanas",
                "Cetonemia: monitorização especial em jejum prolongado e doenças intercorrentes",
                "Internação eletiva para resolução 37–38 semanas (ou antes se complicações)",
                "Endocrinologista + obstetra de alto risco co-gestão",
                "Neonatologia presente no parto (hipoglicemia neonatal frequente)",
            ],
        }

        # ── Treatment guidelines (clinical decision support for physician) ──
        self.treatment_guidelines = {
            "Tolerância Normal à Glicose na Gravidez": [
                "Dieta saudável equilibrada — carboidratos complexos, fibras",
                "30 min/dia de atividade física moderada (caminhada, natação) se sem contraindicação",
                "Manter ganho de peso dentro das metas gestacionais (IOM 2009)",
                "Ácido fólico 400–4000 mcg/dia (dose conforme risco de DTN)",
                "Rastreio de DMG na 24ª–28ª semana obrigatório",
                "Orientar sobre sinais de alerta de DMG",
            ],
            "Diabetes Gestacional — Controlado (Dieta e Atividade Física)": [
                "Terapia Nutricional Médica (TNM) como 1ª linha:",
                "  → Distribuição de carboidratos: 33–40% do VET em 3 refeições + 3 lanches",
                "  → Café da manhã: limitar CHO ≤15-30g (resistência insulínica matinal maior)",
                "  → Índice glicêmico baixo — evitar carboidratos simples e açúcar",
                "  → Gorduras saudáveis e proteínas magras complementam",
                "Metas glicêmicas (SBD/ADA 2024):",
                "  → Glicemia de jejum: <95 mg/dL (ou <92 mg/dL IADPSG)",
                "  → 1h pós-prandial: <140 mg/dL",
                "  → 2h pós-prandial: <120 mg/dL",
                "Atividade física moderada após refeições (caminhada 20–30 min)",
                "Reavaliação após 1–2 semanas: se metas não atingidas → farmacoterapia",
                "Amamentação recomendada: reduz risco pós-parto de T2DM em 40%",
                "Vigilância fetal: CTG e USG conforme protocolo de alto risco",
            ],
            "Diabetes Gestacional — Requer Tratamento Farmacológico (Insulina/Metformina)": [
                "INSULINOTERAPIA — primeira escolha farmacológica no DMG:",
                "  → Hiperglicemia de jejum: insulina NPH 0.1–0.2 UI/kg SC ao deitar",
                "     Titular 2 UI a cada 3 dias até GJ <95 mg/dL",
                "  → Hiperglicemia pós-prandial: análogo rápido (lispro/aspart) antes refeições",
                "     Dose inicial: 4 UI, ajustar conforme glicemia 1h pós-prandial",
                "  → Insulina NPH é segura na gravidez (categoria B ANVISA)",
                "  → Análogos de lispro e aspart: aprovados para uso gestacional",
                "METFORMINA (alternativa — decisão médica):",
                "  → 500 mg 2×/dia com refeições, titular até 1000 mg 2×/dia",
                "  → Atravessa placenta — discutir com paciente benefício/risco",
                "  → Eficácia ligeiramente inferior à insulina; 30–50% necessitam insulina adicional",
                "Gliburida NÃO recomendada (maior taxa hipoglicemia neonatal)",
                "Monitorar: crescimento fetal, LA, bem-estar fetal",
                "Resolução da gravidez: 38–39 semanas se bem controlado; antes se complicações",
                "PARTO: suspender antidiabéticos orais; manter insulina com ajuste para jejum",
                "Pós-parto: suspender insulina imediatamente após o parto em DMG",
                "  → TOTG 6–12 semanas pós-parto obrigatório",
            ],
            "DM Tipo 2 Pré-existente Diagnosticado na Gravidez (DM Manifesto)": [
                "SUSPENDER imediatamente: estatinas, IECA/BRA, SGLT-2i (todos teratogênicos)",
                "INSULINOTERAPIA intensiva desde o diagnóstico:",
                "  → Regime basal-bolus: glargina/detemir ao deitar + análogo rápido pré-prandial",
                "  → Meta HbA1c: 6.0–6.5% (mais rígida que DMG por diagnóstico precoce)",
                "  → Doses aumentam progressivamente (25–100% ao longo da gravidez)",
                "Anti-hipertensivos seguros na gravidez: metildopa, nifedipina, labetalol, hidralazina",
                "  → SUSPENDER IECA/BRA — nefrotoxicidade fetal",
                "Ácido acetilsalicílico 100mg/dia após 12 semanas (reduz pré-eclâmpsia)",
                "Controle rígido desde o 1º trimestre: malformações fetais relacionadas a HbA1c",
                "Suplementação: ácido fólico 4–5 mg/dia (dose alta — DM pré-gestacional)",
                "Resolução: individualizada; geralmente 37–38 semanas",
                "Retomar medicações pré-gravidez pós-parto conforme reavaliação",
            ],
            "DM Tipo 1 Pré-existente na Gravidez": [
                "INSULINOTERAPIA INTENSIVA — única opção farmacológica no T1DM:",
                "  → Manter regime basal-bolus ou bomba de insulina (CSII)",
                "  → CGM fortemente recomendado — reduz macrossomia e hipoglicemia neonatal",
                "  → Doses aumentam 50–100% conforme progressão gestacional (resistência ↑)",
                "  → 1º trimestre: sensibilidade aumentada — atenção à hipoglicemia",
                "  → 2º–3º trimestres: resistência progressiva — aumentar doses",
                "Meta glicêmica mais rígida que T2DM pré-gestacional:",
                "  → GJ: 60–95 mg/dL",
                "  → 1h pós-prandial: <140 mg/dL",
                "  → HbA1c: <6.5% (se seguro sem hipoglicemia severa)",
                "Cetoacidose diabética é emergência obstétrica — risco fetal elevado",
                "  → Protocolo de DKA gestacional com endocrinologia e UTI",
                "SUSPENDER estatinas, IECA/BRA pré-concepção ou imediatamente ao diagnóstico",
                "Hipoglicemia: kit de glucagon prescrito + treinamento familiar",
                "Distúrbios tireoidianos: monitorar TSH trimestralmente (associação autoimune)",
                "Resolução: 37–38 semanas; parto vaginal se sem contraindicações obstétricas",
                "Pós-parto: reduzir doses de insulina imediatamente (resistência cai abruptamente)",
                "  → Amamentação: reduz dose de insulina; monitorar hipoglicemia materna",
                "  → Suporte psicológico: período pós-parto com T1DM de alto risco emocional",
            ],
        }


class DiabetesGestationalFeatureNormalizer:
    """
    Converts raw clinical measurements to [0, 1] for DiabetesGestationalLLM.

    Hemoglobin is inverted: lower Hb (anemia) → higher normalized value (more pathological).
    OGTT values normalized to IADPSG 2010 diagnostic cut-offs.
    """

    LAB_RANGES: Dict[str, tuple] = {
        "fasting_glucose": (60.0, 200.0),      # mg/dL; GDM cut-off: 92 mg/dL
        "ogtt_1h_glucose": (80.0, 300.0),      # mg/dL; GDM cut-off: 180 mg/dL
        "ogtt_2h_glucose": (80.0, 280.0),      # mg/dL; GDM cut-off: 153 mg/dL
        "hba1c_first_trimester": (4.0, 10.0),  # %; Overt DM: ≥6.5%
        "fasting_insulin": (2.0, 60.0),        # µU/mL gestacional
        "thyroid_tsh": (0.1, 10.0),            # mIU/L; normal gestacional: 0.1–4.0
        "urine_glucose": (0.0, 1.0),           # 0=neg, 0.5=traço, 1=pos
        "urine_protein": (0.0, 1.0),           # 0=neg, 1=pos
        "ldl_cholesterol": (50.0, 250.0),      # mg/dL
    }

    @classmethod
    def normalize(cls, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return copy of raw_data with lab_markers normalized to [0, 1]."""
        normalized = {k: v for k, v in raw_data.items()}
        raw_labs = raw_data.get("lab_markers", {})
        norm_labs: Dict[str, float] = {}

        for marker, value in raw_labs.items():
            if marker == "hemoglobin":
                # Invert: lower Hb = anemia = more pathological → normalized toward 1
                # Hb range 8.0 (severe anemia) to 16.0 g/dL (normal)
                norm_labs[marker] = float(
                    min(max((16.0 - float(value)) / (16.0 - 8.0), 0.0), 1.0)
                )
            elif marker == "gestational_age_weeks" and marker not in cls.LAB_RANGES:
                # gestational age: linear 0–42 weeks
                norm_labs[marker] = float(min(max(float(value) / 42.0, 0.0), 1.0))
            elif marker in cls.LAB_RANGES:
                lo, hi = cls.LAB_RANGES[marker]
                norm_labs[marker] = float(
                    min(max((float(value) - lo) / (hi - lo), 0.0), 1.0)
                )
            else:
                norm_labs[marker] = float(min(max(float(value), 0.0), 1.0))

        normalized["lab_markers"] = norm_labs
        return normalized


def create_diabetes_gestational_llm() -> DiabetesGestationalLLM:
    """Factory — returns a ready-to-use DiabetesGestationalLLM instance."""
    return DiabetesGestationalLLM()
