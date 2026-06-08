"""
TensorFlow base classes for specialized diabetes diagnostic LLMs.

Architecture: Feature-Token Transformer (FT-Transformer variant).
Each clinical feature is projected to a d_model-dimensional token,
enabling multi-head self-attention to discover cross-feature correlations
(e.g., absent C-peptide + GAD65+ + young age → strong Type 1 signal).

References:
  - Gorishniy et al. (2021) "Revisiting Deep Learning Models for Tabular Data"
  - Vaswani et al. (2017) "Attention is All You Need"
  - ADA Standards of Medical Care in Diabetes (2024)
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Dict, Any, List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ClinicalTokenAttentionBlock(layers.Layer):
    """
    Multi-head self-attention over per-feature tokens with pre-norm residual.

    Learns pairwise feature dependencies that classical models miss
    (e.g., the joint signal from high HbA1c + absent C-peptide).
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name="mha",
        )
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="pre_attn_norm")
        self.dropout = layers.Dropout(dropout, name="attn_dropout")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        normed = self.norm(x)
        attended = self.mha(normed, normed, training=training)
        return x + self.dropout(attended, training=training)


class ClinicalFeedForwardBlock(layers.Layer):
    """Position-wise FFN: expand → GELU → contract, with pre-norm residual."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="pre_ffn_norm")
        self.dense1 = layers.Dense(d_ff, activation="gelu", name="ffn_expand")
        self.drop1 = layers.Dropout(dropout, name="ffn_drop1")
        self.dense2 = layers.Dense(d_model, name="ffn_contract")
        self.drop2 = layers.Dropout(dropout, name="ffn_drop2")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        out = self.norm(x)
        out = self.dense1(out)
        out = self.drop1(out, training=training)
        out = self.dense2(out)
        out = self.drop2(out, training=training)
        return x + out


class ClinicalTransformerBlock(layers.Layer):
    """Single transformer encoder block: attention + FFN with pre-norm residuals."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attn = ClinicalTokenAttentionBlock(d_model, num_heads, dropout)
        self.ffn = ClinicalFeedForwardBlock(d_model, d_ff, dropout)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.ffn(self.attn(x, training=training), training=training)


class BaseDiabetesTFModel(keras.Model):
    """
    Abstract base class for TensorFlow diabetes-specialized LLMs.

    Each subclass must implement `_initialize_clinical_knowledge()` to populate:
        symptom_mapping      → {symptom_name: feature_index}
        lab_marker_mapping   → {marker_name: feature_index}
        context_mapping      → {context_field: feature_index}
        condition_mapping    → {class_index: diagnosis_label}
        recommended_exams    → {diagnosis_label: [exam_string, ...]}
        treatment_guidelines → {diagnosis_label: [guideline_string, ...]}

    Predictions return a structured dict with:
        primary_diagnosis, confidence, differential_diagnoses,
        recommended_examinations, treatment_suggestions,
        urgency_level, urgency_probabilities, low_confidence_flag,
        clinical_disclaimer.
    """

    CONFIDENCE_THRESHOLD: float = 0.40
    URGENCY_LABELS: List[str] = ["Low", "Medium", "High", "Critical"]

    def __init__(
        self,
        input_size: int,
        d_model: int = 80,
        num_heads: int = 4,
        num_layers: int = 4,
        num_classes: int = 6,
        dropout: float = 0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.d_model = d_model
        self.num_classes = num_classes

        # Project each feature scalar → d_model token (input_size tokens per sample)
        self.token_projection = layers.Dense(d_model, use_bias=True, name="token_proj")
        self.token_norm = layers.LayerNormalization(epsilon=1e-6, name="token_norm")

        # Transformer encoder
        self.transformer_blocks = [
            ClinicalTransformerBlock(
                d_model, num_heads, d_model * 4, dropout, name=f"block_{i}"
            )
            for i in range(num_layers)
        ]

        # Aggregate token sequence → single vector
        self.gap = layers.GlobalAveragePooling1D(name="gap")

        # Diagnostic classification head
        self.clf_dense = layers.Dense(d_model // 2, activation="gelu", name="clf_dense")
        self.clf_drop = layers.Dropout(dropout, name="clf_drop")
        self.clf_out = layers.Dense(num_classes, activation="softmax", name="clf_out")

        # Urgency assessment head: Low / Medium / High / Critical
        self.urg_dense = layers.Dense(32, activation="gelu", name="urg_dense")
        self.urg_out = layers.Dense(4, activation="softmax", name="urg_out")

        # Clinical knowledge populated by subclasses
        self.symptom_mapping: Dict[str, int] = {}
        self.condition_mapping: Dict[int, str] = {}
        self.lab_marker_mapping: Dict[str, int] = {}
        self.context_mapping: Dict[str, int] = {}
        self.recommended_exams: Dict[str, List[str]] = {}
        self.treatment_guidelines: Dict[str, List[str]] = {}

        self._initialize_clinical_knowledge()

    @abstractmethod
    def _initialize_clinical_knowledge(self) -> None:
        """Populate all clinical knowledge dictionaries."""

    def call(self, x: tf.Tensor, training: bool = False):
        # x: (batch, input_size)
        # Treat each feature as a token: (batch, input_size, 1) → (batch, input_size, d_model)
        tokens = tf.expand_dims(x, axis=-1)
        tokens = self.token_projection(tokens)
        tokens = self.token_norm(tokens)

        for block in self.transformer_blocks:
            tokens = block(tokens, training=training)

        pooled = self.gap(tokens)  # (batch, d_model)

        diag = self.clf_dense(pooled)
        diag = self.clf_drop(diag, training=training)
        diag = self.clf_out(diag)

        urg = self.urg_dense(pooled)
        urg = self.urg_out(urg)

        return diag, urg

    def preprocess(self, input_data: Dict[str, Any]) -> tf.Tensor:
        feature_vector = [0.0] * self.input_size

        for k, v in input_data.get("symptoms", {}).items():
            if k in self.symptom_mapping:
                feature_vector[self.symptom_mapping[k]] = float(np.clip(v, 0.0, 1.0))

        for k, v in input_data.get("lab_markers", {}).items():
            if k in self.lab_marker_mapping:
                feature_vector[self.lab_marker_mapping[k]] = float(np.clip(v, 0.0, 1.0))

        for k, v in input_data.get("patient_context", {}).items():
            if k in self.context_mapping:
                feature_vector[self.context_mapping[k]] = float(np.clip(v, 0.0, 1.0))

        return tf.constant([feature_vector], dtype=tf.float32)

    def postprocess(
        self,
        diag_probs: tf.Tensor,
        urg_probs: tf.Tensor,
    ) -> Dict[str, Any]:
        dp = diag_probs.numpy()[0]
        up = urg_probs.numpy()[0]

        predicted_class = int(np.argmax(dp))
        confidence = float(dp[predicted_class])
        diagnosis = self.condition_mapping.get(predicted_class, "Unknown")

        urgency_idx = int(np.argmax(up))
        urgency = self.URGENCY_LABELS[urgency_idx]

        top_k = min(3, self.num_classes)
        top_indices = np.argsort(dp)[::-1][:top_k]
        differential = [
            {
                "condition": self.condition_mapping.get(int(idx), "Unknown"),
                "probability": round(float(dp[idx]), 4),
            }
            for idx in top_indices
        ]

        return {
            "primary_diagnosis": diagnosis,
            "confidence": round(confidence, 4),
            "differential_diagnoses": differential,
            "recommended_examinations": self.recommended_exams.get(diagnosis, []),
            "treatment_suggestions": self.treatment_guidelines.get(diagnosis, []),
            "urgency_level": urgency,
            "urgency_probabilities": {
                label: round(float(up[i]), 4)
                for i, label in enumerate(self.URGENCY_LABELS)
            },
            "low_confidence_flag": confidence < self.CONFIDENCE_THRESHOLD,
            "clinical_disclaimer": (
                "Estes resultados são suporte à decisão clínica (CDSS) apenas. "
                "O médico assistente é o único responsável pelo diagnóstico final "
                "e plano terapêutico. Este modelo não substitui o julgamento clínico. "
                "[These results are clinical decision support only. "
                "The attending physician bears sole responsibility for final diagnosis.]"
            ),
        }

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate_input(input_data):
            return {
                "success": False,
                "error": "Invalid input: provide at least 'symptoms' or 'lab_markers'.",
                "required_fields": ["symptoms or lab_markers"],
                "optional_fields": ["patient_context"],
            }
        try:
            tensor = self.preprocess(input_data)
            diag_probs, urg_probs = self(tensor, training=False)
            result = self.postprocess(diag_probs, urg_probs)
            result["success"] = True
            return result
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return isinstance(input_data, dict) and (
            "symptoms" in input_data or "lab_markers" in input_data
        )

    def get_model_info(self) -> Dict[str, Any]:
        try:
            dummy = tf.zeros((1, self.input_size))
            self(dummy, training=False)
            total_params = int(self.count_params())
        except Exception:
            total_params = -1
        return {
            "model_class": self.__class__.__name__,
            "framework": "TensorFlow/Keras",
            "input_size": self.input_size,
            "d_model": self.d_model,
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "supported_symptoms": list(self.symptom_mapping.keys()),
            "supported_lab_markers": list(self.lab_marker_mapping.keys()),
            "supported_conditions": list(self.condition_mapping.values()),
            "architecture": (
                "Feature-Token Transformer — each clinical feature is an "
                "independent attention token (FT-Transformer variant)"
            ),
        }

    def get_supported_symptoms(self) -> List[str]:
        return list(self.symptom_mapping.keys())

    def get_supported_conditions(self) -> List[str]:
        return list(self.condition_mapping.values())
