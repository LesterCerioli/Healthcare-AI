"""
Base LLM architecture for specialized medical diagnostic models.

Uses transformer-style attention to capture correlations between clinical
features (symptoms, lab markers, patient context) that simple MLPs miss.
"""

from abc import abstractmethod
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from .base_model import BaseMedicalModel




class ClinicalAttentionBlock(nn.Module):
    """
    Multi-head self-attention block for clinical feature correlation.

    Learns which symptoms and lab markers are most relevant to each other,
    enabling the model to capture complex clinical relationships (e.g., the
    combination of polyuria + polydipsia is more diagnostic than either alone).
    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(x, x, x)
        return self.norm(x + self.dropout(attended))


class ClinicalFeedForward(nn.Module):
    """Position-wise feed-forward network applied after attention."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class ClinicalTransformerLayer(nn.Module):
    """Single transformer layer: attention + feed-forward with residuals."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = ClinicalAttentionBlock(d_model, num_heads, dropout)
        self.ffn = ClinicalFeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.attention(x))


class BaseMedicalLLM(nn.Module, BaseMedicalModel):
    """
    Abstract base for all specialized medical LLMs.

    Architecture:
        input features → feature embedding → N transformer layers → classifier head

    Subclasses supply:
        - symptom_mapping / condition_mapping / lab_marker_mapping dicts
        - recommended_exams and treatment_guidelines dicts
        - clinical thresholds for urgency assessment
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.num_classes = num_classes

        # Project raw features into the attention space
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Stack of transformer layers
        self.transformer_layers = nn.ModuleList([
            ClinicalTransformerLayer(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_layers)
        ])

        # Diagnostic classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        # Urgency assessment head (independent of diagnosis)
        self.urgency_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 3),  # Low / Medium / High
        )

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.feature_embedding(x).unsqueeze(1)  # (B, 1, d_model)
        for layer in self.transformer_layers:
            embedded = layer(embedded)
        pooled = embedded.squeeze(1)  # (B, d_model)
        return self.classifier(pooled), self.urgency_head(pooled)

    def preprocess(self, input_data: Dict[str, Any]) -> torch.Tensor:
        total_size = self.input_size
        feature_vector = [0.0] * total_size

        symptoms = input_data.get("symptoms", {})
        for symptom, value in symptoms.items():
            if symptom in self.symptom_mapping:
                idx = self.symptom_mapping[symptom]
                if idx < total_size:
                    feature_vector[idx] = float(min(max(value, 0.0), 1.0))

        lab_markers = input_data.get("lab_markers", {})
        for marker, value in lab_markers.items():
            if marker in self.lab_marker_mapping:
                idx = self.lab_marker_mapping[marker]
                if idx < total_size:
                    feature_vector[idx] = float(min(max(value, 0.0), 1.0))

        patient_context = input_data.get("patient_context", {})
        for key, value in patient_context.items():
            if key in self.context_mapping:
                idx = self.context_mapping[key]
                if idx < total_size:
                    feature_vector[idx] = float(min(max(value, 0.0), 1.0))

        return torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)

    def postprocess(self, predictions: torch.Tensor) -> Dict[str, Any]:
        diag_logits, urgency_logits = predictions
        probabilities = torch.softmax(diag_logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

        diagnosis = self.condition_mapping.get(predicted_class.item(), "Unknown")
        urgency_probs = torch.softmax(urgency_logits, dim=1)
        urgency_labels = ["Low", "Medium", "High"]
        urgency_idx = torch.argmax(urgency_probs, dim=1).item()
        urgency = urgency_labels[urgency_idx]

        top_k = min(3, self.num_classes)
        top_probs, top_classes = torch.topk(probabilities, top_k)
        differential = [
            {
                "condition": self.condition_mapping.get(cls.item(), "Unknown"),
                "probability": round(top_probs[0][i].item(), 4),
            }
            for i, cls in enumerate(top_classes[0])
        ]

        return {
            "primary_diagnosis": diagnosis,
            "confidence": round(confidence.item(), 4),
            "differential_diagnoses": differential,
            "recommended_examinations": self.recommended_exams.get(diagnosis, []),
            "treatment_suggestions": self.treatment_guidelines.get(diagnosis, []),
            "urgency_level": urgency,
            "clinical_disclaimer": (
                "These results are clinical decision support only. "
                "The attending physician is responsible for the final diagnosis and treatment plan."
            ),
        }

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.eval()
        if not self.validate_input(input_data):
            return {
                "error": "Invalid input format",
                "required_fields": ["symptoms"],
                "optional_fields": ["lab_markers", "patient_context"],
            }
        try:
            with torch.no_grad():
                tensor = self.preprocess(input_data)
                diag_logits, urgency_logits = self.forward(tensor)
                result = self.postprocess((diag_logits, urgency_logits))
                result["success"] = True
                return result
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return isinstance(input_data, dict) and (
            "symptoms" in input_data or "lab_markers" in input_data
        )

    def get_model_info(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "model_class": self.__class__.__name__,
            "input_size": self.input_size,
            "d_model": self.d_model,
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "supported_symptoms": list(self.symptom_mapping.keys()),
            "supported_lab_markers": list(self.lab_marker_mapping.keys()),
            "supported_conditions": list(self.condition_mapping.values()),
            "architecture": "Clinical Transformer (attention-based)",
            "framework": "PyTorch",
        }
