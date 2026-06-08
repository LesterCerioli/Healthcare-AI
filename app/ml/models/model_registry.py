
import os
from typing import Dict, Any, Optional
import torch
from app.ml.models.base_model import BaseMedicalModel


class ModelRegistry:
    """Registry for managing AI model instances and versions."""

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self.model_directory = "app/ml/models/saved_models"
        os.makedirs(self.model_directory, exist_ok=True)

    # ── Registration ────────────────────────────────────────────────────────

    def register_model(self, model_name: str, model: Any) -> None:
        """Register a model instance under the given name."""
        self._models[model_name] = model

    def get_model(self, model_name: str) -> Optional[Any]:
        """Return a registered model or None if not found."""
        return self._models.get(model_name)

    def list_models(self) -> list:
        """List all registered model names."""
        return list(self._models.keys())

    # ── Persistence ─────────────────────────────────────────────────────────

    def save_model(self, model_name: str, model: Any) -> str:
        """Persist model weights to disk and return the saved path."""
        model_path = os.path.join(self.model_directory, f"{model_name}.pth")
        torch.save(model.state_dict(), model_path)
        return model_path

    def load_model(
        self,
        model_name: str,
        model_path: str,
        model_instance: Optional[Any] = None,
    ) -> Any:
        """
        Load weights from disk into a model instance.

        If model_instance is None, falls back to the already-registered model
        or a default UltraLightMedicalModel (legacy behaviour).
        For LLM models, always supply a pre-constructed model_instance.
        """
        model = model_instance or self._models.get(model_name)
        if model is None:
            from app.ml.models.ultra_light_model import UltraLightMedicalModel
            model = UltraLightMedicalModel(input_size=30, hidden_size=32, num_classes=5)

        if os.path.exists(model_path):
            model.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu"))
            )
        model.eval()
        self.register_model(model_name, model)
        return model

    # ── Specialised LLM loading ─────────────────────────────────────────────

    def load_diabetes_llm(self, checkpoint_path: Optional[str] = None) -> Any:
        """Return a DiabetesLLM instance, optionally restoring saved weights."""
        from app.ml.models.diabetes_llm import create_diabetes_llm
        model = create_diabetes_llm()
        if checkpoint_path and os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.register_model("diabetes_llm", model)
        return model

    def load_cardiovascular_llm(self, checkpoint_path: Optional[str] = None) -> Any:
        """Return a CardiovascularLLM instance, optionally restoring saved weights."""
        from app.ml.models.cardiovascular_llm import create_cardiovascular_llm
        model = create_cardiovascular_llm()
        if checkpoint_path and os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.register_model("cardiovascular_llm", model)
        return model

    def load_symptom_analysis_llm(self, checkpoint_path: Optional[str] = None) -> Any:
        """Return a SymptomAnalysisLLM instance, optionally restoring saved weights."""
        from app.ml.models.symptom_analysis_llm import create_symptom_analysis_llm
        model = create_symptom_analysis_llm()
        if checkpoint_path and os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.register_model("symptom_analysis_llm", model)
        return model

    # ── TensorFlow Diabetes LLMs ────────────────────────────────────────────

    def load_diabetes_type1_llm_tf(self, checkpoint_path: Optional[str] = None) -> Any:
        """Return a TF DiabetesType1LLM instance, optionally restoring saved weights."""
        from app.ml.models.diabetes_type1_llm_tf import create_diabetes_type1_llm
        model = create_diabetes_type1_llm()
        if checkpoint_path and os.path.exists(checkpoint_path):
            model.load_weights(checkpoint_path)
        self.register_model("diabetes_type1_llm_tf", model)
        return model

    def load_diabetes_type2_llm_tf(self, checkpoint_path: Optional[str] = None) -> Any:
        """Return a TF DiabetesType2LLM instance, optionally restoring saved weights."""
        from app.ml.models.diabetes_type2_llm_tf import create_diabetes_type2_llm
        model = create_diabetes_type2_llm()
        if checkpoint_path and os.path.exists(checkpoint_path):
            model.load_weights(checkpoint_path)
        self.register_model("diabetes_type2_llm_tf", model)
        return model

    def load_diabetes_gestational_llm_tf(self, checkpoint_path: Optional[str] = None) -> Any:
        """Return a TF DiabetesGestationalLLM instance, optionally restoring saved weights."""
        from app.ml.models.diabetes_gestational_llm_tf import create_diabetes_gestational_llm
        model = create_diabetes_gestational_llm()
        if checkpoint_path and os.path.exists(checkpoint_path):
            model.load_weights(checkpoint_path)
        self.register_model("diabetes_gestational_llm_tf", model)
        return model

    # ── Model info ──────────────────────────────────────────────────────────

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Return metadata for a registered model."""
        model = self.get_model(model_name)
        if model is None:
            return {"error": f"Model '{model_name}' not registered"}
        if hasattr(model, "get_model_info"):
            return model.get_model_info()
        return {
            "model_name": model_name,
            "class": type(model).__name__,
            "total_parameters": sum(p.numel() for p in model.parameters()),
        }

    def get_registry_summary(self) -> Dict[str, Any]:
        """Return a summary of all registered models."""
        return {
            name: {
                "class": type(model).__name__,
                "parameters": sum(p.numel() for p in model.parameters()),
            }
            for name, model in self._models.items()
        }
