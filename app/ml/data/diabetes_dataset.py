"""
Synthetic dataset generator for Diabetes Mellitus LLM training.

Generates clinically plausible patient records with symptoms, lab markers,
and patient context, labelled with one of six diabetic condition classes.

In production, replace generate_synthetic_records() with a real data
loader that reads from the PostgreSQL database.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict


CONDITION_LABELS = {
    "Normal Glucose Regulation": 0,
    "Pre-diabetes (Impaired Glucose Tolerance)": 1,
    "Type 2 Diabetes Mellitus": 2,
    "Type 1 Diabetes Mellitus": 3,
    "Gestational Diabetes Mellitus": 4,
    "Maturity-Onset Diabetes of the Young (MODY)": 5,
}

NUM_FEATURES = 26  # Must match DiabetesLLM.input_size


class DiabetesDataset(Dataset):
    """PyTorch dataset for DiabetesLLM training and evaluation."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def _clip(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _generate_normal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Normal glucose regulation: low symptom burden, normal labs."""
    records = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    for i in range(n):
        # Minimal / no metabolic symptoms
        records[i, 0] = _clip(rng.uniform(0.0, 0.1))   # polyuria
        records[i, 1] = _clip(rng.uniform(0.0, 0.1))   # polydipsia
        records[i, 4] = _clip(rng.uniform(0.0, 0.2))   # fatigue (lifestyle)
        # Normal labs
        records[i, 13] = _clip(rng.uniform(0.0, 0.25))  # blood_glucose normal
        records[i, 14] = _clip(rng.uniform(0.0, 0.2))   # hba1c <5.7%
        records[i, 15] = _clip(rng.uniform(0.0, 0.2))   # fasting_glucose normal
        records[i, 16] = _clip(rng.uniform(0.0, 0.4))   # bmi
        records[i, 17] = _clip(rng.uniform(0.4, 0.8))   # insulin normal
        records[i, 18] = _clip(rng.uniform(0.4, 0.8))   # c_peptide normal
        records[i, 20] = _clip(rng.uniform(0.5, 1.0))   # hdl good
        # Context
        records[i, 21] = _clip(rng.uniform(0.0, 0.6))   # age_group
        records[i, 25] = _clip(rng.uniform(0.3, 1.0))   # physical activity
    return records


def _generate_prediabetes(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pre-diabetes: mild symptom elevation, borderline labs."""
    records = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    for i in range(n):
        records[i, 0] = _clip(rng.uniform(0.1, 0.3))   # polyuria mild
        records[i, 1] = _clip(rng.uniform(0.1, 0.3))   # polydipsia mild
        records[i, 4] = _clip(rng.uniform(0.2, 0.5))   # fatigue moderate
        records[i, 13] = _clip(rng.uniform(0.2, 0.5))  # blood_glucose borderline
        records[i, 14] = _clip(rng.uniform(0.2, 0.45)) # hba1c 5.7–6.4%
        records[i, 15] = _clip(rng.uniform(0.2, 0.5))  # fasting_glucose 100–125
        records[i, 16] = _clip(rng.uniform(0.3, 0.7))  # bmi overweight
        records[i, 19] = _clip(rng.uniform(0.3, 0.7))  # triglycerides elevated
        records[i, 21] = _clip(rng.uniform(0.3, 0.8))  # age_group
        records[i, 22] = float(rng.integers(0, 2))     # family_history
    return records


def _generate_type2(n: int, rng: np.random.Generator) -> np.ndarray:
    """Type 2 DM: prominent metabolic syndrome features, elevated labs."""
    records = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    for i in range(n):
        records[i, 0] = _clip(rng.uniform(0.4, 0.9))   # polyuria prominent
        records[i, 1] = _clip(rng.uniform(0.4, 0.9))   # polydipsia
        records[i, 2] = _clip(rng.uniform(0.3, 0.7))   # polyphagia
        records[i, 4] = _clip(rng.uniform(0.4, 0.8))   # fatigue
        records[i, 5] = _clip(rng.uniform(0.2, 0.6))   # blurred vision
        records[i, 6] = _clip(rng.uniform(0.2, 0.6))   # slow healing
        records[i, 7] = _clip(rng.uniform(0.2, 0.6))   # frequent infections
        records[i, 9] = _clip(rng.uniform(0.2, 0.7))   # acanthosis nigricans
        records[i, 13] = _clip(rng.uniform(0.5, 0.95)) # blood_glucose high
        records[i, 14] = _clip(rng.uniform(0.45, 0.9)) # hba1c ≥6.5%
        records[i, 15] = _clip(rng.uniform(0.5, 0.9))  # fasting_glucose ≥126
        records[i, 16] = _clip(rng.uniform(0.5, 1.0))  # bmi obese
        records[i, 18] = _clip(rng.uniform(0.5, 1.0))  # c_peptide elevated
        records[i, 19] = _clip(rng.uniform(0.5, 1.0))  # triglycerides high
        records[i, 21] = _clip(rng.uniform(0.4, 1.0))  # older age
        records[i, 22] = float(rng.integers(0, 2))
        records[i, 23] = float(rng.integers(0, 2))     # hypertension
    return records


def _generate_type1(n: int, rng: np.random.Generator) -> np.ndarray:
    """Type 1 DM: acute onset, low/absent C-peptide, DKA features."""
    records = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    for i in range(n):
        records[i, 0] = _clip(rng.uniform(0.6, 1.0))   # polyuria severe
        records[i, 1] = _clip(rng.uniform(0.6, 1.0))   # polydipsia severe
        records[i, 3] = _clip(rng.uniform(0.5, 0.9))   # weight loss
        records[i, 4] = _clip(rng.uniform(0.5, 0.9))   # fatigue severe
        records[i, 10] = _clip(rng.uniform(0.4, 0.9))  # fruity breath (DKA)
        records[i, 11] = _clip(rng.uniform(0.3, 0.8))  # abdominal pain
        records[i, 13] = _clip(rng.uniform(0.7, 1.0))  # blood_glucose very high
        records[i, 14] = _clip(rng.uniform(0.5, 0.9))  # hba1c elevated
        records[i, 15] = _clip(rng.uniform(0.6, 1.0))  # fasting_glucose very high
        records[i, 16] = _clip(rng.uniform(0.0, 0.4))  # bmi normal/low
        records[i, 17] = _clip(rng.uniform(0.8, 1.0))  # insulin absent → high norm value
        records[i, 18] = _clip(rng.uniform(0.0, 0.2))  # c_peptide absent/low
        records[i, 21] = _clip(rng.uniform(0.0, 0.4))  # younger age
    return records


def _generate_gestational(n: int, rng: np.random.Generator) -> np.ndarray:
    """Gestational DM: female reproductive-age context, moderate lab elevation."""
    records = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    for i in range(n):
        records[i, 0] = _clip(rng.uniform(0.3, 0.7))   # polyuria (pregnancy overlay)
        records[i, 1] = _clip(rng.uniform(0.2, 0.6))   # polydipsia
        records[i, 4] = _clip(rng.uniform(0.3, 0.7))   # fatigue
        records[i, 13] = _clip(rng.uniform(0.3, 0.7))  # blood_glucose borderline-high
        records[i, 14] = _clip(rng.uniform(0.2, 0.5))  # hba1c
        records[i, 15] = _clip(rng.uniform(0.3, 0.65)) # fasting_glucose
        records[i, 16] = _clip(rng.uniform(0.4, 0.8))  # bmi
        records[i, 21] = _clip(rng.uniform(0.2, 0.5))  # reproductive age
        records[i, 24] = 1.0                            # gestational_history
    return records


def _generate_mody(n: int, rng: np.random.Generator) -> np.ndarray:
    """MODY: young, non-obese, strong family history, mild-moderate hyperglycaemia."""
    records = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    for i in range(n):
        records[i, 0] = _clip(rng.uniform(0.1, 0.5))   # polyuria mild
        records[i, 1] = _clip(rng.uniform(0.1, 0.4))   # polydipsia mild
        records[i, 4] = _clip(rng.uniform(0.1, 0.4))   # fatigue mild
        records[i, 13] = _clip(rng.uniform(0.3, 0.65)) # blood_glucose moderate
        records[i, 14] = _clip(rng.uniform(0.25, 0.5)) # hba1c moderate
        records[i, 15] = _clip(rng.uniform(0.25, 0.55))# fasting_glucose
        records[i, 16] = _clip(rng.uniform(0.0, 0.35)) # bmi normal
        records[i, 17] = _clip(rng.uniform(0.3, 0.6))  # insulin present (not T1)
        records[i, 18] = _clip(rng.uniform(0.4, 0.7))  # c_peptide present
        records[i, 21] = _clip(rng.uniform(0.0, 0.35)) # young age
        records[i, 22] = 1.0                            # strong family history
    return records


def generate_synthetic_records(
    n_per_class: int = 300,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate n_per_class synthetic patient records for each of the six
    diabetes condition classes.

    Returns:
        features: (N, 26) float32 array
        labels:   (N,) int64 array
    """
    rng = np.random.default_rng(seed)
    generators = [
        _generate_normal,
        _generate_prediabetes,
        _generate_type2,
        _generate_type1,
        _generate_gestational,
        _generate_mody,
    ]
    all_features: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for class_idx, gen_fn in enumerate(generators):
        feats = gen_fn(n_per_class, rng)
        # Add small Gaussian noise for variation
        feats += rng.normal(0, 0.03, feats.shape).astype(np.float32)
        feats = np.clip(feats, 0.0, 1.0)
        all_features.append(feats)
        all_labels.append(np.full(n_per_class, class_idx, dtype=np.int64))

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Shuffle
    perm = rng.permutation(len(features))
    return features[perm], labels[perm]


def get_diabetes_dataloaders(
    n_per_class: int = 300,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/validation DataLoaders for DiabetesLLM training.

    Args:
        n_per_class: Number of synthetic records per condition class.
        batch_size:  Mini-batch size.
        val_split:   Fraction of data reserved for validation.
        seed:        Random seed for reproducibility.

    Returns:
        (train_loader, val_loader)
    """
    features, labels = generate_synthetic_records(n_per_class, seed)
    split = int(len(features) * (1.0 - val_split))
    train_ds = DiabetesDataset(features[:split], labels[:split])
    val_ds = DiabetesDataset(features[split:], labels[split:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def get_class_distribution(labels: np.ndarray) -> Dict[str, int]:
    """Return a mapping of condition name → sample count."""
    inv = {v: k for k, v in CONDITION_LABELS.items()}
    return {inv[i]: int(np.sum(labels == i)) for i in range(len(CONDITION_LABELS))}
