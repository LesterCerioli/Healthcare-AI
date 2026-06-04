"""
Lightweight medical dataset optimised for low-memory environments.

The dataset stores raw Python lists and converts them to tensors only when
__getitem__ is called, so no large tensor block is allocated upfront.  This
makes it suitable for the 500 MB RAM cloud constraint.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union


class LightweightMedicalDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for medical symptom / label pairs.

    Key design decisions
    --------------------
    * Raw feature data is stored as a plain Python list of lists (not a tensor),
      so there is no single large allocation in VRAM / RAM.
    * Tensors are created on-the-fly in __getitem__; PyTorch's DataLoader
      collate_fn then batches them.
    * An optional in-place normalisation pass clips every feature value to
      [0.0, 1.0] on construction (one-time, O(n) cost) so inference code does
      not need to duplicate the logic.
    * Supports optional per-sample weights for imbalanced class handling.
    """

    def __init__(
        self,
        features: Union[List[List[float]], np.ndarray],
        labels: Union[List[int], np.ndarray],
        sample_weights: Optional[Union[List[float], np.ndarray]] = None,
        normalize: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        features:
            2-D array-like of shape (N, F) where N is the number of samples
            and F is the number of input features.
        labels:
            1-D array-like of integer class indices, length N.
        sample_weights:
            Optional per-sample importance weights, length N.  When provided
            they are accessible via ``get_sample_weights()`` and can be used
            with a WeightedRandomSampler.
        normalize:
            If True (default) every feature value is clipped to [0, 1].
        """
        if len(features) != len(labels):
            raise ValueError(
                f"features and labels must have the same length, "
                f"got {len(features)} and {len(labels)}"
            )

        # Convert to plain Python lists to avoid large contiguous allocations
        if isinstance(features, np.ndarray):
            raw_features: List[List[float]] = features.tolist()
        else:
            raw_features = [list(row) for row in features]

        if normalize:
            raw_features = [
                [min(max(v, 0.0), 1.0) for v in row] for row in raw_features
            ]

        self._features: List[List[float]] = raw_features
        self._labels: List[int] = (
            labels.tolist() if isinstance(labels, np.ndarray) else list(labels)
        )

        if sample_weights is not None:
            self._weights: Optional[List[float]] = (
                sample_weights.tolist()
                if isinstance(sample_weights, np.ndarray)
                else list(sample_weights)
            )
        else:
            self._weights = None

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_tensor = torch.tensor(self._features[idx], dtype=torch.float32)
        label_tensor = torch.tensor(self._labels[idx], dtype=torch.long)
        return feature_tensor, label_tensor

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_sample_weights(self) -> Optional[torch.Tensor]:
        """Return per-sample weights as a 1-D float tensor, or None."""
        if self._weights is None:
            return None
        return torch.tensor(self._weights, dtype=torch.float32)

    def feature_size(self) -> int:
        """Return the number of input features."""
        return len(self._features[0]) if self._features else 0

    def num_classes(self) -> int:
        """Return the number of unique classes in this split."""
        return len(set(self._labels))

    def class_counts(self) -> Dict[int, int]:
        """Return a mapping of class_index -> sample_count."""
        counts: Dict[int, int] = {}
        for lbl in self._labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    def split(self, train_ratio: float = 0.8) -> Tuple["LightweightMedicalDataset", "LightweightMedicalDataset"]:
        """Split the dataset into train and validation subsets.

        Parameters
        ----------
        train_ratio:
            Fraction of samples assigned to the training split (default 0.8).

        Returns
        -------
        Tuple of (train_dataset, val_dataset).
        """
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be strictly between 0 and 1")
        split_idx = int(len(self) * train_ratio)
        train_ds = LightweightMedicalDataset(
            self._features[:split_idx],
            self._labels[:split_idx],
            sample_weights=self._weights[:split_idx] if self._weights else None,
            normalize=False,  # already normalised
        )
        val_ds = LightweightMedicalDataset(
            self._features[split_idx:],
            self._labels[split_idx:],
            sample_weights=self._weights[split_idx:] if self._weights else None,
            normalize=False,
        )
        return train_ds, val_ds


# ---------------------------------------------------------------------------
# Factory helpers used by trainers
# ---------------------------------------------------------------------------

def _generate_synthetic_data(
    num_samples: int = 500,
    num_symptoms: int = 30,
    num_conditions: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic symptom feature vectors with plausible co-occurrence."""
    rng = np.random.default_rng(seed)

    features = rng.integers(0, 2, size=(num_samples, num_symptoms)).astype(np.float32)

    # Inject some condition-specific symptom patterns
    symptom_patterns = {
        0: [0, 2, 8, 9],          # Common Cold: fever, cough, runny_nose, muscle_pain
        1: [0, 1, 3, 9, 10],      # Influenza: fever, headache, fatigue, muscle_pain, chills
        2: [0, 2, 3, 11, 12],     # COVID-19: fever, cough, fatigue, taste/smell loss
        3: [0, 2, 5, 6],          # Pneumonia: fever, cough, chest_pain, dyspnoea
        4: [2, 6, 7],             # Bronchitis: cough, dyspnoea, sore_throat
    }

    labels = rng.integers(0, num_conditions, size=num_samples)
    for i, lbl in enumerate(labels):
        pattern = symptom_patterns.get(int(lbl), [])
        for idx in pattern:
            if idx < num_symptoms:
                features[i, idx] = float(rng.uniform(0.6, 1.0))

    return features, labels


def create_lightweight_dataset(
    num_samples: int = 500,
    num_symptoms: int = 30,
    num_conditions: int = 5,
    seed: int = 42,
) -> LightweightMedicalDataset:
    """Create a synthetic LightweightMedicalDataset for training/testing."""
    features, labels = _generate_synthetic_data(
        num_samples=num_samples,
        num_symptoms=num_symptoms,
        num_conditions=num_conditions,
        seed=seed,
    )
    return LightweightMedicalDataset(features, labels, normalize=True)


def get_lightweight_dataloader(
    batch_size: int = 8,
    num_samples: int = 500,
    num_symptoms: int = 30,
    num_conditions: int = 5,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) built from synthetic medical data.

    This function is the primary entry point used by the optimised trainer and
    the training CLI.

    Parameters
    ----------
    batch_size:
        Mini-batch size for both loaders (default 8, suited for 500 MB RAM).
    num_samples:
        Total synthetic samples to generate (default 500).
    num_symptoms:
        Feature dimensionality (default 30, matches UltraLightMedicalModel).
    num_conditions:
        Number of diagnostic classes (default 5).
    train_ratio:
        Fraction of data used for training (default 0.8).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Tuple of (train_loader, val_loader).
    """
    full_dataset = create_lightweight_dataset(
        num_samples=num_samples,
        num_symptoms=num_symptoms,
        num_conditions=num_conditions,
        seed=seed,
    )
    train_ds, val_ds = full_dataset.split(train_ratio=train_ratio)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=False,  # Avoid extra RAM overhead on CPU-only systems
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )
    return train_loader, val_loader
