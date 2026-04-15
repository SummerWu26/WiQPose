import numpy as np
from typing import List

class CSIPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, csi_samples: List[np.ndarray]):
        all_data = np.concatenate([csi.flatten() for csi in csi_samples])
        self.mean = np.mean(all_data)
        self.std = np.std(all_data)
        print(f"CSI: mean={self.mean:.4f}, std={self.std:.4f}")

    def transform(self, csi_data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return csi_data
        return (csi_data - self.mean) / (self.std + 1e-8)