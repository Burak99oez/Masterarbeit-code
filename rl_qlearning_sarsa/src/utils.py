import os
import random
import numpy as np

def set_seed(seed: int) -> None:
    """Setzt Seeds für Reproduzierbarkeit (Python + NumPy)."""
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
