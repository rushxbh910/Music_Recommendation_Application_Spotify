from typing import List, Dict
import numpy as np

def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row‑wise cosine similarity between A (mxd) and B (nxd)."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T

def topk_indices(scores: np.ndarray, k: int) -> List[int]:
    k = min(k, scores.shape[0])
    return np.argpartition(-scores, range(k))[:k]
