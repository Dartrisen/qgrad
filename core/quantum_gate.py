# core/quantum_gate.py
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from typing import Optional, Union


class QuantumGate:
    """
    Represents a quantum gate with its unitary matrix and optional parameters.
    """

    def __init__(self, matrix: Union[np.ndarray, csr_matrix], name: Optional[str] = None, params: Optional[dict] = None) -> None:
        self.matrix = matrix
        self.name = name or "UnnamedGate"
        self.params = params or {}

    def __repr__(self) -> str:
        return f"QuantumGate(name={self.name}, params={self.params})"
