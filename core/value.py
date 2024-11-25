from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


class QuantumState:
    """
    Represents a quantum state (vector) and its associated gradient information.
    """

    def __init__(self, data: np.ndarray, _children: tuple = (), _op: str = "", label: str = "") -> None:
        self.data = data
        self.grad = np.zeros_like(data, dtype=np.complex128)  # Gradients are complex for quantum states
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._cache = {}

    def __repr__(self) -> str:
        return f"QuantumState(data={self.data}, grad={self.grad})"

    def apply_gate(self, gate: np.ndarray, label: str = "") -> QuantumState:
        """
        Apply a quantum gate (unitary matrix) to the quantum state.
        """
        if isinstance(gate, csr_matrix):
            out_data = gate.dot(self.data)
        else:
            out_data = gate @ self.data

        out = QuantumState(out_data, (self,), "apply_gate", label)

        def _backward() -> None:
            if "gate_grad" not in self._cache:
                self._cache["gate_grad"] = gate.T.conj()
            self.grad += self._cache["gate_grad"] @ out.grad

        out._backward = _backward
        return out

    def measure_expectation(self, observable: np.ndarray, label: str = "") -> QuantumState:
        """
        Measure the expectation value of an observable with respect to the state.
        """
        expectation = np.vdot(self.data, observable @ self.data).real
        out = QuantumState(np.array([expectation]), (self,), "measure", label)

        def _backward() -> None:
            if "obs_grad" not in self._cache:
                self._cache["obs_grad"] = observable @ self.data
            self.grad += 2 * self._cache["obs_grad"].conj() * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        topo = self._tsort(self)

        self.grad = np.ones_like(self.data, dtype=np.complex128)
        for v in reversed(topo):
            v._backward()

    @staticmethod
    def _tsort(start: QuantumState) -> list[QuantumState]:
        """Iterative topological sort for handling large graphs."""
        topo = []
        visited = set()
        stack = [start]

        while stack:
            node = stack.pop()
            if node in visited:
                topo.append(node)
                continue

            visited.add(node)
            stack.append(node)

            for child in node._prev:
                if child not in visited:
                    stack.append(child)

        return topo


# Define quantum gates
def hadamard_gate() -> np.ndarray:
    """Returns a 2x2 Hadamard gate."""
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def pauli_z_gate() -> np.ndarray:
    """Returns a 2x2 Pauli-Z gate."""
    return np.array([[1, 0], [0, -1]])


# Define an observable
def observable_z() -> np.ndarray:
    """Returns the Z observable."""
    return np.array([[1, 0], [0, -1]])


# Example usage
if __name__ == '__main__':
    # Define initial quantum state |0⟩
    state = QuantumState(np.array([1, 0], dtype=np.complex128), label="|0⟩")

    # Apply Hadamard gate
    H = hadamard_gate()
    state_H = state.apply_gate(H, label="H|0⟩")

    # Measure expectation value of Pauli-Z
    Z = observable_z()
    expectation = state_H.measure_expectation(Z, label="⟨Z⟩")

    # Perform backward pass
    expectation.backward()

    # Results
    print("Quantum state after Hadamard:", state_H)
    print("Expectation value of Z:", expectation.data)
    print("Gradient w.r.t. initial state:", state.grad)