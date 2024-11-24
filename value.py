from __future__ import annotations

import numpy as np


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

    def __repr__(self) -> str:
        return f"QuantumState(data={self.data}, grad={self.grad})"

    def apply_gate(self, gate: np.ndarray, label: str = "") -> QuantumState:
        """
        Apply a quantum gate (unitary matrix) to the quantum state.
        """
        out_data = gate @ self.data
        out = QuantumState(out_data, (self,), "apply_gate", label)

        def _backward() -> None:
            self.grad += gate.T.conj() @ out.grad  # Use Hermitian transpose for backward propagation

        out._backward = _backward
        return out

    def measure_expectation(self, observable: np.ndarray, label: str = "") -> QuantumState:
        """
        Measure the expectation value of an observable with respect to the state.
        """
        expectation = np.vdot(self.data, observable @ self.data).real
        out = QuantumState(np.array([expectation]), (self,), "measure", label)

        def _backward() -> None:
            self.grad += 2 * (observable @ self.data).conj() * out.grad  # Chain rule for expectation

        out._backward = _backward
        return out

    def backward(self) -> None:
        """
        Perform backpropagation to compute gradients through the computational graph.
        """
        topo = []
        visited = set()

        def build_topo(v: QuantumState) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data, dtype=np.complex128)  # Start with seed gradient
        for v in reversed(topo):
            v._backward()


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
