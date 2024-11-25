import unittest

import numpy as np
from scipy.sparse import csr_matrix

from core.value import QuantumState, hadamard_gate, pauli_z_gate, observable_z


class TestQuantumState(unittest.TestCase):
    def setUp(self):
        """Set up initial states and gates for testing."""
        self.state_0 = QuantumState(np.array([1, 0], dtype=np.complex128), label="|0⟩")
        self.state_1 = QuantumState(np.array([0, 1], dtype=np.complex128), label="|1⟩")
        self.hadamard = hadamard_gate()
        self.pauli_z = pauli_z_gate()
        self.observable_z = observable_z()

    def test_initial_state(self):
        """Test the initialization of quantum states."""
        self.assertTrue(np.allclose(self.state_0.data, [1, 0]))
        self.assertTrue(np.allclose(self.state_1.data, [0, 1]))

    def test_hadamard_on_0(self):
        """Test applying the Hadamard gate to |0⟩."""
        state_H = self.state_0.apply_gate(self.hadamard, label="H|0⟩")
        expected = np.array([1, 1]) / np.sqrt(2)
        self.assertTrue(np.allclose(state_H.data, expected))

    def test_hadamard_on_1(self):
        """Test applying the Hadamard gate to |1⟩."""
        state_H = self.state_1.apply_gate(self.hadamard, label="H|1⟩")
        expected = np.array([1, -1]) / np.sqrt(2)
        self.assertTrue(np.allclose(state_H.data, expected))

    def test_pauli_z_on_0(self):
        """Test applying the Pauli-Z gate to |0⟩."""
        state_Z = self.state_0.apply_gate(self.pauli_z, label="Z|0⟩")
        expected = np.array([1, 0])  # Pauli-Z does not change |0⟩
        self.assertTrue(np.allclose(state_Z.data, expected))

    def test_pauli_z_on_1(self):
        """Test applying the Pauli-Z gate to |1⟩."""
        state_Z = self.state_1.apply_gate(self.pauli_z, label="Z|1⟩")
        expected = np.array([0, -1])  # Pauli-Z flips the phase of |1⟩
        self.assertTrue(np.allclose(state_Z.data, expected))

    def test_expectation_z_on_hadamard(self):
        """Test expectation value of Z observable on H|0⟩."""
        state_H = self.state_0.apply_gate(self.hadamard, label="H|0⟩")
        expectation = state_H.measure_expectation(self.observable_z, label="⟨Z⟩")
        self.assertAlmostEqual(expectation.data[0], 0.0)

    def test_gradients(self):
        """Test gradients after backpropagation."""
        state_H = self.state_0.apply_gate(self.hadamard, label="H|0⟩")
        expectation = state_H.measure_expectation(self.observable_z, label="⟨Z⟩")
        expectation.backward()
        expected_grad = np.array([0 + 0j, 2 + 0j])  # Analytical gradient w.r.t. |0⟩
        self.assertTrue(np.allclose(self.state_0.grad, expected_grad))

    def test_backward_propagation(self):
        """Test backpropagation of gradients through the computational graph."""
        state_H = self.state_0.apply_gate(self.hadamard, label="H|0⟩")
        expectation = state_H.measure_expectation(self.observable_z, label="⟨Z⟩")
        expectation.backward()

        # Check if gradients are computed correctly
        expected_state_H_grad = np.array([np.sqrt(2), -np.sqrt(2)])
        expected_state_0_grad = np.array([0 + 0j, 2 + 0j])

        # Check state_H gradients
        self.assertTrue(np.allclose(state_H.grad, expected_state_H_grad), f"state_H.grad: {state_H.grad}")
        self.assertTrue(np.allclose(self.state_0.grad, expected_state_0_grad), f"self.state_0.grad: {self.state_0.grad}")

    def test_large_graph(self):
        """Test performance and correctness for a large computational graph."""
        np.random.seed(42)
        size = 1000
        depth = 100

        # Random initial state
        state = QuantumState(np.random.rand(size) + 1j * np.random.rand(size), label="random_state")

        # Random gates
        gates = [csr_matrix(np.random.rand(size, size)) for _ in range(depth)]

        # Build a deep computational graph
        current_state = state
        for i, gate in enumerate(gates):
            current_state = current_state.apply_gate(gate, label=f"Gate_{i}")

        # Random observable
        observable = csr_matrix(np.random.rand(size, size))
        expectation = current_state.measure_expectation(observable, label="Final_expectation")

        # Backpropagation
        expectation.backward()

        # Check that gradients are non-zero
        self.assertTrue(np.any(state.grad != 0), "Gradients did not propagate to the initial state.")

    def test_complex_graph(self):
        """Test a graph with overlapping dependencies."""
        state_H1 = self.state_0.apply_gate(self.hadamard, label="H|0⟩")
        state_H2 = self.state_0.apply_gate(self.hadamard, label="H|0⟩ (again)")
        combined_state = QuantumState(state_H1.data + state_H2.data, (state_H1, state_H2), "combine")

        expectation = combined_state.measure_expectation(self.pauli_z, label="⟨Z⟩")
        expectation.backward()
        print(self.state_0.grad)
        # Ensure gradients are distributed correctly
        self.assertTrue(np.any(self.state_0.grad != 0), "Gradients did not propagate correctly.")
        self.assertTrue(np.any(state_H1.grad != 0) and np.any(state_H2.grad != 0), "Child gradients are missing.")


if __name__ == '__main__':
    unittest.main()
