# optimizers/gradient_descent.py
from core.quantum_state import QuantumState


def gradient_descent(initial_state: QuantumState, compute_loss, learning_rate: float = 0.01, iterations: int = 100):
    """
    Gradient descent for QuantumState.

    :param initial_state: The QuantumState to optimize.
    :param compute_loss: Function that takes a state and returns (loss, gradient).
    :param learning_rate: Learning rate.
    :param iterations: Number of iterations.
    """
    state = initial_state

    for i in range(iterations):
        loss = compute_loss(state)
        state.backward()

        if state.grad is None:
            raise ValueError("Gradients were not computed correctly.")

        state.data -= learning_rate * state.grad
        print(f"Iteration {i}: Loss = {loss}")

    return state
