"""Utility functions: GP kernel, Cholesky decomposition, matrix inversion."""

import math
import numpy as np
from competition.energy_arbitrage.python import constants


class GPKernel:
    """Gaussian Process kernel for price generation."""

    def __init__(self):
        self.sigma_periodic: float = 10.0
        self.length_periodic: float = 2.0
        self.sigma_se: float = 5.0
        self.length_se: float = 4.0
        self.period_hours: float = 24.0

    def evaluate(self, t1: float, t2: float) -> float:
        # Convert step indices to hours
        h1 = t1 * constants.DELTA_T
        h2 = t2 * constants.DELTA_T
        tau = abs(h1 - h2)

        # Periodic component
        periodic = (self.sigma_periodic**2) * math.exp(
            -2.0 * math.sin(math.pi * tau / self.period_hours) ** 2 / self.length_periodic**2
        )

        # Squared exponential component
        se = (self.sigma_se**2) * math.exp(-(tau**2) / (2.0 * self.length_se**2))

        return periodic + se

    def covariance_matrix(self, n: int) -> list[list[float]]:
        k = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                k[i][j] = self.evaluate(float(i), float(j))
                if i == j:
                    k[i][j] += 1e-6  # Numerical stability
        return k


def cholesky(a: list[list[float]]) -> list[list[float]]:
    """Cholesky decomposition (lower triangular) using numpy."""
    n = len(a)
    if n == 0:
        return []
    matrix = np.array(a, dtype=np.float64)
    l_matrix = np.linalg.cholesky(matrix)
    return l_matrix.tolist()


def invert_matrix(a: list[list[float]]) -> list[list[float]]:
    """Matrix inversion via Cholesky decomposition (for symmetric positive definite matrices)."""
    n = len(a)
    if n == 0:
        return []
    matrix = np.array(a, dtype=np.float64)
    l_matrix = np.linalg.cholesky(matrix)
    l_inv = np.linalg.inv(l_matrix)
    inverse = l_inv.T @ l_inv
    return inverse.tolist()
