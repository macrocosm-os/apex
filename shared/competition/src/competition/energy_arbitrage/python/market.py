"""Market price generation matching the Rust energy_arbitrage::market module."""

import math
from dataclasses import dataclass
from competition.energy_arbitrage.python import constants
from competition.energy_arbitrage.python.utils import GPKernel, cholesky


@dataclass
class MarketParams:
    """Market parameters for price generation."""

    volatility: float
    jump_probability: float
    tail_index: float


@dataclass
class Market:
    """Market with day-ahead prices and RT price generation."""

    params: MarketParams
    day_ahead_prices: list[list[float]]  # [time_step][node]

    @staticmethod
    def generate_instance(
        rng,
        params: MarketParams,
        num_nodes: int,
        num_steps: int,
    ) -> "Market":
        """Generate a market instance with given parameters."""
        # Base price curve via Gaussian Process
        kernel = GPKernel()
        k = kernel.covariance_matrix(num_steps)
        l_matrix = cholesky(k)

        # Generate standard normal samples
        z = [rng.gauss(0.0, 1.0) for _ in range(num_steps)]

        base_prices = [0.0] * num_steps
        for i in range(num_steps):
            for j in range(num_steps):
                base_prices[i] += l_matrix[i][j] * z[j]
            # Add diurnal pattern based on 15-min steps
            hour = i * constants.DELTA_T
            base_prices[i] += constants.MEAN_DA_PRICE + constants.DA_AMPLITUDE * math.sin(
                2.0 * math.pi * hour / 24.0 - math.pi / 2.0
            )
            base_prices[i] = max(base_prices[i], constants.LAMBDA_DA_MIN)

        # Generate node offsets (correlated AR(1) residual)
        prices = [[0.0] * num_nodes for _ in range(num_steps)]
        ar_coef = 0.8

        for node in range(num_nodes):
            offset = 5.0 * (rng.random() - 0.5)
            residual = 0.0

            for t in range(num_steps):
                residual = ar_coef * residual + math.sqrt(1.0 - ar_coef * ar_coef) * 2.0 * rng.gauss(0.0, 1.0)
                price = base_prices[t] + offset + residual
                prices[t][node] = max(price, constants.LAMBDA_DA_MIN)

        return Market(params=params, day_ahead_prices=prices)

    def generate_rt_prices(
        self,
        rng,
        time_step: int,
        congestion_indicators: list[bool],
    ) -> list[float]:
        """Generate real-time prices for a given time step."""
        num_nodes = len(self.day_ahead_prices[0])
        prices = []

        # Draw common factor z_t
        z_common = rng.gauss(0.0, 1.0)
        # Draw z'_t for congestion premium
        z_prime = rng.gauss(0.0, 1.0)
        zeta = max(z_prime, 0.0)

        for i in range(num_nodes):
            da_price = self.day_ahead_prices[time_step][i]

            # Draw idiosyncratic shock
            eps_i = rng.gauss(0.0, 1.0)

            # Spatially correlated shock
            rho = constants.RHO_SPATIAL
            xi_i = math.sqrt(rho) * z_common + math.sqrt(1.0 - rho) * eps_i

            # Base price with shock
            mu = constants.MU_BIAS
            sigma = self.params.volatility
            price = da_price * (1.0 + mu + sigma * xi_i)

            # Congestion premium
            if congestion_indicators[i]:
                price += constants.GAMMA_PRICE * zeta

            # Jump component
            u_jump = rng.random()
            if u_jump < self.params.jump_probability:
                u_pareto = max(rng.random(), 1e-10)
                # Pareto: X = (1-U)^(-1/alpha), support [1, inf)
                pareto = (1.0 - u_pareto) ** (-1.0 / self.params.tail_index)
                jump = da_price * pareto
                price += jump

            prices.append(max(constants.LAMBDA_MIN, min(price, constants.LAMBDA_MAX)))

        return prices
