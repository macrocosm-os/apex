"""Network topology and DC power flow matching the Rust energy_arbitrage::network module."""

import math
from dataclasses import dataclass
from competition.energy_arbitrage.python import constants
from competition.energy_arbitrage.python.utils import invert_matrix


@dataclass
class Network:
    """Network topology and DC power flow parameters."""

    num_nodes: int
    num_lines: int
    lines: list[tuple[int, int]]
    susceptances: list[float]
    nominal_flow_limits: list[float]
    flow_limits: list[float]
    ptdf: list[list[float]]
    slack_bus: int
    node_incident_lines: list[list[int]]
    congestion_threshold: float

    @staticmethod
    def generate_instance(rng, num_nodes: int, num_lines: int, gamma_cong: float) -> "Network":
        """Generate a connected network with given parameters."""
        lines: list[tuple[int, int]] = []
        susceptances: list[float] = []
        nominal_flow_limits: list[float] = []

        # Phase 1: Create spanning tree using random edges
        connected = [False] * num_nodes
        connected[0] = True
        connected_count = 1

        while connected_count < num_nodes:
            # Pick a random unconnected node
            unconnected = [i for i in range(num_nodes) if not connected[i]]
            new_node = unconnected[rng.randint(0, len(unconnected) - 1)]

            # Connect to a random connected node
            connected_nodes = [i for i in range(num_nodes) if connected[i]]
            existing = connected_nodes[rng.randint(0, len(connected_nodes) - 1)]

            if new_node < existing:
                from_node, to_node = new_node, existing
            else:
                from_node, to_node = existing, new_node

            lines.append((from_node, to_node))
            susceptances.append(constants.BASE_SUSCEPTANCE * (0.8 + 0.4 * rng.random()))
            nominal_flow_limits.append(constants.NOMINAL_FLOW_LIMIT * (0.8 + 0.4 * rng.random()))

            connected[new_node] = True
            connected_count += 1

        # Phase 2: Add extra lines to reach target
        extra_needed = max(num_lines - len(lines), 0)
        if extra_needed > 0:
            existing_set = set(lines)

            # Build list of all non-existing edges
            candidates = []
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if (i, j) not in existing_set:
                        candidates.append((i, j))

            # Fisher-Yates partial shuffle
            to_add = min(extra_needed, len(candidates))
            for k in range(to_add):
                idx = rng.randint(k, len(candidates) - 1)
                candidates[k], candidates[idx] = candidates[idx], candidates[k]

                from_node, to_node = candidates[k]
                lines.append((from_node, to_node))
                susceptances.append(constants.BASE_SUSCEPTANCE * (0.8 + 0.4 * rng.random()))
                nominal_flow_limits.append(constants.NOMINAL_FLOW_LIMIT * (0.8 + 0.4 * rng.random()))

        # Compute PTDF matrix
        slack_bus = constants.SLACK_BUS
        ptdf = Network._compute_ptdf(num_nodes, lines, susceptances, slack_bus)

        # Apply congestion scaling to get effective limits
        flow_limits = [f * gamma_cong for f in nominal_flow_limits]

        # Build node-to-incident-lines mapping
        node_incident_lines: list[list[int]] = [[] for _ in range(num_nodes)]
        for l, (from_node, to_node) in enumerate(lines):
            node_incident_lines[from_node].append(l)
            node_incident_lines[to_node].append(l)

        return Network(
            num_nodes=num_nodes,
            num_lines=num_lines,
            lines=lines,
            susceptances=susceptances,
            nominal_flow_limits=nominal_flow_limits,
            flow_limits=flow_limits,
            ptdf=ptdf,
            slack_bus=slack_bus,
            node_incident_lines=node_incident_lines,
            congestion_threshold=constants.TAU_CONG,
        )

    @staticmethod
    def _compute_ptdf(
        num_nodes: int,
        lines: list[tuple[int, int]],
        susceptances: list[float],
        slack_bus: int,
    ) -> list[list[float]]:
        """Compute PTDF matrix using DC power flow."""
        if num_nodes == 0 or len(lines) == 0:
            return []

        # Build bus susceptance matrix B (n x n)
        b_matrix = [[0.0] * num_nodes for _ in range(num_nodes)]
        for l, (i, j) in enumerate(lines):
            b = susceptances[l]
            b_matrix[i][i] += b
            b_matrix[j][j] += b
            b_matrix[i][j] -= b
            b_matrix[j][i] -= b

        # Remove slack bus - create reduced (n-1) x (n-1) matrix
        row_map = [i for i in range(num_nodes) if i != slack_bus]
        n_red = len(row_map)
        b_red = [[0.0] * n_red for _ in range(n_red)]

        for ri, i in enumerate(row_map):
            for rj, j in enumerate(row_map):
                b_red[ri][rj] = b_matrix[i][j]

        # Invert reduced matrix
        x_red = invert_matrix(b_red)

        # Build full X matrix (with zeros for slack)
        x = [[0.0] * num_nodes for _ in range(num_nodes)]
        for ri, i in enumerate(row_map):
            for rj, j in enumerate(row_map):
                x[i][j] = x_red[ri][rj]

        # Compute PTDF: PTDF[l,k] = b_l * (X[i,k] - X[j,k])
        num_lines_actual = len(lines)
        ptdf = [[0.0] * num_nodes for _ in range(num_lines_actual)]
        for l, (i, j) in enumerate(lines):
            b = susceptances[l]
            for k in range(num_nodes):
                ptdf[l][k] = b * (x[i][k] - x[j][k])

        return ptdf

    def generate_exogenous_injections(self, rng, num_steps: int) -> list[list[float]]:
        """Generate exogenous nodal injections."""

        injections = [[0.0] * self.num_nodes for _ in range(num_steps)]

        # Time patterns (sinusoidal load curves)
        time_pattern1 = [math.sin(2.0 * math.pi * t / 96.0) for t in range(num_steps)]
        time_pattern2 = [math.sin(2.0 * math.pi * t / 48.0 + math.pi / 4.0) for t in range(num_steps)]

        # Node patterns (random loadings)
        node_pattern1 = [rng.random() - 0.5 for _ in range(self.num_nodes)]
        node_pattern2 = [rng.random() - 0.5 for _ in range(self.num_nodes)]

        # Combine with noise
        base_load = 200.0  # MW
        pattern_scale = 60.0
        noise_scale = 5.0

        for i in range(self.num_nodes):
            if i == self.slack_bus:
                continue
            for t in range(num_steps):
                pattern = pattern_scale * (node_pattern1[i] * time_pattern1[t] + node_pattern2[i] * time_pattern2[t])
                noise = noise_scale * rng.gauss(0.0, 1.0)
                injections[t][i] = base_load * (rng.random() - 0.5) + pattern + noise

        # Balance at slack bus: p_s = -sum_{i!=s} p_i
        for t in range(num_steps):
            total = sum(injections[t][i] for i in range(self.num_nodes) if i != self.slack_bus)
            injections[t][self.slack_bus] = -total

        # Per-timestep rescaling: ensure exogenous flows stay within
        # flow_margin * effective_limit for each timestep independently.
        flow_margin = 0.85
        for t in range(num_steps):
            flows = self.compute_flows(injections[t])
            scale_t = 1.0
            for l, flow in enumerate(flows):
                limit = self.flow_limits[l] * flow_margin
                if abs(flow) > limit:
                    scale_t = min(scale_t, limit / abs(flow))
            if scale_t < 1.0:
                for i in range(self.num_nodes):
                    if i != self.slack_bus:
                        injections[t][i] *= scale_t
                # Re-balance at slack for this timestep
                total = sum(injections[t][i] for i in range(self.num_nodes) if i != self.slack_bus)
                injections[t][self.slack_bus] = -total

        return injections

    def compute_flows(self, injections: list[float]) -> list[float]:
        """Compute line flows from nodal injections using PTDF."""
        return [sum(self.ptdf[l][k] * injections[k] for k in range(self.num_nodes)) for l in range(len(self.ptdf))]

    def verify_flows(self, flows: list[float]) -> None:
        """Verify that the flows are within the flow limits. Raises ValueError on violation."""
        for l, flow in enumerate(flows):
            violation = abs(flow) - self.flow_limits[l]
            if violation > constants.EPS_FLOW * self.flow_limits[l]:
                raise ValueError(f"Line {l} flow limit violated: |{abs(flow):.2f}| > {self.flow_limits[l]:.2f}")

    def generate_congestion_indicators(self, rng, exogenous_injections: list[float]) -> list[bool]:
        """Generate congestion indicators given exogenous injections (stochastic)."""
        indicators = [False] * self.num_nodes
        flows = self.compute_flows(exogenous_injections)
        for l, flow in enumerate(flows):
            p = (abs(flow) / (self.congestion_threshold * self.flow_limits[l])) ** 10.0
            if p > rng.random():
                from_node, to_node = self.lines[l]
                indicators[from_node] = True
                indicators[to_node] = True
        return indicators
