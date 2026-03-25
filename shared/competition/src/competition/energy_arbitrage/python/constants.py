"""Constants matching the Rust energy_arbitrage::constants module."""

# Time step duration in hours (15 minutes)
DELTA_T: float = 0.25

# Slack bus index (0-indexed)
SLACK_BUS: int = 0

# Action quantization step (MW)
Q_U: float = 0.01

# SOC quantization step (MWh)
Q_E: float = 0.01

# Fractional SOC lower bound
E_MIN_FRAC: float = 0.10

# Fractional SOC upper bound
E_MAX_FRAC: float = 0.90

# Initial SOC fraction
E_INIT_FRAC: float = 0.50

# Default charge efficiency
ETA_CHARGE: float = 0.95

# Default discharge efficiency
ETA_DISCHARGE: float = 0.95

# Transaction cost ($/MWh)
KAPPA_TX: float = 0.25

# Degradation scale ($)
KAPPA_DEG: float = 1.00

# Degradation exponent
BETA_DEG: float = 2.0

# RT bias term
MU_BIAS: float = 0.0

# Spatial correlation parameter
RHO_SPATIAL: float = 0.70

# Congestion premium scale ($/MWh)
GAMMA_PRICE: float = 20.0

# Congestion proximity threshold
TAU_CONG: float = 0.90

# Jump probability
RHO_JUMP: float = 0.02

# Pareto tail index
ALPHA_TAIL: float = 3.5

# RT price floor ($/MWh)
LAMBDA_MIN: float = -200.0

# RT price cap ($/MWh)
LAMBDA_MAX: float = 5000.0

# DA price floor ($/MWh)
LAMBDA_DA_MIN: float = 0.0

# Flow feasibility tolerance (per-unit)
EPS_FLOW: float = 1e-6

# SOC feasibility tolerance (MWh)
EPS_SOC: float = 1e-9

# Nominal battery capacity (MWh)
NOMINAL_CAPACITY: float = 100.0

# Nominal battery power (MW)
NOMINAL_POWER: float = 25.0

# Nominal line flow limit (MW)
NOMINAL_FLOW_LIMIT: float = 50.0

# Base susceptance for network generation
BASE_SUSCEPTANCE: float = 10.0

# Mean DA price ($/MWh)
MEAN_DA_PRICE: float = 50.0

# DA price amplitude ($/MWh)
DA_AMPLITUDE: float = 20.0
