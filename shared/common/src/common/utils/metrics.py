from prometheus_client import Counter, Gauge, REGISTRY

label_names = ["project", "subsystem"]

# Buckets for histograms we need have higher duration than typicall web apis
COMMON_HIST_DURATION_BUKCET = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    20.0,
    30.0,
    45.0,
    60.0,
    75.0,
    100.0,
    115.0,
    130.0,
    145.0,
    160.0,
    175.0,
    200.0,
    215.0,
    230.0,
    float("inf"),
)


def GaugeWithParams(metric_name: str, description: str) -> Gauge:
    g = Gauge(
        metric_name,
        description,
        labelnames=label_names,
        registry=REGISTRY,
    )
    return g


def CounterWithParams(metric_name: str, description: str) -> Counter:
    c = Counter(
        metric_name,
        description,
        labelnames=label_names,
        registry=REGISTRY,
    )
    return c
