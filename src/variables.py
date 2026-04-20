INF: float = 1e18

EPS_ZERO: float = 1e-16
EPS_BOUNDARY: float = 1e-12
EPS_COEFF: float = 1e-14
EPS_CONST: float = 1e-9

BIWEIGHT_K_STD: float = 3.0
HUBER_K_STD: float = 1.345

DEFAULT_SCALING_MULTIPLIERS: list[float] = [
    0.01,
    0.1,
    1,
    5,
    10,
    50,
    100,
    500,
    1000,
    5000,
    10000,
    50000,
]
VALID_LOSSES: list[str] = ["huber", "biweight", "l2"]

MIN_SERIES_LENGTH: int = 10
MAX_MISSING_RATIO: float = 0.2

DATA_DIR: str = "data"
S3_ENDPOINT_URL: str = "https://minio.lab.sspcloud.fr"
VAULT_ENDPOINT_URL: str = "https://vault.lab.sspcloud.fr"
