"""Project-wide paths, modelling constants, and business assumptions."""

from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
RANDOM_STATE = 42


@dataclass(frozen=True)
class BusinessAssumptions:
    """Editable decision-model assumptions; values are illustrative INR proxies."""

    sla_ratio: float = 1.20
    late_delivery_penalty_inr: float = 500.0
    time_value_inr_per_minute: float = 2.0
    ftl_fixed_cost_inr: float = 1_200.0
    ftl_cost_per_km_inr: float = 18.0
    carting_fixed_cost_inr: float = 250.0
    carting_cost_per_km_inr: float = 28.0
    risk_transition_minutes: float = 45.0
    hub_upgrade_effectiveness: float = 0.30

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


BUSINESS = BusinessAssumptions()
