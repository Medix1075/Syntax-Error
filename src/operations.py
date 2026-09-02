"""Bottleneck attribution, interventions, and ML-backed route decisions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BusinessAssumptions
from .data import time_bucket
from .graph_features import GraphSnapshot, add_graph_features, aggregate_trip_features
from .models import ETAModel


def _minmax(values: pd.Series) -> pd.Series:
    spread = float(values.max() - values.min())
    if spread == 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.min()) / spread


def _hub_names(legs: pd.DataFrame) -> pd.DataFrame:
    names = pd.concat([
        legs[["source_center", "source_name"]].rename(
            columns={"source_center": "hub", "source_name": "hub_name"}
        ),
        legs[["destination_center", "destination_name"]].rename(
            columns={"destination_center": "hub", "destination_name": "hub_name"}
        ),
    ]).dropna(subset=["hub_name"])
    return names.drop_duplicates("hub", keep="first")


def corridor_audit(
    legs: pd.DataFrame,
    snapshot: GraphSnapshot,
    assumptions: BusinessAssumptions,
) -> pd.DataFrame:
    """Rank corridors by chronic delay and share of network delay exposure."""
    corridors = snapshot.corridors.copy()
    breach_counts = (
        legs.assign(is_breach=legs["delay_ratio"].gt(assumptions.sla_ratio))
        .groupby(["source_center", "destination_center"])
        .agg(
            breached_legs=("is_breach", "sum"),
            breached_trips=("trip_uuid", lambda ids: ids[legs.loc[ids.index, "delay_ratio"].gt(assumptions.sla_ratio)].nunique()),
        )
        .reset_index()
    )
    corridors = corridors.merge(breach_counts, on=["source_center", "destination_center"], how="left")
    total_excess = max(float(corridors["excess_minutes"].sum()), 1.0)
    corridors["sla_breach_contribution_pct"] = corridors["excess_minutes"] / total_excess * 100
    corridors["is_chronic_delay"] = corridors["median_delay_ratio"].gt(assumptions.sla_ratio)
    name_map = _hub_names(legs).set_index("hub")["hub_name"]
    corridors["source_name"] = corridors["source_center"].map(name_map).fillna(corridors["source_center"])
    corridors["destination_name"] = corridors["destination_center"].map(name_map).fillna(corridors["destination_center"])
    corridors["corridor"] = corridors["source_center"] + " → " + corridors["destination_center"]
    # Impact outranks spectacle: a tiny lane with an extreme ratio should not
    # displace a corridor responsible for materially more late minutes.
    corridors["corridor_score"] = (
        0.60 * _minmax(corridors["excess_minutes"])
        + 0.20 * _minmax(corridors["volume"])
        + 0.10 * _minmax(corridors["breach_rate"])
        + 0.10 * _minmax(corridors["median_delay_ratio"].clip(upper=3.0))
    )
    return corridors.sort_values("corridor_score", ascending=False).reset_index(drop=True)


def rank_bottleneck_hubs(
    legs: pd.DataFrame,
    snapshot: GraphSnapshot,
    assumptions: BusinessAssumptions,
    top_k: int = 5,
) -> pd.DataFrame:
    """Attribute half of each delayed leg to each endpoint and combine with topology."""
    working = legs[[
        "trip_uuid", "source_center", "destination_center", "delay_ratio",
        "delay_excess_minutes",
    ]].copy()
    working["is_breach"] = working["delay_ratio"].gt(assumptions.sla_ratio)
    endpoint_rows = []
    for column in ["source_center", "destination_center"]:
        endpoint = working.rename(columns={column: "hub"})[
            ["trip_uuid", "hub", "delay_ratio", "delay_excess_minutes", "is_breach"]
        ].copy()
        endpoint["attribution_weight"] = 0.5
        endpoint_rows.append(endpoint)
    touches = pd.concat(endpoint_rows, ignore_index=True)
    touches["attributed_breaches"] = touches["is_breach"] * touches["attribution_weight"]
    touches["attributed_excess_minutes"] = (
        touches["delay_excess_minutes"] * touches["attribution_weight"]
    )
    hubs = touches.groupby("hub").agg(
        touched_legs=("attribution_weight", "sum"),
        affected_trips=("trip_uuid", "nunique"),
        attributed_breaches=("attributed_breaches", "sum"),
        attributed_excess_minutes=("attributed_excess_minutes", "sum"),
        median_delay_ratio=("delay_ratio", "median"),
    ).reset_index()
    total_breaches = max(float(working["is_breach"].sum()), 1.0)
    hubs["sla_breach_contribution_pct"] = hubs["attributed_breaches"] / total_breaches * 100
    hubs["revenue_at_risk_inr"] = (
        hubs["attributed_breaches"] * assumptions.late_delivery_penalty_inr
    )
    hubs = hubs.merge(snapshot.node_metrics, on="hub", how="left").merge(
        _hub_names(legs), on="hub", how="left"
    ).fillna(0)
    hubs["bottleneck_score"] = (
        0.30 * _minmax(hubs["attributed_excess_minutes"])
        + 0.25 * _minmax(hubs["betweenness"])
        + 0.20 * _minmax(hubs["median_delay_ratio"])
        + 0.15 * _minmax(hubs["attributed_breaches"])
        + 0.10 * _minmax(hubs["in_volume"] + hubs["out_volume"])
    )
    return hubs.sort_values("bottleneck_score", ascending=False).head(top_k).reset_index(drop=True)


def corridor_interventions(
    legs: pd.DataFrame,
    audited_corridors: pd.DataFrame,
    assumptions: BusinessAssumptions,
    top_k: int = 12,
) -> pd.DataFrame:
    """Translate corridor evidence into explicit operational actions and scenarios."""
    route_comparison = (
        legs.groupby(["source_center", "destination_center", "route_type"])
        .agg(route_delay_ratio=("delay_ratio", "median"), route_volume=("trip_uuid", "size"))
        .reset_index()
    )
    lookup: dict[tuple[str, str], pd.DataFrame] = {
        key: group for key, group in route_comparison.groupby(["source_center", "destination_center"])
    }
    dwell_threshold = float(audited_corridors["median_dwell_proxy"].quantile(0.75))
    rows: list[dict[str, object]] = []
    for row in audited_corridors.head(top_k).itertuples(index=False):
        routes = lookup.get((row.source_center, row.destination_center), pd.DataFrame())
        action: str
        rationale: str
        efficacy: float
        if len(routes) >= 2:
            best = routes.sort_values("route_delay_ratio").iloc[0]
            worst = routes.sort_values("route_delay_ratio").iloc[-1]
            improvement = float(worst["route_delay_ratio"] - best["route_delay_ratio"])
        else:
            best, improvement = None, 0.0
        if best is not None and improvement >= 0.10:
            action = f"Shift eligible departures toward {best['route_type']}"
            rationale = f"Observed median delay ratio is {improvement:.2f} lower than the alternative."
            efficacy = 0.25
        elif row.median_dwell_proxy >= dwell_threshold:
            action = "Upgrade endpoint capacity and enforce dispatch slots"
            rationale = f"Median dwell proxy is {row.median_dwell_proxy:.0f} minutes (top quartile)."
            efficacy = 0.30
        else:
            action = "Qualify a parallel lane/carrier and add exception alerts"
            rationale = f"Median actual time is {row.median_delay_ratio:.2f}× OSRM across {row.volume} legs."
            efficacy = 0.20
        estimated_reduction = float(row.breached_trips) * efficacy
        rows.append({
            "corridor": row.corridor,
            "source_center": row.source_center,
            "destination_center": row.destination_center,
            "action": action,
            "supporting_finding": rationale,
            "corridor_score": row.corridor_score,
            "current_breached_trips": int(row.breached_trips),
            "assumed_effectiveness_pct": efficacy * 100,
            "estimated_late_deliveries_reduced": estimated_reduction,
            "estimated_revenue_recovered_inr": estimated_reduction * assumptions.late_delivery_penalty_inr,
        })
    return pd.DataFrame(rows)


def hub_upgrade_scenario(
    legs: pd.DataFrame,
    top_hubs: pd.DataFrame,
    assumptions: BusinessAssumptions,
) -> dict[str, object]:
    """Estimate de-duplicated impact if the top three bottleneck hubs are upgraded."""
    selected = top_hubs.head(3)["hub"].tolist()
    breach = legs["delay_ratio"].gt(assumptions.sla_ratio)
    impacted = breach & (
        legs["source_center"].isin(selected) | legs["destination_center"].isin(selected)
    )
    network_late_trips = int(legs.loc[breach, "trip_uuid"].nunique())
    impacted_late_trips = int(legs.loc[impacted, "trip_uuid"].nunique())
    recovered = impacted_late_trips * assumptions.hub_upgrade_effectiveness
    return {
        "top_3_hubs": selected,
        "network_late_trips": network_late_trips,
        "late_trips_touching_top_3": impacted_late_trips,
        "assumed_effectiveness_pct": assumptions.hub_upgrade_effectiveness * 100,
        "estimated_late_deliveries_reduced": recovered,
        "network_late_delivery_reduction_pct": recovered / max(network_late_trips, 1) * 100,
        "current_revenue_at_risk_inr": network_late_trips * assumptions.late_delivery_penalty_inr,
        "estimated_revenue_recovered_inr": recovered * assumptions.late_delivery_penalty_inr,
        "note": "Scenario estimate, not a causal forecast; change assumptions in src/config.py.",
    }


def _late_probability(predicted_eta: np.ndarray, sla_minutes: np.ndarray, scale: float) -> np.ndarray:
    z = np.clip((predicted_eta - sla_minutes) / max(scale, 1.0), -20, 20)
    return 1.0 / (1.0 + np.exp(-z))


def route_type_decisions(
    test_legs: pd.DataFrame,
    snapshot: GraphSnapshot,
    model: ETAModel,
    assumptions: BusinessAssumptions,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score FTL and Carting counterfactuals with one ETA model and explicit costs.

    The comparison is observational decision support, not a causal estimate: each
    candidate changes route type and route-specific graph priors while holding the
    trip's planned distance, time, and network position fixed.
    """
    original = aggregate_trip_features(add_graph_features(test_legs, snapshot))
    base = original[["trip_uuid", "route_type", "osrm_distance_km", "osrm_minutes", "structural_risk"]].rename(
        columns={"route_type": "observed_route_type"}
    )
    options: list[pd.DataFrame] = []
    for route in ["FTL", "Carting"]:
        candidate = aggregate_trip_features(add_graph_features(test_legs, snapshot, route_override=route))
        predicted = model.predict(candidate)
        sla = candidate["osrm_minutes"].to_numpy() * assumptions.sla_ratio
        late_probability = _late_probability(predicted, sla, assumptions.risk_transition_minutes)
        if route == "FTL":
            transport = assumptions.ftl_fixed_cost_inr + assumptions.ftl_cost_per_km_inr * candidate["osrm_distance_km"].to_numpy()
        else:
            transport = assumptions.carting_fixed_cost_inr + assumptions.carting_cost_per_km_inr * candidate["osrm_distance_km"].to_numpy()
        total = (
            transport
            + assumptions.time_value_inr_per_minute * predicted
            + assumptions.late_delivery_penalty_inr * late_probability
        )
        options.append(pd.DataFrame({
            "trip_uuid": candidate["trip_uuid"],
            f"{route.lower()}_predicted_eta": predicted,
            f"{route.lower()}_late_probability": late_probability,
            f"{route.lower()}_transport_cost_inr": transport,
            f"{route.lower()}_expected_total_cost_inr": total,
        }))
    decisions = base.merge(options[0], on="trip_uuid").merge(options[1], on="trip_uuid")
    use_ftl = decisions["ftl_expected_total_cost_inr"] <= decisions["carting_expected_total_cost_inr"]
    decisions["recommended_route_type"] = np.where(use_ftl, "FTL", "Carting")
    decisions["recommended_expected_cost_inr"] = np.where(
        use_ftl, decisions["ftl_expected_total_cost_inr"], decisions["carting_expected_total_cost_inr"]
    )
    decisions["observed_expected_cost_inr"] = np.where(
        decisions["observed_route_type"].eq("FTL"),
        decisions["ftl_expected_total_cost_inr"],
        decisions["carting_expected_total_cost_inr"],
    )
    decisions["estimated_saving_vs_observed_inr"] = (
        decisions["observed_expected_cost_inr"] - decisions["recommended_expected_cost_inr"]
    )
    decisions["distance_band"] = pd.cut(
        decisions["osrm_distance_km"], [0, 50, 150, 300, 600, np.inf],
        labels=["0–50", "50–150", "150–300", "300–600", "600+"],
    ).astype("string")
    original_by_id = original.set_index("trip_uuid")
    decisions["time_bucket"] = time_bucket(
        original_by_id.loc[decisions["trip_uuid"], "dispatch_hour"].reset_index(drop=True)
    ).to_numpy()
    ranked_risk = decisions["structural_risk"].rank(method="first")
    decisions["structural_risk_band"] = pd.qcut(
        ranked_risk, 3, labels=["low", "medium", "high"]
    ).astype("string")
    policy = decisions.groupby(
        ["distance_band", "time_bucket", "structural_risk_band"],
        observed=True,
    ).agg(
        sample_trips=("trip_uuid", "size"),
        recommended_ftl_share=("recommended_route_type", lambda values: float(values.eq("FTL").mean())),
        ftl_predicted_eta=("ftl_predicted_eta", "median"),
        carting_predicted_eta=("carting_predicted_eta", "median"),
        ftl_expected_total_cost_inr=("ftl_expected_total_cost_inr", "median"),
        carting_expected_total_cost_inr=("carting_expected_total_cost_inr", "median"),
        median_saving_vs_observed_inr=("estimated_saving_vs_observed_inr", "median"),
    ).reset_index()
    policy["recommended_route_type"] = np.where(
        policy["recommended_ftl_share"] >= 0.5, "FTL", "Carting"
    )
    return decisions.sort_values("estimated_saving_vs_observed_inr", ascending=False), policy
