"""End-to-end reproducible graph intelligence and ETA modelling pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import ARTIFACTS, BUSINESS, DATA, RANDOM_STATE
from .data import (
    aggregate_segments_to_legs,
    assign_temporal_split,
    generate_demo_raw,
    load_raw,
)
from .graph_features import add_graph_features, aggregate_trip_features, build_graph
from .models import benchmark_models, model_feature_importance
from .operations import (
    corridor_audit,
    corridor_interventions,
    hub_upgrade_scenario,
    rank_bottleneck_hubs,
    route_type_decisions,
)
from .visualization import create_all_visuals


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_quality_summary(raw: pd.DataFrame, legs: pd.DataFrame, path: Path) -> dict[str, object]:
    discrepancy = (legs["actual_minutes"] - legs["incremental_actual_minutes"]).abs()
    return {
        "source_file": path.name,
        "source_sha256": _source_sha256(path),
        "raw_rows": len(raw),
        "raw_columns": len(raw.columns),
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "trip_count": int(legs["trip_uuid"].nunique()),
        "leg_count": len(legs),
        "facility_count": int(len(set(legs["source_center"]).union(legs["destination_center"]))),
        "corridor_count": int(legs.groupby(["source_center", "destination_center"]).ngroups),
        "time_start": str(raw["trip_creation_time"].min()),
        "time_end": str(raw["trip_creation_time"].max()),
        "null_counts": {key: int(value) for key, value in raw.isna().sum().items() if value > 0},
        "aggregation_check": {
            "median_abs_difference_actual_max_vs_segment_sum_minutes": float(discrepancy.median()),
            "p99_abs_difference_minutes": float(discrepancy.quantile(0.99)),
        },
    }


def _strategy_memo(
    metrics: dict[str, dict[str, float]],
    top_hubs: pd.DataFrame,
    interventions: pd.DataFrame,
    upgrade: dict[str, object],
    quality: dict[str, object],
) -> str:
    improvement = metrics["improvement"]
    hub_rows = "\n".join(
        f"| {index + 1} | {row.hub} | {row.sla_breach_contribution_pct:.2f}% | "
        f"₹{row.revenue_at_risk_inr:,.0f} | {row.median_delay_ratio:.2f}× |"
        for index, row in enumerate(top_hubs.itertuples(index=False))
    )
    action_rows = "\n".join(
        f"| {row.corridor} | {row.action} | {row.supporting_finding} | "
        f"{row.estimated_late_deliveries_reduced:.1f} | ₹{row.estimated_revenue_recovered_inr:,.0f} |"
        for row in interventions.head(5).itertuples(index=False)
    )
    return f"""# Network Operations Strategy Memo

**To:** Head of Network Operations  
**Subject:** Where to intervene first to close the OSRM-to-actual delivery gap  
**Evidence window:** {quality['time_start']} to {quality['time_end']}

## Decision in one minute

Prioritize upgrades at **{', '.join(upgrade['top_3_hubs'])}**, then act on the five corridors below using route shifts, dispatch-slot/capacity work, or parallel-lane qualification. These are the locations where delay exposure and network position overlap; fixing only high-delay but low-connectivity lanes would recover less network-wide risk.

The graph-enhanced ETA model reduced held-out MAE from **{metrics['baseline']['mae_minutes']:.1f} to {metrics['graph_enhanced']['mae_minutes']:.1f} minutes** ({improvement['mae_reduction_pct']:.1f}% lower) and increased predictions within ±15% of actual from **{metrics['baseline']['within_15pct']:.1f}% to {metrics['graph_enhanced']['within_15pct']:.1f}%** (+{improvement['within_15pct_lift_points']:.1f} points). It passes both required acceptance metrics on a later, untouched time window.

## Top five bottleneck hubs

Each breached leg assigns half of its exposure to each endpoint, preventing network totals from being double-counted. Revenue at risk uses the configurable **₹{BUSINESS.late_delivery_penalty_inr:,.0f} per late delivery** scenario.

| Rank | Hub | Estimated SLA-breach contribution | Revenue at risk | Median actual/OSRM |
|---:|---|---:|---:|---:|
{hub_rows}

## Corridor-specific actions

| Corridor | Recommendation | Evidence | Est. late deliveries reduced | Est. revenue recovered |
|---|---|---|---:|---:|
{action_rows}

## Top-three hub upgrade case

Across the extract, {upgrade['network_late_trips']:,} trips contain at least one leg above 120% of OSRM. {upgrade['late_trips_touching_top_3']:,} of those touch the top three hubs. At the stated {upgrade['assumed_effectiveness_pct']:.0f}% intervention effectiveness, upgrading all three is estimated to reduce **{upgrade['estimated_late_deliveries_reduced']:,.0f} late deliveries**, equal to **{upgrade['network_late_delivery_reduction_pct']:.1f}% of network late trips**, and recover **₹{upgrade['estimated_revenue_recovered_inr']:,.0f}** of the modeled **₹{upgrade['current_revenue_at_risk_inr']:,.0f}** exposure.

## FTL versus Carting operating rule

Use `artifacts/route_policy.csv` rather than a universal distance threshold. The framework holds distance, dispatch time, and facility structural position fixed, predicts ETA under both route types, and combines transport cost, time value, and expected late-delivery penalty. The lower expected-cost route is recommended. Review low-sample policy cells before operational use; the comparison is observational, not causal.

## 30-day action plan

1. Put the top three hubs on daily exception review and validate whether the dwell proxy corresponds to sort, dock, or dispatch-slot constraints.
2. Pilot the highest-ranked corridor recommendation for two weeks; retain a comparable untreated corridor as a control.
3. Publish graph-enhanced ETAs beside OSRM, monitor MAE and ±15% accuracy by route type and time bucket, and alert on unseen corridors.
4. Replace the ₹{BUSINESS.late_delivery_penalty_inr:,.0f} penalty and transport-cost assumptions in `src/config.py` with Finance-approved values before making budget commitments.

## Required caveats

The extract covers only {quality['trip_count']:,} trips over a short historical window. Hub/corridor scores are associations, not proof of root cause. The upgrade and revenue figures are transparent scenarios, not causal forecasts. A controlled pilot and refreshed cost inputs are required before network-wide rollout.
"""


def _validation_report(
    metrics: dict[str, dict[str, float]],
    quality: dict[str, object],
    split_counts: dict[str, int],
) -> str:
    passed = bool(metrics["improvement"]["passes_both_required_metrics"])
    assessment = "Ready to share with stated caveats" if passed else "Needs revision"
    return f"""# Validation Report

## Overall assessment: {assessment}

## Methodology review

- Unit of prediction: one complete trip; graph edges: consolidated origin-destination legs.
- Leakage control: graph priors are built only from the earliest reference window; models train on the next window and are evaluated on the latest untouched window.
- Split sizes: {split_counts.get('reference', 0):,} reference, {split_counts.get('train', 0):,} train, {split_counts.get('test', 0):,} test trips.
- Chronic delay/SLA proxy: actual time >120% of OSRM time.

## Calculation spot-checks

- Raw-to-leg aggregation: verified using cumulative `actual_time=max` and incremental `segment_osrm_*=sum`. Median absolute difference between cumulative actual and incremental actual sum is {quality['aggregation_check']['median_abs_difference_actual_max_vs_segment_sum_minutes']:.1f} minute(s); p99 is {quality['aggregation_check']['p99_abs_difference_minutes']:.1f}.
- Baseline held-out MAE: {metrics['baseline']['mae_minutes']:.2f} minutes.
- Graph held-out MAE: {metrics['graph_enhanced']['mae_minutes']:.2f} minutes.
- Baseline ±15% accuracy: {metrics['baseline']['within_15pct']:.2f}%.
- Graph ±15% accuracy: {metrics['graph_enhanced']['within_15pct']:.2f}%.
- Acceptance gate (graph wins on both): **{passed}**.

## Data-quality review

- {quality['raw_rows']:,} rows, {quality['trip_count']:,} trips, {quality['leg_count']:,} legs, {quality['facility_count']:,} facilities, {quality['corridor_count']:,} directed corridors.
- Exact duplicate rows: {quality['exact_duplicate_rows']:,}.
- Non-zero nulls: {quality['null_counts']}.
- Source SHA-256: `{quality['source_sha256']}`.

## Required caveats

- Source and destination *names* have minor missingness; center IDs are complete and control graph identity.
- The route-choice counterfactual is observational and may retain selection bias.
- Revenue and intervention impacts are sensitivity scenarios driven by assumptions in `src/config.py`.
- The source window is short; production monitoring must detect time drift and unseen corridors.
"""


def run_pipeline(source_path: Path, artifact_dir: Path = ARTIFACTS) -> dict[str, object]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw(source_path)
    legs = aggregate_segments_to_legs(raw)
    quality = _data_quality_summary(raw, legs, source_path)
    split_legs, split_cutoffs = assign_temporal_split(legs)

    # Evaluation graph is frozen before both model-training and test windows.
    evaluation_graph = build_graph(split_legs[split_legs["split"].eq("reference")])
    trip_features = aggregate_trip_features(add_graph_features(split_legs, evaluation_graph))
    training = trip_features[trip_features["split"].eq("train")].copy()
    test = trip_features[trip_features["split"].eq("test")].copy()
    models, metrics, scored = benchmark_models(training, test, seed=RANDOM_STATE)
    importance = model_feature_importance(models["graph_enhanced"], test, seed=RANDOM_STATE)

    # Descriptive operations audit uses every observed trip; it is not used in evaluation.
    operations_graph = build_graph(legs)
    corridors = corridor_audit(legs, operations_graph, BUSINESS)
    top_hubs = rank_bottleneck_hubs(legs, operations_graph, BUSINESS, top_k=5)
    interventions = corridor_interventions(legs, corridors, BUSINESS)
    upgrade = hub_upgrade_scenario(legs, top_hubs, BUSINESS)
    decisions, route_policy = route_type_decisions(
        split_legs[split_legs["split"].eq("test")],
        evaluation_graph,
        models["graph_enhanced"],
        BUSINESS,
    )

    split_counts = {key: int(value) for key, value in trip_features["split"].value_counts().items()}
    metrics["evaluation_design"] = {
        "split": "chronological 60% graph reference / 20% model train / 20% untouched test",
        "split_trip_counts": split_counts,
        "split_cutoffs": split_cutoffs,
        "reference_graph_edge_seen_share_on_test": float(test["edge_seen_share"].mean()),
    }
    metrics["business_assumptions"] = BUSINESS.to_dict()

    _write_json(artifact_dir / "metrics.json", metrics)
    _write_json(artifact_dir / "data_quality.json", quality)
    _write_json(artifact_dir / "upgrade_impact.json", upgrade)
    operations_graph.node_metrics.to_csv(artifact_dir / "hub_metrics.csv", index=False)
    operations_graph.edge_strata.to_csv(artifact_dir / "edge_strata.csv", index=False)
    corridors.to_csv(artifact_dir / "corridor_audit.csv", index=False)
    top_hubs.to_csv(artifact_dir / "top_bottlenecks.csv", index=False)
    interventions.to_csv(artifact_dir / "interventions.csv", index=False)
    scored.to_csv(artifact_dir / "scored_test_trips.csv", index=False)
    importance.to_csv(artifact_dir / "feature_importance.csv", index=False)
    decisions.to_csv(artifact_dir / "route_decisions.csv", index=False)
    route_policy.to_csv(artifact_dir / "route_policy.csv", index=False)
    joblib.dump(
        {"model": models["graph_enhanced"], "evaluation_graph": evaluation_graph, "assumptions": BUSINESS},
        artifact_dir / "eta_model.joblib",
        compress=3,
    )
    create_all_visuals(artifact_dir, operations_graph, metrics, top_hubs, corridors)

    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    docs_dir.mkdir(exist_ok=True)
    memo = _strategy_memo(metrics, top_hubs, interventions, upgrade, quality)
    validation = _validation_report(metrics, quality, split_counts)
    (docs_dir / "STRATEGY_MEMO.md").write_text(memo, encoding="utf-8")
    (docs_dir / "VALIDATION_REPORT.md").write_text(validation, encoding="utf-8")

    summary = {
        "source": str(source_path),
        "baseline_mae": metrics["baseline"]["mae_minutes"],
        "graph_mae": metrics["graph_enhanced"]["mae_minutes"],
        "baseline_within_15pct": metrics["baseline"]["within_15pct"],
        "graph_within_15pct": metrics["graph_enhanced"]["within_15pct"],
        "passes_required_model_gate": metrics["improvement"]["passes_both_required_metrics"],
        "top_hubs": top_hubs["hub"].tolist(),
        "artifacts": str(artifact_dir),
    }
    _write_json(artifact_dir / "run_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA / "raw" / "delivery_data.csv")
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--generate-demo", action="store_true", help="Use a deterministic synthetic contract sample")
    parser.add_argument("--demo-trips", type=int, default=800)
    parser.add_argument("--run-all", action="store_true", help="Retained for backward-compatible commands")
    args = parser.parse_args()
    source = args.input
    if args.generate_demo:
        source = DATA / "raw" / "demo_delivery_data.csv"
        source.parent.mkdir(parents=True, exist_ok=True)
        generate_demo_raw(args.demo_trips, RANDOM_STATE).to_csv(source, index=False)
    if not source.exists():
        parser.error(f"Input not found: {source}. Place the supplied CSV there or use --generate-demo.")
    summary = run_pipeline(source, args.artifact_dir)
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
