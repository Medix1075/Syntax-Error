"""Interactive operating view over artifacts produced by the offline pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(page_title="Delivery Network Intelligence", page_icon="◉", layout="wide")


@st.cache_data
def load_artifacts() -> dict[str, object]:
    return {
        "metrics": json.loads((ARTIFACTS / "metrics.json").read_text()),
        "upgrade": json.loads((ARTIFACTS / "upgrade_impact.json").read_text()),
        "hubs": pd.read_csv(ARTIFACTS / "top_bottlenecks.csv"),
        "corridors": pd.read_csv(ARTIFACTS / "corridor_audit.csv"),
        "interventions": pd.read_csv(ARTIFACTS / "interventions.csv"),
        "importance": pd.read_csv(ARTIFACTS / "feature_importance.csv"),
        "scores": pd.read_csv(ARTIFACTS / "scored_test_trips.csv"),
        "policy": pd.read_csv(ARTIFACTS / "route_policy.csv"),
        "nodes": pd.read_csv(ARTIFACTS / "network_nodes.csv"),
        "edges": pd.read_csv(ARTIFACTS / "network_edges.csv"),
    }


def network_figure(nodes: pd.DataFrame, edges: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for chronic, color, width, label in [
        (False, "rgba(130,145,160,.25)", 1, "Other corridor"),
        (True, "rgba(245,158,11,.75)", 2, "Chronic-delay corridor"),
    ]:
        x_values: list[float | None] = []
        y_values: list[float | None] = []
        for row in edges[edges["is_chronic_delay"].eq(chronic)].itertuples(index=False):
            x_values.extend([row.x0, row.x1, None])
            y_values.extend([row.y0, row.y1, None])
        figure.add_trace(go.Scatter(
            x=x_values, y=y_values, mode="lines", line={"color": color, "width": width},
            hoverinfo="skip", name=label,
        ))
    marker_size = 7 + np.sqrt(nodes["total_volume"].clip(lower=1)) * 1.8
    figure.add_trace(go.Scatter(
        x=nodes["x"], y=nodes["y"], mode="markers",
        marker={
            "size": marker_size,
            "color": np.where(nodes["is_top_bottleneck"], "#D64545", "#2F6BFF"),
            "line": {"width": 0.8, "color": "white"},
        },
        customdata=nodes[["hub", "total_volume", "betweenness", "hub_delay_ratio"]],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>Volume: %{customdata[1]:,.0f}"
            "<br>Betweenness: %{customdata[2]:.3f}<br>Median actual/OSRM: %{customdata[3]:.2f}×<extra></extra>"
        ),
        name="Facility",
    ))
    figure.update_layout(
        height=680, margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={"visible": False}, yaxis={"visible": False},
        legend={"orientation": "h", "y": 1.02},
    )
    return figure


required = [
    "metrics.json", "upgrade_impact.json", "top_bottlenecks.csv", "corridor_audit.csv",
    "interventions.csv", "feature_importance.csv", "scored_test_trips.csv",
    "route_policy.csv", "network_nodes.csv", "network_edges.csv",
]
missing = [name for name in required if not (ARTIFACTS / name).exists()]
if missing:
    st.error("Generated artifacts are missing: " + ", ".join(missing))
    st.code("python -m src.pipeline --input data/raw/delivery_data.csv --run-all")
    st.stop()

data = load_artifacts()
metrics = data["metrics"]
upgrade = data["upgrade"]

st.title("Optimizing Delivery ETAs with Graph-Based Network Intelligence")
st.caption("Directed-network risk, leakage-aware ETA benchmarking, and route economics for operations leaders")

overview_tab, network_tab, bottleneck_tab, eta_tab, route_tab = st.tabs([
    "Executive view", "Network risk", "Bottlenecks", "ETA benchmark", "FTL vs Carting",
])

with overview_tab:
    one, two, three, four = st.columns(4)
    one.metric("Graph ETA MAE", f"{metrics['graph_enhanced']['mae_minutes']:.1f} min", f"-{metrics['improvement']['mae_reduction_pct']:.1f}%")
    two.metric("Within ±15%", f"{metrics['graph_enhanced']['within_15pct']:.1f}%", f"+{metrics['improvement']['within_15pct_lift_points']:.1f} pts")
    three.metric("Top-3 late deliveries reduced", f"{upgrade['estimated_late_deliveries_reduced']:,.0f}", f"{upgrade['network_late_delivery_reduction_pct']:.1f}% of network")
    four.metric("Revenue recovered scenario", f"₹{upgrade['estimated_revenue_recovered_inr']:,.0f}")
    st.image(str(ARTIFACTS / "model_comparison.png"), use_container_width=True)
    st.info("Impact figures are scenarios using the transparent assumptions in `src/config.py`; they are not causal forecasts.")

with network_tab:
    st.subheader("High-impact network subgraph")
    st.plotly_chart(network_figure(data["nodes"], data["edges"]), use_container_width=True)
    st.subheader("Latest-window delay risk scores")
    left, right = st.columns(2)
    selected_route = left.multiselect("Route type", ["FTL", "Carting"], default=["FTL", "Carting"])
    minimum_risk = right.slider("Minimum graph delay-risk score", 0.0, 1.0, 0.65, 0.05)
    risk_view = data["scores"][
        data["scores"]["route_type"].isin(selected_route)
        & data["scores"]["graph_delay_risk"].ge(minimum_risk)
    ].sort_values("graph_delay_risk", ascending=False)
    st.dataframe(
        risk_view[[
            "trip_uuid", "route_type", "start_hub", "end_hub", "osrm_minutes",
            "graph_enhanced_prediction", "graph_delay_risk", "edge_seen_share",
        ]].head(100),
        use_container_width=True,
        hide_index=True,
    )

with bottleneck_tab:
    st.image(str(ARTIFACTS / "bottleneck_hubs.png"), use_container_width=True)
    st.dataframe(
        data["hubs"][[
            "hub", "hub_name", "bottleneck_score", "betweenness",
            "sla_breach_contribution_pct", "revenue_at_risk_inr",
        ]], use_container_width=True, hide_index=True,
    )
    st.subheader("Corridor actions")
    st.dataframe(data["interventions"], use_container_width=True, hide_index=True)

with eta_tab:
    left, right = st.columns([1.25, 1])
    left.image(str(ARTIFACTS / "model_comparison.png"), use_container_width=True)
    importance = data["importance"].head(12).sort_values("importance_mean")
    importance_plot = px.bar(
        importance, x="importance_mean", y="feature", orientation="h",
        title="Graph model permutation importance", color_discrete_sequence=["#2F6BFF"],
    )
    importance_plot.update_layout(xaxis_title="Increase in log-MAE when shuffled", yaxis_title=None)
    right.plotly_chart(importance_plot, use_container_width=True)
    st.caption("Evaluation: 60% historical graph reference, 20% model training, 20% untouched chronological test.")

with route_tab:
    st.subheader("ML-backed route-type decision framework")
    policy = data["policy"].copy()
    c1, c2, c3 = st.columns(3)
    distance = c1.selectbox("Distance band (km)", policy["distance_band"].dropna().drop_duplicates().tolist())
    dispatch = c2.selectbox("Dispatch window", policy["time_bucket"].dropna().drop_duplicates().tolist())
    risk = c3.selectbox("Facility structural risk", ["low", "medium", "high"])
    match = policy[
        policy["distance_band"].eq(distance)
        & policy["time_bucket"].eq(dispatch)
        & policy["structural_risk_band"].eq(risk)
    ]
    if match.empty:
        st.warning("No held-out trips match this exact profile. Choose another combination.")
    else:
        row = match.sort_values("sample_trips", ascending=False).iloc[0]
        st.success(f"Recommended route: **{row['recommended_route_type']}** ({int(row['sample_trips'])} supporting held-out trips)")
        ftl, carting = st.columns(2)
        ftl.metric("FTL predicted ETA", f"{row['ftl_predicted_eta']:.0f} min")
        ftl.metric("FTL expected total cost", f"₹{row['ftl_expected_total_cost_inr']:,.0f}")
        carting.metric("Carting predicted ETA", f"{row['carting_predicted_eta']:.0f} min")
        carting.metric("Carting expected total cost", f"₹{row['carting_expected_total_cost_inr']:,.0f}")
        st.caption("Expected total cost = transport proxy + time value + probability-weighted SLA penalty. Replace assumptions before procurement decisions.")
