"""Deterministic, decision-focused visual artifacts for the report and dashboard."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .graph_features import GraphSnapshot


COLORS = {
    "navy": "#17324D",
    "blue": "#2F6BFF",
    "teal": "#00A6A6",
    "orange": "#F59E0B",
    "red": "#D64545",
    "gray": "#AAB4C0",
}


def _spring_layout(nodes: list[str], edges: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """Small force-directed layout for the highest-impact network subgraph."""
    rng = np.random.default_rng(seed)
    n = len(nodes)
    if n == 1:
        return np.zeros((1, 2))
    positions = rng.normal(0, 0.3, size=(n, 2))
    index = {node: i for i, node in enumerate(nodes)}
    edge_pairs = [
        (index[row.source_center], index[row.destination_center])
        for row in edges.itertuples(index=False)
        if row.source_center in index and row.destination_center in index
    ]
    ideal = np.sqrt(1.0 / n)
    temperature = 0.15
    for _ in range(180):
        delta = positions[:, None, :] - positions[None, :, :]
        distance = np.linalg.norm(delta, axis=2) + np.eye(n)
        repulsion = (ideal * ideal / distance**2)[:, :, None] * delta
        displacement = repulsion.sum(axis=1)
        for source, destination in edge_pairs:
            vector = positions[source] - positions[destination]
            length = max(float(np.linalg.norm(vector)), 1e-6)
            attraction = vector / length * (length * length / ideal)
            displacement[source] -= attraction
            displacement[destination] += attraction
        norms = np.linalg.norm(displacement, axis=1).clip(min=1e-9)
        positions += displacement / norms[:, None] * np.minimum(norms, temperature)[:, None]
        positions -= positions.mean(axis=0)
        temperature *= 0.975
    scale = np.abs(positions).max()
    return positions / max(scale, 1e-9)


def network_visual_data(
    snapshot: GraphSnapshot,
    top_hubs: pd.DataFrame,
    max_nodes: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = snapshot.node_metrics.copy()
    metrics["total_volume"] = metrics["in_volume"] + metrics["out_volume"]
    required_order = top_hubs["hub"].tolist()
    required = set(required_order)
    candidate_edges = snapshot.corridors.copy()
    candidate_edges["display_priority"] = (
        0.55 * candidate_edges["excess_minutes"].rank(pct=True)
        + 0.30 * candidate_edges["volume"].rank(pct=True)
        + 0.15 * candidate_edges["median_delay_ratio"].clip(upper=3).rank(pct=True)
    )
    # Grow outward from the five bottlenecks so every displayed node participates
    # in the visible subgraph; independent high-volume nodes otherwise crush the
    # useful network into a corner of the canvas.
    selected_set = set(required)
    while len(selected_set) < max_nodes:
        frontier = candidate_edges[
            candidate_edges["source_center"].isin(selected_set)
            ^ candidate_edges["destination_center"].isin(selected_set)
        ]
        if frontier.empty:
            break
        best = frontier.sort_values("display_priority", ascending=False).iloc[0]
        selected_set.add(best["source_center"])
        selected_set.add(best["destination_center"])
    selected = required_order + sorted(selected_set.difference(required))
    selected = selected[:max_nodes]
    edge_data = snapshot.corridors[
        snapshot.corridors["source_center"].isin(selected)
        & snapshot.corridors["destination_center"].isin(selected)
    ].copy()
    positions = _spring_layout(selected, edge_data)
    position_map = {node: positions[i] for i, node in enumerate(selected)}
    node_data = metrics[metrics["hub"].isin(selected)].copy()
    node_data["x"] = node_data["hub"].map(lambda node: position_map[node][0])
    node_data["y"] = node_data["hub"].map(lambda node: position_map[node][1])
    node_data["is_top_bottleneck"] = node_data["hub"].isin(required)
    rank_map = {hub: rank for rank, hub in enumerate(required_order, start=1)}
    node_data["bottleneck_rank"] = node_data["hub"].map(rank_map)
    edge_data["x0"] = edge_data["source_center"].map(lambda node: position_map[node][0])
    edge_data["y0"] = edge_data["source_center"].map(lambda node: position_map[node][1])
    edge_data["x1"] = edge_data["destination_center"].map(lambda node: position_map[node][0])
    edge_data["y1"] = edge_data["destination_center"].map(lambda node: position_map[node][1])
    edge_data["is_chronic_delay"] = edge_data["median_delay_ratio"] > 1.20
    return node_data, edge_data


def plot_network(nodes: pd.DataFrame, edges: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(13, 9))
    for edge in edges.sort_values("is_chronic_delay").itertuples(index=False):
        axis.plot(
            [edge.x0, edge.x1], [edge.y0, edge.y1],
            color=COLORS["orange"] if edge.is_chronic_delay else COLORS["gray"],
            alpha=0.55 if edge.is_chronic_delay else 0.20,
            linewidth=0.5 + np.log1p(edge.volume) * 0.30,
            zorder=1,
        )
    regular = nodes[~nodes["is_top_bottleneck"]]
    top = nodes[nodes["is_top_bottleneck"]]
    axis.scatter(
        regular["x"], regular["y"],
        s=18 + np.sqrt(regular["total_volume"]) * 6,
        color=COLORS["blue"], alpha=0.72, edgecolor="white", linewidth=0.5, zorder=2,
    )
    axis.scatter(
        top["x"], top["y"],
        s=85 + np.sqrt(top["total_volume"]) * 8,
        color=COLORS["red"], edgecolor="white", linewidth=1.1, zorder=3,
    )
    for row in top.itertuples(index=False):
        axis.annotate(
            str(int(row.bottleneck_rank)), (row.x, row.y),
            ha="center", va="center", color="white", fontsize=8, weight="bold", zorder=4,
        )
    axis.set_title(
        "High-impact logistics subgraph", loc="left", weight="bold", pad=30, fontsize=15
    )
    axis.text(
        0, 1.01, "Red = top-5 hub · orange = median actual time >120% of OSRM",
        transform=axis.transAxes, fontsize=9,
    )
    legend_text = "\n".join(
        f"{int(row.bottleneck_rank)}  {row.hub}" for row in top.sort_values("bottleneck_rank").itertuples(index=False)
    )
    axis.text(
        1.01, 0.98, legend_text, transform=axis.transAxes, va="top", fontsize=8,
        bbox={"boxstyle": "round,pad=.5", "facecolor": "white", "edgecolor": "#DDE3EA"},
    )
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(metrics: dict[str, dict[str, float]], path: Path) -> None:
    labels = ["Baseline", "Graph-enhanced"]
    mae = [metrics["baseline"]["mae_minutes"], metrics["graph_enhanced"]["mae_minutes"]]
    accuracy = [metrics["baseline"]["within_15pct"], metrics["graph_enhanced"]["within_15pct"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    axes[0].bar(labels, mae, color=[COLORS["gray"], COLORS["blue"]])
    axes[0].set_title("MAE (lower is better)", loc="left", weight="bold")
    axes[0].set_ylabel("Minutes")
    axes[1].bar(labels, accuracy, color=[COLORS["gray"], COLORS["teal"]])
    axes[1].set_title("Predictions within ±15% (higher is better)", loc="left", weight="bold")
    axes[1].set_ylabel("Held-out trips (%)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        for bar in axis.patches:
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.1f}", ha="center", va="bottom")
    fig.suptitle("Graph context improves both required ETA metrics", x=0.02, ha="left", weight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_bottlenecks(top_hubs: pd.DataFrame, path: Path) -> None:
    view = top_hubs.sort_values("bottleneck_score")
    fig, axis = plt.subplots(figsize=(10, 5.2))
    bars = axis.barh(view["hub"], view["bottleneck_score"], color=COLORS["red"])
    axis.set_xlabel("Composite bottleneck score (0–1)")
    axis.set_title("Top bottleneck hubs combine structural centrality and SLA exposure", loc="left", weight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    for bar, contribution in zip(bars, view["sla_breach_contribution_pct"]):
        axis.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {contribution:.1f}% breach share", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_corridors(corridors: pd.DataFrame, path: Path, top_k: int = 12) -> None:
    view = corridors[corridors["is_chronic_delay"]].head(top_k).sort_values("corridor_score")
    fig, axis = plt.subplots(figsize=(11, 6.5))
    axis.barh(view["corridor"], view["median_delay_ratio"], color=COLORS["orange"])
    axis.axvline(1.20, color=COLORS["red"], linestyle="--", linewidth=1.2, label="Chronic-delay threshold")
    axis.set_xlabel("Median actual / OSRM time")
    axis.set_title("Highest-impact chronic-delay corridors", loc="left", weight="bold")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_all_visuals(
    artifact_dir: Path,
    snapshot: GraphSnapshot,
    metrics: dict[str, dict[str, float]],
    top_hubs: pd.DataFrame,
    corridors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    nodes, edges = network_visual_data(snapshot, top_hubs)
    nodes.to_csv(artifact_dir / "network_nodes.csv", index=False)
    edges.to_csv(artifact_dir / "network_edges.csv", index=False)
    plot_network(nodes, edges, artifact_dir / "network_map.png")
    plot_model_comparison(metrics, artifact_dir / "model_comparison.png")
    plot_bottlenecks(top_hubs, artifact_dir / "bottleneck_hubs.png")
    plot_corridors(corridors, artifact_dir / "chronic_corridors.png")
    return nodes, edges
