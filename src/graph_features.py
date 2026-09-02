"""Directed graph construction, network metrics, and leakage-safe features."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np
import pandas as pd


BASE_FEATURES = [
    "osrm_minutes", "osrm_distance_km", "leg_count", "segment_count",
    "route_ftl", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

GRAPH_ONLY_FEATURES = [
    "edge_eta_prior", "edge_delay_ratio_mean", "edge_delay_ratio_max",
    "edge_seen_share", "source_betweenness_mean", "destination_betweenness_mean",
    "source_pagerank_mean", "destination_pagerank_mean", "source_in_volume_mean",
    "source_out_volume_mean", "destination_in_volume_mean",
    "destination_out_volume_mean", "source_hub_delay_mean",
    "destination_hub_delay_mean", "structural_risk",
]

GRAPH_FEATURES = BASE_FEATURES + GRAPH_ONLY_FEATURES


@dataclass
class GraphSnapshot:
    """Serializable graph tables learned from a historical reference window."""

    edge_strata: pd.DataFrame
    corridors: pd.DataFrame
    node_metrics: pd.DataFrame
    route_time_prior: pd.DataFrame
    global_delay_ratio: float


def _weighted_betweenness(nodes: list[str], edges: pd.DataFrame) -> dict[str, float]:
    """Exact directed Brandes betweenness using inverse-volume path costs."""
    n = len(nodes)
    if n < 3:
        return {node: 0.0 for node in nodes}
    index = {node: i for i, node in enumerate(nodes)}
    adjacency: list[list[tuple[int, float]]] = [[] for _ in nodes]
    for row in edges.itertuples(index=False):
        adjacency[index[row.source_center]].append(
            (index[row.destination_center], float(row.path_cost))
        )
    centrality = np.zeros(n, dtype=float)
    epsilon = 1e-12
    for source in range(n):
        predecessors: list[list[int]] = [[] for _ in nodes]
        sigma = np.zeros(n, dtype=float)
        sigma[source] = 1.0
        distance = np.full(n, np.inf)
        distance[source] = 0.0
        stack: list[int] = []
        queue: list[tuple[float, int]] = [(0.0, source)]
        while queue:
            dist_v, vertex = heappop(queue)
            if dist_v > distance[vertex] + epsilon:
                continue
            stack.append(vertex)
            for neighbor, cost in adjacency[vertex]:
                candidate = dist_v + cost
                if candidate < distance[neighbor] - epsilon:
                    distance[neighbor] = candidate
                    heappush(queue, (candidate, neighbor))
                    sigma[neighbor] = sigma[vertex]
                    predecessors[neighbor] = [vertex]
                elif abs(candidate - distance[neighbor]) <= epsilon:
                    sigma[neighbor] += sigma[vertex]
                    predecessors[neighbor].append(vertex)
        dependency = np.zeros(n, dtype=float)
        while stack:
            target = stack.pop()
            if sigma[target] > 0:
                coefficient = (1.0 + dependency[target]) / sigma[target]
                for predecessor in predecessors[target]:
                    dependency[predecessor] += sigma[predecessor] * coefficient
            if target != source:
                centrality[target] += dependency[target]
    centrality /= (n - 1) * (n - 2)
    return {node: float(centrality[index[node]]) for node in nodes}


def _pagerank(nodes: list[str], edges: pd.DataFrame, damping: float = 0.85) -> dict[str, float]:
    """Volume-weighted PageRank for the directed facility graph."""
    n = len(nodes)
    if n == 0:
        return {}
    index = {node: i for i, node in enumerate(nodes)}
    outgoing: list[list[tuple[int, float]]] = [[] for _ in nodes]
    out_weight = np.zeros(n, dtype=float)
    for row in edges.itertuples(index=False):
        source, destination, volume = (
            index[row.source_center], index[row.destination_center], float(row.volume)
        )
        outgoing[source].append((destination, volume))
        out_weight[source] += volume
    score = np.full(n, 1.0 / n)
    for _ in range(200):
        updated = np.full(n, (1.0 - damping) / n)
        dangling = float(score[out_weight == 0].sum())
        updated += damping * dangling / n
        for source, links in enumerate(outgoing):
            if out_weight[source] == 0:
                continue
            for destination, weight in links:
                updated[destination] += damping * score[source] * weight / out_weight[source]
        if np.abs(updated - score).sum() < 1e-12:
            score = updated
            break
        score = updated
    return {node: float(score[index[node]]) for node in nodes}


def _clustering(nodes: list[str], edges: pd.DataFrame) -> dict[str, float]:
    """Unweighted local clustering on the undirected projection."""
    neighbors = {node: set() for node in nodes}
    for row in edges.itertuples(index=False):
        if row.source_center != row.destination_center:
            neighbors[row.source_center].add(row.destination_center)
            neighbors[row.destination_center].add(row.source_center)
    result: dict[str, float] = {}
    for node, adjacent in neighbors.items():
        degree = len(adjacent)
        if degree < 2:
            result[node] = 0.0
            continue
        linked_pairs = sum(len(neighbors[other].intersection(adjacent)) for other in adjacent) / 2
        result[node] = float(2 * linked_pairs / (degree * (degree - 1)))
    return result


def build_graph(legs: pd.DataFrame, smoothing: float = 20.0) -> GraphSnapshot:
    """Build a directed weighted graph stratified by route type and time window.

    Edge risk is the smoothed median actual/OSRM ratio. Structural shortest paths
    use inverse square-root volume, so heavily used corridors are treated as
    operationally close instead of incorrectly treating high volume as distance.
    """
    if legs.empty:
        raise ValueError("Cannot build a graph from an empty leg table")
    global_ratio = float(legs["delay_ratio"].median())
    route_time_prior = (
        legs.groupby(["route_type", "time_bucket"], observed=True)
        .agg(fallback_delay_ratio=("delay_ratio", "median"))
        .reset_index()
    )
    edge_strata = (
        legs.groupby(
            ["source_center", "destination_center", "route_type", "time_bucket"],
            observed=True,
        )
        .agg(
            edge_observations=("trip_uuid", "size"),
            unique_trips=("trip_uuid", "nunique"),
            median_delay_ratio=("delay_ratio", "median"),
            median_actual_minutes=("actual_minutes", "median"),
            median_osrm_minutes=("osrm_minutes", "median"),
            breach_rate=("delay_ratio", lambda values: float((values > 1.20).mean())),
            excess_minutes=("delay_excess_minutes", "sum"),
        )
        .reset_index()
        .merge(route_time_prior, on=["route_type", "time_bucket"], how="left")
    )
    edge_strata["fallback_delay_ratio"] = edge_strata["fallback_delay_ratio"].fillna(global_ratio)
    edge_strata["delay_ratio_weight"] = (
        edge_strata["median_delay_ratio"] * edge_strata["edge_observations"]
        + smoothing * edge_strata["fallback_delay_ratio"]
    ) / (edge_strata["edge_observations"] + smoothing)
    corridors = (
        legs.groupby(["source_center", "destination_center"])
        .agg(
            volume=("trip_uuid", "size"),
            unique_trips=("trip_uuid", "nunique"),
            median_delay_ratio=("delay_ratio", "median"),
            breach_rate=("delay_ratio", lambda values: float((values > 1.20).mean())),
            median_actual_minutes=("actual_minutes", "median"),
            median_osrm_minutes=("osrm_minutes", "median"),
            median_distance_km=("osrm_distance_km", "median"),
            excess_minutes=("delay_excess_minutes", "sum"),
            median_dwell_proxy=("dwell_proxy_minutes", "median"),
        )
        .reset_index()
    )
    corridors["path_cost"] = 1.0 / np.sqrt(corridors["volume"].clip(lower=1))
    nodes = sorted(set(corridors["source_center"]).union(corridors["destination_center"]))
    betweenness = _weighted_betweenness(nodes, corridors)
    pagerank = _pagerank(nodes, corridors)
    clustering = _clustering(nodes, corridors)
    in_stats = corridors.groupby("destination_center").agg(
        in_degree=("source_center", "nunique"), in_volume=("volume", "sum")
    )
    out_stats = corridors.groupby("source_center").agg(
        out_degree=("destination_center", "nunique"), out_volume=("volume", "sum")
    )
    touches = pd.concat(
        [
            legs[["source_center", "delay_ratio"]].rename(columns={"source_center": "hub"}),
            legs[["destination_center", "delay_ratio"]].rename(columns={"destination_center": "hub"}),
        ],
        ignore_index=True,
    )
    risk = touches.groupby("hub").agg(
        hub_delay_ratio=("delay_ratio", "median"),
        hub_breach_rate=("delay_ratio", lambda values: float((values > 1.20).mean())),
    )
    node_metrics = pd.DataFrame({"hub": nodes})
    node_metrics = node_metrics.join(in_stats, on="hub").join(out_stats, on="hub").join(risk, on="hub")
    node_metrics = node_metrics.fillna(0.0)
    node_metrics["betweenness"] = node_metrics["hub"].map(betweenness)
    node_metrics["pagerank"] = node_metrics["hub"].map(pagerank)
    node_metrics["clustering"] = node_metrics["hub"].map(clustering)
    return GraphSnapshot(edge_strata, corridors, node_metrics, route_time_prior, global_ratio)


def add_graph_features(
    legs: pd.DataFrame,
    snapshot: GraphSnapshot,
    route_override: str | None = None,
) -> pd.DataFrame:
    """Attach edge priors and endpoint topology learned only from the snapshot."""
    featured = legs.copy()
    if route_override is not None:
        if route_override not in {"FTL", "Carting"}:
            raise ValueError("route_override must be FTL or Carting")
        featured["route_type"] = route_override
    edge_columns = [
        "source_center", "destination_center", "route_type", "time_bucket",
        "edge_observations", "delay_ratio_weight",
    ]
    featured = featured.merge(
        snapshot.edge_strata[edge_columns],
        on=["source_center", "destination_center", "route_type", "time_bucket"],
        how="left",
    ).merge(snapshot.route_time_prior, on=["route_type", "time_bucket"], how="left")
    featured["fallback_delay_ratio"] = featured["fallback_delay_ratio"].fillna(
        snapshot.global_delay_ratio
    )
    featured["edge_seen"] = featured["edge_observations"].notna().astype(float)
    featured["edge_observations"] = featured["edge_observations"].fillna(0.0)
    featured["edge_delay_ratio"] = featured["delay_ratio_weight"].fillna(
        featured["fallback_delay_ratio"]
    )
    featured["edge_eta_prior"] = featured["osrm_minutes"] * featured["edge_delay_ratio"]
    metric_columns = [
        "hub", "in_degree", "out_degree", "in_volume", "out_volume",
        "hub_delay_ratio", "hub_breach_rate", "betweenness", "pagerank", "clustering",
    ]
    for prefix, key in [("source", "source_center"), ("destination", "destination_center")]:
        metrics = snapshot.node_metrics[metric_columns].rename(
            columns={column: f"{prefix}_{column}" for column in metric_columns if column != "hub"}
        ).rename(columns={"hub": key})
        featured = featured.merge(metrics, on=key, how="left")
        added = [column for column in metrics.columns if column != key]
        featured[added] = featured[added].fillna(0.0)
    featured["structural_risk_leg"] = (
        featured["source_betweenness"] + featured["destination_betweenness"]
        + featured["source_pagerank"] + featured["destination_pagerank"]
    )
    return featured


def aggregate_trip_features(featured_legs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate leg-level graph evidence into one row per trip for ETA modelling."""
    ordered = featured_legs.sort_values(["trip_uuid", "od_start_time"])
    trips = (
        ordered.groupby("trip_uuid", sort=False)
        .agg(
            trip_creation_time=("trip_creation_time", "min"),
            route_type=("route_type", "first"),
            split=("split", "first"),
            start_hub=("source_center", "first"),
            end_hub=("destination_center", "last"),
            actual_minutes=("actual_minutes", "sum"),
            osrm_minutes=("osrm_minutes", "sum"),
            osrm_distance_km=("osrm_distance_km", "sum"),
            leg_count=("trip_uuid", "size"),
            segment_count=("segment_count", "sum"),
            edge_eta_prior=("edge_eta_prior", "sum"),
            edge_delay_ratio_mean=("edge_delay_ratio", "mean"),
            edge_delay_ratio_max=("edge_delay_ratio", "max"),
            edge_seen_share=("edge_seen", "mean"),
            source_betweenness_mean=("source_betweenness", "mean"),
            destination_betweenness_mean=("destination_betweenness", "mean"),
            source_pagerank_mean=("source_pagerank", "mean"),
            destination_pagerank_mean=("destination_pagerank", "mean"),
            source_in_volume_mean=("source_in_volume", "mean"),
            source_out_volume_mean=("source_out_volume", "mean"),
            destination_in_volume_mean=("destination_in_volume", "mean"),
            destination_out_volume_mean=("destination_out_volume", "mean"),
            source_hub_delay_mean=("source_hub_delay_ratio", "mean"),
            destination_hub_delay_mean=("destination_hub_delay_ratio", "mean"),
            structural_risk=("structural_risk_leg", "mean"),
        )
        .reset_index()
    )
    trips["dispatch_hour"] = trips["trip_creation_time"].dt.hour
    trips["dispatch_dayofweek"] = trips["trip_creation_time"].dt.dayofweek
    trips["route_ftl"] = trips["route_type"].eq("FTL").astype(float)
    trips["hour_sin"] = np.sin(2 * np.pi * trips["dispatch_hour"] / 24)
    trips["hour_cos"] = np.cos(2 * np.pi * trips["dispatch_hour"] / 24)
    trips["dow_sin"] = np.sin(2 * np.pi * trips["dispatch_dayofweek"] / 7)
    trips["dow_cos"] = np.cos(2 * np.pi * trips["dispatch_dayofweek"] / 7)
    return trips
