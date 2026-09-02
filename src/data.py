"""Loading, validation, segment consolidation, and deterministic data splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW_REQUIRED_COLUMNS = {
    "data", "trip_creation_time", "route_schedule_uuid", "route_type",
    "trip_uuid", "source_center", "destination_center", "od_start_time",
    "od_end_time", "start_scan_to_end_scan", "actual_distance_to_destination",
    "actual_time", "osrm_time", "osrm_distance", "segment_actual_time",
    "segment_osrm_time", "segment_osrm_distance",
}

DATETIME_COLUMNS = [
    "trip_creation_time", "od_start_time", "od_end_time", "cutoff_timestamp",
]

LEG_KEYS = [
    "trip_uuid", "source_center", "destination_center", "od_start_time",
    "od_end_time", "route_type",
]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the Delhivery segment extract and parse available timestamps."""
    frame = pd.read_csv(path)
    for column in DATETIME_COLUMNS:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return validate_raw(frame)


def validate_raw(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the source contract without silently imputing target fields."""
    missing = RAW_REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required source columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("The source dataset is empty")
    if frame["trip_uuid"].isna().any():
        raise ValueError("trip_uuid contains null values")
    numeric_positive = [
        "actual_time", "segment_actual_time", "segment_osrm_time",
        "segment_osrm_distance",
    ]
    invalid = (frame[numeric_positive] <= 0).any(axis=1)
    if invalid.any():
        frame = frame.loc[~invalid].copy()
    if frame.empty:
        raise ValueError("No valid positive-time rows remain after validation")
    return frame


def time_bucket(hours: pd.Series) -> pd.Series:
    """Create operations-friendly dispatch windows."""
    labels = ["night", "morning", "afternoon", "evening", "late_evening"]
    return pd.cut(hours, [-1, 5, 11, 16, 20, 23], labels=labels).astype("string")


def aggregate_segments_to_legs(raw: pd.DataFrame) -> pd.DataFrame:
    """Consolidate cumulative cutoff rows into one directed origin-destination leg.

    `actual_time` is cumulative within a leg, while `segment_*` values are
    incremental. The maximum cumulative actual time and the sum of incremental
    OSRM quantities therefore preserve the correct grain without double-counting.
    """
    legs = (
        raw.sort_values(["trip_uuid", "od_start_time", "cutoff_timestamp"])
        .groupby(LEG_KEYS, dropna=False, sort=False)
        .agg(
            trip_creation_time=("trip_creation_time", "min"),
            route_schedule_uuid=("route_schedule_uuid", "first"),
            source_name=("source_name", "first"),
            destination_name=("destination_name", "first"),
            source_partition=("data", "first"),
            actual_minutes=("actual_time", "max"),
            incremental_actual_minutes=("segment_actual_time", "sum"),
            osrm_minutes=("segment_osrm_time", "sum"),
            osrm_distance_km=("segment_osrm_distance", "sum"),
            actual_distance_km=("actual_distance_to_destination", "max"),
            elapsed_minutes=("start_scan_to_end_scan", "max"),
            segment_count=("trip_uuid", "size"),
        )
        .reset_index()
    )
    legs["dispatch_hour"] = legs["od_start_time"].dt.hour.astype("int16")
    legs["dispatch_dayofweek"] = legs["od_start_time"].dt.dayofweek.astype("int16")
    legs["time_bucket"] = time_bucket(legs["dispatch_hour"])
    legs["delay_ratio"] = (
        legs["actual_minutes"] / legs["osrm_minutes"].clip(lower=1.0)
    ).clip(lower=0.20, upper=10.0)
    legs["delay_excess_minutes"] = (
        legs["actual_minutes"] - legs["osrm_minutes"]
    ).clip(lower=0.0)
    legs["dwell_proxy_minutes"] = (
        legs["elapsed_minutes"] - legs["actual_minutes"]
    ).clip(lower=0.0)
    return legs


def assign_temporal_split(
    legs: pd.DataFrame,
    reference_share: float = 0.60,
    training_share: float = 0.20,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Split whole trips chronologically into graph-reference/train/test windows."""
    if reference_share <= 0 or training_share <= 0 or reference_share + training_share >= 1:
        raise ValueError("Split shares must be positive and leave a non-empty test window")
    trip_times = legs.groupby("trip_uuid")["trip_creation_time"].min().sort_values(kind="stable")
    n = len(trip_times)
    if n < 15:
        reference_share, training_share = 0.50, 0.25
    reference_end = max(1, int(n * reference_share))
    training_end = max(reference_end + 1, int(n * (reference_share + training_share)))
    training_end = min(training_end, n - 1)
    mapping: dict[str, str] = {}
    mapping.update({trip_id: "reference" for trip_id in trip_times.index[:reference_end]})
    mapping.update({trip_id: "train" for trip_id in trip_times.index[reference_end:training_end]})
    mapping.update({trip_id: "test" for trip_id in trip_times.index[training_end:]})
    result = legs.copy()
    result["split"] = result["trip_uuid"].map(mapping)
    cutoffs = {
        name: str(result.loc[result["split"].eq(name), "trip_creation_time"].max())
        for name in ["reference", "train", "test"]
    }
    return result, cutoffs


def generate_demo_raw(n_trips: int = 800, seed: int = 42) -> pd.DataFrame:
    """Generate a contract-compatible dataset for CI and documentation examples."""
    rng = np.random.default_rng(seed)
    hubs = [f"IND{100000 + i:06d}AAA" for i in range(30)]
    names = {hub: f"Demo_Hub_{i:02d} (Synthetic)" for i, hub in enumerate(hubs)}
    start = pd.Timestamp("2018-09-01")
    rows: list[dict[str, object]] = []
    for trip_index in range(n_trips):
        trip_id = f"trip-{trip_index:06d}"
        route = str(rng.choice(["FTL", "Carting"], p=[0.60, 0.40]))
        created = start + pd.Timedelta(minutes=int(rng.integers(0, 30 * 24 * 60)))
        path = list(rng.choice(hubs, size=int(rng.integers(2, 5)), replace=False))
        for leg_index, (source, destination) in enumerate(zip(path, path[1:])):
            distance = float(rng.uniform(15, 500))
            osrm = max(8, int(distance / rng.uniform(35, 60) * 60))
            central_delay = 1.45 if source in hubs[2:5] or destination in hubs[2:5] else 1.05
            route_effect = 1.10 if route == "Carting" and distance > 180 else 0.95
            actual = max(8, int(osrm * central_delay * route_effect + rng.normal(0, 12)))
            od_start = created + pd.Timedelta(minutes=leg_index * 180)
            od_end = od_start + pd.Timedelta(minutes=actual + int(rng.integers(5, 50)))
            segments = int(rng.integers(1, 5))
            elapsed_actual = 0
            for segment_index in range(segments):
                seg_actual = max(1, actual // segments + int(rng.integers(-2, 3)))
                seg_osrm = max(1, osrm // segments)
                elapsed_actual += seg_actual
                rows.append({
                    "data": "training" if trip_index < n_trips * 0.8 else "test",
                    "trip_creation_time": created,
                    "route_schedule_uuid": f"route-{trip_index % 25:03d}",
                    "route_type": route,
                    "trip_uuid": trip_id,
                    "source_center": source,
                    "source_name": names[source],
                    "destination_center": destination,
                    "destination_name": names[destination],
                    "od_start_time": od_start,
                    "od_end_time": od_end,
                    "start_scan_to_end_scan": float((od_end - od_start).total_seconds() / 60),
                    "is_cutoff": segment_index == segments - 1,
                    "cutoff_factor": segment_index + 1,
                    "cutoff_timestamp": od_start + pd.Timedelta(minutes=elapsed_actual),
                    "actual_distance_to_destination": distance * (segment_index + 1) / segments,
                    "actual_time": float(min(actual, elapsed_actual)),
                    "osrm_time": float(osrm * (segment_index + 1) / segments),
                    "osrm_distance": distance * (segment_index + 1) / segments,
                    "factor": actual / max(osrm, 1),
                    "segment_actual_time": float(seg_actual),
                    "segment_osrm_time": float(seg_osrm),
                    "segment_osrm_distance": distance / segments,
                    "segment_factor": seg_actual / max(seg_osrm, 1),
                })
    return pd.DataFrame(rows)
