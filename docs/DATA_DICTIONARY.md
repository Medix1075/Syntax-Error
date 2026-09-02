# Data dictionary and generated evidence

## Required source fields

| Field | Role | Treatment |
|---|---|---|
| `trip_uuid` | trip identifier | groups legs into a trip; never used as a model feature |
| `source_center`, `destination_center` | directed facility IDs | graph endpoints |
| `source_name`, `destination_name` | readable labels | optional; ID is authoritative |
| `trip_creation_time` | temporal ordering | split and calendar features |
| `od_start_time`, `od_end_time` | leg chronology | leg identity and dispatch window |
| `route_type` | FTL/Carting | model and stratified edge feature |
| `actual_time` | cumulative actual leg time | `max` per leg; target component |
| `segment_actual_time` | incremental actual time | aggregation cross-check |
| `segment_osrm_time` | incremental OSRM time | `sum` per leg; baseline feature |
| `segment_osrm_distance` | incremental OSRM distance | `sum` per leg; baseline feature |
| `start_scan_to_end_scan` | leg elapsed time | dwell proxy only |

`factor`, `segment_factor`, and cumulative progress fields are retained in the source but excluded from model inputs because they directly encode actual-time outcomes or in-transit progress.

## Generated artifacts

| Artifact | Grain | Purpose |
|---|---|---|
| `metrics.json` | run | benchmark, split, assumptions, acceptance gate |
| `data_quality.json` | run | SHA-256, counts, nulls, aggregation checks |
| `edge_strata.csv` | corridor × route × time | graph risk weights and coverage |
| `hub_metrics.csv` | facility | degree, volume, centrality, clustering, delay |
| `corridor_audit.csv` | directed corridor | chronic-delay ranking and SLA contribution |
| `top_bottlenecks.csv` | facility | top-five attribution and revenue scenario |
| `interventions.csv` | corridor | action, evidence, estimated impact |
| `scored_test_trips.csv` | held-out trip | actual, both predictions, errors, risk |
| `feature_importance.csv` | model feature | permutation importance |
| `route_decisions.csv` | held-out trip | FTL and Carting counterfactuals |
| `route_policy.csv` | operating profile | dashboard decision cells |
| `upgrade_impact.json` | scenario | de-duplicated top-three upgrade impact |
| `eta_model.joblib` | model bundle | enhanced estimator, graph snapshot, assumptions |
| `network_nodes.csv`, `network_edges.csv` | visual subgraph | dashboard coordinates and styling data |

The raw CSV is intentionally not committed. `data_quality.json` pins the exact analyzed file with SHA-256.
