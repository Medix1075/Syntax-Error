# Deliverables index

| Brief requirement | Implementation | Evidence |
|---|---|---|
| Directed weighted graph, route/time stratification | `src/data.py`, `src/graph_features.py` | `edge_strata.csv`, `hub_metrics.csv` |
| Betweenness, degree, clustering, chronic corridors | `src/graph_features.py`, `src/operations.py` | `corridor_audit.csv`, `top_bottlenecks.csv` |
| Baseline versus graph ETA model | `src/models.py` | `metrics.json`, `model_comparison.png`, `scored_test_trips.csv` |
| Win on MAE and ±15% accuracy | chronological benchmark gate | 18.2% lower MAE; +9.4 pp accuracy |
| FTL versus Carting framework | `route_type_decisions` | `route_decisions.csv`, `route_policy.csv`, dashboard |
| Top-five hubs and SLA contribution | half-edge attribution | `top_bottlenecks.csv`, strategy memo |
| Corridor-specific interventions | evidence-driven decision logic | `interventions.csv` |
| Top-three upgrade impact and revenue recovery | de-duplicated scenario | `upgrade_impact.json`, strategy memo |
| Graph visualizations | `src/visualization.py` | `network_map.png`, `chronic_corridors.png` |
| 1–2 page operations memo | generated from the run | `docs/STRATEGY_MEMO.md` |
| Optional live dashboard | Streamlit, five operating views | `app/dashboard.py` |
| Reproducibility and validation | tests, CI, source hash, method docs | `.github/workflows/tests.yml`, `docs/VALIDATION_REPORT.md` |

All evidence files in the table are under `artifacts/` unless another path is shown.
