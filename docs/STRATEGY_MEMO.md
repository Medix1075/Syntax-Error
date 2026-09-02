# Network Operations Strategy Memo

## Executive summary
The system combines trip-level machine learning with graph topology. The baseline predicts actual trip duration from operational variables; the enhanced model adds origin/destination centrality and local network structure. The intervention process ranks hubs by observed delay, SLA breaches, traffic volume and betweenness.

## Top-5 bottleneck workflow
Run the pipeline and use `artifacts/top_bottlenecks.csv`. For each hub, the intervention file recommends either capacity/dispatch-buffer upgrades for structurally central hubs or corridor schedule rebalancing for less central hubs.

## Revenue-at-risk proxy
For a configurable late-delivery penalty (P), revenue at risk is estimated as `late_trips × P`. The dashboard/pipeline intentionally separates this assumption from model output so an operator can substitute a business-specific penalty.

## Decision logic
- Long, low-risk corridors favour FL.
- Peak-hour or high-risk movements favour Carting for flexibility.
- Recommendations are decision support, not a replacement for contractual or safety constraints.
