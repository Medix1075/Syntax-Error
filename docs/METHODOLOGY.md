# Methodology

## Decision and unit of analysis

The operational decision is where to intervene and which route type to use so that promised ETAs move closer to actual delivery time. The graph unit is a consolidated source-center → destination-center leg. The model unit is a complete `trip_uuid`, because the requested ETA benchmark is trip-level.

## 1. Data pipeline

Raw rows are cutoff observations within a leg. Treating every row as an independent trip would over-count long legs and leak cumulative progress. The pipeline groups by trip, source, destination, leg start/end, and route type:

- `actual_minutes = max(actual_time)` because `actual_time` is cumulative.
- `osrm_minutes = sum(segment_osrm_time)` and `osrm_distance_km = sum(segment_osrm_distance)` because segment fields are incremental.
- `elapsed_minutes = max(start_scan_to_end_scan)` and `dwell_proxy = max(0, elapsed - actual)`.

The source label fields have limited missingness, so center IDs—not names—define graph identity. Invalid non-positive target and OSRM segment values are rejected rather than imputed.

## 2. Evaluation design

Trips are sorted by creation time and never split across windows:

| Window | Share | Use |
|---|---:|---|
| Reference | 60% | learn graph topology and historical edge-delay priors |
| Train | 20% | train both ETA models on identical trips |
| Test | 20% | score both models once on later, untouched trips |

This is intentionally stricter than a random train/test split. Operational graph features are frozen before model training and testing. The full dataset is used only after the benchmark for the descriptive network audit.

## 3. Directed weighted graph

Facilities are nodes. Directed corridors are edges. Each edge is stratified by:

- source and destination center;
- route type (`FTL` or `Carting`);
- dispatch window (`night`, `morning`, `afternoon`, `evening`, `late_evening`).

The operational risk weight is a smoothed median:

\[
w_e = \frac{n_e \cdot \operatorname{median}(actual/OSRM)_e + 20 \cdot prior_{route,time}}{n_e + 20}
\]

Smoothing stops one or two noisy legs from dominating. An unseen edge falls back to its route/time prior and is flagged through `edge_seen_share`.

Betweenness needs a path cost, not a traffic count. The structural projection therefore uses \(1/\sqrt{volume}\): heavily used corridors become operationally close. Weighted in/out volume, unique in/out degree, PageRank, and clustering complete the node audit.

## 4. ETA models

Both models use histogram gradient boosting on `log1p(actual_minutes)`. This captures nonlinear distance/time interactions while reducing domination by a small number of very long trips.

The baseline sees only deployable trip-planning fields: OSRM time/distance, leg and segment counts, route type, dispatch hour, and day of week. The enhanced model adds edge ETA priors, edge coverage, source/destination centrality, traffic volume, hub delay priors, and combined structural risk. High-cardinality trip and facility IDs are not one-hot encoded.

Metrics:

- mean absolute error in minutes;
- median absolute error;
- percentage of trips with absolute percentage error ≤15%;
- percentage within 30%;
- mean bias.

## 5. Bottlenecks and corridors

A chronic delay/SLA proxy is `actual_minutes > 1.20 × osrm_minutes`. Each breached leg assigns half of its count and excess minutes to each endpoint, so hub contributions reconcile to network exposure without double-counting.

Hub rank combines attributed excess minutes, betweenness, median delay ratio, breach count, and traffic volume. Corridor rank deliberately emphasizes total excess minutes and volume; an extreme ratio on a tiny lane cannot outrank a lane that creates materially more delay exposure.

## 6. FTL versus Carting

The graph-enhanced model predicts each held-out trip twice, once with route type and route-specific graph priors set to FTL and once to Carting. Expected cost is:

\[
transport\ proxy + predicted\ minutes \times time\ value + P(late) \times late\ penalty
\]

All rupee assumptions are centralized in `src/config.py`. Because historical route assignment was not randomized, this is decision support—not a causal treatment-effect estimate.

## 7. Impact scenarios

The top-three upgrade case de-duplicates late trips touching any selected hub, applies the configured 30% effectiveness once, and multiplies recovered late deliveries by the configured ₹500 penalty proxy. Corridor actions use explicit 20–30% scenario effectiveness depending on intervention type. These are transparent sensitivities to prioritize pilots, not budget promises.
