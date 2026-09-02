# Validation Report

## Overall assessment: Ready to share with stated caveats

## Methodology review

- Unit of prediction: one complete trip; graph edges: consolidated origin-destination legs.
- Leakage control: graph priors are built only from the earliest reference window; models train on the next window and are evaluated on the latest untouched window.
- Split sizes: 8,890 reference, 2,963 train, 2,964 test trips.
- Chronic delay/SLA proxy: actual time >120% of OSRM time.

## Calculation spot-checks

- Raw-to-leg aggregation: verified using cumulative `actual_time=max` and incremental `segment_osrm_*=sum`. Median absolute difference between cumulative actual and incremental actual sum is 1.0 minute(s); p99 is 23.0.
- Baseline held-out MAE: 72.20 minutes.
- Graph held-out MAE: 59.05 minutes.
- Baseline ±15% accuracy: 36.94%.
- Graph ±15% accuracy: 46.32%.
- Acceptance gate (graph wins on both): **True**.

## Data-quality review

- 142,267 rows, 14,817 trips, 26,369 legs, 1,657 facilities, 2,783 directed corridors.
- Exact duplicate rows: 0.
- Non-zero nulls: {'source_name': 285, 'destination_name': 260, 'cutoff_timestamp': 2985}.
- Source SHA-256: `203535609bb2e2fe60030705b18bb5a2685f137e81bcd3b16579b75891c40422`.

## Required caveats

- Source and destination *names* have minor missingness; center IDs are complete and control graph identity.
- The route-choice counterfactual is observational and may retain selection bias.
- Revenue and intervention impacts are sensitivity scenarios driven by assumptions in `src/config.py`.
- The source window is short; production monitoring must detect time drift and unseen corridors.
