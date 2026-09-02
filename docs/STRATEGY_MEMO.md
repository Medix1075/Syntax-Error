# Network Operations Strategy Memo

**To:** Head of Network Operations  
**Subject:** Where to intervene first to close the OSRM-to-actual delivery gap  
**Evidence window:** 2018-09-12 00:00:16.535741 to 2018-10-03 23:59:42.701692

## Decision in one minute

Prioritize upgrades at **IND000000ACB, IND562132AAA, IND421302AAG**, then act on the five corridors below using route shifts, dispatch-slot/capacity work, or parallel-lane qualification. These are the locations where delay exposure and network position overlap; fixing only high-delay but low-connectivity lanes would recover less network-wide risk.

The graph-enhanced ETA model reduced held-out MAE from **72.2 to 59.1 minutes** (18.2% lower) and increased predictions within ±15% of actual from **36.9% to 46.3%** (+9.4 points). It passes both required acceptance metrics on a later, untouched time window.

## Top five bottleneck hubs

Each breached leg assigns half of its exposure to each endpoint, preventing network totals from being double-counted. Revenue at risk uses the configurable **₹500 per late delivery** scenario.

| Rank | Hub | Estimated SLA-breach contribution | Revenue at risk | Median actual/OSRM |
|---:|---|---:|---:|---:|
| 1 | IND000000ACB | 3.56% | ₹429,000 | 1.67× |
| 2 | IND562132AAA | 2.71% | ₹327,250 | 1.64× |
| 3 | IND421302AAG | 2.91% | ₹350,750 | 2.21× |
| 4 | IND712311AAA | 0.99% | ₹119,500 | 2.42× |
| 5 | IND501359AAE | 1.47% | ₹177,250 | 1.80× |

## Corridor-specific actions

| Corridor | Recommendation | Evidence | Est. late deliveries reduced | Est. revenue recovered |
|---|---|---|---:|---:|
| IND000000ACB → IND562132AAA | Upgrade endpoint capacity and enforce dispatch slots | Median dwell proxy is 193 minutes (top quartile). | 19.8 | ₹9,900 |
| IND000000ACB → IND712311AAA | Upgrade endpoint capacity and enforce dispatch slots | Median dwell proxy is 226 minutes (top quartile). | 14.4 | ₹7,200 |
| IND562132AAA → IND000000ACB | Upgrade endpoint capacity and enforce dispatch slots | Median dwell proxy is 312 minutes (top quartile). | 13.2 | ₹6,600 |
| IND000000ACB → IND501359AAE | Upgrade endpoint capacity and enforce dispatch slots | Median dwell proxy is 247 minutes (top quartile). | 8.7 | ₹4,350 |
| IND000000ACB → IND421302AAG | Upgrade endpoint capacity and enforce dispatch slots | Median dwell proxy is 256 minutes (top quartile). | 9.9 | ₹4,950 |

## Top-three hub upgrade case

Across the extract, 14,079 trips contain at least one leg above 120% of OSRM. 3,903 of those touch the top three hubs. At the stated 30% intervention effectiveness, upgrading all three is estimated to reduce **1,171 late deliveries**, equal to **8.3% of network late trips**, and recover **₹585,450** of the modeled **₹7,039,500** exposure.

## FTL versus Carting operating rule

Use `artifacts/route_policy.csv` rather than a universal distance threshold. The framework holds distance, dispatch time, and facility structural position fixed, predicts ETA under both route types, and combines transport cost, time value, and expected late-delivery penalty. The lower expected-cost route is recommended. Review low-sample policy cells before operational use; the comparison is observational, not causal.

## 30-day action plan

1. Put the top three hubs on daily exception review and validate whether the dwell proxy corresponds to sort, dock, or dispatch-slot constraints.
2. Pilot the highest-ranked corridor recommendation for two weeks; retain a comparable untreated corridor as a control.
3. Publish graph-enhanced ETAs beside OSRM, monitor MAE and ±15% accuracy by route type and time bucket, and alert on unseen corridors.
4. Replace the ₹500 penalty and transport-cost assumptions in `src/config.py` with Finance-approved values before making budget commitments.

## Required caveats

The extract covers only 14,817 trips over a short historical window. Hub/corridor scores are associations, not proof of root cause. The upgrade and revenue figures are transparent scenarios, not causal forecasts. A controlled pilot and refreshed cost inputs are required before network-wide rollout.
