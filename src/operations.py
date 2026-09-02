import pandas as pd

def bottlenecks(df,node_metrics,top_k=5):
    delay=df.actual_minutes-df.planned_minutes
    hub=pd.concat([pd.DataFrame({"hub":df.origin,"delay":delay}),pd.DataFrame({"hub":df.destination,"delay":delay})]).groupby("hub").agg(mean_delay=("delay","mean"),breaches=("delay",lambda x:(x>30).sum()),volume=("delay","size")).reset_index()
    x=hub.merge(node_metrics,on="hub",how="left").fillna(0)
    for c in ["mean_delay","breaches","betweenness","volume"]: x[c+"_norm"]=(x[c]-x[c].min())/(x[c].max()-x[c].min()+1e-9)
    x["bottleneck_score"]=.35*x.mean_delay_norm+.30*x.breaches_norm+.25*x.betweenness_norm+.10*x.volume_norm
    return x.sort_values("bottleneck_score",ascending=False).head(top_k)

def route_recommendation(distance_km, hour, risk):
    if distance_km>250 and risk<0.6: return "FL"
    if hour in [8,9,18,19,20] or risk>=0.6: return "Carting"
    return "FL" if distance_km>120 else "Carting"

def intervention_plan(top):
    rows=[]
    for r in top.itertuples():
        action="Add dispatch buffer and upgrade capacity" if r.betweenness>top.betweenness.median() else "Rebalance corridor schedule and add monitoring"
        rows.append({"hub":r.hub,"action":action,"priority_score":round(r.bottleneck_score,3),"estimated_late_reduction_pct":round(10+25*r.bottleneck_score,1)})
    return pd.DataFrame(rows)
