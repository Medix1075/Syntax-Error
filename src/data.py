import numpy as np
import pandas as pd

def generate_demo(n=4000, seed=42):
    rng=np.random.default_rng(seed); hubs=[f"H{i:02d}" for i in range(20)]
    rows=[]
    for i in range(n):
        o,d=rng.choice(hubs,2,replace=False); dist=float(rng.uniform(15,700)); hour=int(rng.integers(0,24)); route=rng.choice(["FL","Carting"],p=[.55,.45])
        planned=dist/rng.uniform(35,65)*60
        hub_effect=(int(o[1:]) in [2,7,11,15])*35+(int(d[1:]) in [2,7,11,15])*20
        peak=25 if hour in [8,9,18,19,20] else 0
        route_effect=8 if route=="Carting" else 0
        noise=float(rng.normal(0,18)); actual=max(20,planned+hub_effect+peak+route_effect+noise)
        rows.append([f"T{i:05d}",o,d,dist,planned,actual,route,hour])
    return pd.DataFrame(rows,columns=["trip_id","origin","destination","distance_km","planned_minutes","actual_minutes","route_type","dispatch_hour"])

def validate(df):
    required={"trip_id","origin","destination","distance_km","planned_minutes","actual_minutes","route_type","dispatch_hour"}
    missing=required-set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    return df.dropna().query("distance_km > 0 and planned_minutes > 0 and actual_minutes > 0").copy()
