import json,sys
from pathlib import Path
import pandas as pd
import streamlit as st
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.operations import route_recommendation
ROOT=Path(__file__).resolve().parents[1]; A=ROOT/"artifacts"
st.set_page_config(page_title="Delivery ETA Intelligence",layout="wide");st.title("Optimising Delivery ETAs with Graph-Based Network Intelligence")
if not (A/"metrics.json").exists(): st.warning("Run: python -m src.pipeline --generate-demo --run-all");st.stop()
m=json.loads((A/"metrics.json").read_text());c1,c2=st.columns(2);c1.metric("Baseline MAE",f"{m['baseline']['mae']:.2f} min");c2.metric("Graph-enhanced MAE",f"{m['graph_enhanced']['mae']:.2f} min")
st.subheader("Top bottleneck hubs");st.dataframe(pd.read_csv(A/"top_bottlenecks.csv"),use_container_width=True)
st.subheader("Recommended interventions");st.dataframe(pd.read_csv(A/"interventions.csv"),use_container_width=True)
st.subheader("FL vs Carting decision");d=st.slider("Distance (km)",10,700,150);h=st.slider("Dispatch hour",0,23,10);r=st.slider("Delay risk",0.0,1.0,.3);st.success(route_recommendation(d,h,r))
