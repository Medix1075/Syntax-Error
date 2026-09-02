import argparse,json
from pathlib import Path
from .config import DATA,ARTIFACTS,RANDOM_STATE
from .data import generate_demo,validate
from .graph_features import build_graph,node_metrics,add_graph_features
from .models import fit_evaluate
from .operations import bottlenecks,intervention_plan

def main():
 p=argparse.ArgumentParser();p.add_argument("--generate-demo",action="store_true");p.add_argument("--run-all",action="store_true");a=p.parse_args()
 (DATA/"raw").mkdir(parents=True,exist_ok=True);ARTIFACTS.mkdir(exist_ok=True)
 path=DATA/"raw"/"trips.csv"
 if a.generate_demo or not path.exists(): generate_demo(seed=RANDOM_STATE).to_csv(path,index=False)
 df=validate(__import__("pandas").read_csv(path));g=build_graph(df);nm=node_metrics(g);enh=add_graph_features(df,g)
 _,base,_,_=fit_evaluate(df,False);_,graph,_,_=fit_evaluate(enh,True)
 top=bottlenecks(df,nm);plan=intervention_plan(top)
 nm.to_csv(ARTIFACTS/"hub_metrics.csv",index=False);top.to_csv(ARTIFACTS/"top_bottlenecks.csv",index=False);plan.to_csv(ARTIFACTS/"interventions.csv",index=False)
 (ARTIFACTS/"metrics.json").write_text(json.dumps({"baseline":base,"graph_enhanced":graph},indent=2))
 print(json.dumps({"baseline":base,"graph_enhanced":graph,"top_hubs":top.hub.tolist()},indent=2))
if __name__=="__main__": main()
