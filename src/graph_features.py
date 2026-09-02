import networkx as nx
import pandas as pd

def build_graph(df):
    g=nx.DiGraph()
    agg=df.groupby(["origin","destination"]).agg(weight=("trip_id","size"),mean_actual=("actual_minutes","mean")).reset_index()
    for r in agg.itertuples(index=False): g.add_edge(r.origin,r.destination,weight=float(r.weight),mean_actual=float(r.mean_actual))
    return g

def node_metrics(g):
    bc=nx.betweenness_centrality(g,weight="weight",normalized=True)
    cc=nx.clustering(g.to_undirected(),weight="weight")
    return pd.DataFrame({"hub":list(g.nodes),"in_degree":[g.in_degree(x,weight="weight") for x in g.nodes],"out_degree":[g.out_degree(x,weight="weight") for x in g.nodes],"betweenness":[bc[x] for x in g.nodes],"clustering":[cc[x] for x in g.nodes]})

def add_graph_features(df,g):
    m=node_metrics(g).set_index("hub")
    out=df.copy()
    for col in ["in_degree","out_degree","betweenness","clustering"]:
        out["origin_"+col]=out.origin.map(m[col]).fillna(0)
        out["destination_"+col]=out.destination.map(m[col]).fillna(0)
    return out
