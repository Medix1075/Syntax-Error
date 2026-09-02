from src.data import generate_demo,validate
from src.graph_features import build_graph,node_metrics
from src.operations import route_recommendation

def test_demo_contract():
 df=validate(generate_demo(100)); assert len(df)==100

def test_graph_metrics():
 g=build_graph(generate_demo(100)); m=node_metrics(g); assert set(["hub","betweenness","in_degree"]).issubset(m.columns)

def test_route_decision():
 assert route_recommendation(500,10,.2)=="FL"
