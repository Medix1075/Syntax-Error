import unittest

import numpy as np

from src.config import BUSINESS
from src.data import aggregate_segments_to_legs, assign_temporal_split, generate_demo_raw, validate_raw
from src.graph_features import GRAPH_FEATURES, add_graph_features, aggregate_trip_features, build_graph
from src.models import benchmark_models
from src.operations import corridor_audit, hub_upgrade_scenario, rank_bottleneck_hubs


class CorePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = generate_demo_raw(n_trips=120, seed=7)
        cls.legs = aggregate_segments_to_legs(validate_raw(cls.raw))
        cls.split_legs, _ = assign_temporal_split(cls.legs)
        cls.graph = build_graph(cls.split_legs[cls.split_legs["split"].eq("reference")])
        cls.trips = aggregate_trip_features(add_graph_features(cls.split_legs, cls.graph))

    def test_raw_to_leg_grain(self):
        expected = self.raw.groupby([
            "trip_uuid", "source_center", "destination_center", "od_start_time",
            "od_end_time", "route_type",
        ]).ngroups
        self.assertEqual(len(self.legs), expected)
        self.assertTrue((self.legs["actual_minutes"] > 0).all())

    def test_whole_trips_do_not_cross_splits(self):
        counts = self.split_legs.groupby("trip_uuid")["split"].nunique()
        self.assertEqual(int(counts.max()), 1)
        self.assertEqual(set(self.split_legs["split"]), {"reference", "train", "test"})

    def test_graph_contract_and_metrics(self):
        self.assertFalse(self.graph.edge_strata.empty)
        self.assertTrue({
            "delay_ratio_weight", "route_type", "time_bucket",
        }.issubset(self.graph.edge_strata.columns))
        self.assertTrue({
            "hub", "betweenness", "in_degree", "out_degree", "clustering",
        }.issubset(self.graph.node_metrics.columns))
        self.assertTrue(np.isfinite(self.graph.node_metrics["betweenness"]).all())

    def test_trip_features_have_no_missing_model_inputs(self):
        self.assertTrue(set(GRAPH_FEATURES).issubset(self.trips.columns))
        self.assertEqual(int(self.trips[GRAPH_FEATURES].isna().sum().sum()), 0)

    def test_model_benchmark_same_holdout(self):
        train = self.trips[self.trips["split"].eq("train")]
        test = self.trips[self.trips["split"].eq("test")]
        _, metrics, scored = benchmark_models(train, test, seed=7)
        self.assertEqual(metrics["baseline"]["n_test_trips"], len(test))
        self.assertEqual(metrics["graph_enhanced"]["n_test_trips"], len(test))
        self.assertEqual(len(scored), len(test))

    def test_bottleneck_attribution_and_upgrade_no_double_count(self):
        full_graph = build_graph(self.legs)
        corridors = corridor_audit(self.legs, full_graph, BUSINESS)
        top = rank_bottleneck_hubs(self.legs, full_graph, BUSINESS)
        scenario = hub_upgrade_scenario(self.legs, top, BUSINESS)
        self.assertTrue(corridors["corridor_score"].is_monotonic_decreasing)
        self.assertLessEqual(
            scenario["late_trips_touching_top_3"], scenario["network_late_trips"]
        )


if __name__ == "__main__":
    unittest.main()
