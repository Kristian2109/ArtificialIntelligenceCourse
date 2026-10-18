"""Check API responses and faithful playback: python3 -B -m unittest -v test_road_server.py"""

import json
from math import isclose
import unittest

import road_pathfinding as routing
from road_server import RouteService
from test_road_pathfinding import fixture


class TraceTests(unittest.TestCase):
    def test_trace_matches_search_and_queue_order(self):
        graph = routing.load_osm(routing.DEFAULT_OSM)
        start, _ = routing.nearest_node(graph, (42.6977, 23.3219))
        goal, _ = routing.nearest_node(graph, (42.6936, 23.3349))
        events = []
        result = routing.find_route(graph, start, goal, on_step=events.append)
        plain = routing.find_route(graph, start, goal)
        self.assertEqual(result["edge_ids"], plain["edge_ids"])
        self.assertEqual(events[0]["kind"], "start")
        self.assertEqual(events[-1]["kind"], "goal")
        self.assertEqual(sum(event["kind"] == "expand" for event in events), result["expanded_states"])
        self.assertAlmostEqual(events[-1]["g"], result["objective_cost"])
        self.assertEqual(events[-1]["h"], 0)
        for before, after in zip(events, events[1:]):
            self.assertEqual(before["frontier"][0]["node"], after["node"])
            self.assertAlmostEqual(before["frontier"][0]["g"], after["g"])
            self.assertEqual(before["frontier_count"] >= len(before["frontier"]), True)
        json.dumps(events, allow_nan=False)

    def test_trace_shows_rejected_turn(self):
        graph = fixture({1:(0,0), 2:(0,.01), 3:(0,.02), 4:(.01,.01)},
                        [(10,[1,2],{"oneway":"yes"}), (20,[2,3],{"oneway":"yes"}),
                         (30,[2,4,3],{"oneway":"yes"})],
                        [(100,"no_right_turn",10,[("node",2)],20)])
        events = []
        result = routing.find_route(graph, graph.osm_ids.index(1), graph.osm_ids.index(3), on_step=events.append)
        self.assertEqual(result["node_ids"], [1,2,4,3])
        self.assertTrue(any(move["status"] == "restricted" for event in events for move in event["considered"]))

    def test_trace_exhaustion(self):
        graph = fixture({1:(0,0),2:(0,.01)}, [(10,[1,2],{"oneway":"yes"})])
        events = []
        result = routing.find_route(graph, graph.osm_ids.index(2), graph.osm_ids.index(1), on_step=events.append)
        self.assertIsNone(result)
        self.assertEqual(events[-1]["kind"], "no_route")
        self.assertEqual(events[-1]["frontier_count"], 0)


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = RouteService()
        cls.payload = {"start":[42.6977,23.3219], "goal":[42.6936,23.3349], "compare":True}

    def test_real_api_objectives_and_heuristics(self):
        for objective in ("time", "distance"):
            for heuristic in ("alt", "geometric", "dijkstra"):
                result = self.service.route({**self.payload,"objective":objective,"heuristic":heuristic})
                self.assertTrue(isclose(result["route"]["objective_cost"], result["dijkstra"]["objective_cost"]))
                self.assertEqual(result["trace_omitted_frames"], 0)
                self.assertGreater(result["search_without_trace_ms"], 0)
                graph = result["network"]
                node = result["start_node"]
                for eid in result["route"]["edge_ids"]:
                    edge = graph["edges"][eid]
                    self.assertEqual(edge[0], node)
                    node = edge[1]
                self.assertEqual(node, result["goal_node"])
                json.dumps(result, allow_nan=False)

    def test_closures_and_profile_cache(self):
        base = self.service.route(self.payload)
        way = base["network"]["edges"][base["route"]["edge_ids"][0]][2]
        changed = self.service.route({**self.payload,"closed_ways":[way],"turn_delay_s":4,"signal_delay_s":10})
        self.assertTrue(all(edge[2] != way for edge in changed["network"]["edges"]))
        first = self.service.graph_for(routing.Profile())
        self.assertIs(first, self.service.graph_for(routing.Profile()))
        self.assertLessEqual(len(self.service.cache), 6)

    def test_invalid_requests(self):
        invalid = [[], {"start":None}, {"goal":[91,0]}, {"heuristic":"greedy"},
                   {"avoid_tolls":"false"}, {"turn_delay_s":-1}, {"signal_delay_s":float("inf")},
                   {"closed_ways":["123"]}, {"objective":"fuel"}]
        for payload in invalid:
            with self.assertRaises(ValueError):
                self.service.route({**self.payload,**payload} if isinstance(payload,dict) else payload)


if __name__ == "__main__":
    unittest.main()
