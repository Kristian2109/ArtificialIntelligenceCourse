"""Behavioral checks: python3 -m unittest -v test_road_pathfinding.py"""

from heapq import heappop, heappush
import gzip
from io import BytesIO
from math import inf, isclose
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import road_pathfinding as routing


def fixture(coordinates, ways, restrictions=(), node_tags=None, profile=None):
    """Make tiny OSM fixtures; these are synthetic, unlike the bundled Sofia data."""
    root = ET.Element("osm", version="0.6")
    for node, (lat, lon) in coordinates.items():
        element = ET.SubElement(root, "node", id=str(node), lat=str(lat), lon=str(lon))
        for key, value in (node_tags or {}).get(node, {}).items():
            ET.SubElement(element, "tag", k=key, v=value)
    for way, refs, attributes in ways:
        element = ET.SubElement(root, "way", id=str(way))
        for node in refs:
            ET.SubElement(element, "nd", ref=str(node))
        for key, value in {"highway": "residential", **attributes}.items():
            ET.SubElement(element, "tag", k=key, v=value)
    for rid, kind, from_way, via, to_way in restrictions:
        element = ET.SubElement(root, "relation", id=str(rid))
        ET.SubElement(element, "member", type="way", role="from", ref=str(from_way))
        for member_type, ref in via:
            ET.SubElement(element, "member", type=member_type, role="via", ref=str(ref))
        ET.SubElement(element, "member", type="way", role="to", ref=str(to_way))
        for key, value in {"type": "restriction", "restriction": kind}.items():
            ET.SubElement(element, "tag", k=key, v=value)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fixture.osm"
        ET.ElementTree(root).write(path)
        return routing.load_osm(path, profile)


def route(graph, start, goal, **kwargs):
    return routing.find_route(graph, graph.osm_ids.index(start), graph.osm_ids.index(goal), **kwargs)


class RoutingTests(unittest.TestCase):
    coordinates = {1: (0, 0), 2: (0, .01), 3: (0, .02),
                   4: (.01, .01), 5: (.01, .02), 6: (.01, .03), 7: (0, .03)}

    def test_fastest_and_shortest_are_different(self):
        coords = {1: (0, 0), 2: (.02, .01), 3: (0, .01), 4: (0, .02)}
        ways = [(10, [1, 2, 4], {"maxspeed": "120", "oneway": "yes"}),
                (20, [1, 3, 4], {"maxspeed": "20", "oneway": "yes"})]
        fastest = route(fixture(coords, ways), 1, 4)
        shortest = route(fixture(coords, ways, profile=routing.Profile(objective="distance")), 1, 4)
        self.assertEqual(fastest["node_ids"], [1, 2, 4])
        self.assertEqual(shortest["node_ids"], [1, 3, 4])
        self.assertGreater(fastest["distance_m"], shortest["distance_m"])

    def test_oneway_reverse_and_roundabout(self):
        for tags, allowed in [({"oneway": "yes"}, (1, 2)),
                              ({"oneway": "-1"}, (2, 1)),
                              ({"junction": "roundabout"}, (1, 2))]:
            graph = fixture(self.coordinates, [(10, [1, 2], tags)])
            self.assertIsNotNone(route(graph, *allowed))
            self.assertIsNone(route(graph, *allowed[::-1]))

    def test_directional_speed_and_mode_specific_access(self):
        graph = fixture(self.coordinates, [(10, [1, 2], {"access": "no", "motorcar": "yes",
                        "maxspeed:forward": "60", "maxspeed:backward": "30"})])
        self.assertAlmostEqual(route(graph, 2, 1)["estimated_seconds"], 2 * route(graph, 1, 2)["estimated_seconds"])
        graph = fixture(self.coordinates, [(10, [1, 2], {"access": "no", "motorcar:forward": "yes"})])
        self.assertIsNotNone(route(graph, 1, 2))
        self.assertIsNone(route(graph, 2, 1))

    def test_unit_conversion(self):
        self.assertAlmostEqual(routing.speed_kph("30 mph", 1), 48.28032)
        self.assertEqual(routing.speed_kph("50;70", 1), 50)
        self.assertEqual(routing.speed_kph("BG:urban", 30), 30)

    def test_avoidance_and_closures(self):
        for tags, profile in [({"toll": "yes"}, routing.Profile(avoid_tolls=True)),
                              ({"highway": "motorway"}, routing.Profile(avoid_motorways=True)),
                              ({"surface": "gravel"}, routing.Profile(avoid_unpaved=True)),
                              ({}, routing.Profile(closed_ways=frozenset({10}))),
                              ({"access": "private"}, routing.Profile()),
                              ({"access": "destination"}, routing.Profile())]:
            graph = fixture(self.coordinates, [(10, [1, 2, 3], tags),
                            (20, [1, 4, 3], {"oneway": "yes"})], profile=profile)
            self.assertEqual(route(graph, 1, 3)["node_ids"], [1, 4, 3])

    def test_barrier_and_conditional_road_exclusion(self):
        for way_tags, node_tags in [({}, {2: {"barrier": "bollard"}}),
                                   ({"access:conditional": "no @ (Mo-Fr)"}, {})]:
            graph = fixture(self.coordinates, [(10, [1, 2, 3], way_tags),
                            (20, [1, 4, 3], {})], node_tags=node_tags)
            self.assertEqual(route(graph, 1, 3)["node_ids"], [1, 4, 3])

    def test_turn_restriction_requires_distinct_arrival_states(self):
        graph = fixture(self.coordinates, [(10, [1, 2], {"oneway": "yes"}),
                        (20, [2, 3], {"oneway": "yes"}),
                        (30, [1, 4, 2], {"oneway": "yes"})],
                        [(100, "no_right_turn", 10, [("node", 2)], 20)])
        self.assertEqual(route(graph, 1, 3)["node_ids"], [1, 4, 2, 3])

    def test_only_turn_and_closed_required_exit(self):
        ways = [(10, [1, 2], {"oneway": "yes"}), (20, [2, 3], {"oneway": "yes"}),
                (30, [2, 4, 3], {"oneway": "yes"})]
        rules = [(100, "only_right_turn", 10, [("node", 2)], 30)]
        graph = fixture(self.coordinates, ways, rules)
        self.assertEqual(route(graph, 1, 3)["node_ids"], [1, 2, 4, 3])
        graph = fixture(self.coordinates, ways, rules, profile=routing.Profile(closed_ways=frozenset({30})))
        self.assertIsNone(route(graph, 1, 3))
        self.assertIsNotNone(route(graph, 1, 2))  # stopping at the junction is legal

    def test_same_way_u_turn_does_not_forbid_straight_travel(self):
        graph = fixture(self.coordinates, [(10, [1, 2, 3], {})],
                        [(100, "no_u_turn", 10, [("node", 2)], 10)])
        self.assertEqual(route(graph, 1, 3)["node_ids"], [1, 2, 3])
        first = next(i for i, e in enumerate(graph.edges)
                     if graph.osm_ids[e.source] == 1 and graph.osm_ids[e.target] == 2)
        reverse = next(i for i, e in enumerate(graph.edges)
                       if graph.osm_ids[e.source] == 2 and graph.osm_ids[e.target] == 1)
        active = routing.advance_restrictions(graph, (), first)
        self.assertIsNone(routing.advance_restrictions(graph, active, reverse))

    def test_via_way_restriction_spans_shape_nodes(self):
        ways = [(10, [1, 2], {"oneway": "yes"}),
                (20, [2, 4, 5], {"oneway": "yes"}),
                (30, [5, 6], {"oneway": "yes"}),
                (40, [5, 7, 6], {"oneway": "yes"})]
        for kind, expected in [("no_right_turn", [1, 2, 4, 5, 7, 6]),
                               ("only_right_turn", [1, 2, 4, 5, 6])]:
            graph = fixture(self.coordinates, ways, [(100, kind, 10, [("way", 20)], 30)])
            self.assertEqual(route(graph, 1, 6)["node_ids"], expected)
            # Starting inside the via way has no from-way history.
            self.assertEqual(route(graph, 2, 6)["node_ids"], [2, 4, 5, 6])

    def test_multiple_via_ways_and_only_restriction_early_exit(self):
        ways = [(10, [1, 2], {"oneway": "yes"}), (20, [2, 4], {"oneway": "yes"}),
                (30, [4, 5], {"oneway": "yes"}), (40, [5, 6], {"oneway": "yes"}),
                (50, [4, 6], {"oneway": "yes"})]
        rule = [(100, "only_straight_on", 10, [("way", 20), ("way", 30)], 40)]
        graph = fixture(self.coordinates, ways, rule)
        self.assertEqual(route(graph, 1, 6)["node_ids"], [1, 2, 4, 5, 6])

    def test_static_slowdown_and_turn_penalty_change_the_optimum(self):
        ways = [(10, [1, 2, 3], {"oneway": "yes", "maxspeed": "30"}),
                (20, [1, 4], {"oneway": "yes", "maxspeed": "120"}),
                (30, [4, 3], {"oneway": "yes", "maxspeed": "120"})]
        self.assertEqual(route(fixture(self.coordinates, ways), 1, 3)["node_ids"], [1, 4, 3])
        for profile in [routing.Profile(turn_delay_s=300),
                        routing.Profile(slowdown_by_way={20: 10, 30: 10})]:
            self.assertEqual(route(fixture(self.coordinates, ways, profile=profile), 1, 3)["node_ids"], [1, 2, 3])

    def test_signal_delay_and_zero_length_route(self):
        graph = fixture(self.coordinates, [(10, [1, 2, 3], {})],
                        node_tags={2: {"highway": "traffic_signals"}},
                        profile=routing.Profile(signal_delay_s=15))
        result = route(graph, 1, 3)
        self.assertAlmostEqual(result["estimated_seconds"], result["distance_m"] / (30 / 3.6) + 15)
        result = route(graph, 1, 1)
        self.assertEqual(result["objective_cost"], 0)
        self.assertEqual(result["node_ids"], [1])

    def test_invalid_input_and_snap_limit(self):
        for kwargs in [{"turn_delay_s": -1}, {"signal_delay_s": float("nan")},
                       {"slowdown_by_way": {10: .5}}]:
            with self.assertRaises(ValueError):
                routing.Profile(**kwargs)
        graph = fixture(self.coordinates, [(10, [1, 2], {})])
        for point in [(float("nan"), 0), (91, 0), (60, 60)]:
            with self.assertRaises(ValueError):
                routing.nearest_node(graph, point)

    def test_unreachable_landmark_distances(self):
        graph = fixture(self.coordinates, [(10, [1, 2, 3], {"oneway": "yes"}),
                                          (20, [4, 5, 6], {"oneway": "yes"})])
        tables = routing.prepare_landmarks(graph, 4)
        self.assertIsNone(route(graph, 1, 6, landmarks=tables))
        result = route(graph, 1, 3, landmarks=tables)
        self.assertAlmostEqual(result["objective_cost"], route(graph, 1, 3, use_heuristic=False)["objective_cost"])

    def test_downloader_cache_and_failed_response(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roads.osm.gz"
            payload = b'<osm version="0.6"><node id="1" lat="0" lon="0"/></osm>'
            with patch.object(routing, "urlopen", return_value=BytesIO(payload)) as request:
                routing.download_osm(path)
                self.assertIn(b"type", request.call_args.args[0].data)
            self.assertEqual(gzip.decompress(path.read_bytes()), payload)
            saved = path.read_bytes()
            with patch.object(routing, "urlopen", return_value=BytesIO(b'<osm><remark>timeout</remark></osm>')):
                with self.assertRaises(ValueError):
                    routing.download_osm(path)
            self.assertEqual(path.read_bytes(), saved)
            with self.assertRaises(ValueError):
                routing.download_osm(path, (43, 24, 42, 23))

    def test_random_directed_graphs_against_independent_node_dijkstra(self):
        rng = random.Random(42)
        for objective in ("distance", "time"):
            coordinates = {n: (rng.uniform(42.68, 42.71), rng.uniform(23.30, 23.35)) for n in range(16)}
            ways = [(1000 + i, [u, v], {"oneway": "yes", "maxspeed": str(rng.choice([20, 50, 90]))})
                    for i, (u, v) in enumerate((u, v) for u in coordinates for v in coordinates
                                               if u != v and rng.random() < .17)]
            graph = fixture(coordinates, ways, profile=routing.Profile(objective=objective))
            landmarks = routing.prepare_landmarks(graph, 4)
            for start in range(len(graph.coordinates)):
                distances = [inf] * len(graph.coordinates)
                distances[start] = 0
                heap = [(0, start)]
                while heap:
                    cost, node = heappop(heap)
                    if cost != distances[node]:
                        continue
                    for eid in graph.outgoing[node]:
                        edge = graph.edges[eid]
                        candidate = cost + graph.costs[eid]
                        if candidate < distances[edge.target]:
                            distances[edge.target] = candidate
                            heappush(heap, (candidate, edge.target))
                for goal in range(len(graph.coordinates)):
                    for tables in ((), landmarks):
                        result = routing.find_route(graph, start, goal, tables)
                        if distances[goal] == inf:
                            self.assertIsNone(result)
                        else:
                            self.assertTrue(isclose(result["objective_cost"], distances[goal], rel_tol=1e-10, abs_tol=1e-7))

    def test_real_data_legality_costs_and_geometric_consistency(self):
        rng = random.Random(2026)
        for objective in ("time", "distance"):
            graph = routing.load_osm(routing.DEFAULT_OSM, routing.Profile(objective=objective, turn_delay_s=4, signal_delay_s=8))
            self.assertGreater(len(graph.osm_ids), 1000)
            self.assertGreater(len(graph.restrictions), 100)
            tables = routing.prepare_landmarks(graph, 4)
            for _ in range(8):
                start, goal = rng.sample(range(len(graph.coordinates)), 2)
                geometric = routing.make_heuristic(graph, goal)
                self.assertEqual(geometric(goal), 0)
                for eid, edge in enumerate(graph.edges):
                    self.assertLessEqual(geometric(edge.source), graph.costs[eid] + geometric(edge.target) + 1e-7)
                baseline = routing.find_route(graph, start, goal, use_heuristic=False)
                for landmarks in ((), tables):
                    result = routing.find_route(graph, start, goal, landmarks)
                    if baseline is None:
                        self.assertIsNone(result)
                        continue
                    self.assertAlmostEqual(result["objective_cost"], baseline["objective_cost"], places=6)
                    node, active = start, ()
                    for eid in result["edge_ids"]:
                        self.assertEqual(graph.edges[eid].source, node)
                        active = routing.advance_restrictions(graph, active, eid)
                        self.assertIsNotNone(active)
                        node = graph.edges[eid].target
                    self.assertEqual(node, goal)
                    measured = result["estimated_seconds"] if objective == "time" else result["distance_m"]
                    self.assertAlmostEqual(measured, result["objective_cost"], places=6)


if __name__ == "__main__":
    unittest.main()
