#!/usr/bin/env python3
"""Serve the offline road visualizer and the real Python routing API.

Run: python3 road_server.py --port 8000
Open: http://127.0.0.1:8000
"""

import argparse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from math import isclose, isfinite
from pathlib import Path
from threading import Lock
from time import perf_counter
from urllib.parse import urlsplit

from road_pathfinding import (DEFAULT_OSM, Profile, find_route, load_osm,
                              nearest_node, prepare_landmarks)


HTML_PATH = Path(__file__).with_name("road_pathfinding_visualizer.html")
MAX_TRACE_FRAMES = 12_000


def network_data(graph):
    """Compact shared geometry; node/edge indices match this exact profile graph."""
    names, name_indices, edges = [], {}, []
    for edge in graph.edges:
        if edge.name not in name_indices:
            name_indices[edge.name] = len(names)
            names.append(edge.name)
        edges.append([edge.source, edge.target, edge.way, name_indices[edge.name],
                      edge.meters, edge.seconds])
    return {"nodes": [[oid, *coordinate] for oid, coordinate in zip(graph.osm_ids, graph.coordinates)],
            "edges": edges, "names": names, "restrictions": len(graph.restrictions),
            "notes": graph.notes, "attribution": "© OpenStreetMap contributors, ODbL 1.0"}


class RouteService:
    """A small bounded cache keeps immutable profile graphs and ALT tables reusable."""

    def __init__(self, osm_path=DEFAULT_OSM):
        self.osm_path = osm_path
        self.cache = OrderedDict()
        self.lock = Lock()

    def graph_for(self, profile, with_landmarks=False):
        key = (profile.objective, profile.avoid_tolls, profile.avoid_motorways,
               profile.avoid_unpaved, tuple(sorted(profile.closed_ways)),
               profile.turn_delay_s, profile.signal_delay_s)
        with self.lock:
            if key not in self.cache:
                graph = load_osm(self.osm_path, profile)
                self.cache[key] = [graph, None, network_data(graph)]
            entry = self.cache[key]
            if with_landmarks and entry[1] is None:
                entry[1] = prepare_landmarks(entry[0], 4)
            self.cache.move_to_end(key)
            while len(self.cache) > 6:
                self.cache.popitem(last=False)
            return entry

    def route(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("request must be a JSON object")
        objective = payload.get("objective", "time")
        heuristic = payload.get("heuristic", "alt")
        if heuristic not in {"alt", "geometric", "dijkstra"}:
            raise ValueError("heuristic must be alt, geometric, or dijkstra")
        flags = {}
        for name in ("avoid_tolls", "avoid_motorways", "avoid_unpaved", "compare"):
            flags[name] = payload.get(name, False)
            if type(flags[name]) is not bool:
                raise ValueError(f"{name} must be true or false")
        delays = {}
        for name in ("turn_delay_s", "signal_delay_s"):
            value = payload.get(name, 0)
            if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 600:
                raise ValueError(f"{name} must be a finite number from 0 to 600 seconds")
            delays[name] = value
        closed = payload.get("closed_ways", [])
        if not isinstance(closed, list) or len(closed) > 100 or any(type(w) is not int for w in closed):
            raise ValueError("closed_ways must be a list of at most 100 integer OSM way IDs")
        coordinates = []
        for name in ("start", "goal"):
            point = payload.get(name)
            if (not isinstance(point, list) or len(point) != 2
                    or any(type(n) not in (int, float) or not isfinite(n) for n in point)):
                raise ValueError(f"{name} must contain latitude and longitude")
            coordinates.append(point)
        profile = Profile(objective=objective, avoid_tolls=flags["avoid_tolls"],
                          avoid_motorways=flags["avoid_motorways"], avoid_unpaved=flags["avoid_unpaved"],
                          closed_ways=frozenset(closed), **delays)
        begun = perf_counter()
        graph, landmarks, network = self.graph_for(profile, heuristic == "alt")
        prepare_ms = (perf_counter() - begun) * 1000
        start, start_offset = nearest_node(graph, coordinates[0])
        goal, goal_offset = nearest_node(graph, coordinates[1])
        events, total_events = [], 0

        def record(event):
            nonlocal total_events
            total_events += 1
            if len(events) < MAX_TRACE_FRAMES or event["kind"] in {"goal", "no_route"}:
                events.append(event)

        result = find_route(graph, start, goal, landmarks or (), heuristic != "dijkstra", record)
        # Instrumentation has a cost: report a separate run without trace collection.
        plain = find_route(graph, start, goal, landmarks or (), heuristic != "dijkstra")
        comparison = None
        if flags["compare"]:
            baseline = find_route(graph, start, goal, use_heuristic=False)
            if (baseline is None) != (result is None) or (result and not isclose(
                    result["objective_cost"], baseline["objective_cost"], rel_tol=1e-9, abs_tol=1e-7)):
                raise RuntimeError("A* and Dijkstra disagree")
            if baseline:
                comparison = {key: baseline[key] for key in ("objective_cost", "expanded_states", "search_ms")}
        return {"network": network, "route": result, "trace": events,
                "trace_total_frames": total_events, "trace_omitted_frames": total_events - len(events),
                "start_node": start, "goal_node": goal, "heuristic": heuristic,
                "objective": objective, "preparation_ms": prepare_ms,
                "search_without_trace_ms": plain["search_ms"] if plain else None,
                "snap_distance_m": {"start": start_offset, "goal": goal_offset},
                "dijkstra": comparison}


def make_server(host="127.0.0.1", port=8000, osm_path=DEFAULT_OSM):
    service = RouteService(osm_path)
    service.graph_for(Profile())  # Fail early if the extract cannot be loaded.

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, data, content_type="application/json; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, status, value):
            self.send(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode())

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in {"/", "/road_pathfinding_visualizer.html"}:
                self.send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/network":
                self.send_json(200, service.graph_for(Profile())[2])
            elif path == "/favicon.ico":
                self.send(204, b"")
            else:
                self.send_json(404, {"error": "Not found"})

        def do_POST(self):
            if urlsplit(self.path).path != "/api/route":
                self.send_json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32_768:
                    raise ValueError("request body must contain at most 32 KB of JSON")
                payload = json.loads(self.rfile.read(length))
                self.send_json(200, service.route(payload))
            except (ValueError, TypeError, UnicodeError) as error:
                self.send_json(400, {"error": str(error)})
            except (OSError, RuntimeError) as error:
                self.send_json(500, {"error": str(error)})

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--osm", type=Path, default=DEFAULT_OSM)
    args = parser.parse_args()
    server = make_server(args.host, args.port, args.osm)
    print(f"Road Pathfinding Studio: http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
