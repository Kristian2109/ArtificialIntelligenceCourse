#!/usr/bin/env python3
"""Exact A* routing on real OpenStreetMap roads, using only Python's standard library.

Run: python3 road_pathfinding.py --compare --landmarks 4
See ROAD_PATHFINDING.md for the model, guarantees, constraints, and data source.
"""

from __future__ import annotations

import argparse
from array import array
from dataclasses import dataclass, field
import gzip
from heapq import heappop, heappush, nsmallest
from itertools import count
import json
from math import asin, cos, inf, isfinite, radians, sin, sqrt
from pathlib import Path
import re
import sys
from time import perf_counter
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


EARTH_RADIUS_M = 6_371_008.8
DEFAULT_BBOX = (42.685, 23.310, 42.705, 23.340)  # south, west, north, east
DEFAULT_OSM = Path(__file__).with_name("sofia_roads.osm.gz")
# Modeling assumptions, not country-specific legal limits or measured traffic.
DEFAULT_KPH = {
    "motorway": 100, "motorway_link": 50, "trunk": 80, "trunk_link": 40,
    "primary": 50, "primary_link": 35, "secondary": 45, "secondary_link": 30,
    "tertiary": 40, "tertiary_link": 25, "unclassified": 30,
    "residential": 30, "living_street": 10, "service": 15,
}
ACCESS_KEYS = ("motorcar", "motor_vehicle", "vehicle", "access")
ALLOWED_ACCESS = {"yes", "permissive", "designated", "official"}
UNPAVED = {"unpaved", "gravel", "fine_gravel", "dirt", "earth", "ground",
           "sand", "mud", "grass", "compacted", "woodchips"}
TURN_TYPES = {"no_left_turn", "no_right_turn", "no_straight_on", "no_u_turn",
              "only_left_turn", "only_right_turn", "only_straight_on",
              "only_u_turn", "no_entry", "no_exit"}


@dataclass(frozen=True, slots=True)
class Profile:
    objective: str = "time"  # seconds or meters; never mix their units
    avoid_tolls: bool = False
    avoid_motorways: bool = False
    avoid_unpaved: bool = False
    closed_ways: frozenset[int] = frozenset()
    turn_delay_s: float = 0.0  # charged on a change of OSM way, not every shape node
    signal_delay_s: float = 0.0
    slowdown_by_way: dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.objective not in {"time", "distance"}:
            raise ValueError("objective must be 'time' or 'distance'")
        if any(not isfinite(v) or v < 0
               for v in (self.turn_delay_s, self.signal_delay_s)):
            raise ValueError("delays must be finite and nonnegative")
        if any(not isfinite(v) or v < 1 for v in self.slowdown_by_way.values()):
            raise ValueError("slowdown factors must be finite and at least 1")


@dataclass(frozen=True, slots=True)
class Edge:
    source: int
    target: int
    way: int
    name: str
    meters: float
    seconds: float  # includes static slowdown and arrival signal delay


@dataclass(slots=True)
class RoadGraph:
    osm_ids: list[int]
    coordinates: list[tuple[float, float]]  # latitude, longitude
    outgoing: list[list[int]]  # edge IDs; no path copies in the frontier
    incoming: list[list[int]]
    edges: list[Edge]
    costs: list[float]
    max_speed_mps: float
    profile: Profile
    # A trie per turn restriction: children, terminal flags, mandatory/prohibitory.
    restrictions: list[tuple[list[dict[int, int]], list[bool], bool]]
    starts: dict[int, list[tuple[int, int]]]
    notes: dict[str, int]


def great_circle_m(a, b):
    """Spherical straight-line distance, also used to measure road segments."""
    lat1, lon1 = map(radians, a)
    lat2, lon2 = map(radians, b)
    value = sin((lat2 - lat1) / 2) ** 2
    value += cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * asin(sqrt(min(1.0, max(0.0, value))))


def tags(element):
    return {tag.get("k"): tag.get("v") for tag in element.findall("tag")}


def car_access(road_tags, direction=None):
    # Specific vehicle tags override general access tags, including direction.
    for key in ACCESS_KEYS:
        value = road_tags.get(f"{key}:{direction}") if direction else None
        if value is None:
            value = road_tags.get(key)
        if value is not None:
            return value in ALLOWED_ACCESS
    return True


def speed_kph(value, fallback):
    """Accept numeric km/h, mph, knots, and semicolon lists; else use a default."""
    if not value:
        return fallback
    speeds = []
    for part in value.split(";"):
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(km/h|kph|mph|knots)?\s*", part)
        if not match:
            return fallback
        speed = float(match[1]) * {"mph": 1.609344, "knots": 1.852}.get(match[2], 1)
        if speed <= 0 or not isfinite(speed):
            return fallback
        speeds.append(speed)
    return min(speeds)


def download_osm(destination, bbox=DEFAULT_BBOX,
                 endpoint="https://overpass-api.de/api/interpreter"):
    """Download once; subsequent routes read the local XML/XML.gz file offline."""
    south, west, north, east = bbox
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        raise ValueError("bbox must be south west north east in valid degree ranges")
    bounds = ",".join(map(str, bbox))
    query = (f'[out:xml][timeout:60];way["highway"]({bounds})->.roads;'
             '(.roads;rel(bw.roads)["type"="restriction"];);(._;>;);out body;')
    request = Request(endpoint, urlencode({"data": query}).encode(),
                      headers={"User-Agent": "AI-course-road-routing/1.0"})
    with urlopen(request, timeout=90) as response:
        data = response.read()
    root = ET.fromstring(data)
    if root.tag != "osm" or root.find("remark") is not None:
        raise ValueError("Overpass returned an error or incomplete result; file not saved")
    destination = Path(destination)
    payload = gzip.compress(data, mtime=0) if destination.suffix == ".gz" else data
    temporary = destination.with_name(destination.name + ".part")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def load_osm(path, profile=None):
    """Build a directed car graph; keep every road shape point for exact geometry."""
    profile = profile or Profile()
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    nodes, ways, relations = {}, {}, []
    with opener(path, "rb") as stream:
        # Clearing completed top-level elements avoids retaining a second XML tree.
        parser = ET.iterparse(stream, events=("start", "end"))
        _, root = next(parser)
        if root.tag != "osm":
            raise ValueError("expected an OpenStreetMap XML file")
        for event, element in parser:
            if event != "end":
                continue
            if element.tag == "remark":
                raise ValueError("OSM extract contains an Overpass error remark")
            if element.tag == "node":
                nodes[int(element.get("id"))] = (
                    (float(element.get("lat")), float(element.get("lon"))), tags(element))
            elif element.tag == "way":
                ways[int(element.get("id"))] = (
                    [int(nd.get("ref")) for nd in element.findall("nd")], tags(element))
            elif element.tag == "relation":
                relations.append((int(element.get("id")), tags(element),
                                  [dict(m.attrib) for m in element.findall("member")]))
            else:
                continue
            root.clear()

    graph = RoadGraph([], [], [], [], [], [], 0.0, profile, [], {},
                      {"conditional_ways_excluded": 0, "fallback_speed_directions": 0})
    index = {}

    def node_index(osm_id):
        if osm_id not in index:
            index[osm_id] = len(graph.osm_ids)
            graph.osm_ids.append(osm_id)
            graph.coordinates.append(nodes[osm_id][0])
            graph.outgoing.append([])
            graph.incoming.append([])
        return index[osm_id]

    def usable_node(osm_id):
        nt = nodes[osm_id][1]
        # Unknown physical barriers are excluded unless explicitly passable by car.
        if nt.get("barrier") not in {None, "no", "entrance", "cattle_grid", "toll_booth"}:
            if not any(k in nt for k in ACCESS_KEYS):
                return False
        return car_access(nt) and not any(":conditional" in k for k in nt)

    for way_id, (refs, wt) in ways.items():
        highway = wt.get("highway")
        if (highway not in DEFAULT_KPH or wt.get("area") == "yes"
                or way_id in profile.closed_ways
                or wt.get("smoothness") == "impassable"):
            continue
        if profile.avoid_tolls and (wt.get("toll") == "yes" or wt.get("toll:motorcar") == "yes"):
            continue
        if profile.avoid_motorways and highway in {"motorway", "motorway_link"}:
            continue
        if profile.avoid_unpaved and wt.get("surface") in UNPAVED:
            continue
        relevant = (*ACCESS_KEYS, "oneway", "maxspeed")
        if any(":conditional" in k and k.split(":")[0] in relevant for k in wt):
            graph.notes["conditional_ways_excluded"] += 1
            continue
        if any(ref not in nodes for ref in refs):
            raise ValueError(f"way {way_id} is missing nodes; obtain a complete extract")
        default_oneway = "yes" if highway == "motorway" or wt.get("junction") in {
            "roundabout", "circular"} else "no"
        oneway = next((wt[k] for k in ("oneway:motorcar", "oneway:motor_vehicle",
                                      "oneway:vehicle", "oneway") if k in wt), default_oneway)
        if oneway not in {"yes", "1", "true", "-1", "no", "0", "false"}:
            continue  # reversible/alternating roads need a time-dependent model
        for forward, direction in ((True, "forward"), (False, "backward")):
            if forward and oneway == "-1":
                continue
            if not forward and oneway in {"yes", "1", "true"}:
                continue
            if not car_access(wt, direction):
                continue
            limit = wt.get(f"maxspeed:{direction}", wt.get("maxspeed"))
            parsed_speed = speed_kph(limit, None)
            if parsed_speed is None:
                graph.notes["fallback_speed_directions"] += 1
                parsed_speed = DEFAULT_KPH[highway]
            speed = parsed_speed / 3.6
            speed /= profile.slowdown_by_way.get(way_id, 1.0)
            if speed <= 0:
                raise ValueError(f"modeled speed for way {way_id} is too small")
            graph.max_speed_mps = max(graph.max_speed_mps, speed)
            ordered = refs if forward else refs[::-1]
            for start, end in zip(ordered, ordered[1:]):
                if start == end or not usable_node(start) or not usable_node(end):
                    continue
                u, v = node_index(start), node_index(end)
                meters = great_circle_m(graph.coordinates[u], graph.coordinates[v])
                signal = profile.signal_delay_s if nodes[end][1].get("highway") == "traffic_signals" else 0
                seconds = meters / speed + signal
                if not isfinite(seconds):
                    raise ValueError(f"modeled travel time for way {way_id} is not finite")
                edge = Edge(u, v, way_id, wt.get("name", wt.get("ref", highway)),
                            meters, seconds)
                eid = len(graph.edges)
                graph.edges.append(edge)
                graph.costs.append(edge.seconds if profile.objective == "time" else meters)
                graph.outgoing[u].append(eid)
                graph.incoming[v].append(eid)
    if not graph.edges:
        raise ValueError("no usable car roads remain in this extract/profile")
    compile_restrictions(graph, ways, relations)
    return graph


def compile_restrictions(graph, ways, relations):
    """Compile via-node AND via-way restrictions into finite edge-sequence tries.

    A mandatory restriction keeps its alternative legal sequences in ONE trie.
    Its progress belongs to the search state: arriving at a node is insufficient.
    """
    by_way = {}
    for eid, edge in enumerate(graph.edges):
        by_way.setdefault(edge.way, []).append(eid)

    for rid, rt, members in relations:
        if rt.get("type") != "restriction":
            continue
        exceptions = set(rt.get("except", "").replace(" ", "").split(";"))
        if exceptions.intersection(ACCESS_KEYS[:-1]):
            continue
        restriction_keys = ("restriction:motorcar", "restriction:motor_vehicle",
                            "restriction:vehicle", "restriction")
        kind = next((rt[k] for k in restriction_keys if k in rt), None)
        conditional = any(f"{k}:conditional" in rt for k in restriction_keys)
        if kind is None and not conditional:
            continue  # restriction for a different vehicle type
        from_ways = [int(m["ref"]) for m in members if m["role"] == "from" and m["type"] == "way"]
        to_ways = {int(m["ref"]) for m in members if m["role"] == "to" and m["type"] == "way"}
        via = [m for m in members if m["role"] == "via"]
        if not any(w in by_way for w in from_ways):
            continue
        if conditional or kind not in TURN_TYPES or not from_ways or not to_ways or not via:
            raise ValueError(f"unsupported or conditional turn restriction {rid}; cannot route exactly")
        mandatory = kind.startswith("only_")
        children, terminal = [{}], [False]

        def insert(sequence, complete=True):
            position = 0
            for eid in sequence:
                if eid not in children[position]:
                    children[position][eid] = len(children)
                    children.append({})
                    terminal.append(False)
                position = children[position][eid]
            terminal[position] |= complete

        incoming = [eid for w in from_ways for eid in by_way.get(w, [])]
        if len(via) == 1 and via[0]["type"] == "node":
            via_id = int(via[0]["ref"])
            for first in incoming:
                before = graph.edges[first]
                if graph.osm_ids[before.target] != via_id:
                    continue
                if mandatory:
                    insert([first], complete=False)  # even if the required exit is closed
                for last in graph.outgoing[before.target]:
                    after = graph.edges[last]
                    if after.way not in to_ways:
                        continue
                    if before.way == after.way:
                        reverses = before.source == after.target
                        if kind in {"no_u_turn", "only_u_turn"} and not reverses:
                            continue
                        if kind in {"no_straight_on", "only_straight_on"} and reverses:
                            continue
                    insert([first, last])
        elif all(m["type"] == "way" for m in via):
            via_ways = [int(m["ref"]) for m in via]
            if any(w not in ways for w in via_ways):
                raise ValueError(f"restriction {rid} is missing a via way")
            entry_nodes = set(ways[via_ways[0]][0])
            work = 0
            for first in incoming:
                before = graph.edges[first]
                if graph.osm_ids[before.target] not in entry_nodes:
                    continue
                if mandatory:
                    insert([first], complete=False)
                # Iterative enumeration: via ways can be long, so avoid recursion.
                stack = [(before.target, 0, (first,))]
                while stack:
                    node, stage, sequence = stack.pop()
                    work += 1
                    if work > 100_000:
                        raise ValueError(f"restriction {rid} is too complex to compile")
                    previous = graph.edges[sequence[-1]]
                    for eid in graph.outgoing[node]:
                        edge = graph.edges[eid]
                        if eid in sequence:
                            continue
                        if stage == len(via_ways):
                            if edge.way in to_ways:
                                insert((*sequence, eid))
                        else:
                            # Continue on the current via way without doubling back.
                            if stage > 0 and edge.way == via_ways[stage - 1] and edge.target != previous.source:
                                stack.append((edge.target, stage, (*sequence, eid)))
                            if edge.way == via_ways[stage]:
                                stack.append((edge.target, stage + 1, (*sequence, eid)))
                    if stage == len(via_ways):
                        # Still on the final via way; its exit may be several nodes away.
                        for eid in graph.outgoing[node]:
                            edge = graph.edges[eid]
                            if (edge.way == via_ways[-1] and edge.target != previous.source
                                    and eid not in sequence):
                                stack.append((edge.target, stage, (*sequence, eid)))
        else:
            raise ValueError(f"unsupported via members in restriction {rid}")
        if children[0]:
            rule = len(graph.restrictions)
            graph.restrictions.append((children, terminal, mandatory))
            for first, position in children[0].items():
                graph.starts.setdefault(first, []).append((rule, position))


def advance_restrictions(graph, active, next_edge):
    """Return the new automaton state, or None for an illegal maneuver."""
    starting = graph.starts.get(next_edge, ())
    if not active:
        return tuple(starting)
    updated = []
    for rule, position in active:
        children, terminal, mandatory = graph.restrictions[rule]
        child = children[position].get(next_edge)
        if child is None:
            if mandatory:
                return None
        elif terminal[child]:
            if not mandatory:
                return None
        else:
            updated.append((rule, child))
    updated.extend(starting)
    return tuple(sorted(set(updated)))


def nearest_node(graph, coordinate, max_distance_m=500):
    """Snap to a road node, reporting distance; no hidden straight-line connector."""
    if (len(coordinate) != 2 or not all(isfinite(x) for x in coordinate)
            or not -90 <= coordinate[0] <= 90 or not -180 <= coordinate[1] <= 180):
        raise ValueError("coordinates must be finite latitude longitude in degree ranges")
    node = min(range(len(graph.coordinates)),
               key=lambda n: great_circle_m(coordinate, graph.coordinates[n]))
    distance = great_circle_m(coordinate, graph.coordinates[node])
    if distance > max_distance_m:
        raise ValueError(f"nearest road node is {distance:.0f} m away; enlarge the extract or snap limit")
    return node, distance


def dijkstra_distances(graph, start, reverse=False):
    """Relaxed node graph: ignore turn restrictions/delays to get lower bounds."""
    distances = array("d", [inf]) * len(graph.coordinates)
    distances[start] = 0.0
    frontier = [(0.0, start)]
    adjacency = graph.incoming if reverse else graph.outgoing
    while frontier:
        cost, node = heappop(frontier)
        if cost != distances[node]:
            continue
        for eid in adjacency[node]:
            edge = graph.edges[eid]
            neighbor = edge.source if reverse else edge.target
            candidate = cost + graph.costs[eid]
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heappush(frontier, (candidate, neighbor))
    return distances


def prepare_landmarks(graph, number=4):
    """Deterministic geographic farthest-point sampling; reuse tables across queries."""
    if number < 0:
        raise ValueError("landmark count must be nonnegative")
    nearest = [inf] * len(graph.coordinates)
    selected, tables = set(), []
    landmark = min(range(len(nearest)), key=graph.coordinates.__getitem__)
    for _ in range(min(number, len(nearest))):
        selected.add(landmark)
        tables.append((dijkstra_distances(graph, landmark),
                       dijkstra_distances(graph, landmark, reverse=True)))
        for node, coordinate in enumerate(graph.coordinates):
            nearest[node] = min(nearest[node], great_circle_m(coordinate, graph.coordinates[landmark]))
        landmark = max((n for n in range(len(nearest)) if n not in selected),
                       key=nearest.__getitem__, default=landmark)
    return tables


def make_heuristic(graph, goal, landmarks=()):
    """Maximum of the geometric bound and directed ALT triangle inequalities."""
    target = graph.coordinates[goal]
    divisor = graph.max_speed_mps if graph.profile.objective == "time" else 1.0
    cache = {goal: 0.0}

    def heuristic(node):
        if node not in cache:
            bound = great_circle_m(graph.coordinates[node], target) / divisor
            for from_landmark, to_landmark in landmarks:
                if isfinite(from_landmark[goal]) and isfinite(from_landmark[node]):
                    bound = max(bound, from_landmark[goal] - from_landmark[node])
                if isfinite(to_landmark[node]) and isfinite(to_landmark[goal]):
                    bound = max(bound, to_landmark[node] - to_landmark[goal])
            cache[node] = bound
        return cache[node]

    return heuristic


def find_route(graph, start, goal, landmarks=(), use_heuristic=True, on_step=None):
    """Expand the state with minimum g+h; return an optimal route or None.

    State = (incoming edge ID, restriction progress), with (-1, ()) at the start.
    Reopen cheaper states, skip stale heap records, and stop on goal removal.
    An optional on_step callback records actual decisions for the HTML visualizer.
    """
    if not 0 <= start < len(graph.coordinates) or not 0 <= goal < len(graph.coordinates):
        raise ValueError("start and goal must be graph node indices")
    begun = perf_counter()
    heuristic = make_heuristic(graph, goal, landmarks) if use_heuristic else lambda _: 0.0
    initial = (-1, ())
    best, parents = {initial: 0.0}, {}
    order = count()
    frontier = [(heuristic(start), 0.0, next(order), initial)]
    expanded = 0
    peak_frontier = 1
    turn_cost = graph.profile.turn_delay_s if graph.profile.objective == "time" else 0.0
    waiting = {initial: (heuristic(start), 0.0, 0, start, -1)} if on_step else None
    expanded_before = set() if on_step else None

    def emit(kind, node, cost, considered):
        if on_step:
            estimate = heuristic(node)
            on_step({"kind": kind, "node": node, "g": cost, "h": estimate,
                     "f": cost + estimate, "expanded": expanded,
                     "frontier_count": len(waiting), "considered": considered,
                     "frontier": [{"node": n, "edge": e, "g": g, "h": f - g, "f": f}
                                  for f, g, _, n, e in nsmallest(8, waiting.values())]})

    emit("start", start, 0.0, [])
    while frontier:
        _, cost, _, state = heappop(frontier)
        if cost != best.get(state):
            continue
        incoming, active = state
        node = start if incoming == -1 else graph.edges[incoming].target
        if on_step:
            waiting.pop(state, None)
        if node == goal:
            emit("goal", node, cost, [])
            path = []
            while state != initial:
                path.append(state[0])
                state = parents[state]
            path.reverse()
            seconds = sum(graph.edges[e].seconds for e in path)
            seconds += graph.profile.turn_delay_s * sum(
                graph.edges[a].way != graph.edges[b].way for a, b in zip(path, path[1:]))
            return {
                "edge_ids": path,
                "node_ids": [graph.osm_ids[start]] + [graph.osm_ids[graph.edges[e].target] for e in path],
                "distance_m": sum(graph.edges[e].meters for e in path),
                "estimated_seconds": seconds,
                "objective": graph.profile.objective,
                "objective_cost": cost,
                "expanded_states": expanded,
                "discovered_states": len(best),
                "peak_frontier_records": peak_frontier,
                "search_ms": (perf_counter() - begun) * 1000,
            }
        expanded += 1
        if on_step:
            expanded_before.add(state)
        considered = [] if on_step else None
        previous_way = graph.edges[incoming].way if incoming != -1 else None
        for eid in graph.outgoing[node]:
            progress = advance_restrictions(graph, active, eid)
            if progress is None:
                if on_step:
                    considered.append({"edge": eid, "status": "restricted", "g": None})
                continue
            next_state = (eid, progress)
            delay = turn_cost if previous_way is not None and previous_way != graph.edges[eid].way else 0
            candidate = cost + graph.costs[eid] + delay
            previous_cost = best.get(next_state, inf)
            if candidate < previous_cost:
                best[next_state] = candidate
                parents[next_state] = state
                priority = candidate + heuristic(graph.edges[eid].target)
                entry_order = next(order)
                heappush(frontier, (priority,
                                    candidate, entry_order, next_state))
                if on_step:
                    waiting[next_state] = (priority, candidate, entry_order, graph.edges[eid].target, eid)
                    status = "reopened" if next_state in expanded_before else "improved" if isfinite(previous_cost) else "queued"
                    considered.append({"edge": eid, "status": status, "g": candidate})
            elif on_step:
                considered.append({"edge": eid, "status": "unchanged", "g": candidate})
        peak_frontier = max(peak_frontier, len(frontier))
        emit("expand", node, cost, considered)
    emit("no_route", start, 0.0, [])
    return None


def write_geojson(path, graph, route, start):
    nodes = [start] + [graph.edges[e].target for e in route["edge_ids"]]
    points = [[graph.coordinates[n][1], graph.coordinates[n][0]] for n in nodes]
    geometry = ({"type": "LineString", "coordinates": points} if len(points) > 1
                else {"type": "Point", "coordinates": points[0]})
    properties = {k: v for k, v in route.items() if k not in {"edge_ids", "node_ids"}}
    properties["attribution"] = "© OpenStreetMap contributors, ODbL 1.0"
    Path(path).write_text(json.dumps({"type": "Feature", "properties": properties,
                                     "geometry": geometry}, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--osm", type=Path, default=DEFAULT_OSM)
    parser.add_argument("--download", action="store_true", help="refresh --osm with real OSM roads")
    parser.add_argument("--bbox", nargs=4, type=float, default=DEFAULT_BBOX,
                        metavar=("SOUTH", "WEST", "NORTH", "EAST"))
    parser.add_argument("--start", nargs=2, type=float, default=(42.6977, 23.3219), metavar=("LAT", "LON"))
    parser.add_argument("--goal", nargs=2, type=float, default=(42.6936, 23.3349), metavar=("LAT", "LON"))
    parser.add_argument("--objective", choices=("time", "distance"), default="time")
    parser.add_argument("--avoid-tolls", action="store_true")
    parser.add_argument("--avoid-motorways", action="store_true")
    parser.add_argument("--avoid-unpaved", action="store_true")
    parser.add_argument("--closed-way", type=int, action="append", default=[], help="repeat for multiple OSM way IDs")
    parser.add_argument("--turn-delay", type=float, default=0, metavar="SECONDS")
    parser.add_argument("--signal-delay", type=float, default=0, metavar="SECONDS")
    parser.add_argument("--landmarks", type=int, default=0, help="ALT preprocessing count; most useful for repeated queries")
    parser.add_argument("--max-snap", type=float, default=500, metavar="METERS")
    parser.add_argument("--compare", action="store_true", help="verify optimal cost against Dijkstra on the same turn-aware graph")
    parser.add_argument("--geojson", type=Path, help="optional route geometry output")
    args = parser.parse_args()
    try:
        if not isfinite(args.max_snap) or args.max_snap < 0:
            raise ValueError("max-snap must be finite and nonnegative")
        if args.landmarks < 0:
            raise ValueError("landmark count must be nonnegative")
        if args.download:
            download_osm(args.osm, args.bbox)
        if not args.osm.exists():
            raise ValueError(f"missing {args.osm}; use --download or provide --osm FILE")
        profile = Profile(args.objective, args.avoid_tolls, args.avoid_motorways,
                          args.avoid_unpaved, frozenset(args.closed_way), args.turn_delay,
                          args.signal_delay)
        begun = perf_counter()
        graph = load_osm(args.osm, profile)
        load_ms = (perf_counter() - begun) * 1000
        start, start_offset = nearest_node(graph, args.start, args.max_snap)
        goal, goal_offset = nearest_node(graph, args.goal, args.max_snap)
        begun = perf_counter()
        landmarks = prepare_landmarks(graph, args.landmarks)
        preprocessing_ms = (perf_counter() - begun) * 1000
        route = find_route(graph, start, goal, landmarks)
        if route is None:
            print("No legal route between the snapped nodes. Try a larger extract or different constraints.", file=sys.stderr)
            return 2
        output = {"data": "© OpenStreetMap contributors, ODbL 1.0", "osm_file": str(args.osm),
                  "graph_nodes": len(graph.osm_ids), "directed_edges": len(graph.edges),
                  "turn_restrictions": len(graph.restrictions), "model_notes": graph.notes,
                  "load_ms": load_ms, "landmark_preprocessing_ms": preprocessing_ms,
                  "landmarks": len(landmarks),
                  "snap_distance_m": {"start": start_offset, "goal": goal_offset}, **route}
        if args.compare:
            baseline = find_route(graph, start, goal, use_heuristic=False)
            if baseline is None or abs(baseline["objective_cost"] - route["objective_cost"]) > 1e-7 * max(1, route["objective_cost"]):
                raise RuntimeError("A* and Dijkstra disagree")
            output["dijkstra"] = {k: baseline[k] for k in ("objective_cost", "expanded_states", "search_ms")}
        if args.geojson:
            write_geojson(args.geojson, graph, route, start)
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, OSError, ET.ParseError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
