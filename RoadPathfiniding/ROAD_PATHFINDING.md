# Road pathfinding

**Central question:** Which legal road route minimizes distance or estimated driving time?

**Contents:** Run the example · Interactive HTML map · Define the objective · A* and its heuristic · Constraints · Performance and limitations.

## Run the real-data example

Requires **Python 3.10+**, with no third-party packages. Run from the repository root:

```bash
python3 road_pathfinding.py --compare --landmarks 4
```

The included `sofia_roads.osm.gz` is a real OpenStreetMap extract of central Sofia. The default coordinates are near Serdika and Sofia University. This example works offline. Output includes the ordered OSM node IDs, route length, estimated time, search statistics, and the distance each endpoint was snapped to a road node.

```bash
# Shortest route, avoiding tolls and unpaved roads; optionally export the geometry.
python3 road_pathfinding.py --objective distance --avoid-tolls --avoid-unpaved --geojson route.geojson

# Fastest modeled route with estimated junction delays.
python3 road_pathfinding.py --turn-delay 4 --signal-delay 15 --compare

# Other endpoints in the extract: latitude, then longitude.
python3 road_pathfinding.py --start 42.6977 23.3219 --goal 42.6936 23.3349

# Download another area once, then use the saved file offline.
# Bounding box order: SOUTH WEST NORTH EAST. Use a generous margin around both endpoints.
python3 road_pathfinding.py --download --osm my_roads.osm.gz --bbox 42.68 23.30 42.72 23.36
python3 road_pathfinding.py --osm my_roads.osm.gz --start 42.6977 23.3219 --goal 42.6936 23.3349

# Block a road by its OSM way ID. Repeat --closed-way to block several roads.
python3 road_pathfinding.py --closed-way 162764117 --compare
```

`--download` refreshes the specified file using the public Overpass API; ordinary routing makes no network requests. Small teaching extracts suit this service. Use a local data pipeline for large or frequent downloads. The importer accepts OSM XML and gzipped XML, not `.pbf` files.

## Interactive HTML map

Start the standard-library Python server from the repository root:

```bash
python3 road_server.py --port 8000
```

Open **http://127.0.0.1:8000**. The page is `road_pathfinding_visualizer.html`; it uses the real Python router through this server, so opening the HTML as a `file://` URL is not sufficient. No packages, external map tiles, or internet connection are needed.

- Choose **Set start** or **Set destination**, then click a road node. Coordinate fields also support keyboard entry.
- Choose **Close road** and click a segment to toggle closure of its entire OSM way. Closed roads remain visible in red. **Inspect road** shows the road name, way ID, base segment cost, and direction.
- Choose the objective, heuristic, avoidance rules, and junction delays, then **Calculate route**. Changing settings clears the old result until recalculation.
- **Play**, **Back**, **Next**, **Reset**, and the timeline replay the search. **Show route** jumps to its outcome. Change playback speed to move through a larger trace quickly.
- Drag to pan; scroll or use the zoom buttons to zoom. With the map focused, arrow keys pan and `+` / `-` zoom. Outside inputs and the map, Space toggles playback and left/right arrows step.
- **Export GeoJSON** downloads the returned route geometry with attribution.

The canvas draws all 5,099 road nodes and 7,132 directed edges in the default fixture. Arrows indicate one-way travel; faint roads belong to the original network but are excluded by the chosen profile. Expanded nodes, the current state, the top eight live waiting candidates, rejected moves, and the returned route have distinct colors. Several search states can share a node, so expansion counts can exceed unique expanded-node counts.

Playback is recorded by an optional callback in `find_route`, rather than a separate JavaScript search. Every frame shows the chosen state's `g`, `h`, and `f`, live frontier counts, the top eight queue entries, and the outcomes of outgoing moves, including turn restrictions and cheaper arrivals. The start and goal frames are not expansions. Up to 12,000 decision frames are retained plus the final outcome; any omitted frames are explicitly disclosed. Search timing comes from a separate run without trace collection; profile/landmark preparation is reported separately.

The server provides `GET /api/network` and `POST /api/route`, binds to loopback by default, and serves only the visualizer and its API. Graphs and landmark tables for up to six profiles are cached. To use another extract, pass `--osm my_roads.osm.gz` and enter endpoints in that region; the supplied page's location title and default coordinates describe Sofia.

## Define the optimal path

> An optimal path has the lowest chosen cost among all routes permitted by the loaded graph and profile.

| Objective | Edge cost | Additional cost |
| :-- | :-- | :-- |
| `time` (default) | Segment length / modeled speed, plus arrival signal delay | Configurable delay when changing OSM ways |
| `distance` | Segment length in meters | No time penalty in the distance objective |

- Numeric `maxspeed`, directional limits, mph, knots, and semicolon-separated numeric limits are understood. Missing or symbolic speeds use the visible `DEFAULT_KPH` assumptions in the script.
- Traffic can be modeled with **static**, per-way slowdown factors. It is not downloaded or measured live.
- Turn and signal delays default to zero. A change of OSM way is only a simple junction-delay proxy; it does not measure turn angle. Signals are charged when arriving at their tagged node, including the final node.
- Both distance and estimated time are reported, regardless of which is minimized. The shortest route need not be the fastest route.

Reuse the loaded graph and landmark tables for multiple queries:

```python
from road_pathfinding import (
    DEFAULT_OSM, Profile, load_osm, prepare_landmarks, nearest_node, find_route,
)

profile = Profile(
    objective="time",
    avoid_tolls=True,
    turn_delay_s=4,
    signal_delay_s=10,
    slowdown_by_way={162764117: 2.0},  # Twice the free-flow travel time on this way.
)
graph = load_osm(DEFAULT_OSM, profile)
landmarks = prepare_landmarks(graph, number=4)
start, start_offset = nearest_node(graph, (42.6977, 23.3219))
goal, goal_offset = nearest_node(graph, (42.6936, 23.3349))
result = find_route(graph, start, goal, landmarks)
print(result["node_ids"] if result else "No legal route")
```

Keep graph data, costs, and landmark tables together. Rebuild them when closures, speeds, the objective, or other profile settings change; do not mutate a loaded profile's slowdown dictionary.

## A* search

> Expand the waiting state with the smallest `f = g + h`, where `g` is cost already paid and `h` is a lower bound on remaining cost.

- The state stores the **incoming directed edge and active turn-restriction progress**. Two arrivals at the same junction may permit different next moves.
- A min-heap orders candidates. Cheaper arrivals replace previous costs; outdated heap records are skipped, and improved states can be reopened.
- Parent links reconstruct the route after the goal is removed from the heap. Full paths are not copied into frontier entries.
- Via-node and ordered via-way restrictions compile into tries of directed edge sequences. Mandatory restrictions keep alternative allowed sequences together. Their progress is retained across intermediate road shape nodes.
- **Completeness:** yes on the finite loaded state graph, with finite nonnegative costs.
- **Optimality:** least modeled cost between the snapped nodes, with the supplied admissible heuristic and reopening. This guarantee concerns the filtered extract, rather than every road in the real world.
- **Data structure:** adjacency lists, a heap, cost/parent dictionaries, and compact landmark distance arrays.

Let `X` be the number of outgoing transitions examined across all expansions, `R` the number of successful relaxations, and `S` the number of discovered search states. For a fixed landmark count and a bounded number of active restriction prefixes, search time is `O(X + R log(R + 1))`; search memory is `O(S + R)`, in addition to the graph and heuristic cache/tables. Larger restriction-progress tuples add hashing, copying, and sorting overhead. Restriction progress makes the search state graph larger than the road-node graph. Inconsistent landmark bounds can cause reopening, so a bound assuming each state is expanded once would be misleading.

## A safe, stronger heuristic

Let `D(u, t)` be the great-circle straight-line distance from road node `u` to target `t`, and `v_max` the largest modeled directed-road speed in this graph.

| Objective | Geometric lower bound |
| :-- | :-- |
| Distance | `D(u, t)` |
| Time | `D(u, t) / v_max` |

Road segments use the same spherical distance model. The triangle inequality ensures that their total length cannot be smaller than the direct distance. No road segment can be traversed faster than `v_max`, and all added delays are nonnegative. These facts make the geometric bound admissible and consistent, with zero at the target. Using an average speed instead of an upper bound could overestimate the remaining time and lose optimality.

With `--landmarks K`, **ALT** adds road-network information. For each landmark `L`, preprocessing computes distances **from** `L` and **to** `L` on the directed node graph. It uses the selected objective and profile, while relaxing turn restrictions and turn delays. If `d(a, b)` denotes these relaxed shortest-path costs:

```text
h(u) = max(geometric_bound,
           d(L, target) - d(L, u),
           d(u, L) - d(target, L),
           0)
```

Take the maximum across landmarks and omit any difference with an infinite operand. Each finite difference is a lower bound by the directed triangle inequality. The relaxed graph can only make routes cheaper, so the bounds also hold for the turn-aware search. On disconnected directed graphs, omitting infinite terms can make the combined heuristic inconsistent; reopening preserves optimality. This implementation follows the landmark idea from [Goldberg and Harrelson's ALT paper](https://www.microsoft.com/en-us/research/publication/computing-the-shortest-path-a-search-meets-graph-theory/).

For `V` road nodes, `E` directed road edges, and `K` landmarks, preprocessing uses `2K` Dijkstra runs: `O(K(V + E log(E + 1)))` time and `O(KV)` table space. Deterministic geographic farthest-point selection costs another `O(KV)`. Each new heuristic evaluation costs `O(K)` and is cached per node per query. Landmark preprocessing is optional because it can cost more than a single short search.

## Constraints and boundaries

| Constraint | Treatment |
| :-- | :-- |
| Road direction | One-way, reverse one-way, car-specific overrides, motorway and roundabout defaults |
| Car access | Specific vehicle tags override general access; directional access is checked separately |
| Private or purpose-only access | Excluded, including `private`, `destination`, `delivery`, and `customers` |
| Turn restrictions | `no_*` and `only_*` types listed in `TURN_TYPES`, via nodes or ordered via ways, and car-relevant exceptions |
| Closures and preferences | Explicit closed way IDs; optional toll, motorway, and tagged unpaved-road exclusions |
| Physical barriers | Unknown barriers are excluded unless access tags explicitly permit cars |
| Conditional access/direction/speed | Relevant ways are excluded conservatively; conditional access nodes are excluded |
| Conditional/unsupported turn restrictions | Import stops with an explicit error when an applicable restriction cannot be modeled |

The model is for ordinary cars with static costs. It does not implement truck height/weight rules, lane guidance, country-specific implicit restrictions, live traffic, weather, or departure-time-dependent routing. Untagged restrictions cannot be recovered from the data. Missing surface tags do not establish that a road is paved, and missing toll tags do not establish that it is free.

Endpoints snap to the nearest retained **road node**, with a default 500 m limit. Their offsets are reported but are not included in the route cost. This can select a different carriageway or a disconnected service road; precise navigation would project onto accessible directed road segments. A small extract can omit useful detours: enlarge the bounding box if no route exists or a route looks unnecessarily long. Selected ways and restriction members can extend beyond the requested box.

The code favors fast, inspectable Python for teaching and regional extracts. It preserves all geometry nodes and compiles restriction paths explicitly. A pathological restriction exceeding the compiler work limit raises an error rather than producing a partial rule. Very large networks would benefit from graph contraction, spatial indexing, more compact storage, and a dedicated routing engine; no city-to-continent performance claim is made here.

## Data source and attribution

`sofia_roads.osm.gz` contains unmodified OpenStreetMap XML, compressed for this repository:

- **Source:** [Overpass API](https://dev.overpass-api.de/overpass-doc/en/full_data/bbox.html), `https://overpass-api.de/api/interpreter`.
- **OSM snapshot timestamp:** `2026-10-12T20:23:39Z`, taken from the response metadata.
- **Requested bounds:** south `42.685`, west `23.310`, north `42.705`, east `23.340`.
- **Downloaded elements:** 15,643 nodes, 6,209 ways, 217 restriction relations, including referenced members.
- **Default car graph:** 5,099 road nodes, 7,132 directed edges, and 200 compiled restrictions; 18 conditionally tagged ways are excluded.
- **License:** © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/). These terms apply to the included road data; preserve attribution when sharing derived route data.

The exact query is:

```text
[out:xml][timeout:25];
way["highway"](42.685,23.31,42.705,23.34)->.roads;
(.roads;rel(bw.roads)["type"="restriction"];);
(._;>;);
out body;
```

Tag interpretation references: [one-way roads](https://wiki.openstreetmap.org/wiki/Key:oneway), [access](https://wiki.openstreetmap.org/wiki/Key:access), [speed limits](https://wiki.openstreetmap.org/wiki/Key:maxspeed), [turn restrictions](https://wiki.openstreetmap.org/wiki/Relation:restriction), and [conditional restrictions](https://wiki.openstreetmap.org/wiki/Conditional_restrictions).

## Validation

```bash
python3 -B -m unittest -v test_road_pathfinding.py
python3 -B -m unittest -v test_road_server.py
```

Tests check different fastest/shortest routes, directions, access, closures, barriers, turn restrictions spanning nodes and ways, speed units, delays, disconnected components, invalid inputs, and random directed graphs against an independently written node-Dijkstra oracle. On the real extract, sampled queries compare geometric A* and ALT A* with turn-aware Dijkstra; returned edge continuity and accumulated costs are checked. The geometric consistency inequality is checked on every directed edge for sampled goals.

Server tests also check that tracing preserves the route, matches the actual priority-queue order, records prohibited turns and exhausted searches, and reports frame and expansion counts separately. HTTP checks cover the page, network, route, malformed JSON, and file allowlist. Headless browser checks exercised playback, stepping, timeline seeking, zoom, objective changes, clicked road closures and reopening, inspection, and GeoJSON downloads; desktop and 390 px mobile screenshots were inspected, with no horizontal overflow or JavaScript runtime exceptions.

For the default real-data query, four-landmark A* found a **1,607.54 m** route with **115.74 s** estimated time. It expanded **719 states**, compared with **2,969** for turn-aware Dijkstra, at the same objective cost. These are search-state counts, not unique road-node counts. Runtime and preprocessing cost are reported separately by the script; a reduction in expansions does not guarantee a reduction in total runtime for every query.
