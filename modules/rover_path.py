"""
rover_path.py
-------------
Terrain-aware A* rover path planning for lunar south polar exploration.

Plans an optimal path from a landing site to the doubly-shadowed crater
rim / ice-bearing region, respecting:
  - Maximum slope constraints (20°)
  - Solar power corridor preference
  - Hazard avoidance (steep scarps, boulders)
  - Science stop identification along path

Algorithm: Weighted A* on a grid graph with terrain cost function.
"""

import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Cost Function Parameters
# ---------------------------------------------------------------------------
MAX_ROVER_SLOPE      = 20.0    # degrees – absolute limit
OPTIMAL_ROVER_SLOPE  = 10.0    # degrees – beyond this, cost increases
SHADOW_PENALTY       = 5.0     # extra cost per step in shadow (power drain)
ROUGH_PENALTY_SCALE  = 3.0     # TRI roughness penalty multiplier
SCIENCE_BONUS        = -2.0    # negative cost (reward) for ice pixels along path


# ---------------------------------------------------------------------------
# Graph Node
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _Node:
    f: float
    g: float = field(compare=False)
    pos: tuple = field(compare=False)
    parent: Optional["_Node"] = field(default=None, compare=False, repr=False)


# ---------------------------------------------------------------------------
# Terrain Cost Function
# ---------------------------------------------------------------------------

def pixel_cost(slope: np.ndarray,
               roughness: np.ndarray,
               illumination: np.ndarray,
               ice_mask: np.ndarray,
               row: int, col: int,
               is_diagonal: bool = False) -> float:
    """
    Cost of traversing into pixel (row, col).

    Returns np.inf if terrain is impassable.
    """
    s = float(slope[row, col])
    r = float(roughness[row, col])
    illum = float(illumination[row, col])
    ice   = bool(ice_mask[row, col])

    # Hard constraint
    if s > MAX_ROVER_SLOPE:
        return np.inf

    # Base movement cost (diagonal = √2 × orthogonal)
    move_cost = 1.414 if is_diagonal else 1.0

    # Slope penalty: quadratic beyond optimal slope
    if s <= OPTIMAL_ROVER_SLOPE:
        slope_pen = 0.0
    else:
        slope_pen = ((s - OPTIMAL_ROVER_SLOPE) /
                     (MAX_ROVER_SLOPE - OPTIMAL_ROVER_SLOPE)) ** 2 * 5.0

    # Roughness penalty
    rough_pen = min(r * ROUGH_PENALTY_SCALE, 8.0)

    # Shadow penalty (power drain)
    shadow_pen = SHADOW_PENALTY * (1.0 - illum)

    # Science reward (ice pixels are scientifically valuable waypoints)
    science = SCIENCE_BONUS if ice else 0.0

    return move_cost + slope_pen + rough_pen + shadow_pen + science


# ---------------------------------------------------------------------------
# A* Path Planner
# ---------------------------------------------------------------------------

def plan_path(slope: np.ndarray,
              roughness: np.ndarray,
              illumination: np.ndarray,
              ice_mask: np.ndarray,
              start_px: tuple,
              goal_px: tuple,
              pixel_size_m: float = 20.0,
              allow_diagonal: bool = True) -> dict:
    """
    A* path from start_px to goal_px over terrain grid.

    Parameters
    ----------
    slope, roughness, illumination, ice_mask : 2-D terrain arrays
    start_px   : (col, row) start pixel
    goal_px    : (col, row) goal pixel (crater rim / ice target)
    pixel_size_m : pixel spacing
    allow_diagonal : allow 8-connected neighbours (default True)

    Returns
    -------
    dict with:
      path_px     : list of (col, row) pixel coordinates
      path_m      : list of (x, y) in metres from start
      cost        : total path cost
      length_km   : path length in km
      waypoints   : science stop pixel coordinates
      stats       : slope/roughness statistics along path
      success     : bool
    """
    rows, cols = slope.shape
    sc, sr = int(start_px[0]), int(start_px[1])
    gc, gr = int(goal_px[0]), int(goal_px[1])

    # Heuristic: Euclidean distance (admissible – movement cost ≥ 1.0)
    def heuristic(r, c):
        return np.sqrt((r - gr)**2 + (c - gc)**2)

    # Neighbour offsets
    if allow_diagonal:
        neighbours = [(-1,-1,True),(-1,0,False),(-1,1,True),
                      ( 0,-1,False),            ( 0,1,False),
                      ( 1,-1,True),( 1,0,False),( 1,1,True)]
    else:
        neighbours = [(-1,0,False),(0,-1,False),(0,1,False),(1,0,False)]

    # Priority queue: (f, g, (row, col))
    open_set   = []
    g_score    = np.full((rows, cols), np.inf)
    came_from  = {}

    g_score[sr, sc] = 0.0
    h0 = heuristic(sr, sc)
    heapq.heappush(open_set, _Node(f=h0, g=0.0, pos=(sr, sc)))

    while open_set:
        node = heapq.heappop(open_set)
        r, c = node.pos

        if (r, c) == (gr, gc):
            # Reconstruct path
            path = []
            cur  = (r, c)
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append((sr, sc))
            path.reverse()

            path_px = [(p[1], p[0]) for p in path]  # (col, row)
            path_m  = [(p[1] * pixel_size_m, p[0] * pixel_size_m) for p in path]

            # Length
            length_px = 0.0
            for i in range(len(path) - 1):
                dr = path[i+1][0] - path[i][0]
                dc = path[i+1][1] - path[i][1]
                length_px += np.sqrt(dr**2 + dc**2)
            length_km = length_px * pixel_size_m / 1000.0

            # Science waypoints (ice pixels along path)
            waypoints = [(p[1], p[0]) for p in path
                         if ice_mask[p[0], p[1]]]

            # Stats
            path_rows = [p[0] for p in path]
            path_cols = [p[1] for p in path]
            slope_vals = slope[path_rows, path_cols]
            rough_vals = roughness[path_rows, path_cols]
            illum_vals = illumination[path_rows, path_cols]

            return {
                "success"     : True,
                "path_px"     : path_px,
                "path_m"      : path_m,
                "cost"        : float(g_score[gr, gc]),
                "length_km"   : float(length_km),
                "waypoints"   : waypoints,
                "stats"       : {
                    "mean_slope_deg"  : float(slope_vals.mean()),
                    "max_slope_deg"   : float(slope_vals.max()),
                    "mean_roughness"  : float(rough_vals.mean()),
                    "mean_illumination": float(illum_vals.mean()),
                    "n_ice_waypoints" : len(waypoints),
                },
            }

        if g_score[r, c] < node.g:
            continue   # Stale entry

        for dr, dc, diag in neighbours:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            step = pixel_cost(slope, roughness, illumination, ice_mask,
                              nr, nc, is_diagonal=diag)
            if np.isinf(step):
                continue

            tentative_g = g_score[r, c] + step
            if tentative_g < g_score[nr, nc]:
                g_score[nr, nc] = tentative_g
                came_from[(nr, nc)] = (r, c)
                h = heuristic(nr, nc)
                heapq.heappush(open_set,
                               _Node(f=tentative_g + h, g=tentative_g,
                                     pos=(nr, nc)))

    # No path found
    return {"success": False, "path_px": [], "path_m": [],
            "cost": np.inf, "length_km": 0.0, "waypoints": [], "stats": {}}


# ---------------------------------------------------------------------------
# Multi-Stop Path (via waypoints)
# ---------------------------------------------------------------------------

def plan_multi_stop(slope: np.ndarray,
                    roughness: np.ndarray,
                    illumination: np.ndarray,
                    ice_mask: np.ndarray,
                    stops: list[tuple],
                    pixel_size_m: float = 20.0) -> dict:
    """
    Plan a path through multiple stops in sequence.

    Parameters
    ----------
    stops : list of (col, row) pixel coordinates [start, wp1, wp2, ..., goal]

    Returns
    -------
    Merged path dict combining all segments.
    """
    if len(stops) < 2:
        raise ValueError("Need at least start and goal")

    full_path = []
    total_cost = 0.0
    total_km   = 0.0
    all_wp     = []
    all_stats  = {"mean_slope_deg": [], "max_slope_deg": [],
                  "mean_roughness": [], "mean_illumination": []}

    for i in range(len(stops) - 1):
        seg = plan_path(slope, roughness, illumination, ice_mask,
                        stops[i], stops[i+1], pixel_size_m)
        if not seg["success"]:
            return {"success": False,
                    "message": f"No path from stop {i} to stop {i+1}"}

        # Avoid duplicating junction points
        if full_path:
            full_path.extend(seg["path_px"][1:])
        else:
            full_path.extend(seg["path_px"])

        total_cost += seg["cost"]
        total_km   += seg["length_km"]
        all_wp.extend(seg["waypoints"])

        for k in all_stats:
            if k in seg["stats"]:
                all_stats[k].append(seg["stats"][k])

    summary_stats = {k: float(np.mean(v)) if v else 0.0
                     for k, v in all_stats.items()}
    summary_stats["n_ice_waypoints"] = len(all_wp)

    return {
        "success"   : True,
        "path_px"   : full_path,
        "cost"      : total_cost,
        "length_km" : total_km,
        "waypoints" : all_wp,
        "stats"     : summary_stats,
    }


# ---------------------------------------------------------------------------
# Path Utilities
# ---------------------------------------------------------------------------

def path_to_geojson(path_px: list[tuple],
                     pixel_size_m: float,
                     origin_lon: float = -84.0,
                     origin_lat: float = -85.0) -> dict:
    """
    Convert pixel path to approximate GeoJSON LineString.

    Uses simple linear projection from a reference origin.
    For accurate geographic coordinates, apply proper SPICE/LROC projection.
    """
    # Approximate metres per degree on the Moon
    m_per_deg_lat = 30387.0   # 1° lat ≈ 30.4 km on Moon
    m_per_deg_lon = lambda lat: m_per_deg_lat * np.cos(np.radians(lat))

    coords = []
    for (c, r) in path_px:
        x_m = c * pixel_size_m
        y_m = r * pixel_size_m
        lon = origin_lon + x_m / m_per_deg_lon(origin_lat)
        lat = origin_lat + y_m / m_per_deg_lat
        coords.append([lon, lat])

    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {"name": "Rover Traverse Path",
                           "length_km": len(coords) * pixel_size_m / 1000}
        }]
    }
