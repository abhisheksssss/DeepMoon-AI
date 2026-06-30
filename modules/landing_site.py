"""
landing_site.py
---------------
Landing site identification and scoring for lunar south polar missions.

CORE STRATEGY
--------------
Landing zones are scored on ILLUMINATED TERRAIN adjacent to or outside
the doubly-shadowed crater — NOT on the crater floor or deep PSR zones.

Rationale:
  - A lander on unilluminated terrain cannot generate solar power for
    survival or science operations.
  - The crater interior (doubly-shadowed zone / PSR) is the SCIENCE TARGET,
    accessed via a planned rover traverse from the safe landing site.
  - Earth visibility is critical for direct communication and must be
    considered in landing zone selection.

Scoring uses a multi-criteria weighted matrix including:
  - Terrain safety (slope, roughness)
  - Solar illumination (power generation — primary survival criterion)
  - Earth visibility (communication line-of-sight)
  - Proximity to science target (manageable rover traverse distance)
  - Surface properties from OHRC where imaging quality permits

References
----------
- NASA PGDA Lunar Polar Illumination products
- LPI South Pole Atlas
- ISRO Chandrayaan-2 mission planning documentation
"""

import numpy as np
from scipy.ndimage import (label, distance_transform_edt, binary_dilation)


# ---------------------------------------------------------------------------
# Scoring Weights
# ---------------------------------------------------------------------------

WEIGHTS = {
    "slope_safety"    : 0.25,   # Slope ≤ 15° for touchdown
    "illumination"    : 0.25,   # Solar power — MANDATORY for survival
    "earth_visibility": 0.20,   # Direct communication LoS
    "ice_proximity"   : 0.20,   # Science: manageable traverse to ice target
    "roughness"       : 0.10,   # Surface smoothness for landing
}

# Mission parameters
LANDER_ELLIPSE_M   = (150, 100)   # Semi-axes of landing uncertainty ellipse (m)
MAX_ROVER_RANGE_KM = 3.0          # Rover operational range
SAFE_SLOPE_DEG     = 15.0
MIN_ILLUMINATION   = 0.25         # Hard floor: < 25% illumination = no landing


# ---------------------------------------------------------------------------
# Core Scoring
# ---------------------------------------------------------------------------

def score_landing_sites(slope: np.ndarray,
                         roughness: np.ndarray,
                         illumination: np.ndarray,
                         ice_mask: np.ndarray,
                         pixel_size_m: float,
                         crater_center_px: tuple = None,
                         earth_visibility: np.ndarray | None = None,
                         psr_mask: np.ndarray | None = None,
                         weights: dict = None) -> np.ndarray:
    """
    Compute per-pixel landing suitability score in [0, 1].

    Landing zones must be on adequately illuminated terrain OUTSIDE the PSR.
    The illumination criterion is weighted heavily because solar power is
    a survival prerequisite for any lunar south polar lander.

    Parameters
    ----------
    slope            : slope map (degrees)
    roughness        : TRI roughness map (metres)
    illumination     : illumination fraction [0, 1] per orbit
    ice_mask         : boolean ice candidate detection map
    pixel_size_m     : pixel spacing (metres)
    crater_center_px : (col, row) pixel coordinates of target crater centre
    earth_visibility : boolean array, True = has LoS to Earth (optional)
    psr_mask         : boolean PSR mask; PSR pixels penalised for landing
    weights          : override default WEIGHTS dict

    Returns
    -------
    score : float32 array in [0, 1]
    """
    if weights is None:
        weights = WEIGHTS

    rows, cols = slope.shape

    # --- Slope Safety Score ---
    slope_score = np.clip(1.0 - slope / SAFE_SLOPE_DEG, 0, 1) ** 2

    # --- Roughness Score ---
    max_tri = np.percentile(roughness, 95)
    rough_score = np.clip(1.0 - roughness / (max_tri + 1e-6), 0, 1)

    # --- Illumination Score (heavily weighted — survival requirement) ---
    illum_score = np.clip(illumination, 0, 1)

    # --- Earth Visibility Score ---
    if earth_visibility is not None:
        earth_score = earth_visibility.astype(float)
    else:
        # If not provided, use illumination as a proxy (correlated at south pole)
        earth_score = illum_score * 0.8

    # --- Ice Proximity Score ---
    # Science target is the crater ice zone; score by proximity for rover access.
    if ice_mask.any():
        ice_dist = distance_transform_edt(~ice_mask)
        max_dist_px = (MAX_ROVER_RANGE_KM * 1000) / pixel_size_m
        ice_prox = np.clip(1.0 - ice_dist / max_dist_px, 0, 1)
    else:
        ice_prox = np.zeros_like(slope)

    # --- Composite Score ---
    # Combine using provided weights; default weights sum to 1.0.
    # Note: ice_proximity proxies for 'crater_access' in the rover traverse sense.
    score = (weights.get("slope_safety",    0.25) * slope_score  +
             weights.get("illumination",    0.25) * illum_score   +
             weights.get("earth_visibility", 0.20) * earth_score  +
             weights.get("ice_proximity",   0.20) * ice_prox      +
             weights.get("roughness",       0.10) * rough_score)

    # --- Hard masks ---
    # Pixels with slope > safe limit: no landing
    score = np.where(slope > SAFE_SLOPE_DEG, 0.0, score)

    # Pixels inside deep PSR: no landing (no solar power for lander survival).
    # The lander must be placed on illuminated terrain outside the crater.
    if psr_mask is not None:
        score = np.where(psr_mask, 0.0, score)

    # Illumination hard floor: any site with < MIN_ILLUMINATION is infeasible
    score = np.where(illumination < MIN_ILLUMINATION, 0.0, score)

    return score.astype(np.float32)


# ---------------------------------------------------------------------------
# Candidate Extraction
# ---------------------------------------------------------------------------

def extract_landing_candidates(score: np.ndarray,
                                pixel_size_m: float,
                                min_score: float = 0.45,
                                min_diameter_m: float = 300.0,
                                top_n: int = 5) -> list[dict]:
    """
    Extract candidate landing zones as clusters of high-scoring pixels.

    Each candidate must be large enough to accommodate the landing ellipse
    and separated from others to represent distinct options.

    Parameters
    ----------
    score         : landing suitability score array
    pixel_size_m  : pixel spacing
    min_score     : minimum score threshold for candidate inclusion
    min_diameter_m: minimum diameter of viable landing zone (metres)
    top_n         : number of top candidates to return

    Returns
    -------
    List of candidate dicts sorted by mean score descending.
    """
    min_px   = int((min_diameter_m / pixel_size_m) ** 2 * np.pi / 4)
    good     = score > min_score

    labeled, n = label(good)
    candidates = []

    for i in range(1, n + 1):
        region = labeled == i
        count  = region.sum()
        if count < max(min_px, 5):
            continue

        ys, xs   = np.where(region)
        cx, cy   = float(xs.mean()), float(ys.mean())
        area_km2 = count * (pixel_size_m / 1000)**2
        mean_sc  = float(score[region].mean())
        max_sc   = float(score[region].max())

        candidates.append({
            "id"         : i,
            "centroid_px": (cx, cy),
            "area_km2"   : area_km2,
            "mean_score" : mean_sc,
            "max_score"  : max_sc,
            "pixel_count": count,
        })

    candidates.sort(key=lambda c: c["mean_score"], reverse=True)
    return candidates[:top_n]


# ---------------------------------------------------------------------------
# Landing Ellipse Feasibility Check
# ---------------------------------------------------------------------------

def check_ellipse_feasibility(score: np.ndarray,
                               slope: np.ndarray,
                               center_px: tuple,
                               pixel_size_m: float,
                               ellipse_axes_m: tuple = LANDER_ELLIPSE_M) -> dict:
    """
    Check if a landing ellipse centred at center_px is feasible.

    Returns dict with:
      feasible, slope_stats, score_stats, hazard_fraction
    """
    cx, cy = center_px
    a_px   = ellipse_axes_m[0] / pixel_size_m
    b_px   = ellipse_axes_m[1] / pixel_size_m

    rows, cols = score.shape
    Y, X = np.ogrid[:rows, :cols]
    ellipse = ((X - cx) / a_px) ** 2 + ((Y - cy) / b_px) ** 2 <= 1.0

    if not ellipse.any():
        return {"feasible": False, "reason": "Ellipse outside image"}

    sl   = slope[ellipse]
    sc   = score[ellipse]
    haz  = (sl > SAFE_SLOPE_DEG).sum() / ellipse.sum()

    return {
        "feasible"        : bool(haz < 0.05),      # < 5% hazardous pixels
        "hazard_fraction" : float(haz),
        "mean_slope_deg"  : float(sl.mean()),
        "max_slope_deg"   : float(sl.max()),
        "mean_score"      : float(sc.mean()),
        "min_score"       : float(sc.min()),
        "ellipse_px_count": int(ellipse.sum()),
    }


# ---------------------------------------------------------------------------
# Recommended Landing Site Report
# ---------------------------------------------------------------------------

def recommend_landing_site(candidates: list[dict],
                            slope: np.ndarray,
                            score: np.ndarray,
                            pixel_size_m: float) -> dict:
    """
    From a list of candidates, return the best feasible site.
    """
    for cand in candidates:
        check = check_ellipse_feasibility(
            score, slope, cand["centroid_px"], pixel_size_m)
        if check["feasible"]:
            return {**cand, "feasibility": check, "recommended": True}

    # If none strictly feasible, return best by score
    if candidates:
        best = candidates[0]
        check = check_ellipse_feasibility(
            score, slope, best["centroid_px"], pixel_size_m)
        return {**best, "feasibility": check, "recommended": False,
                "note": "No fully feasible site; best available returned"}
    return {}
