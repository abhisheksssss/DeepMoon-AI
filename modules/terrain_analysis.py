"""
terrain_analysis.py
-------------------
Terrain analysis module for lunar south polar DEM data.

Provides:
  - Slope and aspect maps
  - Surface roughness indices (TRI, BPI)
  - Illumination / shadow modelling (vectorised ray-cast)
  - PSR / doubly-shadowed zone mapping
  - Earth-visibility proxy map (LoS to Earth)
  - OHRC-based surface hazard cues (with imaging quality confidence)
  - Safety score per pixel for landing and traversal

Data sources assumed: LOLA DEM (20 m/px or finer) or Chandrayaan-2 TMC2 DEM.

NOTE ON OHRC HAZARD MAPPING
-----------------------------
OHRC provides ~0.25 m GSD imagery, excellent for surface morphology.
However, in permanently shadowed or deeply shadowed regions:
  - Image quality and contrast may be severely limited.
  - Boulder density and small-crater mapping are only reliable where the
    scene is adequately illuminated (Earth-shine, orbital geometry).
  - All OHRC-derived hazard products should carry an illumination-quality
    confidence flag.  Deep-PSR analyses should be flagged as 'limited'.

NOTE ON LANDING STRATEGY
--------------------------
Landing zones are scored on illuminated terrain OUTSIDE or adjacent to
the PSR/doubly-shadowed crater — NOT on the crater floor itself.
The crater interior is the science TARGET, accessed by rover traverse
from a safe, illuminated, Earth-visible landing zone nearby.
"""

import numpy as np
from scipy.ndimage import (uniform_filter, generic_filter,
                            sobel, gaussian_filter, label)
import warnings


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LUNAR_GRAVITY        = 1.62      # m/s²
MAX_SAFE_SLOPE_DEG   = 15.0      # Rover & landing safety limit
MAX_TRAVERSE_SLOPE   = 20.0      # Absolute max for rover traverse
SAFE_ROUGHNESS_TRI   = 0.5       # TRI threshold for safe terrain (metres)


# ---------------------------------------------------------------------------
# Synthetic DEM Generator
# ---------------------------------------------------------------------------

def generate_synthetic_dem(shape: tuple = (512, 512),
                             crater_depth_m: float = 800.0,
                             pixel_scale_m: float = 20.0,
                             seed: int = 42) -> dict:
    """
    Generate a realistic synthetic DEM of a doubly-shadowed crater
    at the lunar south pole.

    Features:
      - Crater bowl with central mound
      - Rim elevated above surroundings
      - Interior depression (doubly-shadowed floor)
      - Procedural noise for realistic roughness
      - Inner sub-crater (doubly-shadowed zone)

    Returns
    -------
    dict with keys: elevation (2-D array, metres), pixel_scale_m, metadata
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape
    Y, X = np.ogrid[:rows, :cols]
    cx, cy = cols // 2, rows // 2

    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

    # Base polar terrain (gently sloping away from pole)
    elevation = 1000.0 - 0.5 * dist * pixel_scale_m / 1000.0

    # Main crater bowl (Gaussian depression)
    crater_r = 145   # pixels
    crater   = -crater_depth_m * np.exp(-(dist**2) / (2 * (crater_r * 0.6)**2))
    elevation += crater

    # Rim (annular ridge)
    rim_r   = crater_r
    rim_w   = 15
    rim_h   = crater_depth_m * 0.3
    rim     = rim_h * np.exp(-((dist - rim_r)**2) / (2 * rim_w**2))
    elevation += rim

    # Inner doubly-shadowed sub-depression
    inner_r = 55
    inner   = -400.0 * np.exp(-(dist**2) / (2 * (inner_r * 0.5)**2))
    elevation += inner

    # Fractal roughness (sum of octaves)
    roughness = np.zeros(shape)
    for octave in range(1, 6):
        freq  = octave * 0.05
        amp   = 20.0 / octave
        noise = rng.standard_normal(shape)
        from scipy.ndimage import gaussian_filter
        roughness += amp * gaussian_filter(noise, sigma=max(1, 10 // octave))
    elevation += roughness

    # Boulders – random point elevations on rim
    n_boulders = 80
    for _ in range(n_boulders):
        bx = int(rng.uniform(cx - rim_r - 20, cx + rim_r + 20))
        by = int(rng.uniform(cy - rim_r - 20, cy + rim_r + 20))
        bx = np.clip(bx, 5, cols - 5)
        by = np.clip(by, 5, rows - 5)
        br = int(rng.uniform(1, 4))
        h  = rng.uniform(1.0, 5.0)
        elevation[max(0, by-br):by+br, max(0, bx-br):bx+br] += h

    metadata = {
        "shape"           : shape,
        "pixel_scale_m"   : pixel_scale_m,
        "center_px"       : (cx, cy),
        "crater_radius_px": crater_r,
        "inner_radius_px" : inner_r,
        "crater_depth_m"  : crater_depth_m,
        "elevation_range" : (float(elevation.min()), float(elevation.max())),
    }

    return {"elevation": elevation.astype(np.float32),
            "pixel_scale_m": pixel_scale_m, "metadata": metadata}


# ---------------------------------------------------------------------------
# Slope & Aspect
# ---------------------------------------------------------------------------

def compute_slope(dem: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """
    Compute slope in degrees from DEM using central differences.

    Parameters
    ----------
    dem          : 2-D elevation array (metres)
    pixel_size_m : pixel spacing (metres)

    Returns
    -------
    slope : 2-D array in degrees [0, 90]
    """
    # Sobel-based gradient (8-connected neighbourhood)
    dz_dx = sobel(dem, axis=1) / (8.0 * pixel_size_m)   # dZ/dX
    dz_dy = sobel(dem, axis=0) / (8.0 * pixel_size_m)   # dZ/dY

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    return np.degrees(slope_rad).astype(np.float32)


def compute_aspect(dem: np.ndarray) -> np.ndarray:
    """Compute aspect (azimuth of steepest descent) in degrees [0, 360)."""
    dz_dx = sobel(dem, axis=1)
    dz_dy = sobel(dem, axis=0)
    aspect = np.degrees(np.arctan2(-dz_dy, dz_dx)) % 360
    return aspect.astype(np.float32)


# ---------------------------------------------------------------------------
# Roughness Indices
# ---------------------------------------------------------------------------

def compute_tri(dem: np.ndarray) -> np.ndarray:
    """
    Terrain Ruggedness Index (TRI) – mean absolute difference
    between a pixel and its 8 neighbours.

    TRI = sqrt( Σ (z_i - z_centre)² ) for i in 8-neighbours
    """
    def tri_kernel(patch):
        centre = patch[4]   # centre of 3x3 flattened
        return np.sqrt(np.mean((patch - centre) ** 2))

    tri = generic_filter(dem.astype(np.float64), tri_kernel,
                         size=3, mode="nearest")
    return tri.astype(np.float32)


def compute_roughness_stddev(dem: np.ndarray, window: int = 5) -> np.ndarray:
    """Local standard deviation of elevation (proxy for surface roughness)."""
    mean = uniform_filter(dem.astype(np.float64), window)
    sq_mean = uniform_filter(dem.astype(np.float64)**2, window)
    var = np.maximum(sq_mean - mean**2, 0)
    return np.sqrt(var).astype(np.float32)


# ---------------------------------------------------------------------------
# Illumination / Shadow Modelling
# ---------------------------------------------------------------------------

def compute_shadow_mask(dem: np.ndarray,
                         pixel_size_m: float,
                         solar_azimuth_deg: float = 0.0,
                         solar_elevation_deg: float = 1.5) -> np.ndarray:
    """
    Ray-casting shadow model for a low-elevation Sun (south polar geometry).

    Simplified ray-march along the solar azimuth direction.

    Parameters
    ----------
    dem                : 2-D elevation array
    pixel_size_m       : pixel spacing
    solar_azimuth_deg  : sun azimuth (0 = north, 90 = east)
    solar_elevation_deg: sun elevation angle above horizon

    Returns
    -------
    shadow : boolean array, True = shadowed (PSR candidate)
    """
    rows, cols = dem.shape
    shadow = np.zeros((rows, cols), dtype=bool)
    tan_elev = np.tan(np.radians(solar_elevation_deg))

    az_rad = np.radians(solar_azimuth_deg)
    step_x = np.sin(az_rad)   # East component
    step_y = -np.cos(az_rad)  # North component (row decreases northward)

    for r in range(rows):
        for c in range(cols):
            # March ray from (r, c) in solar direction; check if any terrain blocks
            z0   = dem[r, c]
            dist = 0.0
            blocked = False
            nr, nc = float(r), float(c)

            while True:
                nr   += step_y
                nc   += step_x
                dist += pixel_size_m
                ri, ci = int(round(nr)), int(round(nc))
                if ri < 0 or ri >= rows or ci < 0 or ci >= cols:
                    break
                z_horizon = z0 + dist * tan_elev
                if dem[ri, ci] > z_horizon:
                    blocked = True
                    break

            shadow[r, c] = blocked

    return shadow


def fast_shadow_mask(dem: np.ndarray,
                      pixel_size_m: float,
                      solar_azimuth_deg: float = 0.0,
                      solar_elevation_deg: float = 1.5) -> np.ndarray:
    """
    Vectorised approximate shadow mask using cumulative maximum horizon.

    Much faster than pixel-by-pixel ray cast; accurate for simple terrains.
    """
    rows, cols = dem.shape
    tan_elev = np.tan(np.radians(max(solar_elevation_deg, 0.01)))

    az_rad  = np.radians(solar_azimuth_deg % 360)
    # Primary direction: collapse to row-wise or col-wise march
    use_rows = abs(np.cos(az_rad)) > abs(np.sin(az_rad))

    shadow = np.zeros((rows, cols), dtype=bool)

    if use_rows:
        # Sun shining predominantly N or S → march along columns
        direction = 1 if np.cos(az_rad) > 0 else -1
        col_range = range(cols) if direction > 0 else range(cols - 1, -1, -1)
        horizon = np.full(rows, -np.inf)
        dist    = np.zeros(rows)
        for c in col_range:
            dist += pixel_size_m
            z_sky = dem[:, c] - dist * tan_elev
            shadow[:, c] = horizon > z_sky
            horizon = np.maximum(horizon, dem[:, c])
    else:
        direction = 1 if np.sin(az_rad) > 0 else -1
        row_range = range(rows) if direction > 0 else range(rows - 1, -1, -1)
        horizon = np.full(cols, -np.inf)
        dist    = np.zeros(cols)
        for r in row_range:
            dist += pixel_size_m
            z_sky = dem[r, :] - dist * tan_elev
            shadow[r, :] = horizon > z_sky
            horizon = np.maximum(horizon, dem[r, :])

    return shadow


def compute_psr_mask(dem: np.ndarray, pixel_size_m: float,
                     n_azimuths: int = 12) -> np.ndarray:
    """
    Estimate PSR (Permanently Shadowed Region) mask by checking shadowing
    over multiple solar azimuths (simulating a full orbital period).

    A pixel is PSR if it is shadowed from ALL solar azimuths.

    Parameters
    ----------
    dem          : 2-D elevation array
    pixel_size_m : pixel spacing
    n_azimuths   : number of azimuth directions to sample (≥8 recommended)

    Returns
    -------
    psr : boolean array, True = permanently shadowed
    """
    azimuths = np.linspace(0, 360, n_azimuths, endpoint=False)
    psr = np.ones(dem.shape, dtype=bool)

    for az in azimuths:
        shadow = fast_shadow_mask(dem, pixel_size_m,
                                   solar_azimuth_deg=az,
                                   solar_elevation_deg=1.5)
        psr &= shadow  # Must be shadowed from ALL directions

    return psr


# ---------------------------------------------------------------------------
# Safety Scoring
# ---------------------------------------------------------------------------

def compute_terrain_safety(slope: np.ndarray,
                            tri: np.ndarray,
                            psr_mask: np.ndarray | None = None,
                            max_slope: float = MAX_SAFE_SLOPE_DEG,
                            max_tri: float = SAFE_ROUGHNESS_TRI) -> np.ndarray:
    """
    Compute a terrain safety score in [0, 1]:
      1.0 = perfectly safe  (flat, smooth)
      0.0 = completely unsafe (too steep or rough)

    Parameters
    ----------
    slope    : slope in degrees
    tri      : TRI roughness (metres)
    psr_mask : if provided, penalise PSR pixels (thermal risk for landing)
    max_slope: slope limit in degrees
    max_tri  : TRI limit in metres

    Returns
    -------
    safety : float32 array in [0, 1]
    """
    slope_score = np.clip(1.0 - slope / max_slope, 0, 1)
    rough_score = np.clip(1.0 - tri   / max_tri,   0, 1)

    safety = 0.6 * slope_score + 0.4 * rough_score

    if psr_mask is not None:
        # Slight penalty for deep PSR (thermal challenge for landing power)
        safety = safety * np.where(psr_mask, 0.85, 1.0)

    return safety.astype(np.float32)


# ---------------------------------------------------------------------------
# Illumination Time Fraction
# ---------------------------------------------------------------------------

def illumination_fraction(dem: np.ndarray,
                           pixel_size_m: float,
                           n_steps: int = 48) -> np.ndarray:
    """
    Estimate fraction of time a pixel is illuminated per orbit
    by sampling multiple solar positions.

    Returns
    -------
    frac : float32 array in [0, 1]; 0 = permanent shadow, 1 = always lit
    """
    azimuths  = np.linspace(0, 360, n_steps, endpoint=False)
    lit_count = np.zeros(dem.shape, dtype=np.float32)

    for az in azimuths:
        shadow = fast_shadow_mask(dem, pixel_size_m,
                                   solar_azimuth_deg=az,
                                   solar_elevation_deg=1.5)
        lit_count += (~shadow).astype(np.float32)

    return (lit_count / n_steps).astype(np.float32)


# ---------------------------------------------------------------------------
# Earth Visibility (LoS proxy)
# ---------------------------------------------------------------------------

def compute_earth_visibility(dem: np.ndarray,
                              pixel_size_m: float,
                              earth_elevation_deg: float = 6.5,
                              earth_azimuth_deg: float = 0.0) -> np.ndarray:
    """
    Compute a proxy Earth-visibility mask using the same ray-cast
    as the shadow model but directed toward Earth's position.

    At the lunar south pole, Earth appears near the horizon (~5–8°
    elevation, roughly northward). Pixels in deep craters lose
    Earth visibility, breaking communication and telemetry links.

    In mission planning, this layer should ideally be derived from
    LOLA-based Earth-visibility products (NASA PGDA). This function
    provides a DEM-based approximation for the dashboard.

    Parameters
    ----------
    dem                : 2-D elevation array
    pixel_size_m       : pixel spacing (metres)
    earth_elevation_deg: elevation of Earth above local horizon (degrees)
    earth_azimuth_deg  : azimuth toward Earth (0 = north, 180 = south)

    Returns
    -------
    earth_vis : boolean array, True = has line-of-sight to Earth
    """
    # Re-use the shadow model: a pixel is Earth-visible if it is NOT
    # blocked along the Earth direction.
    shadow_toward_earth = fast_shadow_mask(
        dem, pixel_size_m,
        solar_azimuth_deg=earth_azimuth_deg,
        solar_elevation_deg=earth_elevation_deg,
    )
    return ~shadow_toward_earth   # True = unobstructed LoS to Earth


# ---------------------------------------------------------------------------
# OHRC Imaging Quality Confidence
# ---------------------------------------------------------------------------

def ohrc_quality_flag(illumination: np.ndarray,
                       psr_mask: np.ndarray,
                       illum_min: float = 0.02) -> np.ndarray:
    """
    Estimate per-pixel OHRC imaging quality confidence.

    OHRC provides very high-resolution imagery (~0.25 m GSD), but
    interpretation in deep shadow / PSR zones is severely limited.
    This flag indicates where OHRC-derived hazard products are reliable.

    Returns
    -------
    quality : float32 array in [0, 1]
      1.0 = well-illuminated, OHRC fully reliable
      0.5 = marginal illumination, reduced confidence
      0.0 = deep PSR, OHRC hazard mapping unreliable
    """
    quality = np.where(
        illumination > illum_min,
        np.clip(illumination / illum_min, 0, 1),
        0.0
    ).astype(np.float32)
    # Hard-zero inside PSR
    quality[psr_mask] = np.minimum(quality[psr_mask], 0.1)
    return quality
