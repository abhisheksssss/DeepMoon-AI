"""
radar_processing.py
-------------------
Chandrayaan-2 DFSAR polarimetric radar processing module.

DATA MODEL REQUIREMENT
----------------------
This pipeline is designed for **full-polarimetric DFSAR products**:
  HH, HV, VH, VV  (calibrated complex scattering matrix elements)

Dual-pol (HH/HV only) data can be processed for CPR but NOT for the
full Stokes-vector-based DOP used in the subsurface ice study.

Computes:
  - Circular Polarization Ratio (CPR)  — from Stokes parameters
  - Degree of Polarization (DOP)       — from full-pol Stokes formulation
  - Backscatter coefficient (sigma-naught, σ°)
  - Speckle-filtered SAR imagery (Lee filter)

NOTE ON ICE THRESHOLDS
----------------------
The CPR > 1 and DOP < 0.13 criterion is a study-specific candidate rule
derived from the Chandrayaan-2 full-polarimetric DFSAR result for
doubly-shadowed south polar craters. It is NOT a universal lunar truth.
High CPR alone can indicate rough/blocky terrain; low DOP must accompany
it to discriminate volume scattering (ice) from surface roughness.
See: Chauhan et al. (2022), npj Microgravity / Space Exploration.

References
----------
- Chauhan et al. (2022) – Chandrayaan-2 DFSAR full-pol ice signatures,
  npj Space Exploration
- Kumar et al. (2023) – Polarimetric indicators of lunar PSR ice
- Nozette et al. (2001) – Clementine bistatic CPR
- Ulaby, Moore & Fung (1986) – Microwave Remote Sensing, Vol. III
"""

import numpy as np
from scipy.ndimage import uniform_filter
import warnings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
S_BAND_FREQ_HZ = 2.5e9          # DFSAR S-band centre frequency
L_BAND_FREQ_HZ = 0.43e9         # DFSAR L-band centre frequency
SPEED_OF_LIGHT = 3e8             # m/s
S_BAND_LAMBDA  = SPEED_OF_LIGHT / S_BAND_FREQ_HZ
L_BAND_LAMBDA  = SPEED_OF_LIGHT / L_BAND_FREQ_HZ


# ---------------------------------------------------------------------------
# Speckle Filtering
# ---------------------------------------------------------------------------

def lee_filter(band: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Lee speckle filter for SAR intensity images.

    Parameters
    ----------
    band        : 2-D array of SAR intensity (linear scale, not dB)
    window_size : sliding window size (odd integer)

    Returns
    -------
    Filtered intensity array (same shape as input)
    """
    band = band.astype(np.float64)
    mean = uniform_filter(band, size=window_size)
    sq_mean = uniform_filter(band ** 2, size=window_size)
    variance = np.maximum(sq_mean - mean ** 2, 0)

    noise_var = np.mean(variance) / (np.mean(mean) ** 2 + 1e-10)
    weight = variance / (variance + noise_var * mean ** 2 + 1e-10)
    filtered = mean + weight * (band - mean)
    return filtered.astype(np.float32)


# ---------------------------------------------------------------------------
# Stokes Parameters  (requires full-pol HH, HV, VH, VV)
# ---------------------------------------------------------------------------

def stokes_from_quad_pol(Shh: np.ndarray, Shv: np.ndarray,
                          Svh: np.ndarray, Svv: np.ndarray,
                          window: int = 7) -> dict:
    """
    Compute Stokes parameters from calibrated full-polarimetric complex
    scattering matrix elements (HH, HV, VH, VV).

    Spatial averaging (window × window) replaces ensemble averaging and
    is required before computing CPR and DOP to suppress speckle noise.

    Formulation (monostatic backscatter convention):
        S1 = <|Shh|²> + <|Svv|²> + 2·<|Shv|²>
        S2 = <|Shh|²> - <|Svv|²>
        S3 = 2·Re(<Shh · Svv*>)
        S4 = 2·Im(<Shh · Svv*>)   [also written S3 in some conventions]

    For the circular-polarization representation (Nozette / Black formalism):
        SC (same-sense circular)    = (S1 - S4) / 2
        OC (opposite-sense circular)= (S1 + S4) / 2
        CPR = SC / OC

    Parameters
    ----------
    Shh, Shv, Svh, Svv : 2-D complex arrays (calibrated amplitudes)
    window              : spatial averaging window size (pixels)

    Returns
    -------
    dict with keys: S1, S2, S3, S4, SC, OC  (all float32)
    """
    def spatial_avg_intensity(arr):
        return uniform_filter(np.abs(arr) ** 2, size=window).astype(np.float64)

    def spatial_avg_cross(a, b):
        prod = a * np.conj(b)
        return (uniform_filter(prod.real, size=window) +
                1j * uniform_filter(prod.imag, size=window))

    # Stokes-like parameters for backscatter (monostatic; Shv ≈ Svh)
    Shv_avg = (Shv + Svh) / 2.0   # Reciprocity assumption for monostatic SAR

    S1 = spatial_avg_intensity(Shh) + spatial_avg_intensity(Svv) + \
         2 * spatial_avg_intensity(Shv_avg)
    S2 = spatial_avg_intensity(Shh) - spatial_avg_intensity(Svv)
    cross_hh_vv = spatial_avg_cross(Shh, Svv)
    S3 = 2.0 * cross_hh_vv.real
    S4 = 2.0 * cross_hh_vv.imag

    # Circular components (right-hand circular transmit convention)
    SC = (S1 - S4) / 2.0   # same-sense (depolarized)
    OC = (S1 + S4) / 2.0   # opposite-sense (co-polarized)

    return {
        "S1": S1.astype(np.float32),
        "S2": S2.astype(np.float32),
        "S3": S3.astype(np.float32),
        "S4": S4.astype(np.float32),
        "SC": SC.astype(np.float32),
        "OC": OC.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# CPR  –  Circular Polarization Ratio  (from Stokes)
# ---------------------------------------------------------------------------

def compute_cpr_from_stokes(stokes: dict) -> np.ndarray:
    """
    Compute CPR from Stokes parameters derived from full-pol DFSAR.

    CPR = SC / OC = (S1 − S4) / (S1 + S4)

    For ice-bearing subsurface: volume scattering enhances the same-sense
    return → CPR > 1.
    For rough/blocky rocky terrain: surface scattering dominates → CPR ≈ 0.5–0.9.
    CPR > 1 alone is NOT sufficient to confirm ice; low DOP must also hold.

    Returns
    -------
    CPR array (float32), NaN where OC ≈ 0
    """
    SC = stokes["SC"].astype(np.float64)
    OC = stokes["OC"].astype(np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cpr = np.where(OC > 1e-12, SC / OC, np.nan)

    return cpr.astype(np.float32)


def compute_cpr_intensity(cross_pol: np.ndarray,
                           co_pol: np.ndarray,
                           apply_filter: bool = True,
                           window: int = 7) -> np.ndarray:
    """
    Approximate CPR from intensity images when only HH and HV are available.

    CPR_approx ≈ σ°_HV / σ°_HH

    WARNING: This is an approximation. The full-pol Stokes-based CPR is
    preferred for the subsurface ice discrimination study. Use this only
    as a fallback for dual-pol data.

    Parameters
    ----------
    cross_pol : HV (or VH) intensity array  [same-sense proxy]
    co_pol    : HH (or VV) intensity array  [opposite-sense proxy]
    """
    if apply_filter:
        cross_pol = lee_filter(cross_pol, window)
        co_pol    = lee_filter(co_pol, window)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cpr = np.where(co_pol > 1e-12,
                       cross_pol.astype(np.float64) / co_pol.astype(np.float64),
                       np.nan)

    return cpr.astype(np.float32)


# ---------------------------------------------------------------------------
# DOP  –  Degree of Polarization  (Stokes-based, full-pol only)
# ---------------------------------------------------------------------------

def compute_dop_from_stokes(stokes: dict) -> np.ndarray:
    """
    Compute Degree of Polarization (DOP) from full-pol Stokes parameters.

    DOP = √(S2² + S3² + S4²) / S1

    This is the standard Stokes-based DOP for a partially polarized
    backscattered wave, aligned with the formulation used in the
    Chandrayaan-2 full-pol DFSAR subsurface ice study.

    Physical interpretation:
      DOP → 1 : fully polarized (specular/surface scattering dominant)
      DOP → 0 : fully depolarized (volume/multiple scattering dominant)

    Ice-bearing subsurface causes strong volume scattering → low DOP.
    Rough rocky terrain can produce moderate DOP (0.3–0.7).
    The literature-backed discriminator for possible ice is DOP < 0.13,
    combined with CPR > 1, in doubly-shadowed south polar craters.

    Parameters
    ----------
    stokes : dict from stokes_from_quad_pol()

    Returns
    -------
    DOP array (float32) in [0, 1]
    """
    S1 = stokes["S1"].astype(np.float64)
    S2 = stokes["S2"].astype(np.float64)
    S3 = stokes["S3"].astype(np.float64)
    S4 = stokes["S4"].astype(np.float64)

    numerator = np.sqrt(S2**2 + S3**2 + S4**2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dop = np.where(S1 > 1e-12, numerator / S1, 0.0)

    dop = np.clip(dop, 0.0, 1.0)
    return dop.astype(np.float32)


# ---------------------------------------------------------------------------
# Backscatter Coefficient (σ°)
# ---------------------------------------------------------------------------

def compute_sigma_naught(amplitude: np.ndarray,
                          incidence_deg: float,
                          cal_constant: float = 1.0) -> np.ndarray:
    """
    Convert calibrated SAR amplitude to backscatter coefficient σ° (dB).

    σ° (linear) = (DN² × cal_constant) / sin(θ)
    σ° (dB)     = 10 × log10(σ° linear)

    Parameters
    ----------
    amplitude    : raw DN amplitude (linear, not dB)
    incidence_deg: radar incidence angle (degrees)
    cal_constant : instrument absolute calibration factor

    Returns
    -------
    σ° in dB (float32 array)
    """
    theta = np.radians(incidence_deg)
    sigma_lin = (amplitude.astype(np.float64) ** 2 * cal_constant) / \
                (np.sin(theta) + 1e-10)
    sigma_lin = np.maximum(sigma_lin, 1e-10)
    sigma_db  = 10.0 * np.log10(sigma_lin)
    return sigma_db.astype(np.float32)


# ---------------------------------------------------------------------------
# Rough-Terrain False-Positive Flagging
# ---------------------------------------------------------------------------

def flag_rough_terrain(cpr: np.ndarray,
                        sigma_db: np.ndarray,
                        cpr_thresh: float = 1.0,
                        sigma_rocky_db: float = -8.0) -> np.ndarray:
    """
    Flag pixels where high CPR is LIKELY from rough/blocky rocky terrain
    rather than subsurface ice.

    Heuristic: rocky surfaces have high backscatter AND high CPR.
    Ice-bearing subsurface typically shows moderate σ° but elevated CPR.

    Parameters
    ----------
    cpr            : CPR array
    sigma_db       : σ° array (dB)
    cpr_thresh     : CPR anomaly threshold
    sigma_rocky_db : σ° level above which high CPR is attributed to rocks

    Returns
    -------
    rocky_flag : boolean array, True = probable rough terrain false positive
    """
    high_cpr    = cpr > cpr_thresh
    high_sigma  = sigma_db > sigma_rocky_db
    rocky_flag  = high_cpr & high_sigma
    return rocky_flag


# ---------------------------------------------------------------------------
# Synthetic Full-Pol Data Generator (demo / testing)
# ---------------------------------------------------------------------------

def generate_synthetic_dfsar(shape: tuple = (512, 512),
                               seed: int = 42) -> dict:
    """
    Generate synthetic **full-polarimetric** SAR data mimicking a
    doubly-shadowed crater with subsurface ice in the inner floor.

    Bands returned: HH, HV, VH, VV (Rayleigh-distributed amplitudes
    with physically motivated inter-channel relationships).

    NOTE: This synthetic dataset is intended ONLY for pipeline testing
    and dashboard demonstration. Real Chandrayaan-2 DFSAR products
    require proper Level-1 calibration and geocoding before use.

    Returns
    -------
    dict with keys: HH, HV, VH, VV, incidence_angle, metadata
    """
    rng = np.random.default_rng(seed)
    rows, cols = shape

    def rayleigh(scale, sz):
        return rng.rayleigh(scale=scale, size=sz).astype(np.float32)

    cx, cy = cols // 2, rows // 2
    Y, X = np.ogrid[:rows, :cols]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

    rim_mask   = (dist > 120) & (dist < 145)
    floor_mask = dist <= 120
    ice_mask   = dist <= 55        # inner doubly-shadowed zone (candidate ice)
    rocky_mask = rim_mask          # rim = rough/blocky terrain (CPR false-positive zone)

    # HH co-pol
    HH = rayleigh(0.15, shape)
    HH[floor_mask] = rayleigh(0.08, np.sum(floor_mask))
    HH[rim_mask]   = rayleigh(0.32, np.sum(rim_mask))   # high backscatter rim
    HH[ice_mask]   = rayleigh(0.18, np.sum(ice_mask))

    # HV cross-pol
    HV = rayleigh(0.04, shape)
    HV[floor_mask] = rayleigh(0.02, np.sum(floor_mask))
    # Rim gets elevated HV too (rough terrain), so CPR rim ≈ 0.7 (not ice)
    HV[rim_mask]   = rayleigh(0.09, np.sum(rim_mask))
    # Ice zone: strong volume scattering → high HV → CPR > 1
    HV[ice_mask]   = rayleigh(0.22, np.sum(ice_mask))

    # VH ≈ HV (reciprocity assumption for monostatic SAR)
    VH = HV + rng.normal(0, 0.005, shape).astype(np.float32)
    VH = np.abs(VH)

    # VV co-pol
    VV = rayleigh(0.13, shape)
    VV[floor_mask] = rayleigh(0.07, np.sum(floor_mask))
    VV[rim_mask]   = rayleigh(0.28, np.sum(rim_mask))
    VV[ice_mask]   = rayleigh(0.16, np.sum(ice_mask))

    terrain_mask = np.zeros(shape, dtype=np.uint8)
    terrain_mask[floor_mask] = 2   # crater floor (shadow)
    terrain_mask[rim_mask]   = 1   # rim (rocky — CPR FP zone)
    terrain_mask[ice_mask]   = 3   # inner floor (candidate ice)

    metadata = {
        "shape"           : shape,
        "center_px"       : (cx, cy),
        "crater_radius_px": 145,
        "floor_radius_px" : 120,
        "ice_radius_px"   : 55,
        "pixel_scale_m"   : 10.0,
        "incidence_angle" : 35.0,
        "band"            : "S",
        "frequency_ghz"   : 2.5,
        "pol_mode"        : "full-quad-pol (HH, HV, VH, VV)",
        "terrain_mask"    : terrain_mask,
        "ice_ground_truth": ice_mask,
        "rocky_rim_mask"  : rocky_mask,
        "data_note"       : (
            "SYNTHETIC demo data only. Real DFSAR requires L1 calibration."
        ),
    }

    return {
        "HH": HH, "HV": HV, "VH": VH, "VV": VV,
        "incidence_angle": 35.0,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# GeoTIFF I/O helpers (requires rasterio)
# ---------------------------------------------------------------------------

def load_geotiff_band(filepath: str, band_idx: int = 1) -> dict:
    """
    Load a single band from a GeoTIFF file.

    Expected input: calibrated DFSAR L1 product, single band.
    For full-pol processing, load all four bands (HH, HV, VH, VV) separately.

    Returns dict with keys: data, transform, crs, nodata
    """
    try:
        import rasterio
        with rasterio.open(filepath) as src:
            data      = src.read(band_idx).astype(np.float32)
            transform = src.transform
            crs       = src.crs
            nodata    = src.nodata
        return {"data": data, "transform": transform,
                "crs": crs, "nodata": nodata}
    except ImportError:
        raise ImportError(
            "rasterio is required to load GeoTIFF files. "
            "Install with: pip install rasterio"
        )


def save_geotiff(data: np.ndarray, filepath: str,
                 transform=None, crs=None, nodata: float = -9999.0):
    """Save a 2-D array as a single-band GeoTIFF."""
    try:
        import rasterio
        from rasterio.transform import from_bounds
        rows, cols = data.shape
        if transform is None:
            transform = from_bounds(0, 0, cols, rows, cols, rows)
        with rasterio.open(
            filepath, "w", driver="GTiff", height=rows, width=cols,
            count=1, dtype=str(data.dtype),
            crs=crs, transform=transform, nodata=nodata
        ) as dst:
            dst.write(data, 1)
    except ImportError:
        raise ImportError(
            "rasterio is required. Install with: pip install rasterio"
        )
