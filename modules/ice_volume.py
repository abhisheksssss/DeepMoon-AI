"""
ice_volume.py
-------------
Subsurface ice volume estimation from radar backscatter data.

Methodology:
  1. Estimate radar penetration depth (skin depth) at S-band and L-band
  2. Apply dielectric mixing model to estimate volumetric ice concentration
  3. Integrate over ice-bearing area to compute total ice volume (m³)
  4. Propagate uncertainties

References:
  - Ulaby et al. (1986) – Microwave Remote Sensing
  - Polder & van Santen (1946) – Dielectric mixing model
  - Campbell et al. (2006) – Lunar polar radar ice
  - Nozette et al. (2001) – Clementine bistatic results
"""

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Dielectric Constants (at 2.5 GHz, lunar-relevant values)
# ---------------------------------------------------------------------------

EPS_REGOLITH_REAL  = 2.7      # Real part of dry regolith permittivity
EPS_REGOLITH_IMAG  = 0.005    # Imaginary part (loss tangent ≈ 0.002)
EPS_ICE_REAL       = 3.15     # Real part for water ice
EPS_ICE_IMAG       = 0.001    # Ice is very low-loss at cold temps
EPS_VACUUM         = 1.0

TAN_DELTA_REGOLITH = EPS_REGOLITH_IMAG / EPS_REGOLITH_REAL
SPEED_OF_LIGHT     = 3e8      # m/s


# ---------------------------------------------------------------------------
# Radar Penetration Depth (Skin Depth)
# ---------------------------------------------------------------------------

def skin_depth(frequency_hz: float,
               eps_real: float,
               eps_imag: float) -> float:
    """
    Radar skin depth (1/e power penetration depth) in metres.

    δ = λ / (4π · tan(δ_loss))
    where tan(δ_loss) = ε'' / ε'

    For low-loss media: δ ≈ c / (2πf · √ε') · 1/tan(δ)

    Parameters
    ----------
    frequency_hz : radar frequency in Hz
    eps_real     : real part of permittivity
    eps_imag     : imaginary part of permittivity (loss)

    Returns
    -------
    Skin depth in metres
    """
    tan_delta = eps_imag / (eps_real + 1e-10)
    wavelength = SPEED_OF_LIGHT / frequency_hz
    delta = wavelength / (4 * np.pi * np.sqrt(eps_real) * tan_delta + 1e-10)
    return float(delta)


def penetration_depth_ice_regolith(ice_fraction: float,
                                    frequency_hz: float = 2.5e9,
                                    depth_m: float = 5.0) -> float:
    """
    Effective penetration depth of radar into ice-regolith mixture.

    Uses linear mixing of loss tangent (conservative approximation).

    Parameters
    ----------
    ice_fraction : volumetric ice fraction [0, 1]
    frequency_hz : radar frequency
    depth_m      : maximum integration depth (problem constraint = 5 m)

    Returns
    -------
    Effective penetration depth in metres (capped at depth_m)
    """
    eps_r_mix = (1 - ice_fraction) * EPS_REGOLITH_REAL + ice_fraction * EPS_ICE_REAL
    eps_i_mix = (1 - ice_fraction) * EPS_REGOLITH_IMAG + ice_fraction * EPS_ICE_IMAG

    depth = skin_depth(frequency_hz, eps_r_mix, eps_i_mix)
    return min(depth, depth_m)


# ---------------------------------------------------------------------------
# Dielectric Mixing Model → Ice Fraction
# ---------------------------------------------------------------------------

def polder_van_santen_ice_fraction(eps_measured: float,
                                    eps_host: float = EPS_REGOLITH_REAL,
                                    eps_inclusion: float = EPS_ICE_REAL) -> float:
    """
    Invert Polder-van Santen mixing formula to estimate volumetric ice fraction.

    The mixing formula for spherical inclusions (Bruggeman):
      ε_mix ≈ ε_host + 3·f·ε_host·(ε_inc - ε_host)/(ε_inc + 2·ε_host)

    Solving for f (ice fraction):
      f = (ε_mix - ε_host) · (ε_inc + 2·ε_host) /
          (3·ε_host·(ε_inc - ε_host))

    Parameters
    ----------
    eps_measured  : measured effective permittivity from CPR/backscatter inversion
    eps_host      : host medium permittivity (dry regolith)
    eps_inclusion : inclusion permittivity (ice)

    Returns
    -------
    Ice fraction [0, 1] (clipped)
    """
    numerator   = (eps_measured - eps_host) * (eps_inclusion + 2 * eps_host)
    denominator = 3 * eps_host * (eps_inclusion - eps_host)
    if abs(denominator) < 1e-10:
        return 0.0
    f = numerator / denominator
    return float(np.clip(f, 0.0, 1.0))


def cpr_to_eps(cpr: float,
               cpr_min: float = 0.5,
               cpr_max: float = 2.5,
               eps_min: float = EPS_REGOLITH_REAL,
               eps_max: float = 3.5) -> float:
    """
    Empirical mapping: CPR value → effective permittivity.

    Linear interpolation between dry regolith and ice-saturated limits.
    (Simplified; more rigorous inversion requires full scattering model.)
    """
    t = np.clip((cpr - cpr_min) / (cpr_max - cpr_min + 1e-10), 0, 1)
    return eps_min + t * (eps_max - eps_min)


def estimate_ice_fraction_from_cpr(cpr_map: np.ndarray) -> np.ndarray:
    """
    Per-pixel ice volumetric fraction from CPR map.

    Pipeline: CPR → ε_eff → ice fraction (Polder-van Santen)

    Returns
    -------
    ice_fraction : float32 array in [0, 1]
    """
    eps_map = cpr_to_eps(cpr_map)

    vfunc = np.vectorize(lambda e: polder_van_santen_ice_fraction(e))
    return vfunc(eps_map).astype(np.float32)


# ---------------------------------------------------------------------------
# Volume Estimation
# ---------------------------------------------------------------------------

def estimate_ice_volume(ice_mask: np.ndarray,
                         ice_fraction_map: np.ndarray,
                         cpr_map: np.ndarray,
                         pixel_size_m: float = 10.0,
                         max_depth_m: float = 5.0,
                         frequency_hz: float = 2.5e9) -> dict:
    """
    Estimate total subsurface ice volume within top `max_depth_m` of regolith.

    V_ice = Σ_pixels [ A_pixel × depth_pixel × concentration_pixel ]

    Parameters
    ----------
    ice_mask         : boolean ice detection mask
    ice_fraction_map : volumetric ice fraction per pixel
    cpr_map          : CPR array (for depth computation)
    pixel_size_m     : pixel footprint size
    max_depth_m      : integration depth limit (5 m per problem statement)
    frequency_hz     : radar frequency

    Returns
    -------
    dict with volume estimates and uncertainty bounds
    """
    pixel_area_m2 = pixel_size_m ** 2

    # Per-pixel effective penetration depth (based on ice fraction)
    depths = np.zeros_like(ice_fraction_map)
    fractions = ice_fraction_map[ice_mask]

    for idx, f in enumerate(fractions):
        flat_idx = np.where(ice_mask.ravel())[0][idx]
        depths.ravel()[flat_idx] = penetration_depth_ice_regolith(
            f, frequency_hz, max_depth_m)

    # Volume per pixel (m³)
    vol_per_px = pixel_area_m2 * depths * ice_fraction_map

    # Only over detected ice pixels
    total_volume_m3 = float(vol_per_px[ice_mask].sum())

    # Uncertainty (±30% typical for radar-inferred ice concentration)
    uncertainty_pct  = 30.0
    vol_low  = total_volume_m3 * (1 - uncertainty_pct / 100)
    vol_high = total_volume_m3 * (1 + uncertainty_pct / 100)

    # Convert to km³ and metric tons (ice density ≈ 917 kg/m³)
    ICE_DENSITY = 917.0   # kg/m³
    total_mass_kg = total_volume_m3 * ICE_DENSITY * np.mean(ice_fraction_map[ice_mask])

    # Mean fraction and depth for reporting
    mean_fraction = float(np.mean(ice_fraction_map[ice_mask])) if ice_mask.any() else 0.0
    mean_depth    = float(np.mean(depths[ice_mask])) if ice_mask.any() else 0.0

    return {
        "total_volume_m3"         : total_volume_m3,
        "volume_low_m3"           : max(0, vol_low),
        "volume_high_m3"          : vol_high,
        "total_mass_metric_tons"  : total_mass_kg / 1000.0,
        "mean_ice_fraction_pct"   : mean_fraction * 100,
        "mean_penetration_depth_m": mean_depth,
        "ice_area_km2"            : float(ice_mask.sum() * (pixel_size_m/1000)**2),
        "uncertainty_pct"         : uncertainty_pct,
        "frequency_ghz"           : frequency_hz / 1e9,
        "max_integration_depth_m" : max_depth_m,
    }


# ---------------------------------------------------------------------------
# Parameter Sensitivity Analysis
# ---------------------------------------------------------------------------

def sensitivity_sweep(ice_mask: np.ndarray,
                       pixel_size_m: float = 10.0,
                       cpr_values: list = None) -> dict:
    """
    Sweep over ice fraction assumptions and compute volume range.

    Returns a dict of {ice_fraction: volume_m3} for plotting sensitivity.
    """
    if cpr_values is None:
        cpr_values = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5]

    results = {}
    pixel_area_m2 = pixel_size_m ** 2
    n_ice_px = int(ice_mask.sum())

    for cpr_val in cpr_values:
        eps = cpr_to_eps(cpr_val)
        f   = polder_van_santen_ice_fraction(eps)
        d   = penetration_depth_ice_regolith(f, 2.5e9, 5.0)
        vol = n_ice_px * pixel_area_m2 * d * f
        results[round(cpr_val, 2)] = {
            "ice_fraction": round(f * 100, 1),
            "depth_m"     : round(d, 2),
            "volume_m3"   : round(vol, 1),
        }

    return results
