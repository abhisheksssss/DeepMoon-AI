"""
ice_potential.py
----------------
Relative Ice Potential & Scenario-Based Resource Index

IMPORTANT SCIENTIFIC FRAMING
------------------------------
This module estimates a RELATIVE ice potential / abundance index, NOT an
absolute volumetric ice inventory.

Converting orbital radar backscatter to an absolute ice volume is highly
model-dependent and subject to large uncertainties:
  - Radar penetration depth depends on unknown ice purity, temperature,
    and grain size.
  - Dielectric mixing models assume idealized geometry.
  - The Chandrayaan-2 result itself is framed as "possible presence" of
    subsurface ice, not a validated volumetric inventory.
  - L-band penetrates deeper than S-band, but both face similar inversion
    ambiguities at orbital standoff distances.

Therefore, all volume outputs are:
  (a) scenario-based (assuming a range of ice fraction values), and
  (b) clearly labelled with ±30–50% uncertainty as a LOWER bound.

Use these outputs as an order-of-magnitude ISRU potential indicator,
not as confirmed reserves.

References
----------
- Chauhan et al. (2022) – Chandrayaan-2 DFSAR, npj Space Exploration
- Campbell et al. (2006) – Radar constraints on lunar polar ice
- Ulaby, Moore & Fung (1986) – Microwave Remote Sensing, Vol. III
- Polder & van Santen (1946) – Dielectric mixing
"""

import numpy as np


# ---------------------------------------------------------------------------
# Physical Constants
# ---------------------------------------------------------------------------

EPS_REGOLITH_REAL = 2.7       # Dry regolith real permittivity
EPS_REGOLITH_IMAG = 0.005     # Dry regolith imaginary permittivity
EPS_ICE_REAL      = 3.15      # Water ice real permittivity (cold, pure)
EPS_ICE_IMAG      = 0.001     # Ice imaginary permittivity (very low loss)
ICE_DENSITY_KG_M3 = 917.0     # kg/m³
SPEED_OF_LIGHT    = 3e8       # m/s

# Uncertainty floor — radar-based ice volume is uncertain at best ±30%.
# In practice this is likely a lower bound on uncertainty.
RADAR_VOLUME_UNCERTAINTY_PCT = 40.0


# ---------------------------------------------------------------------------
# Radar Penetration Depth
# ---------------------------------------------------------------------------

def skin_depth_m(frequency_hz: float,
                 eps_real: float,
                 eps_imag: float) -> float:
    """
    Electromagnetic skin depth (1/e power attenuation) in metres.

    δ = c / (2π f √ε') × 1/tan(δ_loss)

    where tan(δ_loss) = ε'' / ε'

    For ice at S-band (2.5 GHz): δ ≈ tens of metres (very low loss).
    For ice-regolith mixtures: intermediate values, 5–30 m typical.

    NOTE: This gives the physical skin depth of the medium, not the
    achievable radar ranging depth, which also depends on surface roughness,
    SNR, and calibration accuracy.
    """
    tan_delta = eps_imag / (eps_real + 1e-12)
    lam = SPEED_OF_LIGHT / frequency_hz
    delta = lam / (4 * np.pi * np.sqrt(eps_real) * (tan_delta + 1e-12))
    return float(delta)


def mixture_skin_depth(ice_fraction: float,
                        frequency_hz: float = 2.5e9,
                        max_depth_m: float = 5.0) -> float:
    """
    Skin depth for an ice–regolith mixture using linear mixing of loss tangent.
    Capped at max_depth_m (problem statement integration limit).
    """
    eps_r = (1 - ice_fraction) * EPS_REGOLITH_REAL + ice_fraction * EPS_ICE_REAL
    eps_i = (1 - ice_fraction) * EPS_REGOLITH_IMAG + ice_fraction * EPS_ICE_IMAG
    d = skin_depth_m(frequency_hz, eps_r, eps_i)
    return min(d, max_depth_m)


# ---------------------------------------------------------------------------
# Dielectric Mixing → Ice Fraction Estimate
# ---------------------------------------------------------------------------

def cpr_to_eps_effective(cpr: float,
                          cpr_min: float = 0.5,
                          cpr_max: float = 2.5,
                          eps_min: float = EPS_REGOLITH_REAL,
                          eps_max: float = 3.5) -> float:
    """
    Empirical linear mapping: CPR → effective dielectric constant.

    This is a simplified surrogate model. The real inversion requires
    a full scattering model (e.g., IEM, small perturbation, or AIEM)
    calibrated to the specific surface geometry.
    """
    t = np.clip((cpr - cpr_min) / (cpr_max - cpr_min + 1e-10), 0, 1)
    return float(eps_min + t * (eps_max - eps_min))


def bruggeman_ice_fraction(eps_mix: float,
                            eps_host: float = EPS_REGOLITH_REAL,
                            eps_incl: float = EPS_ICE_REAL) -> float:
    """
    Invert Bruggeman (Polder-van Santen) mixing model to estimate
    volumetric ice fraction from effective permittivity.

    Formula (spherical inclusions):
        f = (ε_mix − ε_host)(ε_incl + 2ε_host) /
            [3 ε_host (ε_incl − ε_host)]

    Returns ice fraction in [0, 1]. Values outside this range are clipped
    and should be treated as model breakdown.
    """
    numerator   = (eps_mix - eps_host) * (eps_incl + 2 * eps_host)
    denominator = 3 * eps_host * (eps_incl - eps_host)
    if abs(denominator) < 1e-10:
        return 0.0
    f = numerator / denominator
    return float(np.clip(f, 0.0, 1.0))


def cpr_to_ice_fraction_map(cpr_map: np.ndarray) -> np.ndarray:
    """
    Per-pixel ice fraction estimate derived from CPR via:
      CPR → ε_eff (empirical) → ice fraction (Bruggeman).

    Returns float32 array in [0, 1]. Values are uncertain; treat as
    indicative scenario inputs, not calibrated concentrations.
    """
    eps_map = np.vectorize(cpr_to_eps_effective)(cpr_map)
    frac_map = np.vectorize(bruggeman_ice_fraction)(eps_map)
    return frac_map.astype(np.float32)


# ---------------------------------------------------------------------------
# Scenario-Based Ice Potential Index
# ---------------------------------------------------------------------------

def compute_ice_potential(candidate_mask: np.ndarray,
                           frac_map: np.ndarray,
                           cpr_map: np.ndarray,
                           pixel_size_m: float = 10.0,
                           max_depth_m: float = 5.0,
                           frequency_hz: float = 2.5e9) -> dict:
    """
    Compute a scenario-based ice potential / relative resource index.

    Output is explicitly an ORDER-OF-MAGNITUDE estimate with large
    uncertainty bounds. Use for relative comparison between candidate
    regions, not as an absolute reserve figure.

    Parameters
    ----------
    candidate_mask : boolean ice-candidate mask (from ice_detection)
    frac_map       : per-pixel ice fraction (from cpr_to_ice_fraction_map)
    cpr_map        : CPR array
    pixel_size_m   : pixel footprint (metres)
    max_depth_m    : integration depth limit (5 m per problem statement)
    frequency_hz   : radar frequency

    Returns
    -------
    dict with scenario volume estimates, uncertainty bounds, and metadata
    """
    if not candidate_mask.any():
        return {
            "scenario_volume_m3"       : 0.0,
            "volume_lower_m3"          : 0.0,
            "volume_upper_m3"          : 0.0,
            "candidate_area_km2"       : 0.0,
            "mean_ice_fraction_pct"    : 0.0,
            "mean_penetration_depth_m" : 0.0,
            "estimated_mass_metric_t"  : 0.0,
            "uncertainty_pct"          : RADAR_VOLUME_UNCERTAINTY_PCT,
            "note"                     : "No candidate pixels detected.",
        }

    pixel_area_m2 = pixel_size_m ** 2

    # Per-pixel penetration depth (fraction-dependent)
    depths = np.zeros_like(frac_map, dtype=np.float32)
    for r in range(candidate_mask.shape[0]):
        for c in range(candidate_mask.shape[1]):
            if candidate_mask[r, c]:
                depths[r, c] = mixture_skin_depth(
                    float(frac_map[r, c]), frequency_hz, max_depth_m)

    # Scenario volume: pixel area × depth × ice fraction
    vol_per_px = pixel_area_m2 * depths * frac_map * candidate_mask

    best_vol = float(vol_per_px.sum())
    lower    = best_vol * (1 - RADAR_VOLUME_UNCERTAINTY_PCT / 100)
    upper    = best_vol * (1 + RADAR_VOLUME_UNCERTAINTY_PCT / 100)

    mean_frac  = float(np.mean(frac_map[candidate_mask]))
    mean_depth = float(np.mean(depths[candidate_mask]))
    area_km2   = float(candidate_mask.sum() * (pixel_size_m / 1000) ** 2)
    mass_t     = best_vol * mean_frac * ICE_DENSITY_KG_M3 / 1000.0

    return {
        "scenario_volume_m3"       : best_vol,
        "volume_lower_m3"          : max(0.0, lower),
        "volume_upper_m3"          : upper,
        "candidate_area_km2"       : area_km2,
        "mean_ice_fraction_pct"    : mean_frac * 100,
        "mean_penetration_depth_m" : mean_depth,
        "estimated_mass_metric_t"  : mass_t,
        "uncertainty_pct"          : RADAR_VOLUME_UNCERTAINTY_PCT,
        "frequency_ghz"            : frequency_hz / 1e9,
        "max_integration_depth_m"  : max_depth_m,
        "note"                     : (
            f"Scenario estimate ±{RADAR_VOLUME_UNCERTAINTY_PCT:.0f}% "
            "(likely a lower bound on true uncertainty). "
            "Treat as ISRU potential indicator, not confirmed reserve."
        ),
    }


# ---------------------------------------------------------------------------
# Scenario Sweep (sensitivity analysis)
# ---------------------------------------------------------------------------

def scenario_sweep(candidate_mask: np.ndarray,
                   pixel_size_m: float = 10.0,
                   ice_fractions: list = None,
                   frequency_hz: float = 2.5e9) -> dict:
    """
    Compute ice potential across a range of ice fraction assumptions.

    Returns
    -------
    dict mapping ice_fraction → {depth_m, volume_m3, area_km2}
    """
    if ice_fractions is None:
        ice_fractions = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]

    pixel_area_m2 = pixel_size_m ** 2
    n_px = int(candidate_mask.sum())
    area_km2 = n_px * (pixel_size_m / 1000) ** 2

    results = {}
    for f in ice_fractions:
        d = mixture_skin_depth(f, frequency_hz, max_depth_m=5.0)
        v = n_px * pixel_area_m2 * d * f
        results[round(f, 3)] = {
            "ice_fraction_pct" : round(f * 100, 1),
            "depth_m"          : round(d, 2),
            "volume_m3"        : round(v, 0),
            "area_km2"         : round(area_km2, 4),
        }

    return results
