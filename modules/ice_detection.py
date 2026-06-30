"""
ice_detection.py
----------------
Subsurface ice candidate identification module.

IMPORTANT SCIENTIFIC FRAMING
------------------------------
The dual-criteria rule  CPR > 1.0  AND  DOP < 0.13  is a
study-specific candidate criterion derived from the Chandrayaan-2
full-polarimetric DFSAR result for doubly-shadowed south polar craters.

It is NOT a universal lunar ice detection law. Specifically:
  - High CPR alone can arise from rough or blocky rocky terrain
    (surface scattering increases same-sense return).
  - Low DOP is required alongside high CPR to discriminate volume
    scattering (subsurface ice) from surface roughness.
  - The combined signature is indicative of POSSIBLE subsurface ice;
    ground-truth confirmation requires in-situ sampling.
  - This implementation requires full-polarimetric (HH, HV, VH, VV)
    Stokes-based CPR and DOP inputs, not dual-pol approximations.

References
----------
- Chauhan et al. (2022) – Chandrayaan-2 DFSAR full-pol, npj Space Exploration
- Black et al. (2001) – CPR anomalies in lunar PSRs, GRL
- Campbell et al. (2006) – Dual-pol lunar ice constraints
"""

import numpy as np
from scipy.ndimage import (binary_erosion, binary_dilation,
                            label, binary_fill_holes, uniform_filter)


# ---------------------------------------------------------------------------
# Threshold Criteria
# ---------------------------------------------------------------------------

# Literature-backed candidate thresholds from the Chandrayaan-2 study.
# Applied only within PSRs / doubly-shadowed craters.
CPR_THRESHOLD = 1.0    # CPR > 1.0 → anomalous same-sense scattering
DOP_THRESHOLD = 0.13   # DOP < 0.13 → strong volume depolarization

# Backscatter quality gates
SIGMA_MIN_DB  = -25.0  # Exclude noise-dominated pixels
SIGMA_MAX_DB  = -5.0   # Exclude very bright specular returns (rocks/metal)

# Rough-terrain discriminator: if σ° is high AND CPR > threshold,
# attribute to rocky terrain, not ice.
ROCKY_SIGMA_DB_CUTOFF = -8.0


def apply_candidate_criterion(cpr: np.ndarray,
                               dop: np.ndarray,
                               sigma_db: np.ndarray | None = None,
                               shadow_mask: np.ndarray | None = None,
                               rocky_flag: np.ndarray | None = None,
                               cpr_thresh: float = CPR_THRESHOLD,
                               dop_thresh: float = DOP_THRESHOLD) -> dict:
    """
    Apply the study-specific candidate criterion for possible subsurface ice
    in doubly-shadowed south polar craters.

    Criterion: CPR > cpr_thresh  AND  DOP < dop_thresh
    Both conditions must hold; high CPR without low DOP → NOT flagged as ice.

    Parameters
    ----------
    cpr         : CPR array from Stokes-based computation (full-pol)
    dop         : DOP array from Stokes-based computation (full-pol)
    sigma_db    : σ° (dB) for quality screening
    shadow_mask : boolean, True = within PSR/doubly-shadowed zone
    rocky_flag  : boolean, True = probable rough-terrain CPR false positive
    cpr_thresh  : CPR threshold (literature: 1.0)
    dop_thresh  : DOP threshold (literature: 0.13)

    Returns
    -------
    dict with keys:
      'candidate_mask'  : bool array — pixels passing both criteria
      'cpr_only_mask'   : bool array — high CPR but NOT low DOP (rocky suspects)
      'n_rocky_suspects': int — count of CPR-high but DOP-normal pixels
      'thresholds_used' : dict — record of applied thresholds
    """
    cpr_cond = cpr > cpr_thresh
    dop_cond = dop < dop_thresh

    # Both conditions must hold (key discriminator)
    candidate = cpr_cond & dop_cond
    cpr_only  = cpr_cond & ~dop_cond    # likely rough terrain, NOT ice

    # Backscatter quality screen
    if sigma_db is not None:
        quality   = (sigma_db > SIGMA_MIN_DB) & (sigma_db < SIGMA_MAX_DB)
        candidate = candidate & quality
        cpr_only  = cpr_only  & quality

    # Restrict detections to PSRs / doubly-shadowed zones
    if shadow_mask is not None:
        candidate = candidate & shadow_mask.astype(bool)
        # cpr_only not shadow-masked — useful to show everywhere

    # Remove known rough-terrain false positives
    if rocky_flag is not None:
        candidate = candidate & ~rocky_flag

    return {
        "candidate_mask"  : candidate,
        "cpr_only_mask"   : cpr_only,
        "n_rocky_suspects": int(cpr_only.sum()),
        "thresholds_used" : {
            "cpr_threshold": cpr_thresh,
            "dop_threshold": dop_thresh,
            "sigma_min_db" : SIGMA_MIN_DB,
            "sigma_max_db" : SIGMA_MAX_DB,
        },
    }


def morphological_clean(ice_mask: np.ndarray,
                         erosion_iter: int = 2,
                         dilation_iter: int = 3,
                         min_region_px: int = 20) -> np.ndarray:
    """
    Clean ice candidate detections using morphological operations.

    1. Erosion  → remove isolated speckle / single-pixel false positives
    2. Dilation → restore true extent of surviving candidate clusters
    3. Remove connected components smaller than min_region_px

    Returns cleaned boolean mask.
    """
    struct = np.ones((3, 3), dtype=bool)

    cleaned = binary_erosion(ice_mask, structure=struct, iterations=erosion_iter)
    cleaned = binary_dilation(cleaned, structure=struct, iterations=dilation_iter)
    cleaned = binary_fill_holes(cleaned)

    # Remove small spurious regions
    labeled, n_features = label(cleaned)
    for i in range(1, n_features + 1):
        region = labeled == i
        if region.sum() < min_region_px:
            cleaned[region] = False

    return cleaned


# ---------------------------------------------------------------------------
# Confidence / Relative Indicator Scoring
# ---------------------------------------------------------------------------

def compute_ice_confidence(cpr: np.ndarray,
                            dop: np.ndarray,
                            shadow_dist: np.ndarray | None = None) -> np.ndarray:
    """
    Compute a per-pixel candidate confidence indicator in [0, 1].

    This is a RELATIVE indicator — not an absolute probability of ice.
    It scores how strongly a pixel satisfies the study criterion:
      - CPR far above 1.0 → higher score
      - DOP far below 0.13 → higher score
      - Deeper in PSR (optional) → slight bonus

    Score components:
      CPR score : sigmoid( (CPR − 1.0) / 0.3 )
      DOP score : 1 − sigmoid( (DOP − 0.13) / 0.05 )

    Returns
    -------
    confidence : float32 array in [0, 1]
    """
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    cpr_score = sigmoid((cpr - 1.0) / 0.3)
    dop_score = 1.0 - sigmoid((dop - 0.13) / 0.05)

    if shadow_dist is not None:
        shadow_score = np.exp(-shadow_dist / (shadow_dist.max() + 1e-10))
        confidence = (cpr_score ** 0.4) * (dop_score ** 0.4) * (shadow_score ** 0.2)
    else:
        confidence = (cpr_score ** 0.5) * (dop_score ** 0.5)

    return confidence.astype(np.float32)


# ---------------------------------------------------------------------------
# Region Statistics
# ---------------------------------------------------------------------------

def ice_region_stats(candidate_mask: np.ndarray,
                     cpr: np.ndarray,
                     dop: np.ndarray,
                     pixel_size_m: float = 10.0) -> list[dict]:
    """
    Compute statistics for each labeled ice-candidate connected region.

    Returns
    -------
    List of dicts sorted by area descending. Each dict contains:
      id, area_km2, area_px, centroid_px, mean_cpr, max_cpr, mean_dop, min_dop
    """
    labeled, n = label(candidate_mask)
    regions = []
    pixel_area_km2 = (pixel_size_m / 1000.0) ** 2

    for i in range(1, n + 1):
        mask_i = labeled == i
        ys, xs = np.where(mask_i)
        regions.append({
            "id"         : i,
            "area_km2"   : float(mask_i.sum() * pixel_area_km2),
            "area_px"    : int(mask_i.sum()),
            "centroid_px": (float(xs.mean()), float(ys.mean())),
            "mean_cpr"   : float(np.nanmean(cpr[mask_i])),
            "max_cpr"    : float(np.nanmax(cpr[mask_i])),
            "mean_dop"   : float(np.nanmean(dop[mask_i])),
            "min_dop"    : float(np.nanmin(dop[mask_i])),
        })

    regions.sort(key=lambda r: r["area_km2"], reverse=True)
    return regions


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def detection_summary(candidate_mask: np.ndarray,
                       confidence: np.ndarray,
                       rocky_suspects: int = 0,
                       pixel_size_m: float = 10.0) -> dict:
    """
    Generate a summary dictionary for the candidate ice detection result.
    """
    total_px    = candidate_mask.size
    detected_px = int(candidate_mask.sum())
    area_km2    = detected_px * (pixel_size_m / 1000.0) ** 2

    return {
        "total_pixels"          : total_px,
        "candidate_pixels"      : detected_px,
        "coverage_pct"          : 100.0 * detected_px / total_px,
        "candidate_area_km2"    : area_km2,
        "mean_confidence"       : float(np.nanmean(confidence[candidate_mask]))
                                   if detected_px else 0.0,
        "high_confidence_px"    : int((confidence > 0.7).sum()),
        "rocky_suspect_pixels"  : rocky_suspects,
        "note"                  : (
            "Candidate pixels satisfy CPR > threshold AND DOP < threshold "
            "within PSR. This indicates POSSIBLE subsurface ice; "
            "in-situ confirmation required."
        ),
    }
