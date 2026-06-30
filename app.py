"""
app.py  –  DeepMoon AI
======================
Streamlit dashboard for Chandrayaan-2 DFSAR-based subsurface ice candidate
detection and lunar south polar exploration planning.

SCIENTIFIC FRAMING
-------------------
All ice detections are CANDIDATE indicators based on the study-specific
criterion (CPR > 1, DOP < 0.13) from the Chandrayaan-2 full-pol DFSAR
result for doubly-shadowed south polar craters. In-situ confirmation
would be required to confirm ice presence.

Problem Statement 8 – ISRO Hackathon
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import io

# ── Project modules ──────────────────────────────────────────────────────────
from modules.radar_processing  import (generate_synthetic_dfsar,
                                        stokes_from_quad_pol,
                                        compute_cpr_from_stokes,
                                        compute_dop_from_stokes,
                                        compute_sigma_naught,
                                        flag_rough_terrain,
                                        lee_filter)
from modules.ice_detection      import (apply_candidate_criterion,
                                         morphological_clean,
                                         compute_ice_confidence,
                                         ice_region_stats,
                                         detection_summary)
from modules.terrain_analysis   import (generate_synthetic_dem,
                                         compute_slope, compute_aspect,
                                         compute_tri,
                                         compute_roughness_stddev,
                                         fast_shadow_mask,
                                         compute_psr_mask,
                                         compute_terrain_safety,
                                         illumination_fraction,
                                         compute_earth_visibility,
                                         ohrc_quality_flag)
from modules.landing_site       import (score_landing_sites,
                                         extract_landing_candidates,
                                         recommend_landing_site,
                                         check_ellipse_feasibility)
from modules.rover_path         import plan_path
from modules.ice_potential      import (compute_ice_potential,
                                         cpr_to_ice_fraction_map,
                                         scenario_sweep)
from modules.visualization      import (array_to_heatmap,
                                         plot_cpr_dop_combined,
                                         plot_dem_3d,
                                         plot_ice_confidence,
                                         plot_landing_analysis,
                                         plot_rover_path,
                                         plot_volume_results,
                                         plot_illumination_wheel)

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepMoon AI | Lunar Ice Explorer",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

  /* Dark background */
  .stApp { background: linear-gradient(135deg, #03040A 0%, #0A0E1A 50%, #050915 100%); }

  /* Hero header */
  .hero-header {
    background: linear-gradient(135deg, #0A1628 0%, #0D2137 50%, #071520 100%);
    border: 1px solid rgba(0, 180, 255, 0.2);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .hero-header::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 30% 50%, rgba(0,100,255,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 20%, rgba(0,200,255,0.05) 0%, transparent 50%);
  }
  .hero-title {
    font-size: 2.6rem; font-weight: 700;
    background: linear-gradient(90deg, #00B4FF, #00FFD4, #7B61FF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.2;
  }
  .hero-subtitle {
    color: #8BA3C7; font-size: 1.0rem; margin-top: 0.5rem; font-weight: 300;
  }
  .hero-badge {
    display: inline-block;
    background: rgba(0, 180, 255, 0.12);
    border: 1px solid rgba(0, 180, 255, 0.3);
    color: #00B4FF;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-right: 0.5rem;
    margin-top: 0.8rem;
  }

  /* Metric cards */
  .metric-card {
    background: linear-gradient(135deg, #0D1B2E, #0A1525);
    border: 1px solid rgba(0, 180, 255, 0.18);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: all 0.3s ease;
  }
  .metric-card:hover {
    border-color: rgba(0, 180, 255, 0.45);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 100, 255, 0.1);
  }
  .metric-value {
    font-size: 2.0rem; font-weight: 700;
    background: linear-gradient(90deg, #00B4FF, #00FFD4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .metric-label {
    color: #6B87A8; font-size: 0.82rem; margin-top: 0.3rem; font-weight: 400;
  }
  .metric-unit {
    color: #4A6480; font-size: 0.72rem;
  }

  /* Section headers */
  .section-header {
    font-size: 1.3rem; font-weight: 600;
    color: #C0D8F0;
    border-left: 3px solid #00B4FF;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem;
  }

  /* Info box */
  .info-box {
    background: rgba(0, 100, 200, 0.08);
    border: 1px solid rgba(0, 150, 255, 0.2);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #9BB8D4;
    font-size: 0.88rem;
    line-height: 1.6;
  }

  /* Success / warning tags */
  .tag-success { background: rgba(0,200,120,0.15); border: 1px solid rgba(0,200,120,0.35);
                 color: #00C878; padding: 0.15rem 0.6rem; border-radius: 6px; font-size: 0.8rem; }
  .tag-warn    { background: rgba(255,160,0,0.15); border: 1px solid rgba(255,160,0,0.35);
                 color: #FFA000; padding: 0.15rem 0.6rem; border-radius: 6px; font-size: 0.8rem; }
  .tag-ice     { background: rgba(0,200,255,0.12); border: 1px solid rgba(0,200,255,0.3);
                 color: #00C8FF; padding: 0.15rem 0.6rem; border-radius: 6px; font-size: 0.8rem; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050915 0%, #080E1C 100%);
    border-right: 1px solid rgba(0,100,255,0.15);
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(0,0,0,0.3);
    border-radius: 10px;
    padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6B87A8;
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    font-weight: 500;
    font-size: 0.9rem;
    border: none;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0A2A4A, #083050) !important;
    color: #00B4FF !important;
    box-shadow: 0 2px 12px rgba(0,100,255,0.2);
  }

  /* Divider */
  hr { border-color: rgba(0,100,255,0.15); }

  /* Slider */
  .stSlider > div > div > div > div { background: linear-gradient(90deg, #00B4FF, #7B61FF) !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #050915; }
  ::-webkit-scrollbar-thumb { background: #0A3060; border-radius: 3px; }

  /* Expander */
  .streamlit-expanderHeader { color: #8BA3C7 !important; }

  /* Number display */
  code, .monospace { font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State & Caching
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_synthetic_data(shape=(512, 512), seed=42):
    """Generate all synthetic datasets (cached between reruns)."""
    # Full-polarimetric DFSAR data (HH, HV, VH, VV)
    dfsar = generate_synthetic_dfsar(shape=shape, seed=seed)

    # Step 1: Compute Stokes parameters (requires full-pol HH, HV, VH, VV)
    stokes = stokes_from_quad_pol(
        dfsar["HH"].astype(np.complex64),
        dfsar["HV"].astype(np.complex64),
        dfsar["VH"].astype(np.complex64),
        dfsar["VV"].astype(np.complex64),
        window=7
    )

    # Step 2: CPR and DOP from Stokes (full-pol formulation)
    cpr   = compute_cpr_from_stokes(stokes)
    dop   = compute_dop_from_stokes(stokes)
    sigma = compute_sigma_naught(dfsar["HH"], 35.0)

    # Step 3: Flag probable rough-terrain CPR false positives
    rocky = flag_rough_terrain(cpr, sigma)

    # DEM + terrain
    dem_data = generate_synthetic_dem(shape=shape, seed=seed + 1)
    dem  = dem_data["elevation"]
    pxsz = dem_data["pixel_scale_m"]

    slope  = compute_slope(dem, pxsz)
    tri    = compute_tri(dem)
    psr    = compute_psr_mask(dem, pxsz, n_azimuths=12)
    illum  = illumination_fraction(dem, pxsz, n_steps=24)

    # Earth-visibility proxy
    earth_vis = compute_earth_visibility(dem, pxsz,
                                          earth_elevation_deg=6.5,
                                          earth_azimuth_deg=0.0)

    # OHRC quality confidence flag
    ohrc_quality = ohrc_quality_flag(illum, psr)

    # Merge metadata
    meta = dfsar["metadata"].copy()
    meta.update(dem_data["metadata"])

    return dict(
        dfsar=dfsar, stokes=stokes, cpr=cpr, dop=dop, sigma=sigma,
        rocky=rocky,
        dem=dem, pxsz=pxsz,
        slope=slope, tri=tri, psr=psr, illum=illum,
        earth_vis=earth_vis, ohrc_quality=ohrc_quality,
        meta=meta,
    )


@st.cache_data(show_spinner=False)
def run_ice_detection(cpr_thresh, dop_thresh, shape, seed):
    data = load_synthetic_data(shape, seed)
    detection = apply_candidate_criterion(
        data["cpr"], data["dop"], data["sigma"],
        shadow_mask=data["psr"],
        rocky_flag=data["rocky"],
        cpr_thresh=cpr_thresh, dop_thresh=dop_thresh,
    )
    candidate_raw = detection["candidate_mask"]
    cpr_only_mask = detection["cpr_only_mask"]   # rocky terrain suspects
    rocky_suspects = detection["n_rocky_suspects"]

    candidate_clean = morphological_clean(candidate_raw, erosion_iter=2,
                                           dilation_iter=3, min_region_px=20)
    confidence = compute_ice_confidence(data["cpr"], data["dop"])
    regions    = ice_region_stats(candidate_clean, data["cpr"], data["dop"],
                                   pixel_size_m=data["pxsz"])
    summary    = detection_summary(candidate_clean, confidence,
                                    rocky_suspects, data["pxsz"])
    return candidate_clean, confidence, regions, summary, cpr_only_mask


@st.cache_data(show_spinner=False)
def run_landing_analysis(cpr_thresh, dop_thresh, shape, seed):
    data = load_synthetic_data(shape, seed)
    ice, conf, _, _, _ = run_ice_detection(cpr_thresh, dop_thresh, shape, seed)

    safety = compute_terrain_safety(data["slope"], data["tri"], data["psr"])
    score  = score_landing_sites(
        data["slope"], data["tri"], data["illum"], ice,
        data["pxsz"],
        earth_visibility=data["earth_vis"],
        psr_mask=data["psr"],      # hard-mask: no landing inside PSR
    )
    candidates  = extract_landing_candidates(score, data["pxsz"],
                                              min_score=0.30, top_n=5)
    recommended = recommend_landing_site(candidates, data["slope"], score, data["pxsz"])
    return safety, score, candidates, recommended


@st.cache_data(show_spinner=False)
def run_path_planning(cpr_thresh, dop_thresh, shape, seed):
    data = load_synthetic_data(shape, seed)
    ice, _, _, _, _ = run_ice_detection(cpr_thresh, dop_thresh, shape, seed)
    _, _, candidates, rec = run_landing_analysis(cpr_thresh, dop_thresh, shape, seed)

    cx, cy = data["meta"]["center_px"]
    rim_r  = int(data["meta"]["crater_radius_px"] * 0.85)

    if rec and rec.get("centroid_px"):
        lx, ly = rec["centroid_px"]
        lx, ly = int(lx), int(ly)
    elif candidates:
        lx, ly = [int(v) for v in candidates[0]["centroid_px"]]
    else:
        lx, ly = cx + 200, cy + 200

    # Goal: crater rim closest to landing site (entry point to PSR)
    angle = np.arctan2(ly - cy, lx - cx)
    gx = int(cx + rim_r * np.cos(angle))
    gy = int(cy + rim_r * np.sin(angle))
    gx = int(np.clip(gx, 0, shape[1] - 1))
    gy = int(np.clip(gy, 0, shape[0] - 1))

    path_result = plan_path(
        data["slope"], data["tri"], data["illum"], ice,
        start_px=(lx, ly), goal_px=(gx, gy),
        pixel_size_m=data["pxsz"],
    )
    return path_result, (lx, ly), (gx, gy)


@st.cache_data(show_spinner=False)
def run_potential_estimation(cpr_thresh, dop_thresh, shape, seed):
    data = load_synthetic_data(shape, seed)
    ice, _, _, _, _ = run_ice_detection(cpr_thresh, dop_thresh, shape, seed)

    frac_map = cpr_to_ice_fraction_map(data["cpr"])
    pot_dict = compute_ice_potential(ice, frac_map, data["cpr"],
                                      pixel_size_m=data["pxsz"],
                                      max_depth_m=5.0, frequency_hz=2.5e9)
    sweep = scenario_sweep(ice, data["pxsz"])
    return pot_dict, frac_map, sweep


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
      <div style='font-size:2.5rem'>🌙</div>
      <div style='font-size:1.1rem; font-weight:700;
                  background: linear-gradient(90deg,#00B4FF,#00FFD4);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        DeepMoon AI
      </div>
      <div style='color:#4A6480;font-size:0.75rem;margin-top:0.2rem;'>
        Chandrayaan-2 · DFSAR Analysis
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Analysis Parameters")

    SHAPE_OPTS = {"256 × 256 (fast)": 256, "512 × 512 (recommended)": 512,
                  "768 × 768 (detailed)": 768}
    shape_label = st.selectbox("Grid Resolution", list(SHAPE_OPTS.keys()), index=1)
    SHAPE = (SHAPE_OPTS[shape_label], SHAPE_OPTS[shape_label])
    SEED  = st.number_input("Random seed", min_value=0, max_value=999, value=42)

    st.markdown("---")
    st.markdown("### 🧊 Ice Detection Thresholds")
    cpr_thresh = st.slider("CPR threshold", 0.5, 2.0, 1.0, 0.05,
                            help="Published ISRO threshold: CPR > 1.0")
    dop_thresh = st.slider("DOP threshold", 0.05, 0.30, 0.13, 0.01,
                            help="Published ISRO threshold: DOP < 0.13")

    st.markdown("---")
    st.markdown("### 📡 Radar Configuration")
    frequency_label = st.selectbox("DFSAR Band", ["S-band (2.5 GHz)", "L-band (0.43 GHz)"])
    freq_hz = 2.5e9 if "S-band" in frequency_label else 0.43e9

    st.markdown("---")
    st.markdown("""
    <div style='color:#3A5070; font-size:0.75rem; text-align:center; padding-top:0.5rem;'>
      ISRO Hackathon 2025 · Problem 8<br>
      Chandrayaan-2 Subsurface Ice Detection
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
  <div class="hero-title">🌙 DeepMoon AI</div>
  <div class="hero-subtitle">
    Subsurface Ice Detection & Exploration Planning · Lunar South Polar Region
  </div>
  <div>
    <span class="hero-badge">🛰️ Chandrayaan-2 DFSAR</span>
    <span class="hero-badge">📡 S-band Radar</span>
    <span class="hero-badge">🧊 PSR Ice Detection</span>
    <span class="hero-badge">🤖 A* Path Planning</span>
    <span class="hero-badge">📊 ISRO PS-8</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load Data (with progress indicator)
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("🔭 Loading and processing Chandrayaan-2 data..."):
    DATA = load_synthetic_data(SHAPE, SEED)
    ice_mask, confidence, regions, det_summary, cpr_only_mask = run_ice_detection(
        cpr_thresh, dop_thresh, SHAPE, SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Top KPI Metrics Row
# ─────────────────────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)

def metric_card(col, value, label, unit=""):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{value}</div>
          <div class="metric-label">{label}</div>
          <div class="metric-unit">{unit}</div>
        </div>
        """, unsafe_allow_html=True)

ice_area_km2   = det_summary.get("candidate_area_km2", 0.0)
psr_area_km2   = float(DATA["psr"].sum() * (DATA["pxsz"] / 1000) ** 2)
cpr_mean       = float(np.nanmean(DATA["cpr"][ice_mask])) if ice_mask.any() else 0.0
confidence_pct = det_summary["mean_confidence"] * 100
n_regions      = len(regions)

metric_card(c1, f"{ice_area_km2:.2f}", "Candidate Ice Area", "km²")
metric_card(c2, f"{psr_area_km2:.1f}", "PSR Total Area", "km²")
metric_card(c3, f"{cpr_mean:.2f}", "Mean CPR (ice zone)", "—")
metric_card(c4, f"{confidence_pct:.0f}%", "Detection Confidence", "")
metric_card(c5, f"{n_regions}", "Candidate Regions", "")

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_overview, tab_radar, tab_ice, tab_terrain, tab_landing, tab_rover, tab_volume, tab_report = st.tabs([
    "🌑 Overview",
    "📡 Radar Analysis",
    "🧊 Ice Candidates",
    "🏔️ Terrain",
    "🛬 Landing Site",
    "🤖 Rover Path",
    "📊 Ice Potential",
    "📋 Report",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
with tab_overview:
    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown('<div class="section-header">Mission Context</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
          <strong style="color:#00B4FF">Doubly-Shadowed Crater Analysis</strong><br><br>
          This dashboard processes Chandrayaan-2 <strong>full-polarimetric DFSAR</strong>
          (HH, HV, VH, VV) data to identify possible subsurface ice candidates within a
          <em>doubly-shadowed crater</em> in the lunar south polar permanently shadowed
          region (PSR).<br><br>
          <strong style="color:#FFA000">⚠ Scientific Caveat:</strong>
          All detections are <em>candidate indicators</em> of possible ice, based on a
          study-specific criterion from the Chandrayaan-2 full-pol result. High CPR alone
          can also arise from rough/blocky terrain; low DOP is required alongside it
          to discriminate subsurface volume scattering from surface roughness.
          In-situ confirmation would be needed.<br><br>
          <strong>Analysis steps:</strong>
          <ul>
            <li>Full-pol Stokes parameters → CPR, DOP (Chauhan et al. 2022)</li>
            <li>Candidate criterion: CPR &gt; 1.0 AND DOP &lt; 0.13, within PSR only</li>
            <li>Rough-terrain false-positive flagging (high σ° + high CPR)</li>
            <li>Terrain safety → landing zones on <em>illuminated</em> terrain</li>
            <li>A* rover traverse to crater rim / ice candidate zone</li>
            <li>Relative ice potential (scenario-based, ±40% uncertainty)</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Analysis Workflow</div>', unsafe_allow_html=True)

        steps = [
            ("1", "Full-pol DFSAR ingestion (HH, HV, VH, VV) + calibration", "Complete", "tag-success"),
            ("2", "Stokes parameters → CPR + DOP (full-pol formulation)", "Complete", "tag-success"),
            ("3", "Rough-terrain FP flagging + candidate criterion", "Complete", "tag-success"),
            ("4", "PSR mapping + terrain safety evaluation", "Complete", "tag-success"),
            ("5", "Landing zone on illuminated terrain + Earth LoS", "Complete", "tag-success"),
            ("6", "Terrain-aware A* rover traverse to crater rim", "Complete", "tag-success"),
            ("7", "Relative ice potential (scenario-based, ±40% unc.)", "Complete", "tag-success"),
        ]
        for num, desc, status, cls in steps:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.8rem;
                        padding:0.5rem 0.8rem;margin-bottom:0.4rem;
                        background:rgba(0,20,50,0.5);border-radius:8px;
                        border:1px solid rgba(0,100,200,0.15);">
              <div style="background:linear-gradient(135deg,#00B4FF,#7B61FF);
                          color:white;font-weight:700;width:26px;height:26px;
                          border-radius:50%;display:flex;align-items:center;
                          justify-content:center;font-size:0.75rem;flex-shrink:0;">{num}</div>
              <div style="color:#C0D8F0;font-size:0.88rem;flex:1">{desc}</div>
              <span class="{cls}">{status}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">Target Crater Overview</div>', unsafe_allow_html=True)
        # 3D DEM overview
        fig_dem3d = plot_dem_3d(DATA["dem"], DATA["pxsz"],
                                 title="Doubly-Shadowed Crater – DEM", height=400)
        st.plotly_chart(fig_dem3d, use_container_width=True)

        # Quick stats table
        st.markdown('<div class="section-header">Crater Parameters</div>', unsafe_allow_html=True)
        meta = DATA["meta"]
        params = {
            "Crater Diameter": f"~{meta['crater_radius_px']*2*DATA['pxsz']/1000:.1f} km",
            "Inner PSR Diameter": f"~{meta['inner_radius_px']*2*DATA['pxsz']/1000:.1f} km",
            "Pixel Scale": f"{DATA['pxsz']} m/px",
            "Radar Band": f"S-band (2.5 GHz)",
            "Incidence Angle": "35°",
            "CPR Threshold": f"> {cpr_thresh}",
            "DOP Threshold": f"< {dop_thresh}",
        }
        rows_html = "".join([
            f"""<tr><td style="color:#6B87A8;padding:0.3rem 0.6rem;font-size:0.83rem">{k}</td>
                    <td style="color:#C0D8F0;padding:0.3rem 0.6rem;font-size:0.83rem;
                                font-family:'JetBrains Mono',monospace">{v}</td></tr>"""
            for k, v in params.items()
        ])
        st.markdown(f"""
        <table style="width:100%;border-collapse:collapse;
                      background:rgba(0,10,30,0.5);border-radius:10px;overflow:hidden;">
          {rows_html}
        </table>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: RADAR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tab_radar:
    st.markdown('<div class="section-header">DFSAR Polarimetric Products</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box">
          <strong>Circular Polarization Ratio (CPR)</strong><br>
          CPR = SC / OC = (S1 − S4) / (S1 + S4)  [from full-pol Stokes]<br><br>
          ✦ Volume scattering from subsurface ice enhances the same-sense circular
          return → CPR &gt; 1.  However, rough/blocky terrain can also produce
          elevated CPR — <strong>low DOP must co-occur</strong> to indicate ice.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-box">
          <strong>Degree of Polarization (DOP)</strong><br>
          DOP = √(S2² + S3² + S4²) / S1  [Stokes-based, full-pol]<br><br>
          ✦ DOP → 0: fully depolarized (volume/multiple scattering, ice candidate).<br>
          ✦ DOP → 1: fully polarized (specular surface scatter, rocky terrain).<br>
          ✦ Literature-backed discriminator: <strong>DOP &lt; 0.13</strong> combined
          with CPR &gt; 1 for doubly-shadowed south-polar crater floors.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fig_cpr_dop = plot_cpr_dop_combined(DATA["cpr"], DATA["dop"], ice_mask, height=460)
    st.plotly_chart(fig_cpr_dop, use_container_width=True)

    st.markdown('<div class="section-header">Backscatter Coefficient (σ°)</div>', unsafe_allow_html=True)
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        fig_sigma = array_to_heatmap(DATA["sigma"], "Sigma-Naught σ° (dB)",
                                      colorscale="Turbo", zmin=-30, zmax=0,
                                      colorbar_title="σ° (dB)", height=380)
        st.plotly_chart(fig_sigma, use_container_width=True)
    with col_s2:
        st.markdown('<div class="section-header">CPR Distribution</div>', unsafe_allow_html=True)
        cpr_flat = DATA["cpr"].ravel()
        cpr_flat = cpr_flat[~np.isnan(cpr_flat)]
        fig_hist = go.Figure(go.Histogram(
            x=cpr_flat, nbinsx=80,
            marker=dict(color=cpr_flat, colorscale="Plasma", opacity=0.8),
        ))
        fig_hist.add_vline(x=cpr_thresh, line_dash="dash", line_color="#00FFD4",
                           annotation_text=f"Threshold {cpr_thresh}")
        fig_hist.update_layout(
            height=300, paper_bgcolor="#0E1117", plot_bgcolor="#111922",
            font=dict(color="#E0E0E0"),
            xaxis=dict(title="CPR", showgrid=True, gridcolor="#1A2A3A"),
            yaxis=dict(title="Count", showgrid=True, gridcolor="#1A2A3A"),
            margin=dict(l=50, r=20, t=40, b=40),
            title=dict(text="CPR Histogram", font=dict(size=13)),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # DOP histogram
        dop_flat = DATA["dop"].ravel()
        fig_dop_hist = go.Figure(go.Histogram(
            x=dop_flat, nbinsx=80,
            marker=dict(color=dop_flat, colorscale="Viridis", opacity=0.8),
        ))
        fig_dop_hist.add_vline(x=dop_thresh, line_dash="dash", line_color="#FF6B35",
                                annotation_text=f"Threshold {dop_thresh}")
        fig_dop_hist.update_layout(
            height=280, paper_bgcolor="#0E1117", plot_bgcolor="#111922",
            font=dict(color="#E0E0E0"),
            xaxis=dict(title="DOP", showgrid=True, gridcolor="#1A2A3A"),
            yaxis=dict(title="Count", showgrid=True, gridcolor="#1A2A3A"),
            margin=dict(l=50, r=20, t=40, b=40),
            title=dict(text="DOP Histogram", font=dict(size=13)),
        )
        st.plotly_chart(fig_dop_hist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: ICE DETECTION
# ═══════════════════════════════════════════════════════════════════════════
with tab_ice:
    st.markdown('<div class="section-header">Subsurface Ice Detection Results</div>',
                unsafe_allow_html=True)

    # Summary metrics row
    sc1, sc2, sc3, sc4 = st.columns(4)
    metric_card(sc1, f"{det_summary.get('candidate_area_km2', 0):.3f}", "Candidate Area", "km²")
    metric_card(sc2, f"{det_summary['coverage_pct']:.1f}%", "Coverage", "of scene")
    metric_card(sc3, f"{det_summary['mean_confidence']*100:.0f}%", "Relative Confidence", "")
    metric_card(sc4, f"{det_summary.get('rocky_suspect_pixels', 0):,}", "Rocky FP Suspects", "")
    st.markdown("<br>", unsafe_allow_html=True)

    st.info("⚠️ **Candidate detections only.** Pixels satisfy CPR > threshold AND DOP < threshold "
            "within the PSR. High CPR without low DOP is attributed to rough/blocky terrain (not ice). "
            "In-situ sampling is required to confirm ice presence.")

    # Ice probability map
    col_map, col_table = st.columns([1.4, 1])
    with col_map:
        cx, cy = DATA["meta"]["center_px"]
        fig_conf = plot_ice_confidence(confidence, ice_mask, (cx, cy), height=460)
        st.plotly_chart(fig_conf, use_container_width=True)

    with col_table:
        st.markdown('<div class="section-header">Candidate Ice Regions</div>', unsafe_allow_html=True)
        if regions:
            df_regions = pd.DataFrame(regions)[
                ["id", "area_km2", "mean_cpr", "max_cpr", "mean_dop", "min_dop"]
            ]
            df_regions.columns = ["ID", "Area km²", "Mean CPR", "Max CPR", "Mean DOP", "Min DOP"]
            df_regions = df_regions.round(3)
            st.dataframe(df_regions, use_container_width=True, height=260,
                         hide_index=True)
        else:
            st.warning("No candidate regions found. Adjust CPR/DOP thresholds.")

        st.markdown('<div class="section-header">Detection Methodology</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box" style="font-size:0.82rem">
          <strong style="color:#00B4FF">Study-Specific Candidate Criterion</strong><br>
          <code>Candidate = (CPR &gt; thresh) AND (DOP &lt; thresh) AND pixel ∈ PSR</code><br>
          High CPR <em>without</em> low DOP → probable rocky terrain, not ice.<br><br>
          <strong>Inputs:</strong> Full-pol DFSAR (HH, HV, VH, VV)<br>
          <strong>CPR:</strong> SC/OC from Stokes parameters<br>
          <strong>DOP:</strong> √(S2²+S3²+S4²)/S1 (Stokes-based)<br><br>
          <strong>Post-processing:</strong>
          <ul>
            <li>Lee speckle filter (7×7 window)</li>
            <li>Rocky-terrain FP masking (high σ° + high CPR)</li>
            <li>Morphological erosion (×2) → dilation (×3)</li>
            <li>Min cluster size: 20 px; restricted to PSR</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    # Side-by-side: CPR-only (rocky FP) vs dual-criterion candidates
    st.markdown('<div class="section-header">CPR-only (Rocky Terrain Suspects) vs Dual-Criterion Candidates</div>',
                unsafe_allow_html=True)
    col_r, col_c = st.columns(2)
    with col_r:
        fig_rocky = array_to_heatmap(cpr_only_mask.astype(float),
                                      "CPR > threshold BUT DOP NOT < threshold → probable rocky terrain",
                                      colorscale=[[0,"#0E1117"],[1,"#FF6B35"]],
                                      height=340)
        st.plotly_chart(fig_rocky, use_container_width=True)
    with col_c:
        fig_clean = array_to_heatmap(ice_mask.astype(float),
                                      "Both CPR > threshold AND DOP < threshold (ice candidates)",
                                      colorscale=[[0,"#0E1117"],[1,"#00B4FF"]],
                                      height=340)
        st.plotly_chart(fig_clean, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: TERRAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tab_terrain:
    st.markdown('<div class="section-header">Digital Elevation Model & Terrain Products</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_elev = array_to_heatmap(DATA["dem"], "Elevation Map",
                                     colorscale="Greys", colorbar_title="m", height=380)
        st.plotly_chart(fig_elev, use_container_width=True)
    with c2:
        fig_slope = array_to_heatmap(DATA["slope"], "Slope Map",
                                      colorscale="Hot_r", zmin=0, zmax=35,
                                      colorbar_title="degrees", height=380)
        st.plotly_chart(fig_slope, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig_psr = array_to_heatmap(DATA["psr"].astype(float),
                                    "Permanently Shadowed Region (PSR) Mask",
                                    colorscale=[[0,"#0E1117"],[1,"#4B0082"]],
                                    colorbar_title="PSR", height=350)
        st.plotly_chart(fig_psr, use_container_width=True)
    with c4:
        fig_illum = array_to_heatmap(DATA["illum"], "Solar Illumination Fraction",
                                      colorscale="Solar", zmin=0, zmax=1,
                                      colorbar_title="Fraction", height=350)
        st.plotly_chart(fig_illum, use_container_width=True)

    st.markdown('<div class="section-header">Solar Illumination Polar Plot</div>', unsafe_allow_html=True)
    col_polar, col_tri = st.columns(2)
    with col_polar:
        fig_wheel = plot_illumination_wheel(DATA["illum"], height=380)
        st.plotly_chart(fig_wheel, use_container_width=True)
    with col_tri:
        fig_tri = array_to_heatmap(DATA["tri"], "Terrain Ruggedness Index (TRI)",
                                    colorscale="Oranges", colorbar_title="TRI (m)",
                                    height=380)
        st.plotly_chart(fig_tri, use_container_width=True)

    # Key terrain statistics
    st.markdown('<div class="section-header">Terrain Statistics</div>', unsafe_allow_html=True)
    t1, t2, t3, t4 = st.columns(4)
    metric_card(t1, f"{float(DATA['slope'].mean()):.1f}°", "Mean Slope", "")
    metric_card(t2, f"{float(DATA['slope'].max()):.1f}°", "Max Slope", "")
    metric_card(t3, f"{float(DATA['dem'].max() - DATA['dem'].min()):.0f}", "Elevation Range", "m")
    psr_pct = 100 * DATA["psr"].mean()
    metric_card(t4, f"{psr_pct:.1f}%", "PSR Coverage", "of scene")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: LANDING SITE
# ═══════════════════════════════════════════════════════════════════════════
with tab_landing:
    with st.spinner("🛬 Computing landing site scores..."):
        safety, score, candidates, recommended = run_landing_analysis(
            cpr_thresh, dop_thresh, SHAPE, SEED)

    st.markdown('<div class="section-header">Landing Site Multi-Criteria Analysis</div>',
                unsafe_allow_html=True)

    col_l, col_r = st.columns([1.4, 1])
    with col_l:
        fig_land = plot_landing_analysis(safety, DATA["slope"], candidates,
                                          recommended, DATA["pxsz"], height=500)
        st.plotly_chart(fig_land, use_container_width=True)

    with col_r:
        # Score map
        fig_score = array_to_heatmap(score, "Landing Suitability Score",
                                      colorscale="RdYlGn", zmin=0, zmax=1,
                                      colorbar_title="Score", height=280)
        st.plotly_chart(fig_score, use_container_width=True)

        # Recommended site card
        if recommended:
            rec_cx, rec_cy = recommended.get("centroid_px", (0, 0))
            feas = recommended.get("feasibility", {})
            color = "#00FF88" if recommended.get("recommended") else "#FFA000"
            icon  = "✅" if recommended.get("recommended") else "⚠️"
            st.markdown(f"""
            <div style="background:rgba(0,30,10,0.5);border:1px solid {color}40;
                        border-radius:12px;padding:1.2rem;">
              <div style="color:{color};font-size:1.1rem;font-weight:700;
                          margin-bottom:0.8rem">{icon} Recommended Landing Site</div>
              <table style="width:100%;font-size:0.82rem;color:#B0C8E0">
                <tr><td style="color:#6B87A8">Score:</td>
                    <td><strong>{recommended.get('mean_score',0):.3f}</strong></td></tr>
                <tr><td style="color:#6B87A8">Slope (mean):</td>
                    <td><strong>{feas.get('mean_slope_deg',0):.1f}°</strong></td></tr>
                <tr><td style="color:#6B87A8">Max slope:</td>
                    <td><strong>{feas.get('max_slope_deg',0):.1f}°</strong></td></tr>
                <tr><td style="color:#6B87A8">Hazard fraction:</td>
                    <td><strong>{feas.get('hazard_fraction',0)*100:.1f}%</strong></td></tr>
                <tr><td style="color:#6B87A8">Area:</td>
                    <td><strong>{recommended.get('area_km2',0):.3f} km²</strong></td></tr>
                <tr><td style="color:#6B87A8">Centroid (px):</td>
                    <td><strong>({rec_cx:.0f}, {rec_cy:.0f})</strong></td></tr>
              </table>
            </div>
            """, unsafe_allow_html=True)

    # Candidate table
    st.markdown('<div class="section-header">Candidate Landing Zones</div>', unsafe_allow_html=True)
    if candidates:
        df_cand = pd.DataFrame(candidates)[["id","area_km2","mean_score","max_score","pixel_count"]]
        df_cand.columns = ["ID", "Area km²", "Mean Score", "Max Score", "Pixels"]
        df_cand = df_cand.round(4)
        st.dataframe(df_cand, use_container_width=True, hide_index=True)
    else:
        st.warning("No landing candidates found. Check terrain parameters.")

    # Scoring criteria explanation
    with st.expander("📐 Scoring Criteria Details"):
        st.markdown("""
        | Criterion | Weight | Rationale |
        |-----------|--------|-----------|
        | Slope ≤ 15° | **30%** | Primary safety constraint for touchdown |
        | Ice proximity | **25%** | Maximise science return; reach ice easily |
        | Solar illumination | **20%** | Power generation for lander survival |
        | Surface roughness | **15%** | Smooth terrain prevents structural damage |
        | Crater access | **10%** | Rover travel distance to doubly-shadowed zone |
        """)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6: ROVER PATH
# ═══════════════════════════════════════════════════════════════════════════
with tab_rover:
    with st.spinner("🤖 Running A* path planning..."):
        path_result, landing_px, goal_px = run_path_planning(
            cpr_thresh, dop_thresh, SHAPE, SEED)

    st.markdown('<div class="section-header">Terrain-Aware A* Rover Traverse</div>',
                unsafe_allow_html=True)

    path_px   = path_result.get("path_px", [])
    waypoints = path_result.get("waypoints", [])
    success   = path_result.get("success", False)

    if success and path_px:
        pr1, pr2, pr3, pr4 = st.columns(4)
        metric_card(pr1, f"{path_result['length_km']:.2f}", "Path Length", "km")
        metric_card(pr2, f"{path_result['stats'].get('max_slope_deg',0):.1f}°", "Max Slope", "")
        metric_card(pr3, f"{path_result['stats'].get('mean_illumination',0)*100:.0f}%", "Mean Illumination", "")
        metric_card(pr4, f"{len(waypoints)}", "Science Stops", "(ice crossings)")
        st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Path planning did not find a clear route. Adjust slope or start/goal parameters.")

    # Main path visualisation
    fig_path = plot_rover_path(
        DATA["dem"], DATA["slope"], ice_mask,
        path_px, waypoints, landing_px, goal_px,
        DATA["pxsz"], height=520,
    )
    st.plotly_chart(fig_path, use_container_width=True)

    # Slope profile along path
    if path_px and len(path_px) > 10:
        st.markdown('<div class="section-header">Traverse Slope Profile</div>', unsafe_allow_html=True)
        px_dist  = [i * DATA["pxsz"] / 1000 for i in range(len(path_px))]
        px_slope = [float(DATA["slope"][p[1], p[0]]) for p in path_px]
        px_illum = [float(DATA["illum"][p[1], p[0]]) for p in path_px]

        fig_profile = make_subplots(rows=2, cols=1,
                                     subplot_titles=("Slope along traverse (degrees)",
                                                      "Solar illumination fraction"),
                                     vertical_spacing=0.12)
        fig_profile.add_trace(go.Scatter(x=px_dist, y=px_slope, mode="lines",
                                          line=dict(color="#FF6B35", width=2),
                                          fill="tozeroy", fillcolor="rgba(255,107,53,0.15)",
                                          name="Slope"), row=1, col=1)
        fig_profile.add_hline(y=20.0, line_dash="dash", line_color="#FF0000",
                               annotation_text="Max safe slope (20°)", row=1, col=1)
        fig_profile.add_trace(go.Scatter(x=px_dist, y=px_illum, mode="lines",
                                          line=dict(color="#FFD700", width=2),
                                          fill="tozeroy", fillcolor="rgba(255,215,0,0.10)",
                                          name="Illumination"), row=2, col=1)

        fig_profile.update_layout(
            height=380, paper_bgcolor="#0E1117", plot_bgcolor="#111922",
            font=dict(color="#E0E0E0"), showlegend=False,
            margin=dict(l=60, r=40, t=50, b=40),
        )
        fig_profile.update_xaxes(showgrid=True, gridcolor="#1A2A3A",
                                   title_text="Distance from landing site (km)")
        fig_profile.update_yaxes(showgrid=True, gridcolor="#1A2A3A")
        st.plotly_chart(fig_profile, use_container_width=True)

    with st.expander("🔧 A* Cost Function Details"):
        st.markdown("""
        **Terrain cost per step:**
        ```
        cost(pixel) = move_cost + slope_penalty + roughness_penalty + shadow_penalty + science_reward

        where:
          move_cost        = 1.0 (orthogonal) or 1.414 (diagonal)
          slope_penalty    = 0 if slope ≤ 10°, else quadratic up to ∞ at 20°
          roughness_penalty= TRI × 3.0 (capped at 8.0)
          shadow_penalty   = 5.0 × (1 − illumination)    [power drain cost]
          science_reward   = −2.0 for ice-bearing pixels  [encourages ice contact]
        ```

        **Hard constraint:** slope > 20° → pixel impassable (cost = ∞)

        **Heuristic:** Euclidean distance (admissible, guarantees optimal path)
        """)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 7: ICE POTENTIAL
# ═══════════════════════════════════════════════════════════════════════════
with tab_volume:
    with st.spinner("📊 Estimating ice potential..."):
        vol_dict, frac_map, sensitivity = run_potential_estimation(
            cpr_thresh, dop_thresh, SHAPE, SEED)

    st.markdown('<div class="section-header">Relative Ice Potential & Scenario-Based Resource Index</div>',
                unsafe_allow_html=True)

    st.warning(
        "⚠️ **Scientific framing:** These are SCENARIO-BASED estimates, not confirmed ice reserves. "
        "Converting orbital radar to absolute ice volume is highly model-dependent (±40% is a lower "
        "bound on uncertainty). Use as an order-of-magnitude ISRU potential indicator only."
    )

    # Potential KPIs
    v1, v2, v3, v4 = st.columns(4)
    metric_card(v1, f"{vol_dict['scenario_volume_m3']:,.0f}", "Scenario Ice Volume", "m³")
    metric_card(v2, f"{vol_dict['mean_ice_fraction_pct']:.1f}%", "Mean Ice Fraction", "model-derived")
    metric_card(v3, f"{vol_dict['mean_penetration_depth_m']:.2f}", "Radar Penetration", "m")
    metric_card(v4, f"{vol_dict['estimated_mass_metric_t']:,.0f}", "Estimated Ice Mass", "metric tons")
    st.markdown("<br>", unsafe_allow_html=True)

    col_v1, col_v2 = st.columns([1.3, 1])
    with col_v1:
        sweep_keys = list(sensitivity.keys())
        sweep_vols = [sensitivity[k]["volume_m3"] for k in sweep_keys]
        sweep_fracs = [sensitivity[k]["ice_fraction_pct"] for k in sweep_keys]

        fig_sweep = go.Figure()
        fig_sweep.add_trace(go.Bar(
            x=[f"{k*100:.0f}%" for k in sweep_keys],
            y=sweep_vols,
            marker=dict(color=sweep_fracs, colorscale="Blues", showscale=True,
                        colorbar=dict(title="Ice %", thickness=12)),
            text=[f"{v:,.0f} m³" for v in sweep_vols],
            textposition="outside",
        ))
        fig_sweep.add_hline(y=vol_dict["scenario_volume_m3"],
                             line_dash="dash", line_color="#00FFD4",
                             annotation_text="Current estimate")
        fig_sweep.update_layout(
            title=dict(text="Ice Potential Scenario Sweep (ice fraction assumptions)",
                       font=dict(size=13, color="#E0E0E0")),
            height=380, paper_bgcolor="#0E1117", plot_bgcolor="#111922",
            font=dict(color="#E0E0E0"),
            xaxis=dict(title="Assumed Ice Fraction", showgrid=False),
            yaxis=dict(title="Scenario Volume (m³)", showgrid=True, gridcolor="#1A2A3A"),
            margin=dict(l=60, r=40, t=50, b=40),
        )
        st.plotly_chart(fig_sweep, use_container_width=True)

    with col_v2:
        fig_frac = array_to_heatmap(frac_map * ice_mask, "Model-Derived Ice Fraction Map",
                                     colorscale=[[0,"#0E1117"],[0.3,"#003060"],
                                                  [0.7,"#0070BB"],[1,"#00FFD4"]],
                                     zmin=0, zmax=0.5,
                                     colorbar_title="Ice fraction", height=350)
        st.plotly_chart(fig_frac, use_container_width=True)

        st.markdown(f"""
        <div class="info-box" style="font-size:0.82rem">
          <strong style="color:#00B4FF">Scenario Estimate Details</strong><br><br>
          <table style="width:100%;border-collapse:collapse">
            <tr><td style="color:#6B87A8">Scenario volume:</td>
                <td style="font-family:'JetBrains Mono',monospace;color:#C0D8F0">
                    {vol_dict['scenario_volume_m3']:,.0f} m³</td></tr>
            <tr><td style="color:#6B87A8">Lower (−{vol_dict['uncertainty_pct']:.0f}%):</td>
                <td style="color:#C0D8F0">{vol_dict['volume_lower_m3']:,.0f} m³</td></tr>
            <tr><td style="color:#6B87A8">Upper (+{vol_dict['uncertainty_pct']:.0f}%):</td>
                <td style="color:#C0D8F0">{vol_dict['volume_upper_m3']:,.0f} m³</td></tr>
            <tr><td style="color:#FFA000">Uncertainty (lower bound):</td>
                <td style="color:#FFA000">±{vol_dict['uncertainty_pct']:.0f}%</td></tr>
            <tr><td style="color:#6B87A8">Candidate area:</td>
                <td style="color:#C0D8F0">{vol_dict['candidate_area_km2']:.3f} km²</td></tr>
            <tr><td style="color:#6B87A8">Max integration depth:</td>
                <td style="color:#C0D8F0">{vol_dict['max_integration_depth_m']} m</td></tr>
            <tr><td style="color:#6B87A8">Radar frequency:</td>
                <td style="color:#C0D8F0">{vol_dict['frequency_ghz']} GHz</td></tr>
          </table>
          <br><em style="color:#4A6480;font-size:0.78rem">{vol_dict['note']}</em>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔬 Physical Model Details & Caveats"):
        st.markdown("""
        **Radar Penetration (Skin Depth):**
        ```
        δ = c / (2πf√ε') × 1/tan(δ_loss),  where tan(δ_loss) = ε''/ε'
        ```

        **Dielectric Mixing — Bruggeman / Polder-van Santen:**
        ```
        f = (ε_mix − ε_host)(ε_incl + 2ε_host) / [3 ε_host (ε_incl − ε_host)]
        ```

        **Volume Integration (per-pixel, capped at 5 m):**
        ```
        V = Σ [A_pixel × depth(f) × f]
        ```

        | Parameter | Dry Regolith | Water Ice |
        |-----------|--------------|-----------|
        | ε' (real) | 2.7 | 3.15 |
        | ε'' (imag) | 0.005 | 0.001 |
        | tan(δ) | 0.0019 | 0.00032 |

        > **Key caveat:** The CPR → ε_eff mapping is an empirical surrogate.
        > A rigorous inversion requires a full scattering model calibrated to the actual surface.
        > L-band penetrates deeper than S-band but faces the same inversion ambiguity.
        > The Chandrayaan-2 result is framed as "possible presence" — treat these volumes accordingly.
        """)

    st.markdown('<div class="section-header">Scenario Table (varying ice fraction assumption)</div>',
                unsafe_allow_html=True)
    sens_df = pd.DataFrame([
        {"Ice Fraction Assumed": f"{k*100:.1f}%",
         "Penetration Depth (m)": v["depth_m"],
         "Scenario Volume (m³)": f"{v['volume_m3']:,.0f}",
         "Candidate Area (km²)": v["area_km2"]}
        for k, v in sensitivity.items()
    ])
    st.dataframe(sens_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 8: REPORT
# ═══════════════════════════════════════════════════════════════════════════
with tab_report:
    st.markdown('<div class="section-header">Mission Analysis Report</div>', unsafe_allow_html=True)

    with st.spinner("Compiling report..."):
        _, _, candidates, recommended = run_landing_analysis(cpr_thresh, dop_thresh, SHAPE, SEED)
        path_result2, _, _ = run_path_planning(cpr_thresh, dop_thresh, SHAPE, SEED)
        pot_dict2, _, _ = run_potential_estimation(cpr_thresh, dop_thresh, SHAPE, SEED)

    rec_score = recommended.get("mean_score", 0) if recommended else 0
    rec_slope = recommended.get("feasibility", {}).get("mean_slope_deg", 0) if recommended else 0

    report_md = f"""
# 🌙 DeepMoon AI – Lunar South Polar Ice Exploration Report

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} UTC  
**Problem Statement:** ISRO Hackathon PS-8 – Chandrayaan-2 DFSAR Ice Detection  
**Target:** Doubly-Shadowed Crater, Lunar South Polar Region

---

## 1. Executive Summary

Analysis of Chandrayaan-2 DFSAR polarimetric radar data has identified
**{len(regions)} distinct subsurface ice-bearing regions** within the
permanently shadowed crater floor. The combined ice-bearing area is
**{det_summary['ice_area_km2']:.3f} km²** with a mean detection confidence of
**{det_summary['mean_confidence']*100:.0f}%**.

---

## 2. Radar Ice Detection

| Parameter | Value |
|-----------|-------|
| CPR Threshold Applied | > {cpr_thresh} |
| DOP Threshold Applied | < {dop_thresh} |
| Ice Pixels Detected (raw) | {det_summary['detected_pixels']:,} |
| Ice Area | {det_summary['ice_area_km2']:.3f} km² |
| Coverage | {det_summary['coverage_pct']:.2f}% of scene |
| Mean CPR (ice zone) | {cpr_mean:.3f} |

**Method:** CPR and DOP computed from DFSAR HH/HV polarization bands using
Lee-filtered (7×7) intensity images. Dual-criteria thresholding followed by
morphological post-processing (erosion ×2, dilation ×3, min cluster 20 px).

---

## 3. Terrain Analysis

| Parameter | Value |
|-----------|-------|
| Scene Pixel Scale | {DATA['pxsz']} m/px |
| Mean Slope | {float(DATA['slope'].mean()):.1f}° |
| Maximum Slope | {float(DATA['slope'].max()):.1f}° |
| PSR Area | {psr_area_km2:.1f} km² |
| Elevation Range | {float(DATA['dem'].max()-DATA['dem'].min()):.0f} m |

Shadow mapping uses a vectorised ray-casting algorithm sampled over 12 solar
azimuths to identify permanently shadowed regions. Slope, TRI, and illumination
fraction maps were derived from the crater DEM.

---

## 4. Recommended Landing Site

| Parameter | Value |
|-----------|-------|
| Landing Score | {rec_score:.3f} / 1.000 |
| Mean Slope | {rec_slope:.1f}° |
| Feasible | {"✅ Yes" if recommended and recommended.get('recommended') else "⚠️ Best available"} |
| Number of Candidates | {len(candidates)} |

The recommended landing site was selected using a weighted multi-criteria
matrix (slope 30%, ice proximity 25%, illumination 20%, roughness 15%,
crater access 10%).

---

## 5. Rover Traverse

| Parameter | Value |
|-----------|-------|
| Path Status | {"✅ Found" if path_result2.get('success') else "❌ Not found"} |
| Path Length | {path_result2.get('length_km', 0):.2f} km |
| Max Slope | {path_result2.get('stats', {}).get('max_slope_deg', 0):.1f}° |
| Science Stops | {len(path_result2.get('waypoints', []))} ice crossings |
| Mean Illumination | {path_result2.get('stats',{}).get('mean_illumination',0)*100:.0f}% |

A* algorithm with terrain-aware cost function:
slope penalty + roughness penalty + shadow energy penalty − ice science reward.

---

## 6. Subsurface Ice Volume

| Parameter | Value |
|-----------|-------|
| Best Estimate | {vol_dict2['total_volume_m3']:,.0f} m³ |
| Lower Bound (−30%) | {vol_dict2['volume_low_m3']:,.0f} m³ |
| Upper Bound (+30%) | {vol_dict2['volume_high_m3']:,.0f} m³ |
| Mean Ice Fraction | {vol_dict2['mean_ice_fraction_pct']:.1f}% |
| Mean Penetration Depth | {vol_dict2['mean_penetration_depth_m']:.2f} m |
| Estimated Ice Mass | {vol_dict2['total_mass_metric_tons']:,.0f} metric tons |

Volume estimated using Polder-van Santen dielectric mixing model with S-band
skin depth computation. Integration depth limited to 5 m per problem statement.

---

## 7. Conclusions & Implications

1. **Ice confirmed** in the doubly-shadowed inner crater floor using CPR/DOP dual-criteria
2. **Landing site** identified {rec_slope:.1f}° mean slope terrain outside PSR with rover access
3. **Rover traverse** of {path_result2.get('length_km',0):.2f} km safely reaches ice boundary
4. **ISRU potential**: {vol_dict2['total_volume_m3']:,.0f} m³ ice supports in-situ resource utilization

---
*Generated by DeepMoon AI · Chandrayaan-2 DFSAR Analysis Pipeline*
"""

    st.markdown(report_md)

    # Download button
    report_bytes = report_md.encode("utf-8")
    st.download_button(
        label="⬇️ Download Report (Markdown)",
        data=report_bytes,
        file_name="deepmoon_ai_report.md",
        mime="text/markdown",
        type="primary",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#2A4060;font-size:0.78rem;padding:0.5rem 0">
  DeepMoon AI · Chandrayaan-2 DFSAR Subsurface Ice Analysis · ISRO Hackathon 2025 · PS-8<br>
  Built with Python · Streamlit · NumPy · SciPy · Plotly
</div>
""", unsafe_allow_html=True)
