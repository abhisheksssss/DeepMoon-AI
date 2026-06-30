"""
visualization.py
----------------
Plotly / Matplotlib rendering helpers for the DeepMoon AI dashboard.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import Ellipse


# ---------------------------------------------------------------------------
# Color Scales
# ---------------------------------------------------------------------------

COLORSCALE_CPR   = "Plasma"
COLORSCALE_DOP   = "Viridis_r"
COLORSCALE_ELEV  = "Greys"
COLORSCALE_SLOPE = "Oranges"
COLORSCALE_ICE   = "Blues"

# Ice overlay color (semi-transparent cyan)
ICE_RGBA  = "rgba(0, 220, 255, 0.65)"
PATH_RGBA = "rgba(255, 165, 0, 1.0)"


# ---------------------------------------------------------------------------
# Core Heatmap Helper
# ---------------------------------------------------------------------------

def array_to_heatmap(data: np.ndarray,
                      title: str,
                      colorscale: str = "Viridis",
                      zmin: float = None,
                      zmax: float = None,
                      colorbar_title: str = "",
                      height: int = 500) -> go.Figure:
    """Create a Plotly heatmap from a 2-D numpy array."""
    fig = go.Figure(go.Heatmap(
        z=data,
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        colorbar=dict(title=colorbar_title, thickness=15, len=0.8),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#E0E0E0")),
        height=height,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#E0E0E0"),
        margin=dict(l=60, r=40, t=50, b=40),
        xaxis=dict(showgrid=False, zeroline=False, title="Column (pixels)"),
        yaxis=dict(showgrid=False, zeroline=False, title="Row (pixels)",
                   autorange="reversed"),
    )
    return fig


# ---------------------------------------------------------------------------
# CPR / DOP Combined View
# ---------------------------------------------------------------------------

def plot_cpr_dop_combined(cpr: np.ndarray,
                           dop: np.ndarray,
                           ice_mask: np.ndarray | None = None,
                           height: int = 450) -> go.Figure:
    """Side-by-side CPR and DOP maps with optional ice contour overlay."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Circular Polarization Ratio (CPR)",
                                        "Degree of Polarization (DOP)"),
                        horizontal_spacing=0.06)

    fig.add_trace(go.Heatmap(z=cpr, colorscale=COLORSCALE_CPR,
                              zmin=0, zmax=3,
                              colorbar=dict(title="CPR", x=0.46, thickness=12, len=0.8)),
                  row=1, col=1)
    fig.add_trace(go.Heatmap(z=dop, colorscale=COLORSCALE_DOP,
                              zmin=0, zmax=1,
                              colorbar=dict(title="DOP", x=1.0, thickness=12, len=0.8)),
                  row=1, col=2)

    # Ice contour overlay
    if ice_mask is not None and ice_mask.any():
        contour_z = ice_mask.astype(float)
        for col_idx in [1, 2]:
            fig.add_trace(
                go.Contour(z=contour_z, showscale=False,
                           contours=dict(start=0.5, end=0.5, size=0),
                           line=dict(color="#00FFFF", width=2),
                           name="Ice boundary"),
                row=1, col=col_idx)

    fig.update_layout(
        height=height,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#E0E0E0"),
        title=dict(text="DFSAR Polarimetric Parameters", font=dict(size=15, color="#E0E0E0")),
        showlegend=False,
        margin=dict(l=50, r=60, t=60, b=40),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False, autorange="reversed")
    return fig


# ---------------------------------------------------------------------------
# DEM 3-D Surface Plot
# ---------------------------------------------------------------------------

def plot_dem_3d(dem: np.ndarray,
                pixel_size_m: float,
                title: str = "Crater DEM – 3D View",
                height: int = 550) -> go.Figure:
    """Interactive 3-D surface plot of the DEM."""
    rows, cols = dem.shape
    x = np.arange(cols) * pixel_size_m / 1000  # km
    y = np.arange(rows) * pixel_size_m / 1000  # km

    fig = go.Figure(go.Surface(
        z=dem, x=x, y=y,
        colorscale=COLORSCALE_ELEV,
        colorbar=dict(title="Elevation (m)", thickness=15),
        lighting=dict(ambient=0.6, diffuse=0.8,
                      specular=0.4, roughness=0.5),
        lightposition=dict(x=100, y=200, z=0),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#E0E0E0")),
        height=height,
        paper_bgcolor="#0E1117",
        scene=dict(
            bgcolor="#0E1117",
            xaxis=dict(title="East (km)", gridcolor="#333"),
            yaxis=dict(title="North (km)", gridcolor="#333"),
            zaxis=dict(title="Elevation (m)", gridcolor="#333"),
            camera=dict(eye=dict(x=1.5, y=-2.0, z=1.2)),
        ),
        font=dict(color="#E0E0E0"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ---------------------------------------------------------------------------
# Ice Probability Map
# ---------------------------------------------------------------------------

def plot_ice_confidence(confidence: np.ndarray,
                         ice_mask: np.ndarray,
                         crater_center: tuple = None,
                         height: int = 480) -> go.Figure:
    """Ice confidence map with detection boundary and crater marker."""
    fig = go.Figure()

    # Base confidence layer
    fig.add_trace(go.Heatmap(
        z=confidence,
        colorscale=[[0, "#0E1117"], [0.3, "#003060"], [0.6, "#0070BB"],
                    [0.8, "#00BFFF"], [1.0, "#FFFFFF"]],
        zmin=0, zmax=1,
        colorbar=dict(title="Ice Confidence", thickness=15),
        name="Confidence",
    ))

    # Ice mask contour
    if ice_mask is not None and ice_mask.any():
        fig.add_trace(go.Contour(
            z=ice_mask.astype(float),
            showscale=False,
            contours=dict(start=0.5, end=0.5, size=0),
            line=dict(color="#00FFE0", width=2.5),
            name="Ice boundary",
        ))

    # Crater centre marker
    if crater_center is not None:
        cx, cy = crater_center
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode="markers+text",
            marker=dict(symbol="x", size=14, color="#FF4500",
                        line=dict(width=2, color="#FF4500")),
            text=["Crater centre"], textposition="top center",
            name="Crater centre",
        ))

    fig.update_layout(
        title=dict(text="Subsurface Ice Probability Map",
                   font=dict(size=15, color="#E0E0E0")),
        height=height,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#E0E0E0"),
        margin=dict(l=60, r=40, t=50, b=40),
        xaxis=dict(showgrid=False, zeroline=False, title="Column (pixels)"),
        yaxis=dict(showgrid=False, zeroline=False, title="Row (pixels)",
                   autorange="reversed"),
    )
    return fig


# ---------------------------------------------------------------------------
# Terrain Safety + Landing Sites
# ---------------------------------------------------------------------------

def plot_landing_analysis(safety: np.ndarray,
                           slope: np.ndarray,
                           candidates: list[dict],
                           recommended: dict | None = None,
                           pixel_size_m: float = 20.0,
                           height: int = 500) -> go.Figure:
    """Safety map with landing candidate ellipses overlaid."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Terrain Safety Score", "Slope (degrees)"),
                        horizontal_spacing=0.06)

    fig.add_trace(go.Heatmap(z=safety, colorscale="RdYlGn",
                              zmin=0, zmax=1,
                              colorbar=dict(title="Safety", x=0.46,
                                            thickness=12, len=0.8)),
                  row=1, col=1)
    fig.add_trace(go.Heatmap(z=slope, colorscale="Hot_r",
                              zmin=0, zmax=35,
                              colorbar=dict(title="Slope °", x=1.0,
                                            thickness=12, len=0.8)),
                  row=1, col=2)

    # Candidate landing sites
    for i, cand in enumerate(candidates):
        cx, cy = cand["centroid_px"]
        is_rec = recommended and cand["id"] == recommended.get("id")
        color  = "#00FF88" if is_rec else "#FFD700"
        symbol = "star" if is_rec else "circle"
        label  = f"★ LS-{i+1}" if is_rec else f"LS-{i+1}"

        for col_idx in [1, 2]:
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy],
                mode="markers+text",
                marker=dict(symbol=symbol, size=14, color=color,
                            line=dict(width=2, color="white")),
                text=[label], textposition="top center",
                name=label,
                showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

    fig.update_layout(
        height=height,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#E0E0E0"),
        title=dict(text="Landing Site Analysis", font=dict(size=15, color="#E0E0E0")),
        margin=dict(l=50, r=60, t=60, b=40),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False, autorange="reversed")
    return fig


# ---------------------------------------------------------------------------
# Rover Path Plot
# ---------------------------------------------------------------------------

def plot_rover_path(dem: np.ndarray,
                    slope: np.ndarray,
                    ice_mask: np.ndarray,
                    path_px: list[tuple],
                    waypoints: list[tuple],
                    landing_px: tuple,
                    goal_px: tuple,
                    pixel_size_m: float = 20.0,
                    height: int = 520) -> go.Figure:
    """Terrain context map with rover path, waypoints, and key landmarks."""

    fig = go.Figure()

    # Background: slope with hill-shade
    fig.add_trace(go.Heatmap(
        z=slope, colorscale="Greys_r",
        zmin=0, zmax=30,
        colorbar=dict(title="Slope °", thickness=12),
        name="Slope (bg)",
        opacity=0.8,
    ))

    # Ice overlay
    if ice_mask is not None and ice_mask.any():
        ice_rgba = np.zeros((*ice_mask.shape, 4), dtype=np.uint8)
        ice_rgba[ice_mask] = [0, 210, 255, 160]
        rows, cols = ice_mask.shape
        fig.add_trace(go.Heatmap(
            z=ice_mask.astype(float),
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,210,255,0.5)"]],
            showscale=False, zmin=0, zmax=1, name="Ice",
        ))

    # Path line
    if path_px:
        px_cols = [p[0] for p in path_px]
        px_rows = [p[1] for p in path_px]
        fig.add_trace(go.Scatter(
            x=px_cols, y=px_rows,
            mode="lines",
            line=dict(color="#FFA500", width=3),
            name="Rover path",
        ))

    # Science waypoints (ice intersections)
    if waypoints:
        wp_cols = [w[0] for w in waypoints[:10]]  # max 10 for clarity
        wp_rows = [w[1] for w in waypoints[:10]]
        fig.add_trace(go.Scatter(
            x=wp_cols, y=wp_rows,
            mode="markers",
            marker=dict(symbol="diamond", size=8, color="#00FF88",
                        line=dict(width=1.5, color="white")),
            name="Science stop",
        ))

    # Landing site
    if landing_px:
        fig.add_trace(go.Scatter(
            x=[landing_px[0]], y=[landing_px[1]],
            mode="markers+text",
            marker=dict(symbol="star", size=18, color="#FFD700",
                        line=dict(width=2, color="white")),
            text=["Landing"], textposition="top right",
            name="Landing site",
        ))

    # Goal (crater rim / ice target)
    if goal_px:
        fig.add_trace(go.Scatter(
            x=[goal_px[0]], y=[goal_px[1]],
            mode="markers+text",
            marker=dict(symbol="x-open", size=16, color="#FF4500",
                        line=dict(width=3)),
            text=["Ice target"], textposition="bottom right",
            name="Ice target",
        ))

    fig.update_layout(
        title=dict(text="Rover Traverse Path – Terrain Overview",
                   font=dict(size=15, color="#E0E0E0")),
        height=height,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#E0E0E0"),
        margin=dict(l=60, r=40, t=55, b=40),
        xaxis=dict(showgrid=False, zeroline=False, title="Column (pixels)"),
        yaxis=dict(showgrid=False, zeroline=False, title="Row (pixels)",
                   autorange="reversed"),
        legend=dict(bgcolor="#1A1A2E", bordercolor="#444", borderwidth=1),
    )
    return fig


# ---------------------------------------------------------------------------
# Ice Volume Bar Chart + Sensitivity
# ---------------------------------------------------------------------------

def plot_volume_results(volume_dict: dict, sensitivity: dict) -> go.Figure:
    """Bar chart of volume estimate with uncertainty, plus sensitivity sweep."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Ice Volume Estimate", "Sensitivity: CPR vs Volume"),
                        horizontal_spacing=0.10)

    # Bar: best estimate + error bar
    best = volume_dict["total_volume_m3"]
    low  = volume_dict["volume_low_m3"]
    high = volume_dict["volume_high_m3"]

    fig.add_trace(go.Bar(
        x=["Estimated Ice Volume"],
        y=[best],
        error_y=dict(type="data", symmetric=False,
                     array=[high - best], arrayminus=[best - low]),
        marker=dict(color="#0099CC",
                    line=dict(color="#00CCFF", width=2)),
        text=[f"{best:,.0f} m³"],
        textposition="outside",
        name="Ice Volume",
    ), row=1, col=1)

    # Sensitivity sweep
    cpr_vals  = list(sensitivity.keys())
    vol_vals  = [sensitivity[k]["volume_m3"] for k in cpr_vals]
    frac_vals = [sensitivity[k]["ice_fraction"] for k in cpr_vals]

    fig.add_trace(go.Scatter(
        x=cpr_vals, y=vol_vals,
        mode="lines+markers",
        line=dict(color="#FF6B35", width=2),
        marker=dict(size=8, color=frac_vals, colorscale="Oranges"),
        name="Volume vs CPR",
    ), row=1, col=2)

    # Vertical line at CPR threshold
    fig.add_vline(x=1.0, line_dash="dash", line_color="#AAAAAA",
                  annotation_text="CPR = 1.0 threshold", row=1, col=2)

    fig.update_layout(
        height=420,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#E0E0E0"),
        title=dict(text="Subsurface Ice Volume Analysis",
                   font=dict(size=15, color="#E0E0E0")),
        showlegend=False,
        margin=dict(l=60, r=40, t=60, b=40),
    )
    fig.update_yaxes(showgrid=True, gridcolor="#222", zeroline=False,
                     title_text="Volume (m³)", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="#222", zeroline=False,
                     title_text="Volume (m³)", row=1, col=2)
    fig.update_xaxes(showgrid=False, zeroline=False,
                     title_text="CPR", row=1, col=2)
    return fig


# ---------------------------------------------------------------------------
# Polar PSR Illumination Wheel
# ---------------------------------------------------------------------------

def plot_illumination_wheel(illumination: np.ndarray,
                             height: int = 400) -> go.Figure:
    """Show mean illumination as a function of azimuth (polar plot)."""
    rows, cols = illumination.shape
    cx, cy = cols // 2, rows // 2
    azimuths = np.linspace(0, 360, 36, endpoint=False)
    illum_by_az = []

    for az in azimuths:
        az_rad = np.radians(az)
        # Sample illumination in a wedge from centre
        angles = np.linspace(-5, 5, 10)
        vals = []
        for daz in angles:
            ang = np.radians(az + daz)
            for r in range(10, min(rows, cols) // 2):
                rc = int(cy + r * np.cos(ang))
                cc = int(cx + r * np.sin(ang))
                if 0 <= rc < rows and 0 <= cc < cols:
                    vals.append(illumination[rc, cc])
        illum_by_az.append(np.mean(vals) if vals else 0)

    fig = go.Figure(go.Barpolar(
        r=illum_by_az,
        theta=azimuths,
        width=[10] * len(azimuths),
        marker=dict(
            color=illum_by_az,
            colorscale="Solar",
            showscale=True,
            colorbar=dict(title="Fraction lit"),
        ),
    ))
    fig.update_layout(
        title=dict(text="Solar Illumination by Azimuth",
                   font=dict(size=14, color="#E0E0E0")),
        height=height,
        paper_bgcolor="#0E1117",
        polar=dict(bgcolor="#0E1117",
                   radialaxis=dict(visible=True, range=[0, 1],
                                   color="#888"),
                   angularaxis=dict(color="#888")),
        font=dict(color="#E0E0E0"),
        margin=dict(l=40, r=40, t=55, b=40),
    )
    return fig
