# DeepMoon AI — Lunar Subsurface Ice Detection & Exploration Planning

> Detection and Characterization of Subsurface Ice in Lunar South Polar Regions  
> Using Chandrayaan‑2 Radar and Imagery Data

---

## 🚀 Quick Start

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv

# 2. Activate it
# Windows (CMD)
.venv\Scripts\activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run app.py
# or if streamlit is not in PATH:
python -m streamlit run app.py
```

The dashboard will open at **http://localhost:8501** in your browser.

---

## 🏗️ Project Structure

```
DeepMoon AI/
├── app.py                      ← Streamlit dashboard (main entry point)
├── requirements.txt            ← Python dependencies
├── README.md                   ← This file
│
├── modules/
│   ├── __init__.py
│   ├── radar_processing.py     ← CPR, DOP, σ°, Lee filter, synthetic data
│   ├── ice_detection.py        ← Threshold detection, morphological cleaning
│   ├── terrain_analysis.py     ← Slope, TRI, PSR mask, illumination
│   ├── landing_site.py         ← Multi‑criteria landing site scoring
│   ├── rover_path.py           ← A* terrain‑aware path planner
│   ├── ice_volume.py           ← Dielectric mixing, volume estimation
│   └── visualization.py        ← Plotly charts & heatmaps
│
├── data/
│   ├── sample/                 ← Bundled synthetic data (auto‑generated)
│   └── uploads/                ← Place your real DFSAR GeoTIFFs here
│
└── notebooks/
    └── analysis_walkthrough.ipynb
```

---

## 📡 Methodology

### 1. Radar Processing (DFSAR)

| Product | Formula | Significance |
|---------|---------|--------------|
| CPR | `σ°_HV / σ°_HH` | Volume scattering anomaly from ice |
| DOP | `√(1 − 4·det(T)/tr(T)²)` | Polarization state randomization |
| σ° | `DN² / sin(θ)` (dB) | Calibrated backscatter |

**Lee speckle filter** (7×7 window) is applied before ratio computation.

### 2. Ice Detection

**Dual‑criteria thresholds** (ISRO/literature validated):
```
Ice candidate = (CPR > 1.0) AND (DOP < 0.13) AND (pixel ∈ PSR)
```

Post‑processing:
- Morphological erosion (×2) → removes speckle false positives
- Morphological dilation (×3) → restores true cluster extent  
- Minimum cluster size: 20 pixels
- Confidence score = geometric mean of sigmoid‑mapped CPR & DOP

### 3. Terrain Analysis

- **Slope map**: Central‑difference gradient on DEM (Sobel kernel)
- **TRI**: Terrain Ruggedness Index (8‑neighbour mean absolute deviation)
- **PSR mask**: Shadow ray‑casting over 12 solar azimuths; PSR = always shadowed
- **Illumination fraction**: 24‑step azimuth sampling

### 4. Landing Site Selection

Multi‑criteria weighted scoring:

| Criterion | Weight |
|-----------|--------|
| Slope ≤ 15° | 30% |
| Ice proximity | 25% |
| Solar illumination | 20% |
| Surface roughness | 15% |
| Crater access | 10% |

### 5. Rover Path Planning (A*)

**Cost function per pixel:**
```
f = move_cost + slope_penalty + roughness_penalty + shadow_penalty − science_bonus
```

Hard constraint: slope > 20° → impassable.  
Solar power zones are preferred; ice pixels are rewarded as science stops.

### 6. Ice Volume Estimation

```
V_ice = Σ [A_pixel × penetration_depth × ice_fraction]
```

- **Penetration depth**: Skin depth `δ = λ / (4π√ε' tan δ_loss)` at 2.5 GHz
- **Ice fraction**: Polder‑van Santen Bruggeman mixing model  
  `f = (ε_mix − ε_host)(ε_ice + 2ε_host) / [3ε_host(ε_ice − ε_host)]`
- **Integration depth**: Capped at 5 m (per problem statement)
- **Uncertainty**: ±30% (typical for radar‑based ice concentration)

---

## 🗂️ Using Real DFSAR Data

1. Place your DFSAR GeoTIFF files in `data/uploads/`
2. In the dashboard sidebar, use the **“Upload DFSAR Data”** widget (or the pipeline will automatically scan `data/uploads/`).
3. The system will detect band type (`HH`, `HV`, `VV`) from the filename.

**Expected file format:**
- Single‑band GeoTIFF
- Intensity or amplitude values (linear scale, **not** in dB)
- Naming convention: `*_HH.tif`, `*_HV.tif`, `*_VV.tif` (case‑insensitive)

> **Note**: If no real data is provided, the pipeline automatically generates synthetic sample data on the first run (stored in `data/sample/`) so you can test the dashboard immediately.

---

## 📊 Dashboard Tabs

| Tab | Contents |
|-----|---------|
| 🌑 Overview | Mission context, 3D DEM, workflow status |
| 📡 Radar Analysis | CPR/DOP maps, σ° map, histograms |
| 🧊 Ice Detection | Ice probability map, region statistics |
| 🏔️ Terrain | DEM, slope, PSR, illumination maps |
| 🛬 Landing Site | Scored candidates, recommended site |
| 🤖 Rover Path | A* traverse with slope profile |
| 📦 Ice Volume | Volume estimate + sensitivity analysis |
| 📋 Report | Auto‑generated downloadable report |

---

## 📦 Dependencies

```
streamlit ≥ 1.35     numpy ≥ 1.26      scipy ≥ 1.13
rasterio ≥ 1.3       plotly ≥ 5.22     folium ≥ 0.16
scikit‑image ≥ 0.23  scikit‑learn ≥ 1.4  networkx ≥ 3.3
matplotlib ≥ 3.8     pandas ≥ 2.2      fpdf2 ≥ 2.7
```

> **Python version**: 3.9 or later is recommended.

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **`streamlit: command not found`** | Use `python -m streamlit run app.py` instead. |
| **`ModuleNotFoundError`** | Check that your virtual environment is active and all packages are installed (`pip install -r requirements.txt`). |
| **No data appears** | Ensure your GeoTIFFs follow the naming convention or upload them via the sidebar. Synthetic data will be generated if no files are found. |
| **Permission errors on Windows** | Run PowerShell as Administrator or use CMD. For activation, try `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before running the activation script. |
| **Memory errors with large GeoTIFFs** | The dashboard downsamples large rasters automatically. You can also reduce the image size in the sidebar settings. |

---

## 📚 References

1. Chauhan et al. (2021) – *Chandrayaan‑2 DFSAR observations of the lunar polar regions*, **Icarus**
2. Kumar et al. (2023) – *Polarimetric indicators of lunar subsurface ice*
3. Nozette et al. (2001) – *Integration of lunar polar remote‑sensing datasets*, **JGR**
4. Black et al. (2001) – *Anomalous radar backscatter from the lunar south pole*, **GRL**
5. Ulaby, Moore & Fung (1986) – *Microwave Remote Sensing*, Vol. III
6. Polder & van Santen (1946) – *The effective permeability of mixtures*

---

## Gallery
<img width="1847" height="868" alt="Screenshot 2026-06-30 210440" src="https://github.com/user-attachments/assets/d0062688-ecec-43b7-b952-7f30e3850eb1" />
<img width="1626" height="822" alt="Screenshot 2026-06-30 210556" src="https://github.com/user-attachments/assets/cb37006e-656f-45f8-9ac0-76421accc07e" />
<img width="1662" height="786" alt="Screenshot 2026-06-30 210642" src="https://github.com/user-attachments/assets/38702498-0f45-4299-812d-7c5d88a550f0" />
<img width="1673" height="811" alt="Screenshot 2026-06-30 210525" src="https://github.com/user-attachments/assets/f9758bc3-fbcd-4f80-b188-2a75a5da9ab8" />

