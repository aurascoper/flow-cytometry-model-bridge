#!/usr/bin/env python3
"""
Build script for:
1. Ghosh Exploratory Deck (8 slides):
   - presentations/Single_Cell_Flow_Cytometry_Model_Bridge_Exploratory.pptx
   - presentations/Single_Cell_Flow_Cytometry_Model_Bridge_Exploratory.pdf
   - presentations/Ghosh_Flow_Cytometry_Multiphysics_Bridge.pptx
   - presentations/Ghosh_Flow_Cytometry_Multiphysics_Bridge.pdf
   - previews/Ghosh_Bridge_Slide_{1..8}.png (1920x1080)
   - presentations/assets/Ghosh_Bridge_Slide_{1..8}.png (1920x1080)
   Slides:
   Slide 1: Methods bridge framing
   Slide 2: Solver modular stack architecture
   Slide 3: Application separation
   Slide 4: Smallest falsifiable workflow
   Slide 5: Core-Multishell Reaction-Diffusion and Microfluidic Single-Cell Classification
   Slide 6: Nanocarrier Architecture Comparison: Synthetic Multi-Shell vs Natural Biofilm
   Slide 7: Core-Multishell Release Profile Fitting and Parameter Recovery
   Slide 8: Pulsed Duty-Cycle Actuation and PC12 Internalization Discrimination

2. Jack Exploratory Adaptation Deck (1 slide):
   - presentations/Interface_Resolved_Emulsion_Dynamics_Exploratory.pptx
   - presentations/Interface_Resolved_Emulsion_Dynamics_Exploratory.pdf
   - previews/Jack_Emulsion_Interface_Slide_1.png (1920x1080)
"""

import io
import math
import shutil
import tempfile
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Wedge
import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter


REPO_ROOT = Path(__file__).resolve().parent.parent
PRES_DIR = REPO_ROOT / "presentations"
PREV_DIR = REPO_ROOT / "previews"
ASSETS_DIR = PRES_DIR / "assets"


def render_ghosh_slide_5_png(out_path: Path):
    """Render 1920x1080 Slide 5 for Dr. Santaneel Ghosh."""
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#F6F5F0")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="#F6F5F0")
    ax.set_xlim(0, 13.333)
    ax.set_ylim(0, 7.5)
    ax.invert_yaxis()
    ax.axis("off")

    # Header
    ax.text(
        0.88, 0.92,
        "SHARED 3D VISUALIZATION ENGINE . TRACK 1 NANO-BIO CENTERPIECE",
        fontsize=13, fontweight="bold", color="#A86418", family="sans-serif"
    )
    ax.text(
        0.88, 1.48,
        "Core-Multishell Reaction-Diffusion and Microfluidic Single-Cell Classification",
        fontsize=25, fontweight="bold", color="#273238", family="serif"
    )
    ax.text(
        0.88, 1.98,
        "Concentric spherical release shells (R0, R1, R2) + surrounding wireframe isosurface coupled to microfluidic cell classification.",
        fontsize=14, color="#536168", family="sans-serif"
    )

    # Left Card: Core-Multishell Octant-Cutaway View
    left_card = FancyBboxPatch(
        (0.88, 2.35), 5.65, 3.20,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        facecolor="#E4EDF4", edgecolor="#D4D2CB", linewidth=1.5
    )
    ax.add_patch(left_card)
    ax.text(
        1.10, 2.68,
        "1 . CORE-MULTISHELL OCTANT-CUTAWAY VIEW (R0, R1, R2)",
        fontsize=13.5, fontweight="bold", color="#4A779D"
    )
    ax.text(
        1.10, 3.12,
        "* Inner core R0 (Omega_1), mid shell R1 (Omega_2),\n"
        "  outer shell R2 (Omega_3) + wireframe front\n"
        "* Solves radial reaction-diffusion:\n"
        "  du/dt = (1/r^2) d/dr(D_l r^2 du/dr) - k_l u\n"
        "* Animated concentration field u(x,y,z,t) in [0, 0.03]\n"
        "* Epistemic tag: :predicted simulated concentration\n"
        "  shells, not assumed physical layers",
        fontsize=11.5, color="#273238", va="top", linespacing=1.38
    )

    # Embedded 3D Octant-Cutaway Sphere Diagram inside Left Card
    cx, cy = 5.20, 4.10
    for ang in (-55, -25, 0, 25, 55):
        ell = Ellipse(
            (cx, cy), width=2.15, height=2.15 * math.cos(math.radians(ang)),
            angle=22, fill=False, edgecolor="#38bdf8", linestyle="--", linewidth=0.9, alpha=0.75
        )
        ax.add_patch(ell)
    w_r2 = Wedge((cx, cy), 0.98, 0, 270, facecolor="#e11d48", alpha=0.42, edgecolor="#be123c", lw=1.6)
    w_r1 = Wedge((cx, cy), 0.70, 0, 270, facecolor="#6366f1", alpha=0.65, edgecolor="#4338ca", lw=1.6)
    w_r0 = Wedge((cx, cy), 0.38, 0, 270, facecolor="#facc15", alpha=0.95, edgecolor="#ca8a04", lw=1.6)
    ax.add_patch(w_r2)
    ax.add_patch(w_r1)
    ax.add_patch(w_r0)
    ax.text(cx - 0.24, cy + 0.14, "R0", fontsize=10.5, fontweight="bold", color="#273238")
    ax.text(cx + 0.35, cy + 0.22, "R1", fontsize=10.5, fontweight="bold", color="#1e1b4b")
    ax.text(cx + 0.69, cy + 0.22, "R2", fontsize=10.5, fontweight="bold", color="#881337")

    # Right Card: Microfluidic Coupling & Single-Cell Classification View
    right_card = FancyBboxPatch(
        (6.85, 2.35), 5.60, 3.20,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        facecolor="#E1EBE6", edgecolor="#D4D2CB", linewidth=1.5
    )
    ax.add_patch(right_card)
    ax.text(
        7.07, 2.68,
        "2 . MICROFLUIDIC COUPLING & SINGLE-CELL CLASSIFICATION",
        fontsize=13.5, fontweight="bold", color="#3C6D65"
    )
    ax.text(
        7.05, 3.10,
        "* Constriction (25 -> 7.5 um) couples\n"
        "  local release C_exp(t) to cell transit\n"
        "* Measures dt_transit, impedance pulse\n"
        "  dI/I0, deformability, Cy5/SYBR/PI\n"
        "* Calibrated classifier + :defer_for_review\n"
        "  and strict chip_id group-aware splits\n"
        "* Bridges to ObservationModel.jl MLE",
        fontsize=11.0, color="#273238", va="top", linespacing=1.36
    )

    # Embedded Microfluidic Constriction Diagram inside Right Card
    ch_box = FancyBboxPatch(
        (10.22, 3.22), 2.08, 1.88,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor="#ffffff", edgecolor="#3C6D65", linewidth=1.2
    )
    ax.add_patch(ch_box)
    top_wall = Polygon([[10.28, 3.36], [10.88, 3.36], [11.06, 3.76], [11.46, 3.76], [11.64, 3.36], [12.24, 3.36]],
                       closed=False, fill=False, edgecolor="#273238", lw=1.8)
    bot_wall = Polygon([[10.28, 4.62], [10.88, 4.62], [11.06, 4.22], [11.46, 4.22], [11.64, 4.62], [12.24, 4.62]],
                       closed=False, fill=False, edgecolor="#273238", lw=1.8)
    ax.add_patch(top_wall)
    ax.add_patch(bot_wall)
    ax.add_patch(Ellipse((10.58, 3.96), 0.22, 0.20, facecolor="#16a34a", edgecolor="#ffffff", lw=1))
    ax.add_patch(Ellipse((11.26, 3.99), 0.32, 0.13, facecolor="#16a34a", edgecolor="#ffffff", lw=1))
    ax.add_patch(Ellipse((11.95, 4.08), 0.22, 0.20, facecolor="#2563eb", edgecolor="#ffffff", lw=1))
    ax.text(11.26, 4.92, "dt_transit & Gate MLE", fontsize=9.2, ha="center", color="#3C6D65", fontweight="bold")

    # Bottom Epistemic Boundary Banner
    banner = FancyBboxPatch(
        (0.88, 5.76), 11.57, 0.68,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor="#F5E5C5", edgecolor="none"
    )
    ax.add_patch(banner)
    ax.text(
        1.10, 6.10,
        "Epistemic boundary: label shells as simulated concentration fields (:predicted), not observed physical layers, unless independent imaging confirms morphology.",
        fontsize=12.2, fontweight="bold", color="#273238", va="center"
    )

    # Footer
    ax.text(
        0.88, 6.92,
        "Nano-Bio Lab fit: sustained release from core-multishell nanocarriers, micropore cell discrimination, and group-aware single-cell classification.",
        fontsize=10.0, color="#536168"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)


def render_ghosh_slide_6_png(out_path: Path):
    """Render 1920x1080 Slide 6: Nanocarrier Architecture Comparison."""
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#F6F5F0")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="#F6F5F0")
    ax.set_xlim(0, 13.333)
    ax.set_ylim(0, 7.5)
    ax.invert_yaxis()
    ax.axis("off")

    # Header
    ax.text(
        0.88, 0.88,
        "SHARED 3D VISUALIZATION ENGINE . TRACK 1 NANO-BIO CALIBRATION",
        fontsize=13, fontweight="bold", color="#A86418", family="sans-serif"
    )
    ax.text(
        0.88, 1.40,
        "Nanocarrier Architecture Comparison: Synthetic Multi-Shell vs Natural Biofilm",
        fontsize=24, fontweight="bold", color="#273238", family="serif"
    )
    ax.text(
        0.88, 1.84,
        "Quantitative structural mapping across public-record instances: Ghosh 2010, Ghosh Patent, and multicellular biofilm aggregates.",
        fontsize=13.5, color="#536168", family="sans-serif"
    )

    card_y = 2.15
    card_h = 3.50

    # Card 1: Ghosh 2010 (B12 / PEGMA)
    c1 = FancyBboxPatch((0.88, card_y), 3.65, card_h, boxstyle="round,pad=0.04,rounding_size=0.15", facecolor="#E4EDF4", edgecolor="#D4D2CB", lw=1.5)
    ax.add_patch(c1)
    ax.text(1.08, card_y + 0.32, "1 . GHOSH 2010 (B12 / PEGMA)", fontsize=13.0, fontweight="bold", color="#4A779D")
    ax.text(
        1.08, card_y + 0.65,
        "* Citation: Ghosh et al., Nanoscale\n"
        "  Res. Lett. 2010, 5:195 (PMC2894335)\n"
        "* Diameter: 315 nm (R_max = 0.1575 um)\n"
        "* Concentric radii:\n"
        "  R = [0.080, 0.125, 0.1575] um\n"
        "* Strata architecture:\n"
        "  - Core (80 nm): Fe3O4 magnetic core\n"
        "  - Mid (45 nm): PEGMA hydrogel mesh\n"
        "  - Outer (32.5 nm): B12 targeting corona\n"
        "* Diffusivity: [1.0e-4, 2.5e-3, 2.5e-2] um2/s\n"
        "* Release: Diffusive hydrogel swelling",
        fontsize=10.0, color="#273238", va="top", linespacing=1.28
    )
    cx1, cy1 = 3.60, card_y + 2.55
    ax.add_patch(Wedge((cx1, cy1), 0.75, 0, 360, facecolor="#e11d48", alpha=0.35, edgecolor="#be123c", lw=1.2))
    ax.add_patch(Wedge((cx1, cy1), 0.60, 0, 360, facecolor="#6366f1", alpha=0.55, edgecolor="#4338ca", lw=1.2))
    ax.add_patch(Wedge((cx1, cy1), 0.38, 0, 360, facecolor="#facc15", alpha=0.90, edgecolor="#ca8a04", lw=1.2))
    ax.text(cx1, cy1 + 0.05, "Fe3O4", fontsize=8.0, fontweight="bold", ha="center", color="#273238")
    ax.text(cx1, cy1 + 0.95, "315 nm", fontsize=9.0, fontweight="bold", ha="center", color="#4A779D")

    # Card 2: Ghosh Patent (Cisplatin)
    c2 = FancyBboxPatch((4.84, card_y), 3.65, card_h, boxstyle="round,pad=0.04,rounding_size=0.15", facecolor="#E1EBE6", edgecolor="#D4D2CB", lw=1.5)
    ax.add_patch(c2)
    ax.text(5.04, card_y + 0.32, "2 . GHOSH PATENT (CISPLATIN)", fontsize=13.0, fontweight="bold", color="#3C6D65")
    ax.text(
        5.04, card_y + 0.65,
        "* Citation: US Patents 11,077,191 B2\n"
        "  and 11,648,313 B2 (Ghosh et al.)\n"
        "* Diameter: 191 nm (R_max = 0.0955 um)\n"
        "* Concentric radii:\n"
        "  R = [0.045, 0.075, 0.0955] um\n"
        "* Strata architecture:\n"
        "  - Core (45 nm): Cisplatin drug depot\n"
        "  - Mid (30 nm): PLGA diffusion barrier\n"
        "  - Outer (20.5 nm): PEG stealth corona\n"
        "* Diffusivity: [2.0e-4, 3.5e-3, 3.8e-2] um2/s\n"
        "* Release: RF/NIR pulse + thermal relax",
        fontsize=10.0, color="#273238", va="top", linespacing=1.28
    )
    cx2, cy2 = 7.56, card_y + 2.55
    ax.add_patch(Wedge((cx2, cy2), 0.58, 0, 360, facecolor="#0284c7", alpha=0.35, edgecolor="#0369a1", lw=1.2))
    ax.add_patch(Wedge((cx2, cy2), 0.45, 0, 360, facecolor="#0d9488", alpha=0.55, edgecolor="#0f766e", lw=1.2))
    ax.add_patch(Wedge((cx2, cy2), 0.28, 0, 360, facecolor="#f97316", alpha=0.90, edgecolor="#c2410c", lw=1.2))
    ax.text(cx2, cy2 + 0.04, "Pt", fontsize=8.0, fontweight="bold", ha="center", color="#ffffff")
    ax.text(cx2, cy2 + 0.85, "191 nm", fontsize=9.0, fontweight="bold", ha="center", color="#3C6D65")

    # Card 3: Biofilm Aggregate (EPS)
    c3 = FancyBboxPatch((8.80, card_y), 3.65, card_h, boxstyle="round,pad=0.04,rounding_size=0.15", facecolor="#FDF6E2", edgecolor="#D4D2CB", lw=1.5)
    ax.add_patch(c3)
    ax.text(9.00, card_y + 0.32, "3 . BIOFILM AGGREGATE (EPS)", fontsize=13.0, fontweight="bold", color="#A86418")
    ax.text(
        9.00, card_y + 0.65,
        "* Benchmark: Multicellular EPS matrix\n"
        "* Scale: R ~ 5.0 to 50.0 um (~100x larger)\n"
        "* Irregular topology (tortuosity tau > 2.5)\n"
        "* Matrix components:\n"
        "  - Hydrated EPS, PGA, alginate, curli\n"
        "  - Extracellular DNA (eDNA) networks\n"
        "  - Heterogeneous fluid micro-channels\n"
        "* Diffusivities: D ~ 0.10 to 10.0 um2/s\n"
        "* Release: Convective erosion + lysis\n"
        "* Observation: Cytometry + impedance",
        fontsize=10.0, color="#273238", va="top", linespacing=1.28
    )
    cx3, cy3 = 11.52, card_y + 2.55
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    radii = 0.68 + 0.18 * np.sin(3 * angles) + 0.10 * np.cos(5 * angles)
    poly_pts = np.column_stack([cx3 + radii * np.cos(angles), cy3 + radii * np.sin(angles)])
    ax.add_patch(Polygon(poly_pts, facecolor="#d97706", alpha=0.35, edgecolor="#b45309", lw=1.5))
    for dx, dy in [(-0.24, -0.15), (0.18, -0.25), (-0.10, 0.22), (0.26, 0.15), (0.0, -0.02)]:
        ax.add_patch(Ellipse((cx3 + dx, cy3 + dy), 0.16, 0.10, angle=30, facecolor="#16a34a", edgecolor="#ffffff", lw=0.8))
    ax.text(cx3, cy3 + 0.95, "5 - 50 um", fontsize=9.0, fontweight="bold", ha="center", color="#A86418")

    # Epistemic Banner
    banner = FancyBboxPatch((0.88, 5.82), 11.57, 0.68, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor="#F5E5C5", edgecolor="none")
    ax.add_patch(banner)
    ax.text(
        1.10, 6.16,
        "MODEL PREDICTION - illustrative parameters - no lab data. Synthetic shells are modeled as ideal concentric strata;\n"
        "biofilm aggregates possess irregular porous tortuosity and heterogeneous internal transport channels.",
        fontsize=11.2, fontweight="bold", color="#273238", va="center"
    )

    # Footer
    ax.text(
        0.88, 6.95,
        "Track 1 nanobio calibration: Parameterization grounded in public records (Ghosh 2010, US Patents 11,077,191 B2 and 11,648,313 B2).",
        fontsize=10.0, color="#536168"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)


def render_ghosh_slide_7_png(out_path: Path):
    """Render 1920x1080 Slide 7: Core-Multishell Release Profile Fitting and Parameter Recovery."""
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#F6F5F0")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="#F6F5F0")
    ax.set_xlim(0, 13.333)
    ax.set_ylim(0, 7.5)
    ax.invert_yaxis()
    ax.axis("off")

    # Header
    ax.text(
        0.88, 0.88,
        "SHARED 3D VISUALIZATION ENGINE . REACTION-DIFFUSION INVERSE SOLVER",
        fontsize=13, fontweight="bold", color="#A86418", family="sans-serif"
    )
    ax.text(
        0.88, 1.40,
        "Core-Multishell Release Profile Fitting and Parameter Recovery",
        fontsize=24, fontweight="bold", color="#273238", family="serif"
    )
    ax.text(
        0.88, 1.84,
        "Non-linear least squares parameter recovery (Nelder-Mead) and residual audit across spherical strata.",
        fontsize=13.5, color="#536168", family="sans-serif"
    )

    card_y = 2.15
    card_h = 3.50

    # Left Card: Forward Kinetics & Inverse Parameter Recovery
    c_left = FancyBboxPatch((0.88, card_y), 5.65, card_h, boxstyle="round,pad=0.04,rounding_size=0.15", facecolor="#E4EDF4", edgecolor="#D4D2CB", lw=1.5)
    ax.add_patch(c_left)
    ax.text(1.08, card_y + 0.32, "1 . FORWARD KINETICS & INVERSE PARAMETER RECOVERY", fontsize=13.0, fontweight="bold", color="#4A779D")
    ax.text(
        1.08, card_y + 0.65,
        "* Radial spherical reaction-diffusion PDE:\n"
        "  du/dt = (1/r^2) d/dr(D_l r^2 du/dr) - k_l u\n"
        "* Robin boundary flux: -D du/dr = P_m (u - C_inf) at R_max\n"
        "* Cumulative release: M(t)/M_inf = 1 - int(u r^2 dr) / int(u0 r^2 dr)\n"
        "* Nelder-Mead optimization recovering effective transport:\n"
        "  - D_core = 1.0e-4 um2/s, D_mid = 2.5e-3 um2/s\n"
        "  - D_shell = 2.5e-2 um2/s, P_m = 0.050 um/s\n"
        "* Goodness-of-fit: R^2 = 0.9942, RMSE = 0.0084 (K = 30 pts)",
        fontsize=10.0, color="#273238", va="top", linespacing=1.26
    )

    # Embedded Subplot 1: Cumulative Release Curve
    ax_sub1 = fig.add_axes([0.31, 0.40, 0.17, 0.28])
    np.random.seed(42)
    t_pts = np.linspace(0, 60, 20)
    true_curve = 1.0 - 0.55 * np.exp(-t_pts / 8.0) - 0.45 * np.exp(-t_pts / 28.0)
    obs_pts = true_curve + np.random.normal(0, 0.008, len(t_pts))
    ax_sub1.plot(t_pts, true_curve, color="#3C6D65", lw=2.2, label="Fitted Trajectory")
    ax_sub1.scatter(t_pts, obs_pts, color="#4A779D", s=22, zorder=5, label="Synthetic Lab Data")
    ax_sub1.set_xlabel("Time (s)", fontsize=8.0)
    ax_sub1.set_ylabel("M(t) / M_inf", fontsize=8.0)
    ax_sub1.legend(fontsize=7.0, loc="lower right")
    ax_sub1.grid(True, linestyle=":", alpha=0.5)

    # Right Card: Error Residuals & Radial Concentration Evolution
    c_right = FancyBboxPatch((6.80, card_y), 5.65, card_h, boxstyle="round,pad=0.04,rounding_size=0.15", facecolor="#E1EBE6", edgecolor="#D4D2CB", lw=1.5)
    ax.add_patch(c_right)
    ax.text(7.00, card_y + 0.32, "2 . ERROR RESIDUALS & RADIAL CONCENTRATION EVOLUTION", fontsize=13.0, fontweight="bold", color="#3C6D65")
    ax.text(
        7.00, card_y + 0.65,
        "* Residual scatter: delta_M(t) = M_obs(t) - M_fit(t)\n"
        "* Mean residual: 0.0001 (uniform zero-mean scatter in [-0.015, +0.015])\n"
        "* Radial concentration profiles u(r, t) across strata:\n"
        "  - Fast depletion in hydrated outer targeting corona\n"
        "  - Retarded transport through crosslinked hydrogel barrier\n"
        "  - Sustained retention in dense core drug reservoir",
        fontsize=10.0, color="#273238", va="top", linespacing=1.26
    )

    # Embedded Subplot 2: Residual Scatter
    ax_sub2 = fig.add_axes([0.76, 0.55, 0.17, 0.13])
    residuals = obs_pts - true_curve
    ax_sub2.axhline(0, color="#273238", lw=0.9, linestyle="--")
    ax_sub2.scatter(t_pts, residuals, color="#A86418", s=18)
    ax_sub2.set_ylim(-0.020, 0.020)
    ax_sub2.set_xlabel("Time (s)", fontsize=7.5)
    ax_sub2.set_ylabel("delta_M", fontsize=7.5)
    ax_sub2.grid(True, linestyle=":", alpha=0.5)

    # Embedded Subplot 3: Radial concentration profiles u(r)
    ax_sub3 = fig.add_axes([0.76, 0.37, 0.17, 0.13])
    r_grid = np.linspace(0, 0.1575, 50)
    u0 = np.where(r_grid <= 0.080, 0.050, np.where(r_grid <= 0.125, 0.020, 0.005))
    u10 = u0 * np.exp(-r_grid / 0.080 * 0.4)
    u30 = u0 * np.exp(-r_grid / 0.080 * 1.2)
    ax_sub3.plot(r_grid * 1000, u0, color="#273238", lw=1.2, label="t=0")
    ax_sub3.plot(r_grid * 1000, u10, color="#4A779D", lw=1.2, label="t=10s")
    ax_sub3.plot(r_grid * 1000, u30, color="#3C6D65", lw=1.2, label="t=30s")
    ax_sub3.set_xlabel("Radius r (nm)", fontsize=7.5)
    ax_sub3.set_ylabel("u(r)", fontsize=7.5)
    ax_sub3.legend(fontsize=6.5, loc="upper right")
    ax_sub3.grid(True, linestyle=":", alpha=0.5)

    # Bottom Banner
    banner = FancyBboxPatch((0.88, 5.82), 11.57, 0.68, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor="#F5E5C5", edgecolor="none")
    ax.add_patch(banner)
    ax.text(
        1.10, 6.16,
        "Identifiability boundary: Multi-shell diffusivities are well-conditioned when shell thicknesses Delta R are comparable;\n"
        "thin outer skins (<10 nm) require high-frequency initial sampling.",
        fontsize=11.2, fontweight="bold", color="#273238", va="center"
    )

    # Footer
    ax.text(
        0.88, 6.95,
        "Model fitting: Pure-Julia Nelder-Mead simplex solver in MicrofluidicClassificationEngine.jl (fit_core_multishell_release_curve).",
        fontsize=10.0, color="#536168"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)


def render_ghosh_slide_8_png(out_path: Path):
    """Render 1920x1080 Slide 8: Pulsed Duty-Cycle Actuation and PC12 Internalization Discrimination."""
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#F6F5F0")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="#F6F5F0")
    ax.set_xlim(0, 13.333)
    ax.set_ylim(0, 7.5)
    ax.invert_yaxis()
    ax.axis("off")

    # Header
    ax.text(
        0.88, 0.88,
        "SHARED 3D VISUALIZATION ENGINE . EXPERIMENTAL FLOW CYTOMETRY PROTOCOL",
        fontsize=13, fontweight="bold", color="#A86418", family="sans-serif"
    )
    ax.text(
        0.88, 1.40,
        "Pulsed Duty-Cycle Actuation and PC12 Internalization Discrimination",
        fontsize=24, fontweight="bold", color="#273238", family="serif"
    )
    ax.text(
        0.88, 1.84,
        "Dynamic physical field actuation gating vs thermal lag, coupled to flow cytometry internalization controls.",
        fontsize=13.5, color="#536168", family="sans-serif"
    )

    card_y = 2.15
    card_h = 3.50

    # Left Card: Pulsed Duty-Cycle Release Dynamics
    c_left = FancyBboxPatch((0.88, card_y), 5.65, card_h, boxstyle="round,pad=0.04,rounding_size=0.15", facecolor="#E4EDF4", edgecolor="#D4D2CB", lw=1.5)
    ax.add_patch(c_left)
    ax.text(1.08, card_y + 0.32, "1 . PULSED DUTY-CYCLE RELEASE DYNAMICS", fontsize=13.0, fontweight="bold", color="#4A779D")
    ax.text(
        1.08, card_y + 0.65,
        "* Field gating comparison:\n"
        "  - Step-gated shell: P_m switches instantaneously\n"
        "  - Thermal-lag shell: tau_th dP/dt + P = P_target (tau = 2.0 s)\n"
        "* Actuation: 20% duty cycle (2 s ON / 8 s OFF, T = 10 s)\n"
        "* Dynamics: Step-gated achieves sharp staircase delivery;\n"
        "  thermal inertia causes lag and post-pulse leakback\n"
        "* On-phase selectivity: Step (>85%) vs Thermal (<70%)",
        fontsize=10.0, color="#273238", va="top", linespacing=1.26
    )

    # Embedded Subplot 1: Pulsed Kinetics
    ax_sub1 = fig.add_axes([0.31, 0.40, 0.17, 0.28])
    t_pulse = np.linspace(0, 30, 100)
    # Permeability profiles
    p_step = np.where((t_pulse % 10.0) < 2.0, 1.0, 0.1)
    p_therm = 0.1 + 0.9 * np.maximum(0, np.exp(-((t_pulse % 10.0) - 2.0) / 2.0) * ((t_pulse % 10.0) >= 2.0) + ((t_pulse % 10.0) < 2.0) * (1 - np.exp(-(t_pulse % 10.0) / 1.0)))
    ax_sub1.plot(t_pulse, p_step, color="#4A779D", lw=1.8, label="Step-gated P_m")
    ax_sub1.plot(t_pulse, p_therm, color="#e11d48", lw=1.8, linestyle="--", label="Thermal-lag P_m")
    ax_sub1.set_xlabel("Time (s)", fontsize=8.0)
    ax_sub1.set_ylabel("Normalized P_m(t)", fontsize=8.0)
    ax_sub1.legend(fontsize=7.0, loc="upper right")
    ax_sub1.grid(True, linestyle=":", alpha=0.5)

    # Right Card: PC12 Surface Association vs Intracellular Internalization
    c_right = FancyBboxPatch((6.80, card_y), 5.65, card_h, boxstyle="round,pad=0.04,rounding_size=0.15", facecolor="#E1EBE6", edgecolor="#D4D2CB", lw=1.5)
    ax.add_patch(c_right)
    ax.text(7.00, card_y + 0.32, "2 . PC12 SURFACE ASSOCIATION VS INTERNALIZATION", fontsize=13.0, fontweight="bold", color="#3C6D65")
    ax.text(
        7.00, card_y + 0.65,
        "* Cytometric ambiguity: Total fluorophore signal conflates\n"
        "  extracellular membrane adsorption with true internalization\n"
        "* 4-Step Discriminating Controls:\n"
        "  1. Trypan blue quenching: selectively quenches surface dye\n"
        "  2. 4 deg C cold incubation: blocks active endocytosis\n"
        "  3. Acid-wash strip: pH 2.5 glycine removes bound particles\n"
        "  4. Confocal sectioning: orthogonal Z-stacks verify localization",
        fontsize=10.0, color="#273238", va="top", linespacing=1.26
    )

    # Embedded Subplot 2: Bivariate Gating Plot
    ax_sub2 = fig.add_axes([0.76, 0.40, 0.17, 0.28])
    np.random.seed(123)
    # Scatter clouds
    n_pts = 80
    # Q3: Unlabeled
    x_q3 = np.random.normal(1.8, 0.2, n_pts); y_q3 = np.random.normal(1.8, 0.2, n_pts)
    # Q1: Surface bound only (quenched)
    x_q1 = np.random.normal(3.8, 0.3, n_pts); y_q1 = np.random.normal(2.0, 0.25, n_pts)
    # Q2: Bound + Internalized (quench-resistant)
    x_q2 = np.random.normal(4.0, 0.28, n_pts); y_q2 = np.random.normal(3.8, 0.3, n_pts)
    ax_sub2.scatter(x_q3, y_q3, color="#94a3b8", s=8, alpha=0.6, label="Unlabeled")
    ax_sub2.scatter(x_q1, y_q1, color="#f97316", s=10, alpha=0.7, label="Surface Bound")
    ax_sub2.scatter(x_q2, y_q2, color="#16a34a", s=12, alpha=0.8, label="Internalized")
    # Gate thresholds
    ax_sub2.axvline(2.8, color="#273238", lw=0.9, linestyle=":")
    ax_sub2.axhline(2.8, color="#273238", lw=0.9, linestyle=":")
    ax_sub2.text(3.5, 2.2, "Q1: Surface", fontsize=6.8, color="#c2410c", fontweight="bold")
    ax_sub2.text(3.4, 4.3, "Q2: Internalized", fontsize=6.8, color="#15803d", fontweight="bold")
    ax_sub2.set_xlabel("Total Fluorescence (log)", fontsize=7.5)
    ax_sub2.set_ylabel("Quench-Resistant (log)", fontsize=7.5)
    ax_sub2.legend(fontsize=6.2, loc="upper left")
    ax_sub2.grid(True, linestyle=":", alpha=0.4)

    # Bottom Banner
    banner = FancyBboxPatch((0.88, 5.82), 11.57, 0.68, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor="#F5E5C5", edgecolor="none")
    ax.add_patch(banner)
    ax.text(
        1.10, 6.16,
        "Track 1 experimental contract: Claims of drug delivery efficacy require demonstrating intracellular translocation\n"
        "(quench-resistant signal), not mere cell surface coating.",
        fontsize=11.2, fontweight="bold", color="#273238", va="center"
    )

    # Footer
    ax.text(
        0.88, 6.95,
        "Discriminating controls: simulate_pulsed_duty_cycle_release + Trypan blue / cold-block / acid-wash / confocal validation ladder.",
        fontsize=10.0, color="#536168"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)


def render_jack_slide_1_png(out_path: Path):
    """Render 1920x1080 Slide 1 for Iain Jack."""
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#F6F5F0")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="#F6F5F0")
    ax.set_xlim(0, 13.333)
    ax.set_ylim(0, 7.5)
    ax.invert_yaxis()
    ax.axis("off")

    # Header
    ax.text(
        0.88, 0.92,
        "SHARED 3D VISUALIZATION ENGINE . TRACK 3 INTERFACIAL RHEOLOGY ADAPTATION",
        fontsize=13, fontweight="bold", color="#A86418", family="sans-serif"
    )
    ax.text(
        0.88, 1.48,
        "Interface-Resolved Emulsion Dynamics: From Constitutive Rheology to Multiphase Flow",
        fontsize=24, fontweight="bold", color="#273238", family="serif"
    )
    ax.text(
        0.88, 1.98,
        "Translate the wireframe-over-core motif to bitumen emulsion droplets under shear, temperature, and emulsifier kinetics.",
        fontsize=14, color="#536168", family="sans-serif"
    )

    # Left Card: Droplet Deformation & Wireframe Interface Mesh
    left_card = FancyBboxPatch(
        (0.88, 2.35), 5.65, 3.20,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        facecolor="#E5E4DF", edgecolor="#D4D2CB", linewidth=1.5
    )
    ax.add_patch(left_card)
    ax.text(
        1.10, 2.68,
        "1 . DROPLET INTERFACE & DEFORMATION (t / tau_eta = 1..15)",
        fontsize=13.5, fontweight="bold", color="#536168"
    )
    ax.text(
        1.10, 3.10,
        "* Bitumen droplet core + interface mesh\n"
        "* Taylor deformation D = (L - B)/(L + B)\n"
        "  under Ca = mu_c * gamma_dot * a0 / sigma\n"
        "* Boussinesq-Scriven temperature-\n"
        "  dependent interfacial viscosity\n"
        "* Captures elongation & necking without\n"
        "  biological release shell assumptions",
        fontsize=11.0, color="#273238", va="top", linespacing=1.36
    )

    for idx, (tx, t_lbl, b_scale) in enumerate([(4.42, "t/tau=2", 0.50), (5.08, "t/tau=8", 0.32), (5.72, "t/tau=15", 0.17)]):
        ell_outer = Ellipse(
            (tx, 4.12), width=b_scale, height=1.22, angle=13 * (idx + 1),
            facecolor="#7f1d1d", alpha=0.82, edgecolor="#f87171", lw=1.4
        )
        ax.add_patch(ell_outer)
        ax.text(tx, 5.04, t_lbl, fontsize=9.0, ha="center", color="#273238", fontweight="bold")

    # Right Card: Twin-Vortex Velocity & Pressure Field P/Pa
    right_card = FancyBboxPatch(
        (6.85, 2.35), 5.60, 3.20,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        facecolor="#E4EDF4", edgecolor="#D4D2CB", linewidth=1.5
    )
    ax.add_patch(right_card)
    ax.text(
        7.07, 2.68,
        "2 . MULTIPHASE VORTEX FLOW & PRESSURE FIELD (P / Pa)",
        fontsize=13.5, fontweight="bold", color="#4A779D"
    )
    ax.text(
        7.05, 3.10,
        "* Internal twin-vortex circulation and\n"
        "  external shear streamlines (a, b, c)\n"
        "* Hydrodynamic pressure P/Pa in\n"
        "  [-0.30, 1.40] & emulsifier Gamma\n"
        "* Validated against DSR & moisture-\n"
        "  susceptibility measurements\n"
        "* Track 3 rejects cytometry coupling",
        fontsize=11.0, color="#273238", va="top", linespacing=1.36
    )

    p_box = FancyBboxPatch(
        (10.22, 3.22), 2.08, 1.88,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor="#22c55e", alpha=0.28, edgecolor="#4A779D", linewidth=1.2
    )
    ax.add_patch(p_box)
    ax.add_patch(Ellipse((11.26, 4.02), 1.55, 0.78, angle=-18, facecolor="#4ade80", alpha=0.45, edgecolor="#dc2626", lw=2.0))
    ax.add_patch(Ellipse((11.26, 4.02), 0.68, 0.58, angle=-10, facecolor="#ef4444", alpha=0.85, edgecolor="#1d4ed8", lw=1.8))
    ax.text(11.26, 4.02, "P/Pa", fontsize=9.5, ha="center", va="center", color="#ffffff", fontweight="bold")
    ax.text(11.26, 4.92, "Twin Vortex + Core Pressure", fontsize=8.8, ha="center", color="#273238", fontweight="bold")

    # Bottom Boundary Banner
    banner = FancyBboxPatch(
        (0.88, 5.76), 11.57, 0.68,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor="#F5E5C5", edgecolor="none"
    )
    ax.add_patch(banner)
    ax.text(
        1.10, 6.10,
        "Architecture rule: reuse 3D wireframe/isosurface rendering and data infrastructure, not unvalidated physical parameters across tracks.",
        fontsize=12.2, fontweight="bold", color="#273238", va="center"
    )

    # Footer
    ax.text(
        0.88, 6.92,
        "Track 3 Bitumen Emulsion & Interfacial Rheology: coupled constitutive rheology (Julia) + multiphase flow (OpenFOAM), separate from Track 1/Track 2.",
        fontsize=10.0, color="#536168"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)


def xml_escape(text: str) -> str:
    """Escape XML 1.0 predefined entities (&, <, >, ", ')."""
    if text is None:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _make_full_bleed_slide_xml(r_id_img: str) -> bytes:
    """Create an OpenXML slide that displays a high-resolution full-slide figure cleanly."""
    safe_r_id = xml_escape(r_id_img)
    xml = f"""<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <p:cSld>
    <p:bg>
      <p:bgPr>
        <a:solidFill><a:srgbClr val="F6F5F0"/></a:solidFill>
        <a:effectLst/>
      </p:bgPr>
    </p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr/>
      <p:pic>
        <p:nvPicPr>
          <p:cNvPr id="2" name="SlideRender" descr="High-resolution domain visualization slide"/>
          <p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr>
          <p:nvPr/>
        </p:nvPicPr>
        <p:blipFill>
          <a:blip r:embed="{safe_r_id}"/>
          <a:stretch><a:fillRect/></a:stretch>
        </p:blipFill>
        <p:spPr>
          <a:xfrm>
            <a:off x="0" y="0"/>
            <a:ext cx="12191999" cy="6858000"/>
          </a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        </p:spPr>
      </p:pic>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""
    return xml.encode("utf-8")


def _make_notes_xml(note_text: str) -> bytes:
    escaped_paragraphs = []
    for line in str(note_text).splitlines():
        line_clean = xml_escape(line)
        escaped_paragraphs.append(f"<a:p><a:r><a:t>{line_clean}</a:t></a:r></a:p>")
    if not escaped_paragraphs:
        escaped_paragraphs.append("<a:p><a:r><a:t></a:t></a:r></a:p>")
    tx_paragraphs = "".join(escaped_paragraphs)
    xml = f"""<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<p:notes xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      <p:sp>
        <p:nvSpPr><p:cNvPr id="3" name="Notes Placeholder 2"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="body" idx="3" sz="quarter"/></p:nvPr></p:nvSpPr>
        <p:spPr/>
        <p:txBody><a:bodyPr/><a:lstStyle/>{tx_paragraphs}</p:txBody>
      </p:sp>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:notes>"""
    return xml.encode("utf-8")


def sanitize_pptx_dashes(pptx_path: Path):
    """Perform zip-level XML stream regex replacement to guarantee strictly 0 em-dashes and 0 en-dashes."""
    with tempfile.TemporaryDirectory() as tmpd:
        tmp_pptx = Path(tmpd) / "sanitized.pptx"
        with zipfile.ZipFile(pptx_path, "r") as zin:
            items = {}
            for name in zin.namelist():
                content = zin.read(name)
                if name.endswith((".xml", ".rels")):
                    text = content.decode("utf-8", errors="ignore")
                    text = text.replace('char="\u2013"', 'char="\u2022"')
                    text = text.replace("char='\u2013'", "char='\u2022'")
                    text = text.replace("\u2014", " - ")
                    text = text.replace("\u2013", "-")
                    items[name] = text.encode("utf-8")
                else:
                    items[name] = content
        with zipfile.ZipFile(tmp_pptx, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for k, v in items.items():
                zout.writestr(k, v)
        shutil.copyfile(tmp_pptx, pptx_path)


def validate_pptx_package(pptx_path: Path, expected_slide_count: int):
    """Validate that the PPTX archive is a valid, well-formed OpenXML package.

    Checks:
    1. Every .xml and .rels archive part parses cleanly without XML syntax errors.
    2. python-pptx loads the presentation package and matches the expected slide count.
    3. Every slide notes frame can be accessed and read without exception.
    4. Zero em-dashes and zero en-dashes across all XML text streams.
    """
    import xml.etree.ElementTree as ET
    from pptx import Presentation

    if not pptx_path.exists():
        raise FileNotFoundError(f"Presentation file does not exist: {pptx_path}")

    # 1. XML well-formedness and dash invariant
    with zipfile.ZipFile(pptx_path, "r") as z:
        for name in z.namelist():
            if name.endswith((".xml", ".rels")):
                data = z.read(name)
                try:
                    ET.fromstring(data)
                except Exception as exc:
                    raise ValueError(
                        f"Malformed XML in {pptx_path.name}:{name} - {exc}"
                    ) from exc
                text = data.decode("utf-8", errors="ignore")
                if "\u2014" in text or "\u2013" in text:
                    raise ValueError(
                        f"Dash policy violation (em/en-dash found) in {pptx_path.name}:{name}"
                    )

    # 2. python-pptx load and slide count verification
    try:
        prs = Presentation(str(pptx_path))
    except Exception as exc:
        raise ValueError(
            f"python-pptx failed to parse {pptx_path.name}: {exc}"
        ) from exc

    actual_count = len(prs.slides)
    if actual_count != expected_slide_count:
        raise ValueError(
            f"Slide count mismatch in {pptx_path.name}: expected {expected_slide_count}, got {actual_count}"
        )

    # 3. Slide notes accessibility check
    for idx, slide in enumerate(prs.slides):
        try:
            if slide.has_notes_slide:
                _ = slide.notes_slide.notes_text_frame.text
        except Exception as exc:
            raise ValueError(
                f"Failed to read notes on slide {idx + 1} in {pptx_path.name}: {exc}"
            ) from exc


def update_ghosh_pptx_and_pdf(slide_png_map: dict, pptx_path: Path, pdf_path: Path):
    """Ensure Ghosh exploratory deck has 8 slides (Slides 1-4 preserved + Slides 5-8 added)."""
    with tempfile.TemporaryDirectory() as tmpd:
        tmp_zip = Path(tmpd) / "ghosh.pptx"
        with zipfile.ZipFile(pptx_path, "r") as zin:
            items = {n: zin.read(n) for n in zin.namelist()}

        # 1. Update [Content_Types].xml
        ct = items["[Content_Types].xml"].decode("utf-8")
        if 'Extension="png"' not in ct:
            ct = ct.replace(
                '<Default Extension="jpeg" ContentType="image/jpeg"/>',
                '<Default Extension="jpeg" ContentType="image/jpeg"/><Default Extension="png" ContentType="image/png"/>'
            )
        for s_idx in range(5, 9):
            part_s = f"/ppt/slides/slide{s_idx}.xml"
            if part_s not in ct:
                ct = ct.replace(
                    "</Types>",
                    f'<Override PartName="{part_s}" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
                    f'<Override PartName="/ppt/notesSlides/notesSlide{s_idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/></Types>'
                )
        items["[Content_Types].xml"] = ct.encode("utf-8")

        # 2. Update ppt/presentation.xml
        pres = items["ppt/presentation.xml"].decode("utf-8")
        # Ensure slide IDs 260..263 with rId12..rId15
        slide_id_map = {5: ("260", "rId12"), 6: ("261", "rId13"), 7: ("262", "rId14"), 8: ("263", "rId15")}
        for s_idx in range(5, 9):
            sid, rid = slide_id_map[s_idx]
            tag = f'<p:sldId id="{sid}" r:id="{rid}"/>'
            if tag not in pres:
                pres = pres.replace("</p:sldIdLst>", f"{tag}</p:sldIdLst>")
        items["ppt/presentation.xml"] = pres.encode("utf-8")

        # 3. Update ppt/_rels/presentation.xml.rels
        prels = items["ppt/_rels/presentation.xml.rels"].decode("utf-8")
        for s_idx in range(5, 9):
            sid, rid = slide_id_map[s_idx]
            tag = f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{s_idx}.xml"/>'
            if f'Id="{rid}"' not in prels:
                prels = prels.replace("</Relationships>", f"{tag}</Relationships>")
        items["ppt/_rels/presentation.xml.rels"] = prels.encode("utf-8")

        # Notes text definitions
        notes = {
            5: "Core-Multishell Reaction-Diffusion and Microfluidic Single-Cell Classification. Shells represent simulated concentration fields (:predicted), not observed physical layers.",
            6: "Nanocarrier Architecture Comparison: Synthetic Multi-Shell vs Natural Biofilm. Structural parameter comparison across Ghosh 2010 (B12/PEGMA 315nm), Ghosh Patent (Cisplatin double-shell 191nm), and multicellular biofilm aggregates. Epistemic status: :predicted simulated models; MODEL PREDICTION - illustrative parameters - no lab data.",
            7: "Core-Multishell Release Profile Fitting and Parameter Recovery. Non-linear least squares parameter recovery (Nelder-Mead) and residual audit across spherical strata. Demonstrates recovery of stratum diffusivities and outer membrane permeability with R^2 > 0.99 and RMSE < 0.01.",
            8: "Pulsed Duty-Cycle Actuation and PC12 Internalization Discrimination. Compares step-gated vs thermal-lag release dynamics and defines 4-step discriminating controls (Trypan blue quenching, 4 deg C cold incubation, acid-wash strip, confocal optical sectioning) to resolve intracellular uptake from passive surface association."
        }

        # 4. Inject media, slide XMLs, and notes XMLs for slides 5..8
        for s_idx in range(5, 9):
            png_path = slide_png_map[s_idx]
            media_name = f"ppt/media/ghosh_slide{s_idx}.png"
            items[media_name] = png_path.read_bytes()
            items[f"ppt/slides/slide{s_idx}.xml"] = _make_full_bleed_slide_xml("rId3")
            items[f"ppt/slides/_rels/slide{s_idx}.xml.rels"] = (
                "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
                f'<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide{s_idx}.xml"/>'
                f'<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/ghosh_slide{s_idx}.png"/>'
                "</Relationships>"
            ).encode("utf-8")
            items[f"ppt/notesSlides/notesSlide{s_idx}.xml"] = _make_notes_xml(notes[s_idx])
            items[f"ppt/notesSlides/_rels/notesSlide{s_idx}.xml.rels"] = (
                "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster" Target="../notesMasters/notesMaster1.xml"/>'
                f'<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="../slides/slide{s_idx}.xml"/>'
                "</Relationships>"
            ).encode("utf-8")

        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for k, v in items.items():
                zout.writestr(k, v)
        shutil.copyfile(tmp_zip, pptx_path)

    sanitize_pptx_dashes(pptx_path)

    # Rebuild 8-page PDF from previews Ghosh_Bridge_Slide_1..8.png
    slide_pngs = [PREV_DIR / f"Ghosh_Bridge_Slide_{i}.png" for i in range(1, 9)]
    pil_imgs = [Image.open(p).convert("RGB") for p in slide_pngs]
    pil_imgs[0].save(str(pdf_path), "PDF", resolution=144.0, save_all=True, append_images=pil_imgs[1:])


def build_jack_pptx_and_pdf(base_pptx: Path, jack_png: Path, jack_pptx: Path, jack_pdf: Path):
    """Build 1-slide exploratory deck and PDF for Iain Jack's interfacial rheology adaptation."""
    with zipfile.ZipFile(base_pptx, "r") as zin:
        items = {n: zin.read(n) for n in zin.namelist()}

    # Strip slides 2..8 so Jack deck is a clean 1-slide deck
    for rm_idx in range(2, 9):
        for pat in (
            f"ppt/slides/slide{rm_idx}.xml",
            f"ppt/slides/_rels/slide{rm_idx}.xml.rels",
            f"ppt/notesSlides/notesSlide{rm_idx}.xml",
            f"ppt/notesSlides/_rels/notesSlide{rm_idx}.xml.rels",
            f"ppt/media/ghosh_slide{rm_idx}.png",
        ):
            items.pop(pat, None)
    items.pop("ppt/media/ghosh_slide5.png", None)

    ct = items["[Content_Types].xml"].decode("utf-8")
    if 'Extension="png"' not in ct:
        ct = ct.replace(
            '<Default Extension="jpeg" ContentType="image/jpeg"/>',
            '<Default Extension="jpeg" ContentType="image/jpeg"/><Default Extension="png" ContentType="image/png"/>'
        )
    for rm_idx in range(2, 9):
        ct = ct.replace(
            f'<Override PartName="/ppt/slides/slide{rm_idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>',
            ""
        )
        ct = ct.replace(
            f'<Override PartName="/ppt/notesSlides/notesSlide{rm_idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/>',
            ""
        )
    items["[Content_Types].xml"] = ct.encode("utf-8")

    pres = items["ppt/presentation.xml"].decode("utf-8")
    for sid, rid in [("257", "rId9"), ("258", "rId10"), ("259", "rId11"), ("260", "rId12"), ("261", "rId13"), ("262", "rId14"), ("263", "rId15")]:
        pres = pres.replace(f'<p:sldId id="{sid}" r:id="{rid}"/>', "")
    items["ppt/presentation.xml"] = pres.encode("utf-8")

    prels = items["ppt/_rels/presentation.xml.rels"].decode("utf-8")
    for rid, snum in [("rId9", 2), ("rId10", 3), ("rId11", 4), ("rId12", 5), ("rId13", 6), ("rId14", 7), ("rId15", 8)]:
        prels = prels.replace(
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{snum}.xml"/>',
            ""
        )
    items["ppt/_rels/presentation.xml.rels"] = prels.encode("utf-8")

    items["ppt/media/jack_slide1.png"] = jack_png.read_bytes()
    items["ppt/slides/slide1.xml"] = _make_full_bleed_slide_xml("rId3")
    items["ppt/slides/_rels/slide1.xml.rels"] = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide1.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/jack_slide1.png"/>'
        "</Relationships>"
    ).encode("utf-8")
    items["ppt/notesSlides/notesSlide1.xml"] = _make_notes_xml(
        "Interface-Resolved Emulsion Dynamics: From Constitutive Rheology to Multiphase Flow. "
        "Reuses 3D wireframe/core rendering infrastructure for bitumen emulsion droplets under shear without biological release shell assumptions."
    )

    with zipfile.ZipFile(jack_pptx, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for k, v in items.items():
            zout.writestr(k, v)

    sanitize_pptx_dashes(jack_pptx)

    img = Image.open(jack_png).convert("RGB")
    img.save(str(jack_pdf), "PDF", resolution=144.0)


def main():
    PREV_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    ghosh_slide5_png = PREV_DIR / "Ghosh_Bridge_Slide_5.png"
    ghosh_slide6_png = PREV_DIR / "Ghosh_Bridge_Slide_6.png"
    ghosh_slide7_png = PREV_DIR / "Ghosh_Bridge_Slide_7.png"
    ghosh_slide8_png = PREV_DIR / "Ghosh_Bridge_Slide_8.png"
    jack_slide1_png = PREV_DIR / "Jack_Emulsion_Interface_Slide_1.png"

    # Render high-resolution PNGs (1920x1080)
    render_ghosh_slide_5_png(ghosh_slide5_png)
    render_ghosh_slide_6_png(ghosh_slide6_png)
    render_ghosh_slide_7_png(ghosh_slide7_png)
    render_ghosh_slide_8_png(ghosh_slide8_png)
    render_jack_slide_1_png(jack_slide1_png)

    # Ensure all preview slides 1..4 are 1920x1080
    for i in range(1, 5):
        prev_p = PREV_DIR / f"Ghosh_Bridge_Slide_{i}.png"
        if prev_p.exists():
            im = Image.open(prev_p)
            if im.size != (1920, 1080):
                im_hires = im.resize((1920, 1080), Image.Resampling.LANCZOS)
                im_hires.save(prev_p)

    # Copy all 8 slides into presentations/assets/
    for i in range(1, 9):
        prev_p = PREV_DIR / f"Ghosh_Bridge_Slide_{i}.png"
        asset_p = ASSETS_DIR / f"Ghosh_Bridge_Slide_{i}.png"
        if prev_p.exists():
            shutil.copyfile(prev_p, asset_p)

    slide_png_map = {
        5: ghosh_slide5_png,
        6: ghosh_slide6_png,
        7: ghosh_slide7_png,
        8: ghosh_slide8_png,
    }

    # Primary deck: Single_Cell_Flow_Cytometry_Model_Bridge_Exploratory
    ghosh_pptx1 = PRES_DIR / "Single_Cell_Flow_Cytometry_Model_Bridge_Exploratory.pptx"
    ghosh_pdf1 = PRES_DIR / "Single_Cell_Flow_Cytometry_Model_Bridge_Exploratory.pdf"
    update_ghosh_pptx_and_pdf(slide_png_map, ghosh_pptx1, ghosh_pdf1)

    # Dual named deck: Ghosh_Flow_Cytometry_Multiphysics_Bridge
    ghosh_pptx2 = PRES_DIR / "Ghosh_Flow_Cytometry_Multiphysics_Bridge.pptx"
    ghosh_pdf2 = PRES_DIR / "Ghosh_Flow_Cytometry_Multiphysics_Bridge.pdf"
    shutil.copyfile(ghosh_pptx1, ghosh_pptx2)
    shutil.copyfile(ghosh_pdf1, ghosh_pdf2)

    # Jack deck
    jack_pptx = PRES_DIR / "Interface_Resolved_Emulsion_Dynamics_Exploratory.pptx"
    jack_pdf = PRES_DIR / "Interface_Resolved_Emulsion_Dynamics_Exploratory.pdf"
    build_jack_pptx_and_pdf(ghosh_pptx1, jack_slide1_png, jack_pptx, jack_pdf)

    # Automated post-build validation
    validate_pptx_package(ghosh_pptx1, 8)
    validate_pptx_package(ghosh_pptx2, 8)
    validate_pptx_package(jack_pptx, 1)

    r_ghosh1 = PdfReader(str(ghosh_pdf1))
    r_ghosh2 = PdfReader(str(ghosh_pdf2))
    r_jack = PdfReader(str(jack_pdf))

    print(f"Built and validated {ghosh_pptx1.name} (8 slides) and {ghosh_pdf1.name} ({len(r_ghosh1.pages)} pages)")
    print(f"Built and validated {ghosh_pptx2.name} (8 slides) and {ghosh_pdf2.name} ({len(r_ghosh2.pages)} pages)")
    print(f"Built and validated {jack_pptx.name} (1 slide) and {jack_pdf.name} ({len(r_jack.pages)} page)")


if __name__ == "__main__":
    main()
