#!/usr/bin/env python3
"""
Build script for:
1. Ghosh Exploratory Deck (5 slides):
   - presentations/Single_Cell_Flow_Cytometry_Model_Bridge_Exploratory.pptx
   - presentations/Single_Cell_Flow_Cytometry_Model_Bridge_Exploratory.pdf
   - previews/Ghosh_Bridge_Slide_5.png (1920x1080)
   Slide 5 title: "Core-Multishell Reaction-Diffusion and Microfluidic Single-Cell Classification"
   Includes 3D octant-cutaway core-multishell (R0, R1, R2) + wireframe isosurface
   paired with microfluidic channel single-cell trajectory coupling.

2. Jack Exploratory Adaptation Deck (1 slide):
   - presentations/Interface_Resolved_Emulsion_Dynamics_Exploratory.pptx
   - presentations/Interface_Resolved_Emulsion_Dynamics_Exploratory.pdf
   - previews/Jack_Emulsion_Interface_Slide_1.png (1920x1080)
   Slide 1 title: "Interface-Resolved Emulsion Dynamics: From Constitutive Rheology to Multiphase Flow"
   Includes deformed bitumen double-emulsion droplet + wireframe interface mesh under shear
   (t/tau_eta = 1..15) paired with internal twin-vortex velocity and pressure field P/Pa.
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

    # Embedded 3D Octant-Cutaway Sphere Diagram inside Left Card (right half)
    cx, cy = 5.20, 4.10
    # Wireframe outer isosurface rings
    for ang in (-55, -25, 0, 25, 55):
        ell = Ellipse(
            (cx, cy), width=2.15, height=2.15 * math.cos(math.radians(ang)),
            angle=22, fill=False, edgecolor="#38bdf8", linestyle="--", linewidth=0.9, alpha=0.75
        )
        ax.add_patch(ell)
    # Outer shell R2 (red/rose)
    w_r2 = Wedge((cx, cy), 0.98, 0, 270, facecolor="#e11d48", alpha=0.42, edgecolor="#be123c", lw=1.6)
    # Mid shell R1 (indigo/violet)
    w_r1 = Wedge((cx, cy), 0.70, 0, 270, facecolor="#6366f1", alpha=0.65, edgecolor="#4338ca", lw=1.6)
    # Inner core R0 (gold/yellow)
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
    # Constriction walls
    top_wall = Polygon([[10.28, 3.36], [10.88, 3.36], [11.06, 3.76], [11.46, 3.76], [11.64, 3.36], [12.24, 3.36]],
                       closed=False, fill=False, edgecolor="#273238", lw=1.8)
    bot_wall = Polygon([[10.28, 4.62], [10.88, 4.62], [11.06, 4.22], [11.46, 4.22], [11.64, 4.62], [12.24, 4.62]],
                       closed=False, fill=False, edgecolor="#273238", lw=1.8)
    ax.add_patch(top_wall)
    ax.add_patch(bot_wall)
    # Cells in channel
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

    # Deformed Droplet Sequence Diagram inside Left Card
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

    # Embedded Double-Emulsion Vortex/Pressure Schematic inside Right Card
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


def _make_full_bleed_slide_xml(r_id_img: str) -> bytes:
    """Create an OpenXML slide that displays a high-resolution full-slide figure cleanly."""
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
          <a:blip r:embed="{r_id_img}"/>
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
    xml = f"""<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<p:notes xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      <p:sp>
        <p:nvSpPr><p:cNvPr id="3" name="Notes Placeholder 2"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="body" idx="3" sz="quarter"/></p:nvPr></p:nvSpPr>
        <p:spPr/>
        <p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>{note_text}</a:t></a:r></a:p></p:txBody>
      </p:sp>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:notes>"""
    return xml.encode("utf-8")


def update_ghosh_pptx_and_pdf(slide5_png: Path, pptx_path: Path, pdf_path: Path):
    """Ensure Ghosh exploratory deck has 5 slides (Slides 1-4 preserved + Slide 5 added)."""
    with tempfile.TemporaryDirectory() as tmpd:
        tmp_zip = Path(tmpd) / "ghosh.pptx"
        with zipfile.ZipFile(pptx_path, "r") as zin:
            names = zin.namelist()
            items = {n: zin.read(n) for n in names}

        # Add png Default to [Content_Types].xml if not present, plus slide5 & notesSlide5 Overrides
        ct = items["[Content_Types].xml"].decode("utf-8")
        if 'Extension="png"' not in ct:
            ct = ct.replace(
                '<Default Extension="jpeg" ContentType="image/jpeg"/>',
                '<Default Extension="jpeg" ContentType="image/jpeg"/><Default Extension="png" ContentType="image/png"/>'
            )
        if "/ppt/slides/slide5.xml" not in ct:
            ct = ct.replace(
                "</Types>",
                '<Override PartName="/ppt/slides/slide5.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
                '<Override PartName="/ppt/notesSlides/notesSlide5.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/></Types>'
            )
        items["[Content_Types].xml"] = ct.encode("utf-8")

        # Update ppt/presentation.xml
        pres = items["ppt/presentation.xml"].decode("utf-8")
        if 'r:id="rId12"' not in pres:
            pres = pres.replace(
                '<p:sldId id="259" r:id="rId11"/></p:sldIdLst>',
                '<p:sldId id="259" r:id="rId11"/><p:sldId id="260" r:id="rId12"/></p:sldIdLst>'
            )
        items["ppt/presentation.xml"] = pres.encode("utf-8")

        # Update ppt/_rels/presentation.xml.rels
        prels = items["ppt/_rels/presentation.xml.rels"].decode("utf-8")
        if 'Id="rId12"' not in prels:
            prels = prels.replace(
                "</Relationships>",
                '<Relationship Id="rId12" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide5.xml"/></Relationships>'
            )
        items["ppt/_rels/presentation.xml.rels"] = prels.encode("utf-8")

        # Add media, slide5.xml, slide5.xml.rels, notesSlide5.xml, notesSlide5.xml.rels
        items["ppt/media/ghosh_slide5.png"] = slide5_png.read_bytes()
        items["ppt/slides/slide5.xml"] = _make_full_bleed_slide_xml("rId3")
        items["ppt/slides/_rels/slide5.xml.rels"] = (
            "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide5.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/ghosh_slide5.png"/>'
            "</Relationships>"
        ).encode("utf-8")
        items["ppt/notesSlides/notesSlide5.xml"] = _make_notes_xml(
            "Core-Multishell Reaction-Diffusion and Microfluidic Single-Cell Classification. "
            "Shells represent simulated concentration fields (:predicted), not observed physical layers."
        )
        items["ppt/notesSlides/_rels/notesSlide5.xml.rels"] = (
            "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster" Target="../notesMasters/notesMaster1.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="../slides/slide5.xml"/>'
            "</Relationships>"
        ).encode("utf-8")

        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for k, v in items.items():
                zout.writestr(k, v)
        shutil.copyfile(tmp_zip, pptx_path)

    # Rebuild 5-page PDF from previews Ghosh_Bridge_Slide_1..5.png
    slide_pngs = [PREV_DIR / f"Ghosh_Bridge_Slide_{i}.png" for i in range(1, 6)]
    pil_imgs = [Image.open(p).convert("RGB") for p in slide_pngs]
    pil_imgs[0].save(str(pdf_path), "PDF", resolution=144.0, save_all=True, append_images=pil_imgs[1:])


def build_jack_pptx_and_pdf(base_pptx: Path, jack_png: Path, jack_pptx: Path, jack_pdf: Path):
    """Build 1-slide exploratory deck and PDF for Iain Jack's interfacial rheology adaptation."""
    with zipfile.ZipFile(base_pptx, "r") as zin:
        items = {n: zin.read(n) for n in zin.namelist()}

    # Strip slides 2..5 so Jack deck is a clean 1-slide deck
    for rm_idx in (2, 3, 4, 5):
        for pat in (
            f"ppt/slides/slide{rm_idx}.xml",
            f"ppt/slides/_rels/slide{rm_idx}.xml.rels",
            f"ppt/notesSlides/notesSlide{rm_idx}.xml",
            f"ppt/notesSlides/_rels/notesSlide{rm_idx}.xml.rels",
        ):
            items.pop(pat, None)
    items.pop("ppt/media/ghosh_slide5.png", None)

    ct = items["[Content_Types].xml"].decode("utf-8")
    if 'Extension="png"' not in ct:
        ct = ct.replace(
            '<Default Extension="jpeg" ContentType="image/jpeg"/>',
            '<Default Extension="jpeg" ContentType="image/jpeg"/><Default Extension="png" ContentType="image/png"/>'
        )
    for rm_idx in (2, 3, 4, 5):
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
    for tag in (
        '<p:sldId id="257" r:id="rId9"/>',
        '<p:sldId id="258" r:id="rId10"/>',
        '<p:sldId id="259" r:id="rId11"/>',
        '<p:sldId id="260" r:id="rId12"/>',
    ):
        pres = pres.replace(tag, "")
    items["ppt/presentation.xml"] = pres.encode("utf-8")

    prels = items["ppt/_rels/presentation.xml.rels"].decode("utf-8")
    for rid, snum in (("rId9", 2), ("rId10", 3), ("rId11", 4), ("rId12", 5)):
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

    img = Image.open(jack_png).convert("RGB")
    img.save(str(jack_pdf), "PDF", resolution=144.0)


def main():
    ghosh_slide5_png = PREV_DIR / "Ghosh_Bridge_Slide_5.png"
    jack_slide1_png = PREV_DIR / "Jack_Emulsion_Interface_Slide_1.png"

    render_ghosh_slide_5_png(ghosh_slide5_png)
    render_jack_slide_1_png(jack_slide1_png)

    ghosh_pptx = PRES_DIR / "Single_Cell_Flow_Cytometry_Model_Bridge_Exploratory.pptx"
    ghosh_pdf = PRES_DIR / "Single_Cell_Flow_Cytometry_Model_Bridge_Exploratory.pdf"
    update_ghosh_pptx_and_pdf(ghosh_slide5_png, ghosh_pptx, ghosh_pdf)

    jack_pptx = PRES_DIR / "Interface_Resolved_Emulsion_Dynamics_Exploratory.pptx"
    jack_pdf = PRES_DIR / "Interface_Resolved_Emulsion_Dynamics_Exploratory.pdf"
    build_jack_pptx_and_pdf(ghosh_pptx, jack_slide1_png, jack_pptx, jack_pdf)

    r_ghosh = PdfReader(str(ghosh_pdf))
    r_jack = PdfReader(str(jack_pdf))
    print(f"Built {ghosh_pptx.name} (5 slides) and {ghosh_pdf.name} ({len(r_ghosh.pages)} pages)")
    print(f"Built {jack_pptx.name} (1 slide) and {jack_pdf.name} ({len(r_jack.pages)} page)")


if __name__ == "__main__":
    main()
