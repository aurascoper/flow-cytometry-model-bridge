from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement
from pathlib import Path
import tempfile
import zipfile
import shutil
import subprocess
from PIL import Image
from pypdf import PdfReader, PdfWriter

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_PATH = REPO_ROOT / "source" / "Original_20_slide_Clay_Microbes_Metal_Deck.pptx"
OUT_22_PPTX = REPO_ROOT / "presentations" / "Clay_Microbes_Metal_22slides_Flow_Cytometry_Addendum.pptx"
OUT_22_PDF = REPO_ROOT / "presentations" / "Clay_Microbes_Metal_22slides_Flow_Cytometry_Addendum.pdf"
OUT_8_PPTX = REPO_ROOT / "presentations" / "Clay_Microbes_Metal_Wan_Hoffman_Meeting_PreRead.pptx"
OUT_8_PDF = REPO_ROOT / "presentations" / "Clay_Microbes_Metal_Wan_Hoffman_Meeting_PreRead.pdf"

# Deck palette sampled/derived from the source deck.
BG = 'F6F5F0'
INK = '273238'
MUTED = '536168'
OCHRE = 'A86418'
TEAL = '3C6D65'
PALE_TEAL = 'E1EBE6'
PALE_BLUE = 'E4EDF4'
PALE_GOLD = 'F5E5C5'
PALE_GRAY = 'E5E4DF'
WHITE = 'FFFFFF'
BORDER = 'D4D2CB'
BLUE = '4A779D'
RED = 'A34C43'


def rgb(h):
    h = h.lstrip('#')
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def set_bg(slide, color=BG):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = rgb(color)


def add_text(slide, text, x, y, w, h, size=14, color=INK, font='Arial', bold=False,
             align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP, margin=0.02, italic=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin)
    tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    for i, line in enumerate(text.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.space_after = Pt(5 if len(text.split('\n')) > 1 else 0)
        for r in p.runs:
            r.font.name = font
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.italic = italic
            r.font.color.rgb = rgb(color)
    return box


def add_card(slide, x, y, w, h, fill, title, body, title_color=INK, body_size=13.1, title_size=16):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill)
    sh.line.color.rgb = rgb(BORDER); sh.line.width = Pt(0.8)
    add_text(slide, title, x+0.18, y+0.14, w-0.36, 0.38, size=title_size, color=title_color, bold=True, font='Arial')
    add_text(slide, body, x+0.18, y+0.61, w-0.36, h-0.76, size=body_size, color=INK, font='Arial')
    return sh


def add_arrow(slide, x1, y1, x2, y2, color=TEAL, width=2):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    line.line.color.rgb = rgb(color)
    line.line.width = Pt(width)
    ln = line._element.spPr.ln
    head = OxmlElement('a:headEnd')
    head.set('type', 'none')
    end = OxmlElement('a:tailEnd')
    end.set('type', 'triangle')
    ln.append(head)
    return line


def add_arrowhead(slide, x, y, color=TEAL, direction='right', size=0.16):
    tri = slide.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE, Inches(x), Inches(y), Inches(size), Inches(size))
    tri.fill.solid(); tri.fill.fore_color.rgb = rgb(color)
    tri.line.fill.background()
    tri.rotation = {'up': 0, 'right': 90, 'down': 180, 'left': 270}[direction]
    return tri


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
                    # Replace en-dash bullet character with bullet dot
                    text = text.replace('char="\u2013"', 'char="\u2022"')
                    text = text.replace("char='\u2013'", "char='\u2022'")
                    # Replace em-dashes with ' - '
                    text = text.replace("\u2014", " - ")
                    # Replace en-dashes with '-'
                    text = text.replace("\u2013", "-")
                    items[name] = text.encode("utf-8")
                else:
                    items[name] = content
        with zipfile.ZipFile(tmp_pptx, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for k, v in items.items():
                zout.writestr(k, v)
        shutil.copyfile(tmp_pptx, pptx_path)


def sanitize_pdf_dashes(pdf_path: Path):
    """Sanitize all embedded ToUnicode CMaps in PDF to map em-dashes and en-dashes to standard ASCII hyphens."""
    if not pdf_path.exists():
        return
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    writer.append(reader)
    modified = False
    for page in writer.pages:
        fonts = page.get("/Resources", {}).get("/Font", {})
        for fname, fobj in fonts.items():
            f = fobj.get_object()
            if "/ToUnicode" in f:
                tounicode_obj = f["/ToUnicode"].get_object()
                data = tounicode_obj.get_data()
                if b"<2013>" in data or b"<2014>" in data:
                    new_data = data.replace(b"<2013>", b"<002D>").replace(b"<2014>", b"<002D>")
                    tounicode_obj.set_data(new_data)
                    modified = True
    if modified:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            writer.write(str(tmp_path))
        shutil.copyfile(tmp_path, pdf_path)
        tmp_path.unlink(missing_ok=True)
        print(f"Sanitized PDF dashes: {pdf_path}")


def update_appendix_previews():
    """Ensure Appendix Slide 21 and 22 preview PNGs exist at minimum 1920x1080 resolution."""
    prev_dir = REPO_ROOT / "previews"
    p21 = prev_dir / "Appendix_Slide_21_Flow_Cytometry.png"
    p22 = prev_dir / "Appendix_Slide_22_Model_Bridge.png"
    rendered = False
    pdftoppm = shutil.which("pdftoppm") or "/opt/homebrew/bin/pdftoppm"
    if OUT_22_PDF.exists() and Path(pdftoppm).exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                subprocess.check_call([
                    pdftoppm, "-png", "-r", "150", "-f", "21", "-l", "22",
                    str(OUT_22_PDF), f"{tmpdir}/slide"
                ])
                im21 = Image.open(f"{tmpdir}/slide-21.png").resize((1920, 1080), Image.Resampling.LANCZOS)
                im22 = Image.open(f"{tmpdir}/slide-22.png").resize((1920, 1080), Image.Resampling.LANCZOS)
                im21.save(p21)
                im22.save(p22)
                rendered = True
                print("Rendered high-resolution appendix previews via pdftoppm (1920x1080)")
            except Exception as e:
                print(f"Warning: pdftoppm render failed: {e}")
    if not rendered:
        for p in [p21, p22]:
            if p.exists():
                im = Image.open(p)
                if im.size != (1920, 1080):
                    im_hires = im.resize((1920, 1080), Image.Resampling.LANCZOS)
                    im_hires.save(p)
                    print(f"Upscaled preview image to 1920x1080: {p}")


def validate_pptx_package(pptx_path: Path, expected_slide_count: int):
    """Validate that the PPTX archive is a valid, well-formed OpenXML package with zero em/en-dashes."""
    import xml.etree.ElementTree as ET

    if not pptx_path.exists():
        raise FileNotFoundError(f"Presentation file does not exist: {pptx_path}")

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


def build_decks():
    # 1. Load source 20-slide presentation
    prs = Presentation(str(BASE_PATH))
    SW, SH = prs.slide_width, prs.slide_height

    # Slide 21 - optional measurement addendum.
    slide21 = prs.slides.add_slide(prs.slide_layouts[0])
    set_bg(slide21)
    add_text(slide21, 'PROPOSED MEASUREMENT LAYER . FLOW CYTOMETRY', 0.88, 0.79, 11.85, 0.28,
             size=11.8, color=OCHRE, font='Arial')
    add_text(slide21, 'Flow cytometry adds cell-state observations to the interface model', 0.88, 1.28, 11.95, 0.78,
             size=24.5, color=INK, font='Georgia', bold=True)
    add_text(slide21, 'It measures suspended-cell phenotypes; by itself, it does not preserve where cells sat inside clay or a biofilm.',
             0.9, 2.04, 11.9, 0.38, size=13.4, color=MUTED)

    x0, w, gap, y, h = 0.88, 3.82, 0.26, 2.65, 2.75
    add_card(slide21, x0, y, w, h, PALE_BLUE, 'WHAT IT CAN MEASURE',
             '• Gated cell-event counts and concentration using counting beads\n'
             '• Dye-defined membrane-integrity classes, after strain-specific validation\n'
             '• Scatter and per-cell fluorescence distributions\n'
             '• Optional metabolic / ROS panels only after assay validation', title_color=BLUE, body_size=12.35)
    add_card(slide21, x0+w+gap, y, w, h, PALE_TEAL, 'WHAT IT CONSTRAINS',
             '• Suspended and detached-cell abundance\n'
             '• Fractions in measured physiological-state gates\n'
             '• Net growth/death or attachment/detachment rates when time series are available\n'
             '• Equifinality: similar bulk chemistry can arise from different cell-state histories', title_color=TEAL, body_size=12.35)
    add_card(slide21, x0+2*(w+gap), y, w, h, PALE_GOLD, 'LIMITS',
             '• Spatial position in the clay pore network\n'
             '• EPS cross-link density or mineral phase\n'
             '• Corrosion rate / causal mechanism\n'
             '• "Live" status from fluorescence alone without controls; clay particles and aggregates can distort gates', title_color=OCHRE, body_size=12.35)

    banner21 = slide21.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.88), Inches(5.62), Inches(11.94), Inches(0.72))
    banner21.fill.solid(); banner21.fill.fore_color.rgb = rgb(PALE_GOLD); banner21.line.fill.background()
    add_text(slide21, 'MINIMUM VIABLE ADD-ON: absolute counts + a validated membrane-integrity panel at preselected timepoints; keep maximum pit depth as the primary corrosion endpoint.',
             1.08, 5.78, 11.54, 0.43, size=13.1, color=INK, bold=True, valign=MSO_ANCHOR.MIDDLE)
    add_text(slide21, 'Sources: Kerstens et al. (2015), doi:10.1007/s12223-015-0400-4; Grainha et al. (2020), doi:10.3390/antibiotics9110741; Yip et al. (2022), doi:10.1371/journal.pcbi.1010533.',
             0.88, 6.82, 11.95, 0.28, size=9.6, color=MUTED)
    try:
        slide21.notes_slide.notes_text_frame.text = (
            'OPTIONAL APPENDIX SLIDE. Flow cytometry is an observation layer, not a fluid or geochemical solver. '
            'It generally requires cell suspensions. For attached biofilm/clay, validate detachment and recovery and report extraction protocol. '
            'Kerstens et al. (2015) demonstrated a flow-cytometric biofilm quantification workflow in E. coli and C. albicans; do not transfer its staining interpretation automatically to a sulfate-reducing strain. '
            'Grainha et al. (2020) document difficulties caused by strain, morphology and matrix effects in mixed biofilms. Use clay-only, unstained, stain-only, positive/negative viability, counting-bead and spike-recovery controls. '
            'Do not interpret this as proof that flow cytometry directly measures EPS cross-links or spatial location. Maximum pit depth remains the primary corrosion endpoint.'
        )
    except Exception:
        pass

    # Slide 22 - coupling architecture.
    slide22 = prs.slides.add_slide(prs.slide_layouts[0])
    set_bg(slide22)
    add_text(slide22, 'PROPOSED MODEL INTERFACE . NOT YET IMPLEMENTED', 0.88, 0.79, 11.85, 0.28,
             size=11.8, color=OCHRE, font='Arial')
    add_text(slide22, 'Keep the physics solvers; add a cytometry observation operator', 0.88, 1.28, 11.95, 0.72,
             size=25.0, color=INK, font='Georgia', bold=True)
    add_text(slide22, 'OpenFOAM and PHREEQC drive the forward model; flow cytometry constrains the biological state through measured summaries.',
             0.9, 1.98, 11.9, 0.34, size=13.4, color=MUTED)

    # Sources (left, stacked)
    add_card(slide22, 0.88, 2.55, 2.48, 1.28, PALE_GRAY, 'OPENFOAM',
             'FLOW: u(x,t), shear, residence time, transport / detachment cues', title_color=MUTED, body_size=11.2, title_size=14)
    add_card(slide22, 0.88, 4.10, 2.48, 1.28, PALE_GOLD, 'PHREEQC',
             'AQUEOUS CHEMISTRY: pH / pe, ionic strength, Ca/Mg, Fe/S speciation, mineral saturation', title_color=OCHRE, body_size=11.0, title_size=14)
    add_card(slide22, 4.05, 3.16, 2.65, 1.62, PALE_TEAL, 'JULIA . CELL STATES',
             'Attached / suspended cells; membrane-intact, compromised and dead-like states; EPS proxy; growth, death and detachment rates', title_color=TEAL, body_size=11.0, title_size=14)
    add_card(slide22, 7.37, 3.16, 2.34, 1.62, PALE_BLUE, 'PREDICTED FCM SUMMARY',
             'Predict gate fractions and absolute cell concentration from the latent state and gate calibration', title_color=BLUE, body_size=11.0, title_size=13.2)
    add_card(slide22, 10.33, 2.55, 2.48, 1.28, WHITE, 'MEASURED FCM DATA',
             'Gate counts, bead events, dilution, sample origin, extraction recovery', title_color=BLUE, body_size=10.2, title_size=13.0)
    add_card(slide22, 10.33, 4.10, 2.48, 1.28, PALE_GOLD, 'LIKELIHOOD + CHECKS',
             'Compare observed with predicted counts / fractions; check uncertainty, predictive fit and rival mechanisms', title_color=OCHRE, body_size=9.8, title_size=12.3)

    add_arrow(slide22, 3.36, 3.18, 4.00, 3.68, color=MUTED, width=1.8)
    add_arrowhead(slide22, 3.88, 3.60, color=MUTED, direction='right', size=0.16)
    add_arrow(slide22, 3.36, 4.72, 4.00, 4.22, color=OCHRE, width=1.8)
    add_arrowhead(slide22, 3.88, 4.14, color=OCHRE, direction='right', size=0.16)
    add_arrow(slide22, 6.70, 3.97, 7.32, 3.97, color=TEAL, width=2.0)
    add_arrowhead(slide22, 7.22, 3.89, color=TEAL, direction='right', size=0.16)
    add_arrow(slide22, 9.67, 4.54, 10.28, 4.65, color=BLUE, width=1.8)
    add_arrowhead(slide22, 10.18, 4.57, color=BLUE, direction='right', size=0.16)
    add_arrow(slide22, 11.57, 3.87, 11.57, 4.05, color=OCHRE, width=1.8)
    add_arrowhead(slide22, 11.49, 3.95, color=OCHRE, direction='down', size=0.16)

    add_text(slide22, 'OBSERVED DATA', 10.43, 3.79, 2.20, 0.18, size=8.7, color=BLUE, bold=True, align=PP_ALIGN.CENTER)

    banner22 = slide22.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.88), Inches(5.72), Inches(11.94), Inches(0.68))
    banner22.fill.solid(); banner22.fill.fore_color.rgb = rgb(PALE_GOLD); banner22.line.fill.background()
    add_text(slide22, 'BOUNDARY: cytometry loses spatial location; pair it with microscopy / endpoint surface analysis. Do not force OpenFOAM or PHREEQC into a nanocarrier assay unless those physical and aqueous-chemistry fields are genuinely part of that experiment.',
             1.08, 5.82, 11.50, 0.46, size=12.1, color=INK, bold=True, valign=MSO_ANCHOR.MIDDLE)
    add_text(slide22, 'Calibration framework: Yip et al. (2022), PLoS Comput Biol 18(10):e1010533. Lab context: Nano-Bio Lab, Single-Cell Analysis and computational cell classification.',
             0.88, 6.82, 11.95, 0.28, size=9.6, color=MUTED)
    try:
        slide22.notes_slide.notes_text_frame.text = (
            'OPTIONAL APPENDIX SLIDE. This is an architecture proposal, not a claimed integration already in code. '
            'OpenFOAM produces hydrodynamic fields; PHREEQC produces aqueous speciation / mineral chemistry; Julia couples the biological state equations and builds the observation operator. '
            'Flow cytometry is not a spatial sensor: model output must be aggregated to the sampled compartment (effluent, planktonic suspension, or detached biofilm) and passed through a gate-calibration / sampling model. '
            'A simple first likelihood can use multinomial gate counts conditioned on total gated events and a separate count model for absolute concentration from counting beads, with uncertainty and extraction recovery. Avoid double-counting the same event total as two independent likelihood terms. '
            'The current 20-slide deck is explicitly for Ellen Wan and Andrew Hoffman; these two slides are optional appendices for the clay/biofilm pilot. For Dr. Ghosh, use a separate, lab-specific framing around single-cell/nanobio measurements. '
            'Do not imply that flow cytometry directly measures EPS bridges, local mineral phases or spatial cell distribution.'
        )
    except Exception:
        pass

    # Save 22-slide deck
    OUT_22_PPTX.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_22_PPTX))
    sanitize_pptx_dashes(OUT_22_PPTX)
    sanitize_pdf_dashes(OUT_22_PDF)
    print(f"Built and sanitized 22-slide deck: {OUT_22_PPTX}")

    # Build 8-slide meeting pre-read deck
    prs8 = Presentation(str(OUT_22_PPTX))
    # Indices in 22-slide deck: 0: Slide 1, 1: Slide 2, 2: Slide 3, 7: Slide 8, 12: Slide 13, 20: Slide 21, 14: Slide 15, 15: Slide 16
    keep_indices = [0, 1, 2, 7, 12, 20, 14, 15]
    keep_sldIds = [prs8.slides._sldIdLst[i] for i in keep_indices]
    keep_set = set(keep_sldIds)

    for s in list(prs8.slides._sldIdLst):
        if s not in keep_set:
            rId = s.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
            prs8.part.drop_rel(rId)
        prs8.slides._sldIdLst.remove(s)

    prs8.slides._sldIdLst[:] = keep_sldIds

    # Enhance Slide 1 (title & framing bridge)
    s1 = prs8.slides[0]
    add_text(s1, "FRAMING BRIDGE: Connecting the October 26 pool-water coupon concept with the compacted clay-EPS-metal interface as a complementary option, not a unilateral replacement.",
             0.88, 5.85, 11.5, 0.40, size=11.5, color=OCHRE, bold=True)

    # Enhance Slide 4 (formerly slide 8: mechanism separation & equifinality)
    s4 = prs8.slides[3]
    add_text(s4, "EQUIFINALITY NOTE: Distinct causal pathways (microbial sulfide, abiotic redox, clay transport) can produce similar corroded coupon endpoints, requiring discriminating controls. Biofilms and abiotic mineral compartments are comparable spatially structured reaction-diffusion systems; biofilms themselves are biological.",
             0.88, 6.45, 11.5, 0.45, size=9.8, color=OCHRE, bold=True)

    prs8.save(str(OUT_8_PPTX))
    sanitize_pptx_dashes(OUT_8_PPTX)
    print(f"Built and sanitized 8-slide pre-read deck: {OUT_8_PPTX}")

    # Assemble 8-page PDF from 22-slide PDF if available
    if OUT_22_PDF.exists():
        reader = PdfReader(str(OUT_22_PDF))
        if len(reader.pages) >= 22:
            writer = PdfWriter()
            for idx in keep_indices:
                writer.add_page(reader.pages[idx])
            with open(str(OUT_8_PDF), "wb") as f:
                writer.write(f)
            sanitize_pdf_dashes(OUT_8_PDF)
            print(f"Built and sanitized 8-page pre-read PDF: {OUT_8_PDF}")

    # Update appendix preview assets to (1920, 1080)
    update_appendix_previews()

    # Automated post-build validation
    validate_pptx_package(OUT_22_PPTX, 22)
    validate_pptx_package(OUT_8_PPTX, 8)


if __name__ == "__main__":
    build_decks()
