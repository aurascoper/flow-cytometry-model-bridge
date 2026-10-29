from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement
from pathlib import Path

base = Path('/mnt/data/clay_microbes_metal_20slides_4axis_rubric_gifs(1).pptx')
out = Path('/mnt/data/Clay_Microbes_Metal_22slides_Flow_Cytometry_Addendum.pptx')
prs = Presentation(str(base))
SW, SH = prs.slide_width, prs.slide_height

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
    # allow multiple paragraphs using newline
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
    # soften default roundness via shape is okay
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

# Slide 21 - optional measurement addendum.
slide = prs.slides.add_slide(prs.slide_layouts[0])
set_bg(slide)
add_text(slide, 'PROPOSED MEASUREMENT LAYER · FLOW CYTOMETRY', 0.88, 0.79, 11.85, 0.28,
         size=11.8, color=OCHRE, font='Arial')
add_text(slide, 'Flow cytometry adds cell-state observations to the interface model', 0.88, 1.28, 11.95, 0.78,
         size=24.5, color=INK, font='Georgia', bold=True)
add_text(slide, 'It measures suspended-cell phenotypes; by itself, it does not preserve where cells sat inside clay or a biofilm.',
         0.9, 2.04, 11.9, 0.38, size=13.4, color=MUTED)

x0, w, gap, y, h = 0.88, 3.82, 0.26, 2.65, 2.75
add_card(slide, x0, y, w, h, PALE_BLUE, 'WHAT IT CAN MEASURE',
         '• Gated cell-event counts and concentration using counting beads\n'
         '• Dye-defined membrane-integrity classes, after strain-specific validation\n'
         '• Scatter and per-cell fluorescence distributions\n'
         '• Optional metabolic / ROS panels only after assay validation', title_color=BLUE, body_size=12.35)
add_card(slide, x0+w+gap, y, w, h, PALE_TEAL, 'WHAT IT CONSTRAINS',
         '• Suspended and detached-cell abundance\n'
         '• Fractions in measured physiological-state gates\n'
         '• Net growth/death or attachment/detachment rates when time series are available\n'
         '• Equifinality: similar bulk chemistry can arise from different cell-state histories', title_color=TEAL, body_size=12.35)
add_card(slide, x0+2*(w+gap), y, w, h, PALE_GOLD, 'LIMITS',
         '• Spatial position in the clay pore network\n'
         '• EPS cross-link density or mineral phase\n'
         '• Corrosion rate / causal mechanism\n'
         '• “Live” status from fluorescence alone without controls; clay particles and aggregates can distort gates', title_color=OCHRE, body_size=12.35)

banner = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.88), Inches(5.62), Inches(11.94), Inches(0.72))
banner.fill.solid(); banner.fill.fore_color.rgb = rgb(PALE_GOLD); banner.line.fill.background()
add_text(slide, 'MINIMUM VIABLE ADD-ON: absolute counts + a validated membrane-integrity panel at preselected timepoints; keep maximum pit depth as the primary corrosion endpoint.',
         1.08, 5.78, 11.54, 0.43, size=13.1, color=INK, bold=True, valign=MSO_ANCHOR.MIDDLE)
add_text(slide, 'Sources: Kerstens et al. (2015), doi:10.1007/s12223-015-0400-4; Grainha et al. (2020), doi:10.3390/antibiotics9110741; Yip et al. (2022), doi:10.1371/journal.pcbi.1010533.',
         0.88, 6.82, 11.95, 0.28, size=9.6, color=MUTED)
try:
    slide.notes_slide.notes_text_frame.text = (
        'OPTIONAL APPENDIX SLIDE. Flow cytometry is an observation layer, not a fluid or geochemical solver. '
        'It generally requires cell suspensions. For attached biofilm/clay, validate detachment and recovery and report extraction protocol. '
        'Kerstens et al. (2015) demonstrated a flow-cytometric biofilm quantification workflow in E. coli and C. albicans; do not transfer its staining interpretation automatically to a sulfate-reducing strain. '
        'Grainha et al. (2020) document difficulties caused by strain, morphology and matrix effects in mixed biofilms. Use clay-only, unstained, stain-only, positive/negative viability, counting-bead and spike-recovery controls. '
        'Do not interpret this as proof that flow cytometry directly measures EPS cross-links or spatial location. Maximum pit depth remains the primary corrosion endpoint.'
    )
except Exception:
    pass

# Slide 22 - coupling architecture.
slide = prs.slides.add_slide(prs.slide_layouts[0])
set_bg(slide)
add_text(slide, 'PROPOSED MODEL INTERFACE · NOT YET IMPLEMENTED', 0.88, 0.79, 11.85, 0.28,
         size=11.8, color=OCHRE, font='Arial')
add_text(slide, 'Keep the physics solvers; add a cytometry observation operator', 0.88, 1.28, 11.95, 0.72,
         size=25.0, color=INK, font='Georgia', bold=True)
add_text(slide, 'OpenFOAM and PHREEQC drive the forward model; flow cytometry constrains the biological state through measured summaries.',
         0.9, 1.98, 11.9, 0.34, size=13.4, color=MUTED)

# Sources (left, stacked)
add_card(slide, 0.88, 2.55, 2.48, 1.28, PALE_GRAY, 'OPENFOAM',
         'FLOW: u(x,t), shear, residence time, transport / detachment cues', title_color=MUTED, body_size=11.2, title_size=14)
add_card(slide, 0.88, 4.10, 2.48, 1.28, PALE_GOLD, 'PHREEQC',
         'AQUEOUS CHEMISTRY: pH / pe, ionic strength, Ca/Mg, Fe/S speciation, mineral saturation', title_color=OCHRE, body_size=11.0, title_size=14)
# Julia model, observation model and data
add_card(slide, 4.05, 3.16, 2.65, 1.62, PALE_TEAL, 'JULIA · CELL STATES',
         'Attached / suspended cells; membrane-intact, compromised and dead-like states; EPS proxy; growth, death and detachment rates', title_color=TEAL, body_size=11.0, title_size=14)
add_card(slide, 7.37, 3.16, 2.34, 1.62, PALE_BLUE, 'PREDICTED FCM SUMMARY',
         'Predict gate fractions and absolute cell concentration from the latent state and gate calibration', title_color=BLUE, body_size=11.0, title_size=13.2)
add_card(slide, 10.33, 2.55, 2.48, 1.28, WHITE, 'MEASURED FCM DATA',
         'Gate counts, bead events, dilution, sample origin, extraction recovery', title_color=BLUE, body_size=10.2, title_size=13.0)
add_card(slide, 10.33, 4.10, 2.48, 1.28, PALE_GOLD, 'LIKELIHOOD + CHECKS',
         'Compare observed with predicted counts / fractions; check uncertainty, predictive fit and rival mechanisms', title_color=OCHRE, body_size=9.8, title_size=12.3)
# Arrows
add_arrow(slide, 3.36, 3.18, 4.00, 3.68, color=MUTED, width=1.8)
add_arrowhead(slide, 3.88, 3.60, color=MUTED, direction='right', size=0.16)
add_arrow(slide, 3.36, 4.72, 4.00, 4.22, color=OCHRE, width=1.8)
add_arrowhead(slide, 3.88, 4.14, color=OCHRE, direction='right', size=0.16)
add_arrow(slide, 6.70, 3.97, 7.32, 3.97, color=TEAL, width=2.0)
add_arrowhead(slide, 7.22, 3.89, color=TEAL, direction='right', size=0.16)
add_arrow(slide, 9.67, 4.54, 10.28, 4.65, color=BLUE, width=1.8)
add_arrowhead(slide, 10.18, 4.57, color=BLUE, direction='right', size=0.16)
add_arrow(slide, 11.57, 3.87, 11.57, 4.05, color=OCHRE, width=1.8)
add_arrowhead(slide, 11.49, 3.95, color=OCHRE, direction='down', size=0.16)
# Flow-cytometry data enter the likelihood; the operator output enters from the left.
add_text(slide, 'OBSERVED DATA', 10.43, 3.79, 2.20, 0.18, size=8.7, color=BLUE, bold=True, align=PP_ALIGN.CENTER)

banner = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.88), Inches(5.72), Inches(11.94), Inches(0.68))
banner.fill.solid(); banner.fill.fore_color.rgb = rgb(PALE_GOLD); banner.line.fill.background()
add_text(slide, 'BOUNDARY: cytometry loses spatial location; pair it with microscopy / endpoint surface analysis. Do not force OpenFOAM or PHREEQC into a nanocarrier assay unless those physical and aqueous-chemistry fields are genuinely part of that experiment.',
         1.08, 5.82, 11.50, 0.46, size=12.1, color=INK, bold=True, valign=MSO_ANCHOR.MIDDLE)
add_text(slide, 'Calibration framework: Yip et al. (2022), PLoS Comput Biol 18(10):e1010533. Lab context: Nano-Bio Lab, Single-Cell Analysis and computational cell classification.',
         0.88, 6.82, 11.95, 0.28, size=9.6, color=MUTED)
try:
    slide.notes_slide.notes_text_frame.text = (
        'OPTIONAL APPENDIX SLIDE. This is an architecture proposal, not a claimed integration already in code. '
        'OpenFOAM produces hydrodynamic fields; PHREEQC produces aqueous speciation / mineral chemistry; Julia couples the biological state equations and builds the observation operator. '
        'Flow cytometry is not a spatial sensor: model output must be aggregated to the sampled compartment (effluent, planktonic suspension, or detached biofilm) and passed through a gate-calibration / sampling model. '
        'A simple first likelihood can use multinomial gate counts conditioned on total gated events and a separate count model for absolute concentration from counting beads, with uncertainty and extraction recovery. Avoid double-counting the same event total as two independent likelihood terms. '
        'The current 20-slide deck is explicitly for Ellen Wan and Andrew Hoffman; these two slides are optional appendices for the clay/biofilm pilot. For Dr. Ghosh, use a separate, lab-specific framing around single-cell/nanobio measurements. '
        'Do not imply that flow cytometry directly measures EPS bridges, local mineral phases or spatial cell distribution.'
    )
except Exception:
    pass

prs.save(str(out))
print(out)
print('slides:', len(prs.slides))
