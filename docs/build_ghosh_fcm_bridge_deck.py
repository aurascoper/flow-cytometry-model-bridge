from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement
from pathlib import Path

out=Path('/mnt/data/Single_Cell_Flow_Cytometry_Model_Bridge_Exploratory.pptx')
prs=Presentation()
prs.slide_width= Inches(13.333333)
prs.slide_height= Inches(7.5)
BG='F6F5F0'; INK='273238'; MUTED='536168'; OCHRE='A86418'; TEAL='3C6D65'; BLUE='4A779D'; RED='A34C43'; BORDER='D4D2CB'; WHITE='FFFFFF'; PALE_TEAL='E1EBE6'; PALE_BLUE='E4EDF4'; PALE_GOLD='F5E5C5'; PALE_GRAY='E5E4DF'; PALE_RED='F3E2DE'
def rgb(h): return RGBColor(int(h[0:2],16),int(h[2:4],16),int(h[4:6],16))
def bg(sl):
    sl.background.fill.solid(); sl.background.fill.fore_color.rgb=rgb(BG)
def text(sl,t,x,y,w,h,size=14,color=INK,font='Arial',bold=False,align=PP_ALIGN.LEFT,valign=MSO_ANCHOR.TOP,margin=.02):
    sh=sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf=sh.text_frame; tf.clear(); tf.word_wrap=True
    tf.margin_left=Inches(margin); tf.margin_right=Inches(margin); tf.margin_top=Inches(margin); tf.margin_bottom=Inches(margin); tf.vertical_anchor=valign
    for i,line in enumerate(t.split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph(); p.text=line; p.alignment=align
        p.space_after=Pt(4)
        for r in p.runs: r.font.name=font; r.font.size=Pt(size); r.font.bold=bold; r.font.color.rgb=rgb(color)
    return sh
def card(sl,x,y,w,h,fill,title,body,accent=INK,body_size=12.5,title_size=15):
    sh=sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb=rgb(fill); sh.line.color.rgb=rgb(BORDER); sh.line.width=Pt(.8)
    text(sl,title,x+.16,y+.13,w-.32,.34,size=title_size,color=accent,bold=True)
    text(sl,body,x+.16,y+.54,w-.32,h-.65,size=body_size,color=INK)
    return sh
def arrow(sl,x1,y1,x2,y2,color=TEAL):
    sh=sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)); sh.line.color.rgb=rgb(color); sh.line.width=Pt(2)
    tri=sl.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE, Inches(x2-.08), Inches(y2-.08), Inches(.16), Inches(.16)); tri.fill.solid(); tri.fill.fore_color.rgb=rgb(color); tri.line.fill.background(); tri.rotation=90
    return sh
def header(sl,eyebrow,title,subtitle=None):
    text(sl,eyebrow.upper(),.88,.78,11.8,.28,size=11.5,color=OCHRE)
    text(sl,title,.88,1.22,11.9,.75,size=25,color=INK,font='Georgia',bold=True)
    if subtitle: text(sl,subtitle,.9,1.98,11.8,.42,size=13.2,color=MUTED)
def footer(sl,s): text(sl,s,.88,6.86,11.9,.24,size=9.2,color=MUTED)
def notes(sl,s):
    try: sl.notes_slide.notes_text_frame.text=s
    except Exception: pass

# Slide 1
sl=prs.slides.add_slide(prs.slide_layouts[0]); bg(sl)
header(sl,'Exploratory methods bridge · not a validated model','The missing bridge is an observation model','A simulation predicts hidden states; flow cytometry measures a gated sample of cell-level outcomes.')
card(sl,.88,2.72,3.75,2.58,PALE_GRAY,'1 · FORWARD MODEL','OpenFOAM: flow, shear and residence time\nPHREEQC: aqueous speciation and mineral equilibria\nJulia: cell-state, reaction and transport dynamics',accent=MUTED,body_size=12.1,title_size=15)
card(sl,4.82,2.72,3.75,2.58,PALE_BLUE,'2 · OBSERVATION OPERATOR','Project latent state to sample-specific outputs:\n• absolute gated-cell concentration\n• membrane-integrity gate fractions\n• fluorescence / scatter distributions\nInclude calibration, sampling and recovery.',accent=BLUE,body_size=12.1,title_size=15)
card(sl,8.77,2.72,3.75,2.58,PALE_TEAL,'3 · INFERENCE & FALSIFICATION','Compare predicted and measured summaries.\nEstimate uncertainty in rates.\nTest rival mechanisms and held-out time points.\nDo not treat one good fit as unique causality.',accent=TEAL,body_size=12.1,title_size=15)
arrow(sl,4.65,4.0,4.78,4.0,TEAL); arrow(sl,8.60,4.0,8.72,4.0,TEAL)
ban=sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(.88), Inches(5.62), Inches(11.94), Inches(.68)); ban.fill.solid(); ban.fill.fore_color.rgb=rgb(PALE_GOLD); ban.line.fill.background()
text(sl,'Flow cytometry belongs in the measurement / likelihood layer, not inside the CFD or geochemical solver.',1.1,5.77,11.5,.38,size=13.1,bold=True,valign=MSO_ANCHOR.MIDDLE)
footer(sl,'Calibration precedent: Yip et al. (2022), PLOS Comput Biol 18(10):e1010533, https://doi.org/10.1371/journal.pcbi.1010533')
notes(sl,'This slide is an architecture proposal. The Nano-Bio Lab public research page emphasizes single-cell analysis, computational cell classification and signal processing; the user indicates flow cytometry is also part of Dr. Ghosh’s work. Flow cytometry observes sampled cell populations, not the full spatial field. Yip et al. review calibration against microscopy and flow cytometry and note that rigorous validation remains challenging.')

# Slide 2
sl=prs.slides.add_slide(prs.slide_layouts[0]); bg(sl)
header(sl,'Architecture · keep the solvers in their lanes','One modular stack, two independent kinds of inputs','Only couple a solver if its fields represent a real part of the target experiment.')
card(sl,.88,2.55,2.55,1.45,PALE_GRAY,'OPENFOAM','Velocity u(x,t), pressure, shear, residence time and transport.',accent=MUTED,body_size=11.4,title_size=14)
card(sl,.88,4.28,2.55,1.45,PALE_GOLD,'PHREEQC','pH / pe, ionic strength, aqueous species, Fe/S chemistry and mineral saturation.',accent=OCHRE,body_size=11.2,title_size=14)
card(sl,4.05,3.31,2.65,1.70,PALE_TEAL,'JULIA STATE MODEL','Attached / suspended cells; state fractions; growth, death, attachment, detachment and EPS proxy.',accent=TEAL,body_size=11.5,title_size=14)
card(sl,7.40,3.31,2.45,1.70,PALE_BLUE,'FCM PREDICTION','Predict the exact quantities the gates report, including calibration and sample-origin effects.',accent=BLUE,body_size=11.4,title_size=13.2)
card(sl,10.53,2.55,2.29,1.45,WHITE,'MEASURED FCM','Raw events, saved gates, counting beads, dilution and recovery.',accent=BLUE,body_size=10.8,title_size=13)
card(sl,10.53,4.28,2.29,1.45,PALE_RED,'MODEL CHECK','Compare distributions; quantify uncertainty and check rival models.',accent=RED,body_size=10.8,title_size=13)
arrow(sl,3.43,3.25,4.01,3.79,MUTED); arrow(sl,3.43,4.94,4.01,4.42,OCHRE); arrow(sl,6.70,4.16,7.35,4.16,TEAL); arrow(sl,9.85,4.60,10.48,4.94,BLUE); arrow(sl,11.66,4.02,11.66,4.25,OCHRE)
ban=sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(.88), Inches(5.93), Inches(11.94), Inches(.51)); ban.fill.solid(); ban.fill.fore_color.rgb=rgb(PALE_GOLD); ban.line.fill.background()
text(sl,'FCM loses spatial position: pair cell-state distributions with microscopy and endpoint surface / mineral analysis.',1.06,6.01,11.55,.34,size=12.5,bold=True,valign=MSO_ANCHOR.MIDDLE)
footer(sl,'Nano-Bio Lab research overview: https://www.nanobiolab.org/research · Model-calibration review: https://doi.org/10.1371/journal.pcbi.1010533')
notes(sl,'OpenFOAM and PHREEQC are conditional, not mandatory. They fit only if a study has fluid-mechanical and aqueous-speciation questions. A nanocarrier/cell experiment may need neither solver, while the clay/corrosion model may need both. Do not transplant the parameters between the two application branches.')

# Slide 3
sl=prs.slides.add_slide(prs.slide_layouts[0]); bg(sl)
header(sl,'Separate applications · share the software pattern, not the parameters','Two possible flow-cytometry branches should stay separate','The same observation-model idea can serve different biology, but the targets and calibration differ.')
card(sl,.88,2.60,5.75,2.77,PALE_BLUE,'A · NANO-BIO / CELL-ASSOCIATION ASSAY','Possible observables:\n• percent fluorescent-positive cells\n• per-cell fluorescence distribution\n• validated cell-state / membrane-integrity markers\n• scatter / morphology proxies\n\nKey control: distinguish surface association from internalization; cytometry alone may not localize the signal.',accent=BLUE,body_size=12.1,title_size=15)
card(sl,6.87,2.60,5.95,2.77,PALE_TEAL,'B · MICROBIAL / CLAY-BIOFILM ASSAY','Possible observables:\n• absolute gated-cell concentration\n• membrane-integrity state fractions\n• detached vs planktonic sample differences\n• optional validated metabolic panel\n\nKey control: clay-only blanks, disaggregation recovery, staining controls and oxygen-safe sampling.',accent=TEAL,body_size=12.1,title_size=15)
ban=sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(.88), Inches(5.67), Inches(11.94), Inches(.69)); ban.fill.solid(); ban.fill.fore_color.rgb=rgb(PALE_GOLD); ban.line.fill.background()
text(sl,'Do not merge the hypotheses: a nanocarrier uptake model is not the clay–corrosion model. Re-estimate chemistry, kinetics, gates and parameters for each system.',1.07,5.80,11.55,.46,size=12.5,bold=True,valign=MSO_ANCHOR.MIDDLE)
footer(sl,'Flow-cytometric biofilm methods: Kerstens et al. (2015), https://doi.org/10.1007/s12223-015-0400-4; Grainha et al. (2020), https://doi.org/10.3390/antibiotics9110741')
notes(sl,'For a Ghosh-facing conversation, lead with the single-cell/nanobio branch only if it matches the actual assay he runs. For Ellen Wan and Andrew Hoffman, the microbial/clay branch is relevant. The Nano-Bio Lab public page describes single-cell analysis and computational analysis of cell morphology; do not state that the current project uses a particular cytometer protocol unless confirmed.')

# Slide 4
sl=prs.slides.add_slide(prs.slide_layouts[0]); bg(sl)
header(sl,'Smallest falsifiable workflow · discussion draft','Make the cytometry link testable before fitting the model','The goal is to distinguish mechanisms, not to make a visual match look like a validated prediction.')
steps=[
('1 · PREDECLARE','Sample origin, time points, stains, gates, instrument settings, primary summaries and exclusion rules.'),
('2 · CALIBRATE','Unstained / stain-only, single-stain or compensation controls, live-like / dead-like controls, blank matrix and counting beads.'),
('3 · MATCH THE STATE','Map each sample to effluent, planktonic or detached-biofilm model state. Record extraction recovery and dilution.'),
('4 · TEST PREDICTION','Fit on designated observations; check posterior predictive distributions or held-out time points. Report non-identifiable alternatives.')]
for i,(title,body) in enumerate(steps):
    x=.88+(i%2)*6.0; y=2.55+(i//2)*1.52
    card(sl,x,y,5.65,1.26,[PALE_GRAY,PALE_BLUE,PALE_TEAL,PALE_GOLD][i],title,body,accent=[MUTED,BLUE,TEAL,OCHRE][i],body_size=11.5,title_size=14)
ban=sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(.88), Inches(5.80), Inches(11.94), Inches(.56)); ban.fill.solid(); ban.fill.fore_color.rgb=rgb(PALE_RED); ban.line.fill.background()
text(sl,'Falsifier: if predicted cell-state distributions do not track the measurements under the predeclared calibration, revise or reject the state model; do not tune gates after seeing the desired result.',1.07,5.90,11.55,.38,size=11.9,bold=True,valign=MSO_ANCHOR.MIDDLE)
footer(sl,'Assay cautions: Kerstens et al. (2015); Grainha et al. (2020). Calibration concepts: Yip et al. (2022).')
notes(sl,'This is a discussion plan, not a validated protocol. In the current clay coupon pilot, cytometry would be an exploratory secondary readout and must not displace the preselected maximum-pit-depth endpoint. Repeated sampling can change volume, headspace, redox and biomass; consider sacrificial or companion vessels if sampling disturbs the system.')

prs.save(out)
print(out, len(prs.slides))
