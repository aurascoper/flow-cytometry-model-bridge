# Flow Cytometry + Model Bridge — Complete Delivery Bundle

Build date: 2026-10-29

This bundle contains the files from the prior response in one archive, plus PDF review copies, preview images, the original 20-slide source deck, and build scripts so that the handoff does not depend on separate file links.

## Presentations
- `presentations/Clay_Microbes_Metal_22slides_Flow_Cytometry_Addendum.pptx` — 22-slide deck: the original 20 slides plus two optional flow-cytometry/model-observation appendices (slides 21–22).
- `presentations/Clay_Microbes_Metal_22slides_Flow_Cytometry_Addendum.pdf` — PDF review copy.
- `presentations/Single_Cell_Flow_Cytometry_Model_Bridge_Exploratory.pptx` — separate 4-slide exploratory discussion deck.
- `presentations/Single_Cell_Flow_Cytometry_Model_Bridge_Exploratory.pdf` — PDF review copy.
- `source/Original_20_slide_Clay_Microbes_Metal_Deck.pptx` — source deck, included to make the relationship between source and addendum explicit.

## Code and data template
- `code/FlowCytometryObservationModel.jl` — proposed standalone Julia observation-model adapter; not merged into the user's existing OpenFOAM, PHREEQC or project Julia solvers.
- `code/test_FlowCytometryObservationModel.jl` — tests for the proposed adapter. Julia was not available in the prior execution context, so they are not claimed as run.
- `code/flow_cytometry_samples_template.csv` — empty/sample schema template, not real experimental data.

## Specification and build files
- `docs/flow_cytometry_model_bridge_spec.md` — integration design and epistemic limits.
- `docs/build_flow_cytometry_addendum.py` and `docs/build_ghosh_fcm_bridge_deck.py` — presentation build scripts.
- `docs/clay_polymer_infrastructure_epistemic_synthesis.md` — related source-bounded transfer synthesis for soil stabilization; not an empirical performance claim.

## Previews
`previews/` contains render previews of slides 21–22 and all four slides of the separate four-slide discussion deck.

## Scientific boundary
Flow cytometry is treated as an observation layer for sampled cell populations. It does not directly measure in situ pore-scale spatial structure, EPS cross-link density, mineral phases or corrosion rates. Cytometry gates, extraction recovery, staining and debris/aggregate controls need assay-specific validation. The model adapter is a proposal, not an empirically calibrated model or a completed integration with OpenFOAM/PHREEQC.

## Integrity
`SHA256SUMS.txt` provides SHA-256 digests for every other file in this bundle. `DELIVERY_MANIFEST.json` records relative paths, sizes, and presentation slide counts.
