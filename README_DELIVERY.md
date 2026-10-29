# Flow Cytometry + Model Bridge - Complete Delivery Bundle

Build date: 2026-10-29

This bundle contains the presentations, PDF review copies, preview images, original 20-slide source deck, build scripts, specification documents, and the standalone Julia observation-model adapter and verification test suite.

## Presentations
- `presentations/Clay_Microbes_Metal_22slides_Flow_Cytometry_Addendum.pptx` - 22-slide deck: the original 20 slides plus two optional flow-cytometry/model-observation appendices (slides 21-22).
- `presentations/Clay_Microbes_Metal_22slides_Flow_Cytometry_Addendum.pdf` - PDF review copy.
- `presentations/Single_Cell_Flow_Cytometry_Model_Bridge_Exploratory.pptx` - separate 4-slide exploratory discussion deck.
- `presentations/Single_Cell_Flow_Cytometry_Model_Bridge_Exploratory.pdf` - PDF review copy.
- `source/Original_20_slide_Clay_Microbes_Metal_Deck.pptx` - source deck, included to make the relationship between source and addendum explicit.

## Code and data template
- `code/FlowCytometryObservationModel.jl` - standalone Julia observation-model adapter (`MODEL_ADAPTER_VERSION = "0.2.0"`), implementing explicit volume-convention handling (`:aliquot_post_dilution`, `:undiluted_equivalent`, `:beads_co_diluted`), exact MLE concentration-likelihood consistency checks, Poisson-Multinomial and Negative-Binomial/Dirichlet-Multinomial likelihoods, synthetic-data simulation and parameter recovery, MESF and bead-size calibration curves (`MESFCalibrationCurve`, `BeadSizeCalibration`), independent control gate calibration with clay-blank subtraction, uncertainty propagation, identifiability/equifinality audits, held-out posterior-predictive checks, solver alignment contracts (`OpenFOAMTransportSnap`, `PHREEQCSpeciationSnap`, `StakeholderTrackContract`), and epistemic status accounting (`EpistemicQuantity`, `ProductionAuditRecord`). Not merged into external OpenFOAM or PHREEQC solver repositories.
- `code/test_FlowCytometryObservationModel.jl` - unit and verification test suite executed locally with Julia 1.13.1 (`julia --startup-file=no code/test_FlowCytometryObservationModel.jl`, `118/118` passing assertions across all 7 audit priorities).
- `code/flow_cytometry_samples_template.csv` - empty sample schema template validated against `REQUIRED_CSV_COLUMNS`, not real experimental data.

## Specification and build files
- `docs/flow_cytometry_model_bridge_spec.md` - integration design, mathematical observation model, 3 stakeholder track contracts, and epistemic limits.
- `docs/build_flow_cytometry_addendum.py` and `docs/build_ghosh_fcm_bridge_deck.py` - presentation build scripts.
- `docs/clay_polymer_infrastructure_epistemic_synthesis.md` - related source-bounded transfer synthesis for soil stabilization; not an empirical performance claim.

## Previews
`previews/` contains render previews of slides 21-22 and all four slides of the separate four-slide discussion deck.

## Scientific boundary
Flow cytometry is treated as an observation layer for sampled cell populations (`:effluent`, `:planktonic`, `:detached_biofilm`). It does not directly measure in situ pore-scale spatial structure, EPS cross-link density, mineral phases, or corrosion rates. Cytometry gates, extraction recovery, staining, and debris/aggregate controls require assay-specific empirical validation. The software adapter and its 118/118 passing unit tests verify mathematical consistency, synthetic parameter recovery, calibration algebra, and data-contract enforcement; they do not substitute for empirical wet-lab calibration or coupled multiphysics validation.

## Integrity
`SHA256SUMS.txt` provides SHA-256 digests for every other file in this bundle. `DELIVERY_MANIFEST.json` records relative paths, byte sizes, SHA-256 digests, presentation slide counts, and the local Julia test execution record.
