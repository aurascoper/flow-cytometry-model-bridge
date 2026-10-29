# Flow cytometry as the observation layer for coupled flow–geochemistry–biofilm models

**Status:** proposed interface specification, not yet integrated into the user's local solver repositories.

## Why this belongs in the model

Flow cytometry can add high-throughput, single-cell population measurements to models whose current output is mainly fluid transport, aqueous chemistry, or bulk reaction rates. A 2022 review specifically discusses calibrating spatiotemporal microbial models with flow-cytometry and microscopy data, while noting that systematic calibration and validation remain challenging (Yip et al., *PLOS Computational Biology* 18(10):e1010533, https://doi.org/10.1371/journal.pcbi.1010533).

For biofilms, flow cytometry can quantify gated cell events, cell-state fractions and distributions of scatter/fluorescence. Kerstens et al. demonstrated a cytometric biofilm-quantification workflow for *E. coli* and *Candida albicans*, with a specific extraction/dissociation protocol; that staining performance must not be assumed to transfer to a different strain or clay matrix (2015, https://doi.org/10.1007/s12223-015-0400-4). Grainha et al. describe interpretive difficulties caused by strain, cell morphology and matrix effects in mixed biofilms (2020, https://doi.org/10.3390/antibiotics9110741).

## Responsibility by component

| Component | Forward-model responsibility | Outputs consumed by the coupling layer |
|---|---|---|
| OpenFOAM | Velocity/pressure, wall shear, residence times, transport, and where represented, detachment or deposition cues | `u(x,t)`, shear, residence time, aqueous transport fields and boundary fluxes |
| PHREEQC | Aqueous speciation and mineral equilibria | pH, pe/Eh assumptions, ionic strength, Ca/Mg, Fe(II)/Fe(III), sulfate/sulfide, saturation indices and dissolved concentrations |
| Julia biological/reaction model | Couple chemical drivers to attached/suspended cell states, growth/death, attachment/detachment and a clearly defined EPS proxy | Latent state per sample compartment and time point: concentration plus fractions for modelled cell states |
| Flow-cytometry adapter | Transform latent states into predictions that are comparable to the actual sample protocol and gate configuration | Predicted event concentration and gate-category probabilities, with calibration and uncertainty |
| Inference/calibration layer | Compare predictions with observed cytometry and chemistry data; quantify uncertainty and equifinality | Parameter posterior or profile likelihood, predictive checks, and alternative mechanisms still consistent with data |

**Important:** flow cytometry is not a spatial sensor. Do not infer pore location or local EPS cross-link density from a cytometry sample alone. Pair it with endpoint microscopy/confocal imaging and the planned Raman, SEM/EDS and XRD observations. The FCM adapter must be told whether each sample came from effluent, planktonic suspension or detached biofilm; these are different observation compartments.

## Minimal observation schema

One row per vessel/time-point/sample-origin/gating version should retain at least:

- `sample_id`, `vessel_id`, `arm`, `time_h`, `sample_origin` (`effluent`, `planktonic`, `detached_biofilm`)
- `gate_id`, cytometer/instrument configuration, stain panel, compensation controls, dilution factor
- gated event counts for the pre-registered classes, bead events, known beads added, represented sample volume
- extraction/recovery estimate for detached biofilm and clay-associated samples
- blank, unstained, single-stain/compensation, live-like and dead-like control identifiers; a clay-only matrix blank is important because mineral particles can contribute scatter or fluorescence
- raw FCS file path/hash and the saved gate template/version, rather than only the final percentages

Do not call a dye-defined membrane-integrity gate “viability” without validating its interpretation in the target strain and matrix. Flow cytometry after detachment also depends on disaggregation: check that the processing protocol does not selectively destroy cells, change membrane integrity, or leave large aggregates. See Kerstens et al. (2015) and Grainha et al. (2020).

## Observation likelihood

A simple, auditable first implementation is provided in `FlowCytometryObservationModel.jl`.

1. A Julia model produces `CellStatePrediction(total_cells_per_ml, (f_intact, f_compromised, f_dead))` for the *sampled compartment*.
2. A validated 3×3 gate-confusion matrix maps latent fractions to observed-gate probabilities. Its columns represent the probability of each observed gate given a latent state. The columns must sum to one; a separate detection-efficiency parameter is explicit.
3. A counting-bead ratio yields the gated-cell concentration, corrected only for recorded dilution and a validated extraction-recovery fraction.
4. The likelihood factorizes into a count term and a conditional multinomial gate-composition term. The implementation does not count the same event total twice as independent data. If overdispersion is evident, upgrade the multinomial to a Dirichlet-multinomial and the count model to a negative-binomial or an empirically justified alternative.
5. The sampler (for example Metropolis–Hastings) can update biological rates only after prior/sensitivity/identifiability checks. A good fit is not proof of a unique mechanism; compare posterior-predictive distributions and, where possible, reserve a time point or vessel for validation.

## What should be added to each solver

- **OpenFOAM:** no cytometry physics is added to the CFD solver. Export transport and shear fields and define the sampling compartment/boundary flux that corresponds to the cytometry sample.
- **PHREEQC:** retain it as aqueous speciation/mineral-equilibrium engine. Export the chemical variables used by biological rate laws; do not make PHREEQC directly predict fluorescence gates.
- **Julia:** add explicit attached/suspended and cell-state variables, a data table keyed by vessel/time/sample origin, the observation operator, a likelihood, and posterior-predictive checks. If the existing Julia code does not yet expose cell states, add them as separate state variables rather than inferring them silently from bulk sulfide.

## Pilot design caution

Flow cytometry can be an **exploratory secondary endpoint**; it should not displace maximum pit depth as the preselected primary corrosion endpoint in the current coupon study. Repeated sampling may perturb an anaerobic vessel, remove biomass, or change headspace/volume. Use a documented, low-volume, oxygen-controlled sampling protocol only if it preserves the experiment, or use sacrificial/companion vessels for time points and endpoint detachment. The assay, extraction method, time points, gating rules and QC criteria must be predeclared before data collection.

## Separate application boundary for Dr. Ghosh

The SEMO Nano-Bio Lab describes work in single-cell analysis, computational cell classification and imaging/measurement pipelines (https://www.nanobiolab.org/research). Given the user's note that Dr. Ghosh uses flow cytometry, a Ghosh-facing pitch can position cytometry as a single-cell observation/validation layer. For a nanocarrier or opto-magnetic assay, the observables might instead be per-cell fluorescence distribution, percentage-positive cells, cell association/uptake and validated cell-state markers. OpenFOAM belongs only if the specific apparatus requires fluid-mechanical modeling; PHREEQC belongs only if aqueous speciation/mineral equilibria are actually part of that experiment. Do not force the clay–corrosion model or its parameters into a nanobio system.

## Acceptance criteria before calling the bridge integrated

- [ ] Raw cytometry files and gate-template versions are immutable and traceable.
- [ ] Calibration confusion matrix, detection efficiency and extraction recovery come from independent controls, not a fit to the same target data.
- [ ] Model predictions match sample origin and time exactly.
- [ ] Unit tests cover concentration, gate mapping, invalid inputs, and missing/zero-count cases.
- [ ] At least one posterior-predictive or held-out comparison is reported.
- [ ] The slides and UI label all outputs as observed, modelled, calibrated, or assumed.
