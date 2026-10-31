# Track 3 Collaborator Brief: SoftComplexFluids Bitumen Emulsion and Interfacial Rheology Modeling

**Target Collaborator:** Iain Jack (SoftComplexFluids)  
**Project:** Flow Cytometry Model Bridge  
**Track:** Track 3 (Bitumen Emulsion and Interfacial Rheology)  
**Status:** Architecture Specification and Computational Fluid Mechanics Brief  
**Boundary Rule (Control 5):** Pure multiphase fluid dynamics; strictly isolated from biological biofilm models  

---

## 1. Executive Summary and Modeling Boundary

This brief establishes the collaborative modeling interface with Iain Jack (SoftComplexFluids) for interfacial rheology and multiphase droplet dynamics in bitumen emulsions and water-in-oil-in-water double emulsions.

### 1.1 Strict Physical and Biological Isolation (Control 5)
In accordance with **Control 5**, Track 3 strictly isolates industrial bitumen emulsion interfacial rheology from the biological biofilm and single-cell cytometry models used in Tracks 1 and 2:
1. **Zero Biological Release Shells:** The concentric strata used in Track 1 for drug release are completely prohibited. In `code/MicrofluidicClassificationEngine.jl`, `EmulsionDropletSpec` enforces `track == :track3_bitumen_rheology` and `has_biological_release_shells == false`.
2. **Zero Cytometry or Viability Coupling:** Bitumen droplets are non-living colloidal entities. The model enforces zero cell viability gates, zero cell stress markers, and zero fluorescence acquisition channels.
3. **Zero Geochemical Speciation Coupling:** Track 3 does not couple to PHREEQC mineral speciation or biofilm matrix swelling mechanics.
4. **Architectural Code Reuse:** Track 3 reuses only the domain-separated 3D wireframe-over-core rendering pipeline, tensor transformations, and OpenFOAM multiphase fluid dynamic infrastructure developed across the platform.

---

## 2. Mathematical and Rheological Formulation

The computational physics implemented in `SharedVisualizationPhysics` (`solve_emulsion_interfacial_droplet`) models the deformation, orientation, internal circulation, and filament necking of bitumen droplets under simple shear flow.

### 2.1 Boussinesq-Scriven Interfacial Fluid Mechanics
The droplet interface behaves as a two-dimensional Boussinesq-Scriven surface fluid endowed with surface tension $\sigma$, surface shear viscosity $\mu_s$, and surface dilatational viscosity $\kappa_s$:
$$\boldsymbol{\tau}_s = \sigma \mathbf{I}_s + (\kappa_s - \mu_s) (\nabla_s \cdot \mathbf{v}_s) \mathbf{I}_s + 2 \mu_s \mathbf{D}_s$$

1. **Boussinesq Number:** The dimensionless resistance to interfacial shear deformation relative to bulk viscous shear is given by:
   $$Bq = \frac{\mu_s}{\mu_c a_0}$$
   where $\mu_c$ is continuous-phase dynamic viscosity and $a_0$ is unperturbed droplet radius.

2. **Effective Viscosity Ratio ($\lambda_{\text{eff}}$):**
   Interfacial viscosities create an effective drag that augments the internal dispersed-phase viscosity $\mu_d$:
   $$\lambda_{\text{eff}} = \frac{\mu_d(T)}{\mu_c} + \frac{6 \mu_s(T) + 4 \kappa_s(T)}{5 \mu_c a_0}$$

3. **Temperature Dependence:**
   Thermal softening of heavy bitumen components (asphaltenes and maltenes) is represented via an Arrhenius-like scaling around reference temperature $T_0 = 25^\circ\text{C}$:
   $$\mu_d(T) = \mu_{d,0} \exp\left[-0.022 (T - 25.0)\right]$$
   $$\mu_s(T) = \mu_{s,0} \exp\left[-0.022 (T - 25.0)\right]$$
   $$\kappa_s(T) = \kappa_{s,0} \exp\left[-0.022 (T - 25.0)\right]$$

### 2.2 Taylor Droplet Deformation and Orientation
Under simple shear flow with shear rate $\dot{\gamma}$, hydrodynamic forces deform the droplet into an ellipsoid characterized by major axis $L$, minor axis $B$, and thickness $W$:

1. **Capillary Number ($\text{Ca}$):**
   $$\text{Ca} = \frac{\mu_c \dot{\gamma} a_0}{\sigma(T)}$$

2. **Taylor Deformation Parameter ($D$):**
   $$D = \frac{L - B}{L + B}$$
   Steady-state deformation balances viscous elongation against interfacial tension restoration:
   $$D_{\text{steady}} = \min\left(0.82, \frac{19 \lambda_{\text{eff}} + 16}{16 \lambda_{\text{eff}} + 16} \cdot \text{Ca}\right)$$

3. **Transient Evolution across Dimensionless Time ($t / \tau_\eta \in [1.0, 15.0]$):**
   $$D(t) = \min\left(0.88, D_{\text{steady}} \left(1 - \exp\left[-\frac{t/\tau_\eta}{1.8 + 0.35 \lambda_{\text{eff}}}\right]\right) + \Delta D_{\text{elong}}\right)$$
   where $\tau_\eta = \frac{\mu_c a_0}{\sigma}$ is the emulsion relaxation timescale.

4. **Droplet Orientation Angle ($\theta$):**
   The angle of the major axis relative to the flow direction aligns towards the shear axis as deformation increases:
   $$\theta(t) = \frac{\pi}{4} - \frac{1}{2} \arctan\left(\frac{19 \lambda_{\text{eff}} \text{Ca}}{20}\right) \left(1 - \exp\left[-\frac{t/\tau_\eta}{1.8 + 0.35 \lambda_{\text{eff}}}\right]\right)$$

5. **Volume Conservation:**
   Incompressible droplet volume $V_0 = \frac{4}{3} \pi a_0^3 = \frac{4}{3} \pi L B W$ with $B = W$:
   $$\frac{L}{B} = \frac{1 + D}{1 - D} \implies L = a_0 \left(\frac{1 + D}{1 - D}\right)^{2/3}, \quad B = W = a_0 \left(\frac{1 + D}{1 - D}\right)^{-1/3}$$

### 2.3 Internal Twin-Vortex Recirculation and Pressure Perturbations
1. **Twin-Vortex Circulation:**
   Shear stresses transmitted across the Boussinesq-Scriven interface drive internal counter-rotating recirculation vortices whose strength scales inversely with effective viscosity:
   $$\Omega_{\text{vortex}} = \dot{\gamma} \left(\frac{1}{1 + 0.25 \lambda_{\text{eff}}}\right) (1 - 0.3 D)$$

2. **Hydrodynamic Pressure Perturbations ($P/\text{Pa}$):**
   Viscous flow around the deformed interface generates a dynamic pressure field bounded within the physical reference range $P/\text{Pa} \in [-0.30, 1.40]$:
   $$P_{\max} = 0.12 \tau_{\text{shear}} (1 + 0.45 D)$$
   $$P_{\min} = -0.026 \tau_{\text{shear}} (1 + 0.35 D)$$
   where $\tau_{\text{shear}} = \mu_c \dot{\gamma}$.

3. **Surfactant / Asphaltene Tip Enrichment:**
   Interfacial shear sweeps surface-active asphaltenes toward droplet tips, lowering local surface tension and promoting tip streaming:
   $$\Gamma_{\text{tip}} = 1.0 + 0.65 D$$

### 2.4 Breakup, Filament Necking, and Pinch-Off
1. **Critical Capillary Number ($\text{Ca}_{\text{crit}}$):**
   $$\text{Ca}_{\text{crit}} = 0.42 \left(1.0 + 0.12 \left[\log_{10}(\max(\lambda_{\text{eff}}, 0.1))\right]^2\right)$$

2. **Necking Ratio:**
   At high deformation or extended shear exposure ($t/\tau_\eta \ge 4.0$), the central column undergoes filament thinning:
   $$\text{necking}(t/\tau_\eta) = \max\left(0.12, 1.0 - 0.055 \max(0, t/\tau_\eta - 4.0) \frac{\text{Ca}}{\max(\text{Ca}_{\text{crit}}, 0.1)}\right)$$

3. **Pinch-Off Transition:**
   A transition to irreversible droplet breakup and satellite droplet pinch-off is identified when:
   $$\text{necking} < 0.38 \quad \text{or} \quad (t/\tau_\eta \ge 12.0 \text{ and } \text{Ca} \ge 0.85 \text{Ca}_{\text{crit}})$$

---

## 3. Experimental Validation Basis

To ensure the computational model connects to physical experimental data from SoftComplexFluids, the parameters are benchmarked against standardized rheological characterizations:
1. **Dynamic Shear Rheometry (DSR):** Oscillatory frequency sweeps ($0.1$ to $100\,\text{rad/s}$) measuring linear viscoelastic moduli $G'(\omega)$ and $G''(\omega)$ across temperatures from $10^\circ\text{C}$ to $60^\circ\text{C}$.
2. **Interfacial Shear Rheometry:** Double-wall ring (DWR) or bicone geometries measuring Boussinesq interfacial viscosity $\mu_s$ at the water-bitumen interface.
3. **Microfluidic Droplet Deformation Imaging:** High-speed optical microscopy in microfluidic cross-slots and flow-focusing channels measuring real-time $D(t)$ and critical pinch-off capillary numbers.
4. **Moisture Susceptibility and Stripping:** Emulsion stability and phase inversion resistance under environmental moisture conditioning.

---

## 4. Deliverables and Presentation Package

1. **Exploratory Deck (`presentations/Interface_Resolved_Emulsion_Dynamics_Exploratory.pptx` and `.pdf`):**
   - High-resolution Slide 1: Deformed bitumen double-emulsion droplet with wireframe interface mesh under shear ($t/\tau_\eta = 1 \dots 15$), internal twin-vortex velocity vectors, and hydrodynamic pressure field $P/\text{Pa} \in [-0.30, 1.40]$.
2. **Interactive 3D Workbench Mode:**
   - Dedicated Track 3 tab in `workbench/microfluidic_cytometry_workbench.html` demonstrating parametric variations of $\text{Ca}$, $\lambda_{\text{eff}}$, and temperature $T$.
