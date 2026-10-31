# Track 2 Collaborator Brief and Communication Package: Microbiologically Influenced Corrosion (MIC) Coupon and Microfluidic Complement

**Target Collaborators:** Dr. Caixia (Ellen) Wan (Bioprocess / Microbial Systems) and Dr. Andrew Hoffman (Corrosion Science / Materials Engineering)  
**Project:** Flow Cytometry Model Bridge  
**Track:** Track 2 (Clay, Microbes, and Metal Interface Modeling)  
**Status:** Architecture Brief, Experimental Interface Contract, and Protocol Sequence  

---

## 1. MURR Outside-Activity Disclosure Boundary

> **MANDATORY INSTITUTIONAL DISCLOSURE:**  
> "This modeling framework, technical brief, and associated communications represent an independent outside professional activity conducted in a personal capacity. This work does not constitute an official activity, sponsored research project, or endorsement by the University of Missouri or the University of Missouri Research Reactor (MURR). No nonpublic reactor design information, proprietary facility operational data, or university-administered resources have been used or disclosed in this project."

This brief and all associated technical memos, presentation decks, and correspondence operate strictly within this boundary. No university equipment, facilities, staff time, or nonpublic reactor information are involved.

---

## 2. Executive Summary and Experimental Positioning

This brief establishes the collaborative research interface with Dr. Caixia Wan and Dr. Andrew Hoffman for a planned 24-vessel experimental pilot investigating Microbiologically Influenced Corrosion (MIC) of metal coupons embedded in compacted clay-biofilm environments.

### 2.1 Microfluidics and Flow Cytometry as Complementary Observers
A critical design requirement of Track 2 is positioning microfluidics and flow cytometry as **complementary to standardized metallurgical coupon endpoints, not as a replacement**:

1. **Coupon Endpoints Remain Primary:**
   - Physical corrosion damage on metal coupons constitutes the definitive primary endpoint.
   - All corrosion evaluations must adhere strictly to established engineering standards:
     * **ASTM G1-03:** "Standard Practice for Preparing, Cleaning, and Evaluating Corrosion Test Specimens."
     * **NACE TM0194:** "Field Monitoring of Bacterial Growth in Oil and Gas Systems."
   - The primary quantitative metrics are:
     * **Maximum pit depth per coupon ($\mu\text{m}$):** Measured via 3D optical profilometry, white-light interferometry, or laser scanning confocal microscopy.
     * **Gravimetric mass loss and uniform corrosion rate ($\text{mm/year}$ or $\text{mpy}$):** Determined post-exposure after chemical cleaning and oxide film stripping per ASTM G1-03 Section 7 (inhibited hydrochloric acid pickling with ultrasonic agitation).
     * **Inductively Coupled Plasma (ICP-OES / ICP-MS) metal release:** Measuring dissolved iron ($\text{Fe}^{2+}, \text{Fe}^{3+}$), alloying element release, and total cation inventory in vessel effluent and squeezed porewater.
     * **Surface finish and edge finish effects:** Standardized 600-grit SiC finish, edge chamfering, and an explicit $1.5\,\text{mm}$ peripheral margin exclusion zone to eliminate artificial edge-effect pitting.

2. **Flow Cytometry and Microfluidics as Secondary / Effluent Observers:**
   - Single-cell flow cytometry and microfluidic transit sensors provide dynamic observations of suspended, detached, and effluent cell populations.
   - What flow cytometry can measure:
     * Absolute cell concentration ($\text{cells/mL}$) using calibrated fluorescent counting beads.
     * Physiological viability state distributions via dual-dye membrane integrity panels (e.g. SYTO 9 and propidium iodide), validated specifically on the target BSL-1 strain.
     * Planktonic vs detached biofilm population kinetics over time.
   - Explicit physical limitations:
     * Cytometry requires suspended single cells. Disaggregating clay-biofilm matrices destroys spatial pore geometry.
     * Flow cytometry **cannot** measure local spatial cell coordinates within compacted montmorillonite pores.
     * Flow cytometry **cannot** measure extracellular polymeric substance (EPS) cross-link density, mineral phase speciation, or pit initiation sites.

---

## 3. Equifinality and Mechanism Separation

### 3.1 Equifinality in Corrosion and Reaction-Diffusion Systems
In corrosion science and multi-component reaction-diffusion systems, multiple distinct physical, chemical, and biological causal histories can produce identical macroscopic coupon damage:
- **Biogenic Pathway:** Microbial reduction of sulfate by sulfate-reducing bacteria (SRB) producing biogenic hydrogen sulfide ($\text{H}_2\text{S}/\text{HS}^-$), which precipitates iron sulfides ($\text{FeS}$) and accelerates localized pitting through cathodic depolarization.
- **Abiotic Pathway:** Abiotic redox cycling driven by differential aeration cells, dissolved oxygen gradients, and iron oxyhydroxide passivity breakdown.
- **Organo-Mineral Transport Pathway:** Smectite clay edge-site protonation, selective cation exchange ($\text{Na}^+$ vs $\text{Ca}^{2+}$), and moisture-retention gradients altering local electrolyte conductivity and mass transport.

Because these distinct mechanisms generate similar pit depth and mass loss endpoints, the experimental pilot must incorporate rigorous discriminating controls.

### 3.2 Epistemic Clarification on Biofilms and Mineral Systems
The comparison between biofilms and abiotic/prebiotic mineral compartments is strictly an analogy of **spatially structured reaction-diffusion systems** sharing transferable forward mathematical architectures (site speciation, mass transport, and network formation vs degradation). This comparison explicitly does **not** assert that biofilms themselves are abiotic. Biofilms are living biological consortia exhibiting active gene expression, metabolic regulation, and dynamic adaptation.

### 3.3 The 24-Vessel Factorial Pilot Design
To resolve equifinality, the pilot employs a 24-vessel factorial design ($n = 4$ independent replicate vessels per cell):

| Vessel Cell ($n=4$) | Clay Matrix | Inoculation | Exchange Cation | Primary Mechanism Isolated |
|---|---|---|---|---|
| **Cell 1: Ca-Live** | Montmorillonite | Live BSL-1 inoculum | $\text{Ca}^{2+}$ exchanged | Synergistic organo-mineral MIC with divalent cation bridging |
| **Cell 2: Ca-Sterile** | Montmorillonite | Autoclaved / Gamma sterile | $\text{Ca}^{2+}$ exchanged | Abiotic corrosion baseline under $\text{Ca}$-montmorillonite |
| **Cell 3: Na-Live** | Montmorillonite | Live BSL-1 inoculum | $\text{Na}^+$ exchanged | Microbial MIC under unconstrained osmotic swelling conditions |
| **Cell 4: Na-Sterile** | Montmorillonite | Autoclaved / Gamma sterile | $\text{Na}^+$ exchanged | Abiotic corrosion baseline under $\text{Na}$-montmorillonite |
| **Cell 5: Broth-Live** | No clay (solution only) | Live BSL-1 inoculum | Bulk electrolyte | Unconstrained planktonic/biofilm MIC without mineral matrix |
| **Cell 6: Broth-Sterile**| No clay (solution only) | Sterile medium | Bulk electrolyte | Baseline uniform aqueous corrosion |

---

## 4. Feasibility Pilot Framing and Bounded Readouts

### 4.1 Short Pilot Feasibility (2 to 4 Weeks)
In alignment with Slide 13 of the project deck:
- The planned 2 to 4 week feasibility pilot (Day 0 to Day 28) tests **experimental feasibility, protocol reproducibility, and measurement sensitivity**.
- A 28-day pilot **does not establish a long-term biological threshold, repository barrier lifetime, or permanent durability standard**. Long-term containment projections require multi-year exposures under full subterranean hydrostatic and geochemical confinement.

### 4.2 Exploratory Day-28 Secondary Readout on Residual Waste Clay
To maximize scientific yield without compromising primary coupon endpoints, waste clay remaining after coupon harvest and profilometry may undergo three exploratory secondary bench tests:
1. **Drying Shrinkage ($\varepsilon_{\text{shrink}}$ at $40\%$ RH):** Evaluates whether live BSL-1 biofilm exposure alters desiccation crack patterns relative to sterile controls.
2. **Flat-Plunger Penetration Resistance (kPa):** Measures indentation resistance on dried clay pucks. Reported strictly as **penetration resistance**, not standardized ASTM Unconfined Compressive Strength (UCS), which requires tall cylindrical specimens.
3. **24-Hour Immersion-Retained Mass Fraction ($I_{\text{slake}} = m_{\text{retained}} / m_{\text{initial}}$):** Discriminates whether organo-mineral cohesion survives water immersion or rapidly slakes via osmotic dispersion.

---

## 5. Biological Safety and Radiological Boundaries

1. **BSL-1 Microbial Constraint:**
   - All experimental work is strictly restricted to Biosafety Level 1 (BSL-1) non-pathogenic organisms (e.g. *Shewanella oneidensis* MR-1, non-pathogenic environmental isolates, or BSL-1 model soil bacteria).
   - No human pathogens, opportunistic clinical strains, or select agents will be utilized.
2. **Non-Radioactive Baseline:**
   - The pilot uses non-radioactive stable isotopes and benign laboratory reagents.
   - Zero radioactive tracers, nuclear materials, or irradiated reactor components.

---

## 6. Track 2 Correspondence Protocol and Communication Sequence

To protect relationships and maintain rigorous review standards, the following two-step communication sequence must be strictly enforced:

### 6.1 Step 1 (Current: Strict Hold Over the Weekend)
- **Do NOT email Dr. Wan or Dr. Hoffman over the weekend.**
- Await completion of Zelda's weekend red-team review of the 20-slide core presentation deck, equifinality framing, and meeting materials.
- Await confirmation of the proposed November 2 to 6, 2026 three-way meeting schedule.

### 6.2 Step 2 (Pre-Read Package Once Meeting Schedule is Confirmed)
- Once the meeting time is set and Zelda's red-team edits are incorporated, send the focused 8-slide meeting pre-read deck:
  `presentations/Clay_Microbes_Metal_Wan_Hoffman_Meeting_PreRead.pptx` (and `.pdf`).
- Accompany the deck with a concise cover note mirroring the collegial, highly effective structure of the October 29-30 email sent to Dr. Ghosh.

### 6.3 Draft Pre-Read Email Package (Ready for Contributor Review)

```text
Subject: Pre-read: 8-slide framing for our Nov [Date] meeting on clay, microbes, and metal coupon corrosion

Dear Ellen and Andrew,

Thank you for agreeing to connect on [Confirmed Date, Nov 2-6, 2026] at [Confirmed Time] for our three-way discussion on microbiologically influenced corrosion and clay-metal interfaces.

To make the most of our 45 minutes together, I have attached a concise 8-slide pre-read deck (Clay_Microbes_Metal_Wan_Hoffman_Meeting_PreRead.pptx). 

Key points framed in the deck:
1. Framing Bridge: Connecting our initial October 26 coupon discussion with the compacted clay-biofilm interface as a refined, complementary candidate option for testing localized corrosion under realistic mineral barrier conditions.
2. Primary Focus: Preserving physical coupon metallurgy (maximum pit depth, mass loss, and ICP metal release per ASTM G1-03 / NACE TM0194) as the definitive primary endpoints.
3. Specific Input Needed:
   - For Ellen: Suitable BSL-1 environmental strains, biofilm cultivation in smectite matrices, and biological growth limits.
   - For Andrew: Preferred surface finishes (120 vs 600-grit), edge exclusion criteria, trusted corrosion product signatures (Raman/SEM-EDS), and pit mapping protocols.
4. Transparent Limits: A 2 to 4 week pilot tests experimental feasibility and measurement sensitivity, not long-term repository lifetime. Modeling provides hypothesis structuring and equifinality separation, not empirical proof.
5. Compliance: All proposed work is strictly BSL-1, non-radioactive, and conducted as an independent outside professional activity in a personal capacity.

Please look over the slides when you have a moment. We look forward to your critical feedback and to adjusting the pilot design based on your expertise.

Best regards,

Hunter
[Personal Contact Information]
(Independent Outside Professional Activity)
```
