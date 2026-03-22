# Synthetic Validation Systems

A diagnostic pipeline that claims to separate local confinement from inter-site coherence must prove that it can. Not by assertion, but by recovering known ground truth from controlled inputs where every answer is determined before the computation begins.

This document describes fourteen such inputs. Each is an analytically controlled synthetic ensemble whose directional statistics are fully specified by construction. No material system, force field, or physical model is involved. The validation is purely geometric and statistical. The pipeline either recovers the known answer or it does not.

The systems are organised around four claims the pipeline must substantiate. D_dir must recover the correct local confinement geometry across the full range from uniaxial to isotropic. Δ_coh must vanish when all sites share a common axis and grow when their axes disperse. The persistence metrics ⟨A⟩ and ⟨B⟩ must track a known switching rate without conflating temporal instability with spatial incoherence. And the robustness suite must flag pathological inputs rather than silently accept them.


## Shared Construction

All systems use d = 3 spatial dimensions, T = 10,000 frames per site, N = 8 sites, and W = 4 persistence windows unless stated otherwise. Directional distributions are drawn from the von Mises–Fisher (vMF) family on the unit sphere S², parameterised by a concentration κ and a mean direction μ. At κ = 0 the distribution is uniform; as κ increases, it concentrates into a tighter cap around μ. The vMF second-moment tensor has eigenvalues λ₁ = (1 + 2ρ²)/3 and λ₂ = λ₃ = (1 − ρ²)/3, where ρ = coth(κ) − 1/κ, yielding a closed-form D_dir(κ). Every directional ground truth in this suite traces back to this formula.

All systems use deterministic seeds. Five independent seed realisations produce deviations in D_dir and Δ_coh below 0.01. The reported values reflect intended geometry, not Monte Carlo noise.

Persistence classification uses both metrics: LOCKED requires A > 0.8 and B > 0.8; FAST requires A < 0.4; SWITCHING is everything between. These thresholds are calibrated against the switching model in System H. They are not tuned to any material dataset.


## Baseline Dimensionality (A, B, C)

Three systems span the full eigenvalue simplex. All sites share identical statistical properties, so Δ_coh ≈ 0 throughout. The only thing that varies is the shape of the local confinement ellipsoid.

### System A — Isotropic

All sites draw directions uniformly on S². The eigenvalue spectrum is (1/3, 1/3, 1/3). D_dir = 3. The spectral gap is zero, so degeneracy is flagged and axis-based metrics are suppressed. The displacement ellipsoid is a perfect sphere — every direction on the unit sphere is equally likely.

This is the simplest possible case. If the pipeline cannot recover D_dir = 3 here, nothing downstream can be trusted.

### System B — Uniaxial

All sites draw from vMF(ẑ, κ = 50) with a shared axis. D_dir ≈ 1.08 — strongly confined along ẑ. The displacement ellipsoid is a prolate spheroid. Δ_coh ≈ 0 because all sites share the same axis.

The full κ-sweep {1, 5, 10, 20, 50, 100, 200} traces D_dir continuously from 3 (isotropic limit) down to 1 (perfect uniaxial confinement), with each recovered value matching the analytic vMF prediction.

### System C — Planar

Directions are confined to a great circle in the xy-plane. The eigenvalue spectrum is (1/2, 1/2, 0). D_dir = 2. The displacement ellipsoid is an oblate disc — isotropic within the plane, vanishing perpendicular to it.

Together, A, B, and C span the canonical isotropic, uniaxial, and planar limits of the rank-2 directional problem.


## Coherence Hierarchy (D, E, F)

These three systems hold local confinement fixed at κ = 50 and vary only the inter-site axis structure. This is the scientific core of the validation: it tests the pipeline's central claim that Δ_coh measures coherence, not confinement.

### System D — Incoherent (cube vertices)

Eight sites, each drawing from vMF(μ_i, 50), where the axes μ_i are the eight cube-vertex directions {±1, ±1, ±1}/√3. Every site has the same local confinement — D_dir_site ≈ 1.08, the same prolate shape as System B. But the axes point in maximally different directions.

When you pool eight rotated copies of a prolate tensor, the result is nearly a sphere. D_dir_pooled ≈ 3.00. The coherence gap Δ_coh = 1.92 — the largest value possible for this confinement strength. A standard pooled analysis would report "isotropic." The pipeline reports "strongly confined, but incoherent."

This is the central coherence test. Systems B and D share the same local confinement but differ in inter-site axis organisation. A successful pipeline must recover nearly identical local D_dir values while sharply separating them in Δ_coh. Failure here would directly undermine the claimed separation between confinement and coherence.

### System E — Interpolation (tunable κ_inter)

A two-level vMF hierarchy fills the continuum between coherent and incoherent. Site axes μ_i are drawn from vMF(ẑ, κ_inter), then directions from vMF(μ_i, 50). As κ_inter decreases from infinity (all axes aligned, Δ_coh → 0) through intermediate values to zero (axes uniformly random, Δ_coh → 1.92), the coherence gap varies smoothly.

This proves the pipeline does not discretise coherence into binary "aligned or not aligned." It tracks the full continuum.

### System F — Mixed κ

Each site has a different concentration: κ ∈ {20, 50, 100} across sites. This breaks the equal-spectrum assumption used in the Jensen inequality proof for Δ_coh ≥ 0. The empirical result confirms non-negativity even under heterogeneous spectra: Δ_coh = 1.90.

The theoretical guarantee covers identical spectra. System F extends the test beyond its provenance. The result holds, but it is empirical — not proven for the general case.


## Temporal Persistence (G, H, I)

A large coherence gap could arise from stable orientational heterogeneity or from temporal instability. Distinguishing these cases requires a separate diagnostic. Systems G, H, and I calibrate the persistence metrics that make this distinction.

### System G — Static

Each site draws a fixed axis from vMF(ẑ, 50) at time zero and holds it constant for all 10,000 frames. Directions fluctuate around the fixed axis but the axis itself never moves. ⟨A⟩ = 1.00, ⟨B⟩ = 1.00, classification LOCKED.

This sets the persistence ceiling. A system with no temporal dynamics must register as fully persistent.

### System H — Axis switching (rate sweep)

Each site switches its preferred axis at random times with rate r per frame. At r = 0, there is no switching — identical to System G. As r increases, windows begin to contain frames from both axis orientations. The per-window principal axis becomes less stable. ⟨A⟩ decreases monotonically.

The calibration curve r → ⟨A⟩ defines the LOCKED/SWITCHING/FAST boundaries. Rate sweep: {0.0001, 0.0005, 0.001, 0.003, 0.008, 0.02}. At rate 0.0001, the system is still LOCKED (A = 0.83). By rate 0.003, it crosses into FAST (A = 0.34). The thresholds are placed at the crossings of this curve — calibrated against a known stochastic process, not tuned to any real material.

### System I — Axis drift (continuous)

The preferred axis rotates at constant angular velocity ω around a great circle. All sites drift coherently — the same rotation, the same rate. This produces a distinctive signature: consecutive windows remain correlated (moderate ⟨A⟩), while the block axis deviates from the full-trajectory axis (lower ⟨B⟩). Crucially, Δ_coh remains small. All sites drift together, so coherence is preserved even though persistence is reduced.

This is the decoupling test. A pipeline that conflates "drifting axis" with "incoherent sites" fails here. Coherence and persistence are independent diagnostics.

Drift sweep: {5×10⁻⁵, 0.0001, 0.0002, 0.0005, 0.001} rad/frame.


## Edge Cases and Sensitivity (J, K, L, M, N)

Five systems probe corners where pipelines commonly fail. Not because the physics is exotic, but because the numerics are fragile or the assumptions break down.

### System J — Mixed confinement (per-site)

Different sites carry different confinement geometries. Four sites draw from high-κ vMF distributions producing uniaxial confinement (D_dir_site ≈ 1); the remaining four draw from great-circle distributions producing planar confinement (D_dir_site ≈ 2). The resulting heterogeneous site-level D_dir distribution, combined with dispersed axes, produces a non-trivial coherence gap. The pipeline must handle heterogeneous local structure without assuming all sites are identical. See `synthetic/systems.py` for the full parameterisation.

### System K — Near-isotropic

A system near the isotropic limit, with D_dir ≈ 3.00. The eigenvalues are nearly equal. The spectral gap is small. Degeneracy handling must behave correctly: the pipeline should report near-isotropic dimensionality without numerical artefacts from ill-conditioned eigenvector estimation.

### System L — Finite-size (N = 4–64)

System D variant with varying number of sites. As N increases from 4 to 64, D_dir_pooled converges toward 3.0 and Δ_coh toward 1.92, while D_dir_site remains constant at 1.08. Convergence is well developed by N = 8 and essentially complete by N = 16.

At small N, accidental anisotropy in the pooled tensor produces large variance — a known limitation, not a bug. This system documents the boundary between reliable and unreliable pooling.

### System M — Gating-sensitive (low amplitude)

Displacement amplitudes are drawn near the gating threshold δ₀, so that approximately 83% of frames are excluded by the amplitude gate. The directional statistics of the surviving frames may appear normal; the pathology is in the fraction retained, not the directions themselves. The robustness suite flags this as **SENSITIVE**. The point is that the suite detects gating pathology — it does not just confirm success on well-conditioned inputs.

### System N — Temporally unstable (high CV)

T = 50 frames (pathologically short). For the first 25 frames, all sites draw from vMF(ẑ, 50). For the final 25 frames, all sites draw from vMF(x̂, 50) — a 90° abrupt rotation of the preferred axis midway through the trajectory. The coherence gap computed over the full trajectory is not representative of either half. The coefficient of variation of segment-level Δ_coh is CV = 0.36, above the 0.30 threshold. The robustness suite flags this as **SENSITIVE**. The pipeline must distinguish a stationary property from a transient artefact.


## Validation Summary

| System | Description | ⟨D_dir_site⟩ | D_dir_pooled | Δ_coh | Robust |
|--------|------------|--------------|-------------|-------|--------|
| A | Isotropic | 3.000 | 3.000 | 0.000 | ✓ |
| B | Uniaxial (κ=50) | 1.082 | 1.082 | <0.001 | ✓ |
| C | Planar | 2.001 | 2.002 | <0.001 | ✓ |
| D | Incoherent (cube vertices) | 1.082 | 3.000 | 1.918 | ✓ |
| E | Interpolation (κ_inter=5) | 1.082 | 2.239 | 1.157 | ✓ |
| F | Mixed κ (20/50/100) | 1.100 | 2.999 | 1.899 | ✓ |
| G | Static locked axes | 1.083 | 3.000 | 1.917 | ✓ |
| H | Axis switching (rate sweep) | 2.157 | 2.880 | 0.723 | ✓ |
| I | Axis drift (continuous) | 1.531 | 2.908 | 1.378 | ✓ |
| J | Mixed confinement (per-site) | 1.525 | 2.697 | 1.172 | ✓ |
| K | Near-isotropic | 2.998 | 2.998 | 0.000 | ✓ |
| L | Finite-size (N=8) | 1.082 | 2.954 | 1.872 | ✓ |
| M | Gating-sensitive | 1.082 | 1.082 | <0.001 | × |
| N | Temporally unstable | 2.079 | 2.079 | <0.001 | × |

12 systems ROBUST. 2 SENSITIVE — M and N, deliberately constructed to trigger sensitivity flags.


## Persistence Calibration

| System | Control | ⟨A⟩ | ⟨B⟩ | Classification |
|--------|---------|------|------|----------------|
| G | rate=0 (static) | 1.0000 | 1.0000 | LOCKED |
| H | rate=0.0001 | 0.8336 | 0.8795 | LOCKED |
| H | rate=0.0005 | 0.6151 | 0.7340 | SWITCHING |
| H | rate=0.001 | 0.5541 | 0.6946 | SWITCHING |
| H | rate=0.003 | 0.3371 | 0.6757 | FAST |
| H | rate=0.008 | 0.2572 | 0.6194 | FAST |
| H | rate=0.02 | 0.1926 | 0.7005 | FAST |
| I | drift=5×10⁻⁵ | 0.9935 | 0.9919 | LOCKED |
| I | drift=0.0001 | 0.9583 | 0.9490 | LOCKED |
| I | drift=0.0002 | 0.8442 | 0.8216 | LOCKED |
| I | drift=0.0005 | 0.4206 | 0.6727 | SWITCHING |
| I | drift=0.001 | 0.3205 | 0.7084 | FAST |


## Design Logic

The fourteen systems implement an axis-wise controlled validation spanning five independent dimensions:

**Local geometry** (A, B, C). The concentration κ varies while all axes are aligned. This isolates D_dir recovery from any coherence or temporal effect.

**Coherence** (D, E, F). Site axes vary while local confinement is held fixed. This isolates Δ_coh from D_dir. System D demonstrates that identical local confinement can produce sharply different pooled statistics depending on axis alignment.

**Temporal dynamics** (G, H, I). Switching and drift rates vary while spatial geometry is fixed. This isolates persistence from coherence. The decoupling test — System I — demonstrates that coherence and persistence are independent.

**Statistical sampling** (L). The number of sites N varies while all physics is held fixed. This maps the boundary between reliable and unreliable pooling.

**Robustness** (J, K, M, N). Edge cases and adversarial inputs that should trigger explicit sensitivity flags rather than silent acceptance. The pipeline is required to detect pathology, not just confirm success.

Each axis is primarily isolated, with controlled secondary effects. This is an axis-wise controlled design, not a full factorial study. The goal is not to reproduce a particular dataset. It is to establish that the pipeline recovers known ground truth under controlled perturbations — and fails explicitly when it should.
