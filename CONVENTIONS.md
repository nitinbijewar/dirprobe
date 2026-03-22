# dirprobe Conventions — LOCKED for v1.0

These conventions govern the `dirprobe` v1.0 release and its companion
CMS manuscript. They are enforced by `tests/test_conventions_locked.py`.
Changing any convention requires updating the manuscript, the tests,
and this document simultaneously.

## 1. Covariance-level pooling

The pooled tensor is the arithmetic mean of site-level covariance tensors:

    C_pool = (1/N) Σ_i C_i

Vector concatenation across sites is not used for any manuscript-facing
diagnostic. This preserves trace normalisation and avoids conflating
frame counts with directional content.

## 2. Absolute amplitude gate

The gating threshold δ₀ is specified in absolute physical units (Å),
not as a percentile of the amplitude distribution. This ensures a fixed,
material-independent interpretation. The manuscript default is δ₀ = 0.20 Å
for the KNN demonstration; sensitivity is tested at 0.05, 0.10, and 0.20 Å.

## 3. Squared-dot nematic alignment

All axis-alignment metrics use the squared dot product:

    S_ij = |ê_i · ê_j|²

This makes the metric head-tail symmetric (ê and −ê are equivalent),
appropriate for nematic-type orientational order where the principal
axis has no intrinsic sign.

## 4. Cartesian coordinates

All displacement vectors are computed in Cartesian coordinates after
minimum-image unwrapping. No fractional or reduced coordinates are
used in any diagnostic computation.

## 5. Degeneracy epsilon

    EPS_DEGENERACY = 0.05

on the trace-normalised spectrum (Σλ = 1). Sites with spectral gap
λ₁ − λ₂ < ε have axis-based metrics (S_align, persistence A and B,
pairwise S_ij) suppressed. D_dir and Δ_coh remain well-defined.

Note: The reference implementation used ε = 0.01 internally. Both thresholds produce
identical degeneracy masks on the KNN dataset (minimum spectral gap =
0.0607 > 0.05). Parity tests verify this equivalence explicitly.

## 6. Persistence windows: regime-dependent

Persistence metrics (A, B) require stable principal eigenvectors within
each trajectory segment. The number of segments W is chosen based on
trajectory length to ensure reliable covariance estimation:

- **Synthetic calibration (Table 3):** W = 4.
  T = 10,000 frames → segments of ≥ 2,500 frames.
- **KNN demonstration (Tables 4, S5):** W = 2 (half-split).
  Per-site gated frames ≈ 80–160 → segments of ≈ 40–80 frames.

Both use `np.array_split` for frame-preserving partition (distributes
remainder frames rather than dropping them).

Classification thresholds (LOCKED > 0.8, SWITCHING 0.4–0.8, FAST < 0.4)
are calibrated against synthetic systems at W = 4. For short-trajectory
data analysed with W = 2, these thresholds serve as qualitative reference
points rather than strict classification boundaries.

The reproduce scripts set W explicitly per table.

## 7. Independence from the reference implementation

dirprobe contains zero imports from the reference implementation. The two codebases are
independent implementations of the same mathematical framework. Cross-
validation is performed via exported fixtures, not shared code.

## 8. Bundled data policy

Bundled KNN data under `src/dirprobe/data/knn/` are manuscript-frozen,
minimal, and read-only. They contain only what is required for referee-mode
reproduction of Tables 2, 3, and 4.

Every bundled file appears in `DATA_SHA256SUMS.txt`. Checksum verification
runs automatically via `dirprobe.data.verify_bundled_data()`.

Raw trajectory data are not bundled. They are available from the
corresponding author upon reasonable request.
