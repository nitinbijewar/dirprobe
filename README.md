# dirprobe

Directional diagnostics for site-resolved vector-valued trajectory data.

Standard covariance analyses pool site-level displacement vectors into a single tensor and extract dominant axes or effective dimensionalities. This conflates three logically distinct quantities: local directional confinement, inter-site axis dispersion, and temporal persistence. A large eigenvalue in a pooled tensor may reflect strong uniaxial confinement at every site. It may also reflect the incoherent averaging of strongly confined but misaligned sites. The two are not equivalent, and a pooled analysis cannot distinguish them.

**dirprobe** separates these contributions. From trace-normalised second-moment tensors at each site, it computes:

- **D_dir** — directional dimensionality (1 = uniaxial, 2 = planar, 3 = isotropic)
- **Δ_coh** — coherence gap: how much pooling inflates the apparent dimensionality
- **S_align** — nematic alignment of principal axes across sites
- **⟨A⟩, ⟨B⟩** — windowed temporal persistence metrics

A structured robustness suite evaluates stability under gating threshold variation, pooling convention choice, and temporal segmentation. The pipeline operates on generic trajectory data in arbitrary spatial dimension, without material-specific hardcoding.

## Quick Start

```bash
git clone https://github.com/nitinbijewar/dirprobe.git
cd dirprobe
pip install -e .
```

Reproduce the manuscript tables:

```bash
dirprobe reproduce              # Tables 2, 3, and 4 from bundled data
dirprobe reproduce --from-generators   # regenerate synthetic systems from scratch
dirprobe synthetic --smoke-test        # quick validation (Systems A, B, D)
```

After PyPI publication:

```bash
pip install dirprobe
```

## What It Solves

Consider eight crystallographic sites, each with a well-defined preferred displacement direction. If all eight axes are aligned, the pooled covariance tensor reflects the true local confinement — D_dir_pooled ≈ D_dir_site, and Δ_coh ≈ 0. If the eight axes point in different directions, pooling averages the rotated ellipsoids into an inflated, more isotropic tensor — D_dir_pooled ≫ D_dir_site, and Δ_coh is large. The local physics has not changed. Only the inter-site coherence has.

A standard analysis reports the pooled dimensionality and calls it "the" confinement geometry. This pipeline reports both the site-level and pooled values, quantifies the gap between them, and tests whether the result is temporally stable.

## Validation

Fourteen analytically controlled synthetic systems validate the pipeline against known ground truth. The systems span five independent dimensions: local confinement geometry (A, B, C), inter-site coherence (D, E, F), temporal dynamics (G, H, I), finite-size effects (L), and adversarial robustness (J, K, M, N). Twelve systems pass all robustness tests. Two (M, N) are deliberately constructed to trigger sensitivity flags.

See [SYNTHETIC_SYSTEMS.md](SYNTHETIC_SYSTEMS.md) for the full description, physics, and design logic of all fourteen systems.

## Reproduction Modes

The package ships five reproduction modes:

| Mode | Command | What it does |
|------|---------|-------------|
| Frozen | `dirprobe reproduce` | Reads bundled JSON, prints Tables 2–4 |
| KNN end-to-end | `dirprobe reproduce --from-displacements` | Recomputes Table 4 from bundled displacement arrays |
| Synthetic end-to-end | `dirprobe reproduce --from-generators` | Generates fresh synthetic systems, runs the full pipeline |
| Combined | `dirprobe reproduce --from-generators --from-displacements` | Both synthetic and KNN paths |
| Smoke test | `dirprobe synthetic --smoke-test` | Quick check: Systems A, B, D |

Frozen mode reproduces the published tables exactly. Generator mode validates the pipeline against analytic ground truth with fresh random seeds — values will differ from published tables, but the diagnostic behaviour must match.

## Using the Pipeline

```python
from dirprobe.pipeline import run_pipeline
import numpy as np

# displacements: (T, N, 3) array — T frames, N sites, 3 spatial dimensions
result = run_pipeline(displacements, gating_threshold=0.20)

print(f"D_dir (site mean): {result['d_dir_site_mean']:.3f}")
print(f"D_dir (pooled):    {result['d_dir_pooled']:.3f}")
print(f"Coherence gap:     {result['delta_coh']:.3f}")
print(f"S_align:           {result['s_align']:.3f}")
print(f"Persistence:       A={result['a_mean']:.3f}, B={result['b_mean']:.3f}")
print(f"Classification:    {result['classification']}")
```

The pipeline accepts any array of shape (T, N, d). No material-specific configuration is required beyond the gating threshold δ₀ and the number of persistence windows W.

## Package Structure

```
src/dirprobe/
    pipeline.py          # Single rank-2 pipeline assembler
    moments2/            # Core diagnostics: D_dir, Δ_coh, S_align
    time/                # Temporal persistence: ⟨A⟩, ⟨B⟩, classification
    gating/              # Amplitude gating
    robustness/          # Four-test sensitivity suite
    synthetic/
        systems.py       # 14 validation systems (A–N)
        vmf.py           # von Mises–Fisher sampling and analytic formulas
        generators.py    # Demo generation functions
        bundle.py        # Save/load/verify generation bundles
    reproduce/           # Manuscript table reproduction scripts
    data/                # Bundled frozen JSON and KNN displacement arrays
```

## Dependencies

Required: `numpy >= 1.20`, `scipy >= 1.7`

Optional: `matplotlib >= 3.5` (figures only)

Python >= 3.9.

## Citation

If you use this package, please cite:

```
N. Bijewar, "A Reproducible Diagnostic Pipeline for Separating Local Directional
Confinement from Inter-Site Coherence in Vector-Valued Trajectory Data,"
Computational Materials Science (submitted, 2026).
```

## Acknowledgments

Computing was provided by C-DAC on the PARAM AIRAWAT and PARAM SIDDHI facilities.

Development of this package was assisted by AI coding tools for implementation, testing, and documentation. All scientific design, validation, and editorial decisions were made by the author.

## License

MIT. See [LICENSE](LICENSE).
