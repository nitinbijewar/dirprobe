# KNN Displacement Data

Nb B-site displacement vectors for 10 KNN configurations.

Each `config_XX.npz` contains:

- `displacements`: shape `(N_frames, 8, 3)` float64 — Nb position minus
  centroid of 6 nearest O neighbours, in Cartesian coordinates (angstrom),
  minimum-image unwrapped.
- `config_id`: int (1-10)
- `ordering`: str (e.g., "rock-salt", "anti-phase")
- `temperature_range`: `[300.0, 325.0]` — frames selected from the 300-325 K
  window of NPT MLFF-accelerated MD trajectories.

Site axis (dim 1) order corresponds to the 8 B-sites (Nb, indices 8-15)
in the 2x2x2 perovskite supercell (K4Na4Nb8O24).

Pre-gating: raw displacement amplitudes are preserved.  Apply amplitude
gating (e.g., delta_0 = 0.20 angstrom) before computing covariance tensors.

Source: first-principles MLFF molecular dynamics (VASP, PBE functional).
