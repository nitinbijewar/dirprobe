"""14 CMS manuscript synthetic systems (A-N).

Each system returns a dict with displacements (T, N, 3), parameters,
ground_truth, and label. Ground truth is derived from construction,
NOT copied from frozen JSON.

SEED_BASE = 20260321 (independent of archival manuscript seeds).
"""

from __future__ import annotations

import numpy as np

from dirprobe.synthetic.vmf import sample_vmf, vmf_d_dir_3d

SEED_BASE = 20260321
_T = 10_000
_N = 8

# Cube-vertex axes (8 normalised directions)
_CUBE_AXES = np.array([
    [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
    [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
], dtype=float)
_CUBE_AXES /= np.linalg.norm(_CUBE_AXES, axis=1, keepdims=True)


def _rng(index: int) -> np.random.Generator:
    return np.random.default_rng(SEED_BASE + index)


# ── System A: Isotropic ──────────────────────────────────────────────

def system_A(T: int = _T, N: int = _N) -> dict:
    """Isotropic: uniform on S². D_dir = 3.0 (exact)."""
    rng = _rng(0)
    mu = np.array([0.0, 0.0, 1.0])
    displacements = np.stack(
        [sample_vmf(mu, 0.0, T, rng) for _ in range(N)], axis=1
    )
    return {
        "displacements": displacements,
        "parameters": {"kappa": 0.0, "axes": "uniform"},
        "ground_truth": {
            "kind": "analytic",
            "D_dir_site": 3.0,
            "D_dir_pooled": 3.0,
            "Delta_coh": 0.0,
            "S_align": None,  # degeneracy flagged
            "robust": "ROBUST",
            "tolerance": {"D_dir": 0.05, "Delta_coh": 0.05},
        },
        "label": "A",
    }


# ── System B: Uniaxial ──────────────────────────────────────────────

def system_B(T: int = _T, N: int = _N, kappa: float = 50.0) -> dict:
    """Uniaxial: vMF(z, kappa=50), all axes aligned."""
    rng = _rng(1)
    mu = np.array([0.0, 0.0, 1.0])
    displacements = np.stack(
        [sample_vmf(mu, kappa, T, rng) for _ in range(N)], axis=1
    )
    d_dir = vmf_d_dir_3d(kappa)
    return {
        "displacements": displacements,
        "parameters": {"kappa": kappa, "axes": "aligned_z"},
        "ground_truth": {
            "kind": "analytic",
            "D_dir_site": d_dir,
            "D_dir_pooled": d_dir,
            "Delta_coh": 0.0,
            "S_align": 1.0,
            "robust": "ROBUST",
            "tolerance": {"D_dir": 0.05, "Delta_coh": 0.05},
        },
        "label": "B",
    }


# ── System C: Planar ─────────────────────────────────────────────────

def system_C(T: int = _T, N: int = _N) -> dict:
    """Planar: directions in xy-plane. D_dir = 2.0 (exact)."""
    rng = _rng(2)
    displacements = np.zeros((T, N, 3))
    for i in range(N):
        theta = rng.uniform(0, 2 * np.pi, T)
        dirs = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(T)])
        displacements[:, i, :] = dirs
    return {
        "displacements": displacements,
        "parameters": {"type": "planar"},
        "ground_truth": {
            "kind": "analytic",
            "D_dir_site": 2.0,
            "D_dir_pooled": 2.0,
            "Delta_coh": 0.0,
            "S_align": 1.0,
            "robust": "ROBUST",
            "tolerance": {"D_dir": 0.05, "Delta_coh": 0.05},
        },
        "label": "C",
    }


# ── System D: Incoherent ─────────────────────────────────────────────

def system_D(T: int = _T, N: int = _N, kappa: float = 50.0) -> dict:
    """Incoherent: cube-vertex axes, kappa=50. Large Delta_coh."""
    rng = _rng(3)
    displacements = np.stack(
        [sample_vmf(_CUBE_AXES[i], kappa, T, rng) for i in range(N)], axis=1
    )
    d_dir_site = vmf_d_dir_3d(kappa)
    return {
        "displacements": displacements,
        "parameters": {"kappa": kappa, "axes": "cube_vertices"},
        "ground_truth": {
            "kind": "analytic",
            "D_dir_site": d_dir_site,
            "D_dir_pooled": 3.0,  # 8 cube-vertex axes → isotropic pooled
            "Delta_coh": 3.0 - d_dir_site,
            "S_align": None,  # computed empirically
            "robust": "ROBUST",
            "tolerance": {"D_dir": 0.05, "Delta_coh": 0.1},
        },
        "label": "D",
    }


# ── System E: Interpolation ──────────────────────────────────────────

def system_E(T: int = _T, N: int = _N, kappa_inter: float = 5.0) -> dict:
    """Two-level vMF hierarchy. mu_i ~ vMF(z, kappa_inter), u ~ vMF(mu_i, 50)."""
    rng = _rng(4)
    mu_z = np.array([0.0, 0.0, 1.0])
    site_axes = sample_vmf(mu_z, kappa_inter, N, rng)
    displacements = np.stack(
        [sample_vmf(site_axes[i], 50.0, T, rng) for i in range(N)], axis=1
    )
    d_dir_site = vmf_d_dir_3d(50.0)
    return {
        "displacements": displacements,
        "parameters": {"kappa_inner": 50.0, "kappa_inter": kappa_inter},
        "ground_truth": {
            "kind": "semi_analytic",
            "D_dir_site": d_dir_site,
            "D_dir_pooled": None,  # depends on drawn axes
            "Delta_coh": None,
            "robust": "ROBUST",
            "tolerance": {"D_dir": 0.05, "Delta_coh": 0.2},
        },
        "label": "E",
    }


# ── System F: Mixed kappa ────────────────────────────────────────────

def system_F(T: int = _T, N: int = _N) -> dict:
    """Heterogeneous kappa per site on cube vertices."""
    rng = _rng(5)
    kappas = [20, 20, 50, 50, 50, 100, 100, 100]
    displacements = np.stack(
        [sample_vmf(_CUBE_AXES[i], float(kappas[i]), T, rng) for i in range(N)],
        axis=1,
    )
    return {
        "displacements": displacements,
        "parameters": {"kappas": kappas, "axes": "cube_vertices"},
        "ground_truth": {
            "kind": "empirical",
            "D_dir_site": None,  # varies per site
            "D_dir_pooled": None,
            "Delta_coh": None,  # empirical, expected >= 0
            "robust": "ROBUST",
            "tolerance": {"Delta_coh": 0.2},
        },
        "label": "F",
    }


# ── System G: Static ─────────────────────────────────────────────────

def system_G(T: int = _T, N: int = _N, kappa: float = 50.0) -> dict:
    """Static locked axes. A ~ 1, B ~ 1, LOCKED."""
    rng = _rng(6)
    displacements = np.stack(
        [sample_vmf(_CUBE_AXES[i], kappa, T, rng) for i in range(N)], axis=1
    )
    return {
        "displacements": displacements,
        "parameters": {"kappa": kappa, "rate": 0.0},
        "ground_truth": {
            "kind": "analytic",
            "classification": "LOCKED",
            "robust": "ROBUST",
            "tolerance": {"A": 0.05, "B": 0.05},
        },
        "label": "G",
    }


# ── System H: Switching ──────────────────────────────────────────────

def system_H(
    T: int = _T, N: int = _N, kappa: float = 50.0, rate: float = 0.001,
) -> dict:
    """Bernoulli switching between z and x at rate r per frame.

    Each site switches independently. Switching decisions and vMF sampling
    use separate RNG streams to prevent coupling.
    """
    # Deterministic per-rate child RNGs (switching and sampling separated)
    rate_idx = TABLE3_H_RATES.index(rate) if rate in TABLE3_H_RATES else 0
    parent = np.random.default_rng(SEED_BASE + 7 * 1000 + rate_idx)
    child_seeds = parent.integers(0, 2**62, size=2)
    rng_switch = np.random.default_rng(child_seeds[0])
    rng_sample = np.random.default_rng(child_seeds[1])

    axis_z = np.array([0.0, 0.0, 1.0])
    axis_x = np.array([1.0, 0.0, 0.0])

    displacements = np.zeros((T, N, 3))
    for i in range(N):
        current = axis_z.copy()
        # Pre-generate all switch decisions for this site
        switch_mask = rng_switch.random(T) < rate
        # Pre-generate replacement axes
        n_switches = int(switch_mask.sum())
        if n_switches > 0:
            new_axes = rng_switch.standard_normal((n_switches, 3))
            new_axes /= np.linalg.norm(new_axes, axis=1, keepdims=True)

        # Build segments of constant axis, then batch-sample vMF
        segments: list[tuple[np.ndarray, int, int]] = []
        seg_start = 0
        switch_count = 0
        for t in range(T):
            if switch_mask[t]:
                if t > seg_start:
                    segments.append((current.copy(), seg_start, t))
                current = new_axes[switch_count]
                switch_count += 1
                seg_start = t
        if seg_start < T:
            segments.append((current.copy(), seg_start, T))

        for axis, s, e in segments:
            n_frames = e - s
            displacements[s:e, i, :] = sample_vmf(axis, kappa, n_frames, rng_sample)

    return {
        "displacements": displacements,
        "parameters": {"kappa": kappa, "rate": rate},
        "ground_truth": {
            "kind": "calibration",
            "classification": None,  # depends on rate
            "robust": "ROBUST",
            "tolerance": {"A": 0.1, "B": 0.1},
        },
        "label": "H",
    }


# Table 3 rates (from submitted manuscript Table 3)
TABLE3_H_RATES = [0.0001, 0.0005, 0.001, 0.003, 0.008, 0.02]


# ── System I: Drift ──────────────────────────────────────────────────

def system_I(
    T: int = _T, N: int = _N, kappa: float = 50.0, omega: float = 0.0002,
) -> dict:
    """Continuous axis drift at omega rad/frame around great circle.

    Each site drifts around an independent rotation axis.
    Drift and vMF sampling use separate RNG streams.
    """
    drift_idx = TABLE3_I_DRIFTS.index(omega) if omega in TABLE3_I_DRIFTS else 0
    parent = np.random.default_rng(SEED_BASE + 8 * 1000 + drift_idx)
    child_seeds = parent.integers(0, 2**62, size=2)
    rng_drift = np.random.default_rng(child_seeds[0])
    rng_sample = np.random.default_rng(child_seeds[1])

    displacements = np.zeros((T, N, 3))

    for i in range(N):
        axis = _CUBE_AXES[i].copy()
        rot_ax = rng_drift.standard_normal(3)
        rot_ax /= np.linalg.norm(rot_ax)

        # Pre-compute all drifted axes
        axes = np.zeros((T, 3))
        for t in range(T):
            c, s = np.cos(omega), np.sin(omega)
            axis = (axis * c +
                    np.cross(rot_ax, axis) * s +
                    rot_ax * np.dot(rot_ax, axis) * (1 - c))
            axis /= np.linalg.norm(axis)
            axes[t] = axis

        # Batch-sample: one frame at a time (axis changes each frame)
        for t in range(T):
            displacements[t, i, :] = sample_vmf(axes[t], kappa, 1, rng_sample)[0]

    return {
        "displacements": displacements,
        "parameters": {"kappa": kappa, "omega": omega},
        "ground_truth": {
            "kind": "calibration",
            "classification": None,  # depends on omega
            "robust": "ROBUST",
            "tolerance": {"A": 0.1, "B": 0.1},
        },
        "label": "I",
    }


# Table 3 drifts (from submitted manuscript Table 3)
TABLE3_I_DRIFTS = [5e-05, 0.0001, 0.0002, 0.0005, 0.001]


# ── System J: Mixed confinement ───────────────────────────────────────

def system_J(T: int = _T, N: int = _N) -> dict:
    """Mixed confinement: sites 0-3 uniaxial (kappa=100), sites 4-7 planar."""
    rng = _rng(9)
    displacements = np.zeros((T, N, 3))
    mu_z = np.array([0.0, 0.0, 1.0])

    for i in range(N):
        if i < 4:
            # Uniaxial
            displacements[:, i, :] = sample_vmf(mu_z, 100.0, T, rng)
        else:
            # Planar
            theta = rng.uniform(0, 2 * np.pi, T)
            sigma_z = 0.05
            eta = rng.standard_normal(T)
            dirs = np.column_stack([np.cos(theta), np.sin(theta), sigma_z * eta])
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            displacements[:, i, :] = dirs

    return {
        "displacements": displacements,
        "parameters": {"type": "mixed_confinement"},
        "ground_truth": {
            "kind": "empirical",
            "D_dir_site": None,  # mixed
            "Delta_coh": None,  # empirical
            "robust": "ROBUST",
            "tolerance": {"D_dir": 0.3, "Delta_coh": 0.3},
        },
        "label": "J",
    }


# ── System K: Near-isotropic ─────────────────────────────────────────

def system_K(T: int = _T, N: int = _N, kappa: float = 0.5) -> dict:
    """Near-isotropic: low kappa. D_dir ~ 3.0."""
    rng = _rng(10)
    mu = np.array([0.0, 0.0, 1.0])
    displacements = np.stack(
        [sample_vmf(mu, kappa, T, rng) for _ in range(N)], axis=1
    )
    d_dir = vmf_d_dir_3d(kappa)
    return {
        "displacements": displacements,
        "parameters": {"kappa": kappa},
        "ground_truth": {
            "kind": "analytic",
            "D_dir_site": d_dir,
            "D_dir_pooled": d_dir,
            "Delta_coh": 0.0,
            "robust": "ROBUST",
            "tolerance": {"D_dir": 0.05, "Delta_coh": 0.05},
        },
        "label": "K",
    }


# ── System L: Finite-size ────────────────────────────────────────────

def system_L(T: int = _T, N: int = 8, kappa: float = 50.0) -> dict:
    """System D variant at N=8 (representative). Finite-size convergence."""
    rng = _rng(11)
    # Quasi-uniform axes via fibonacci sphere
    axes = _fibonacci_sphere(N)
    displacements = np.stack(
        [sample_vmf(axes[i], kappa, T, rng) for i in range(N)], axis=1
    )
    d_dir_site = vmf_d_dir_3d(kappa)
    return {
        "displacements": displacements,
        "parameters": {"kappa": kappa, "N": N, "axes": "fibonacci"},
        "ground_truth": {
            "kind": "semi_analytic",
            "D_dir_site": d_dir_site,
            "robust": "ROBUST",
            "tolerance": {"D_dir": 0.05, "Delta_coh": 0.2},
        },
        "label": "L",
    }


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n quasi-uniform points on S² via fibonacci spiral."""
    indices = np.arange(n, dtype=float)
    phi = np.arccos(1 - 2 * (indices + 0.5) / n)
    theta = np.pi * (1 + 5**0.5) * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])


# ── System M: Gating-sensitive ────────────────────────────────────────

def system_M(T: int = _T, N: int = _N, kappa: float = 50.0) -> dict:
    """Low amplitude: most frames gated out. SENSITIVE."""
    rng = _rng(12)
    mu = np.array([0.0, 0.0, 1.0])
    dirs = np.stack(
        [sample_vmf(mu, kappa, T, rng) for _ in range(N)], axis=1
    )
    # Amplitudes ~ Uniform(0, 0.06): at delta_0=0.05, ~17% retained
    amps = rng.uniform(0, 0.06, (T, N))
    displacements = dirs * amps[:, :, np.newaxis]

    return {
        "displacements": displacements,
        "parameters": {"kappa": kappa, "amp_max": 0.06, "gating": 0.05},
        "ground_truth": {
            "kind": "empirical",
            "robust": "SENSITIVE",
            "tolerance": {},
        },
        "label": "M",
    }


# ── System N: Temporally unstable ─────────────────────────────────────

def system_N(T: int = 50, N: int = _N, kappa: float = 50.0) -> dict:
    """Short trajectory with abrupt axis change at midpoint. SENSITIVE."""
    rng = _rng(13)
    mu_z = np.array([0.0, 0.0, 1.0])
    mu_x = np.array([1.0, 0.0, 0.0])
    mid = T // 2

    displacements = np.zeros((T, N, 3))
    for i in range(N):
        displacements[:mid, i, :] = sample_vmf(mu_z, kappa, mid, rng)
        displacements[mid:, i, :] = sample_vmf(mu_x, kappa, T - mid, rng)

    return {
        "displacements": displacements,
        "parameters": {"kappa": kappa, "T": T, "regime_change_at": mid},
        "ground_truth": {
            "kind": "empirical",
            "robust": "SENSITIVE",
            "tolerance": {},
        },
        "label": "N",
    }


# ── Batch generators ─────────────────────────────────────────────────

ALL_CMS_SYSTEMS = {
    "A": system_A, "B": system_B, "C": system_C, "D": system_D,
    "E": system_E, "F": system_F, "G": system_G, "H": system_H,
    "I": system_I, "J": system_J, "K": system_K, "L": system_L,
    "M": system_M, "N": system_N,
}


def generate_all_table2() -> dict[str, dict]:
    """Generate all 14 systems for Table 2 (one representative each)."""
    results = {}
    for label, fn in ALL_CMS_SYSTEMS.items():
        results[label] = fn()
    return results


def generate_all_table3() -> list[dict]:
    """Generate Table 3 rows: G (1) + H (6 rates) + I (5 drifts)."""
    rows = []

    # G: static (rate=0)
    g = system_G()
    rows.append({"system_data": g, "control_name": "rate", "control_value": 0.0,
                 "label": "G"})

    # H: 6 switching rates
    for rate in TABLE3_H_RATES:
        h = system_H(rate=rate)
        rows.append({"system_data": h, "control_name": "rate", "control_value": rate,
                     "label": f"H (rate={rate})"})

    # I: 5 drift rates
    for omega in TABLE3_I_DRIFTS:
        i_sys = system_I(omega=omega)
        rows.append({"system_data": i_sys, "control_name": "drift", "control_value": omega,
                     "label": f"I (drift={omega})"})

    return rows
