"""Fifteen synthetic test systems for dirprobe validation.

Each system_X function returns (site_directions, ground_truth) where
site_directions is list[np.ndarray] of (T_i, 3) unit-vector arrays and
ground_truth is a dict with at minimum:
    system_name, d_dir_expected, delta_coh_expected, n_sites, n_frames.

System index mapping (for seeding: rng = default_rng(42 + index)):
    a=0, b=1, c=2, c2=3, d=4, e=5, f=6, g=7, h=8, i=9, j=10,
    k=11, l=12, m=13, n=14.
"""

from __future__ import annotations

import numpy as np

from dirprobe._constants import EPS_DEGENERACY
from dirprobe.synthetic.vmf import sample_vmf, vmf_d_dir_3d


# ── Helpers ──────────────────────────────────────────────────────────


def _make_rng(system_index: int) -> np.random.Generator:
    return np.random.default_rng(42 + system_index)


def _unit_normalise(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-30)
    return arr / norms


# ── System A: Uniaxial, all sites along z ────────────────────────────


def system_a(
    n_sites: int = 8, n_frames: int = 800, kappa: float = 50.0,
) -> tuple[list[np.ndarray], dict]:
    """Strongly uniaxial: all sites share z-axis, high kappa."""
    rng = _make_rng(0)
    mu = np.array([0.0, 0.0, 1.0])
    sites = [sample_vmf(mu, kappa, n_frames, rng) for _ in range(n_sites)]
    return sites, {
        "system_name": "a",
        "d_dir_expected": 1.0,
        "delta_coh_expected": 0.0,
        "n_sites": n_sites,
        "n_frames": n_frames,
        "description": "uniaxial_aligned",
        "kappa": kappa,
    }


# ── System B: Isotropic (uniform on S²) ─────────────────────────────


def system_b(
    n_sites: int = 8, n_frames: int = 800,
) -> tuple[list[np.ndarray], dict]:
    """Isotropic: uniform random on S², D_dir ≈ 3."""
    rng = _make_rng(1)
    sites = []
    for _ in range(n_sites):
        dirs = rng.normal(0, 1, (n_frames, 3))
        dirs = _unit_normalise(dirs)
        sites.append(dirs)
    return sites, {
        "system_name": "b",
        "d_dir_expected": 3.0,
        "delta_coh_expected": 0.0,
        "n_sites": n_sites,
        "n_frames": n_frames,
        "description": "isotropic",
    }


# ── System C: Moderate concentration ─────────────────────────────────


def system_c(
    n_sites: int = 8, n_frames: int = 800, kappa: float = 5.0,
) -> tuple[list[np.ndarray], dict]:
    """Moderate concentration along z, D_dir ≈ 1.5–2.5."""
    rng = _make_rng(2)
    mu = np.array([0.0, 0.0, 1.0])
    sites = [sample_vmf(mu, kappa, n_frames, rng) for _ in range(n_sites)]
    return sites, {
        "system_name": "c",
        "d_dir_expected": vmf_d_dir_3d(kappa),
        "delta_coh_expected": 0.0,
        "n_sites": n_sites,
        "n_frames": n_frames,
        "description": "moderate_concentration",
        "kappa": kappa,
    }


# ── System C2: Same kappa as C, different axis per site ──────────────


def system_c2(
    n_sites: int = 8, n_frames: int = 800, kappa: float = 5.0,
) -> tuple[list[np.ndarray], dict]:
    """Same kappa as C but each site has a DIFFERENT random axis → delta_coh > 0."""
    rng = _make_rng(3)
    sites = []
    for _ in range(n_sites):
        mu = rng.normal(0, 1, 3)
        mu /= np.linalg.norm(mu)
        sites.append(sample_vmf(mu, kappa, n_frames, rng))
    return sites, {
        "system_name": "c2",
        "d_dir_expected": vmf_d_dir_3d(kappa),
        "delta_coh_expected": None,  # depends on axis spread
        "n_sites": n_sites,
        "n_frames": n_frames,
        "description": "multi_axis_moderate",
        "kappa": kappa,
    }


# ── System D: Planar (two degenerate eigenvalues on top) ─────────────


def system_d(
    n_sites: int = 8, n_frames: int = 800,
) -> tuple[list[np.ndarray], dict]:
    """Planar: directions confined to xy-plane, D_dir ≈ 2."""
    rng = _make_rng(4)
    sites = []
    for _ in range(n_sites):
        theta = rng.uniform(0, 2 * np.pi, n_frames)
        dirs = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(n_frames)])
        noise = rng.normal(0, 0.01, dirs.shape)
        dirs += noise
        dirs = _unit_normalise(dirs)
        sites.append(dirs)
    return sites, {
        "system_name": "d",
        "d_dir_expected": 2.0,
        "delta_coh_expected": 0.0,
        "n_sites": n_sites,
        "n_frames": n_frames,
        "description": "planar_isotropic",
    }


# ── System E: High kappa, different axes → high delta_coh ───────────


def system_e(
    n_sites: int = 8, n_frames: int = 800, kappa: float = 50.0,
) -> tuple[list[np.ndarray], dict]:
    """High concentration but each site has orthogonal-ish axes → large delta_coh."""
    rng = _make_rng(5)
    sites = []
    for si in range(n_sites):
        # Distribute axes evenly on sphere
        phi = 2 * np.pi * si / n_sites
        theta = np.pi * (si + 1) / (n_sites + 1)
        mu = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ])
        sites.append(sample_vmf(mu, kappa, n_frames, rng))
    return sites, {
        "system_name": "e",
        "d_dir_expected": 1.0,
        "delta_coh_expected": None,  # large positive
        "n_sites": n_sites,
        "n_frames": n_frames,
        "description": "high_kappa_diverse_axes",
        "kappa": kappa,
    }


# ── System F: Two-group split ────────────────────────────────────────


def system_f(
    n_sites: int = 8, n_frames: int = 800, kappa: float = 30.0,
) -> tuple[list[np.ndarray], dict]:
    """Two groups: half along z, half along x → moderate delta_coh."""
    rng = _make_rng(6)
    mu_z = np.array([0.0, 0.0, 1.0])
    mu_x = np.array([1.0, 0.0, 0.0])
    sites = []
    for si in range(n_sites):
        mu = mu_z if si < n_sites // 2 else mu_x
        sites.append(sample_vmf(mu, kappa, n_frames, rng))
    return sites, {
        "system_name": "f",
        "d_dir_expected": 1.0,
        "delta_coh_expected": None,  # moderate positive
        "n_sites": n_sites,
        "n_frames": n_frames,
        "description": "two_group_split",
        "kappa": kappa,
    }


# ── System G: Gradual tilt across sites ──────────────────────────────


def system_g(
    n_sites: int = 8, n_frames: int = 800, kappa: float = 30.0,
) -> tuple[list[np.ndarray], dict]:
    """Gradual tilt: site axes progressively rotate from z to x."""
    rng = _make_rng(7)
    sites = []
    for si in range(n_sites):
        theta = (np.pi / 2) * si / max(n_sites - 1, 1)
        mu = np.array([np.sin(theta), 0.0, np.cos(theta)])
        sites.append(sample_vmf(mu, kappa, n_frames, rng))
    return sites, {
        "system_name": "g",
        "d_dir_expected": 1.0,
        "delta_coh_expected": None,  # moderate positive from axis spread
        "n_sites": n_sites,
        "n_frames": n_frames,
        "description": "gradual_tilt",
        "kappa": kappa,
    }


# ── System H: One outlier site ───────────────────────────────────────


def system_h(
    n_sites: int = 8, n_frames: int = 800, kappa: float = 50.0,
) -> tuple[list[np.ndarray], dict]:
    """One outlier: 7 sites along z, 1 along x → small delta_coh."""
    rng = _make_rng(8)
    mu_z = np.array([0.0, 0.0, 1.0])
    mu_x = np.array([1.0, 0.0, 0.0])
    sites = []
    for si in range(n_sites):
        mu = mu_x if si == 0 else mu_z
        sites.append(sample_vmf(mu, kappa, n_frames, rng))
    return sites, {
        "system_name": "h",
        "d_dir_expected": 1.0,
        "delta_coh_expected": None,  # small positive
        "n_sites": n_sites,
        "n_frames": n_frames,
        "description": "one_outlier",
        "kappa": kappa,
    }


# ── System I: Very high kappa (near-perfect alignment) ───────────────


def system_i(
    n_sites: int = 8, n_frames: int = 800, kappa: float = 500.0,
) -> tuple[list[np.ndarray], dict]:
    """Very high kappa: near-perfect uniaxial, D_dir → 1."""
    rng = _make_rng(9)
    mu = np.array([0.0, 0.0, 1.0])
    sites = [sample_vmf(mu, kappa, n_frames, rng) for _ in range(n_sites)]
    return sites, {
        "system_name": "i",
        "d_dir_expected": 1.0,
        "delta_coh_expected": 0.0,
        "n_sites": n_sites,
        "n_frames": n_frames,
        "description": "very_high_kappa",
        "kappa": kappa,
    }


# ── System J: Low kappa (near-isotropic) ─────────────────────────────


def system_j(
    n_sites: int = 8, n_frames: int = 800, kappa: float = 0.5,
) -> tuple[list[np.ndarray], dict]:
    """Low kappa: near-isotropic, D_dir → 3."""
    rng = _make_rng(10)
    mu = np.array([0.0, 0.0, 1.0])
    sites = [sample_vmf(mu, kappa, n_frames, rng) for _ in range(n_sites)]
    return sites, {
        "system_name": "j",
        "d_dir_expected": vmf_d_dir_3d(kappa),
        "delta_coh_expected": 0.0,
        "n_sites": n_sites,
        "n_frames": n_frames,
        "description": "low_kappa",
        "kappa": kappa,
    }


# ── System K: Near-degenerate (axis_degenerate ≈ EPS boundary) ───────


def system_k(
    n_sites: int = 8, n_frames: int = 800,
) -> tuple[list[np.ndarray], dict]:
    """Near-degenerate: eigenvalues near 1/3 with gap < EPS (DETERMINISTIC).

    Constructs directions by sampling from vMF with very low kappa
    so that the covariance is near-isotropic (λ₁ ≈ λ₂ ≈ 1/3),
    triggering axis_degenerate = True.
    """
    rng = _make_rng(11)
    # kappa small enough that λ_parallel - λ_perp < EPS_DEGENERACY
    # λ_parallel = 1 - 2*L(κ)/κ, λ_perp = (1 - λ_parallel)/2
    # For small κ: L(κ) ≈ κ/3, so λ_parallel ≈ 1/3 and gap → 0
    kappa = 0.05  # very low → near-isotropic → degenerate
    mu = np.array([0.0, 0.0, 1.0])
    sites = [sample_vmf(mu, kappa, n_frames, rng) for _ in range(n_sites)]

    from dirprobe.synthetic.vmf import vmf_eigenvalues_3d
    analytic_evals = vmf_eigenvalues_3d(kappa)

    return sites, {
        "system_name": "k",
        "d_dir_expected": float(1.0 / np.sum(analytic_evals**2)),
        "delta_coh_expected": 0.0,
        "n_sites": n_sites,
        "n_frames": n_frames,
        "description": "near_degenerate",
        "kappa": kappa,
    }


# ── System L: Switching dynamics ─────────────────────────────────────


def system_l(
    n_sites: int = 8, n_frames: int = 800, kappa: float = 50.0,
) -> tuple[list[np.ndarray], dict]:
    """Switching: axis alternates between z and x every 50 frames."""
    rng = _make_rng(12)
    mu_z = np.array([0.0, 0.0, 1.0])
    mu_x = np.array([1.0, 0.0, 0.0])
    sites = []
    block_size = 50  # = DEFAULT_WINDOW_SIZE → every window sees different axis
    for _ in range(n_sites):
        blocks = []
        for bi in range(n_frames // block_size):
            mu = mu_z if bi % 2 == 0 else mu_x
            blocks.append(sample_vmf(mu, kappa, block_size, rng))
        dirs = np.vstack(blocks)
        sites.append(dirs)
    return sites, {
        "system_name": "l",
        "d_dir_expected": 2.0,
        "delta_coh_expected": 0.0,
        "n_sites": n_sites,
        "n_frames": n_frames,
        "description": "switching",
        "kappa": kappa,
        "persistence_expected": "FAST",
    }


# ── System M: Drifting dynamics ──────────────────────────────────────


def system_m(
    n_sites: int = 8, n_frames: int = 800, kappa: float = 30.0,
) -> tuple[list[np.ndarray], dict]:
    """Drifting: axis smoothly rotates from z toward x over trajectory."""
    rng = _make_rng(13)
    sites = []
    for _ in range(n_sites):
        dirs = np.empty((n_frames, 3))
        for t in range(n_frames):
            theta = (np.pi / 2) * t / n_frames
            mu = np.array([np.sin(theta), 0.0, np.cos(theta)])
            dirs[t] = sample_vmf(mu, kappa, 1, rng)[0]
        sites.append(dirs)
    return sites, {
        "system_name": "m",
        "d_dir_expected": None,  # depends on drift rate
        "delta_coh_expected": 0.0,
        "n_sites": n_sites,
        "n_frames": n_frames,
        "description": "drifting",
        "kappa": kappa,
        "persistence_expected": "SWITCHING",
    }


# ── System N: Pathologically short + nonstationary ───────────────────


def system_n(
    n_sites: int = 8, n_frames: int = 50, kappa: float = 50.0,
) -> tuple[list[np.ndarray], dict]:
    """Pathological: 50 frames, axis rotates 90° at frame 25.

    Must produce FAIL from run_robustness_suite.
    """
    rng = _make_rng(14)
    mu_z = np.array([0.0, 0.0, 1.0])
    mu_x = np.array([1.0, 0.0, 0.0])
    sites = []
    for _ in range(n_sites):
        first_half = sample_vmf(mu_z, kappa, n_frames // 2, rng)
        second_half = sample_vmf(mu_x, kappa, n_frames - n_frames // 2, rng)
        dirs = np.vstack([first_half, second_half])
        sites.append(dirs)
    return sites, {
        "system_name": "n",
        "d_dir_expected": None,
        "delta_coh_expected": None,
        "n_sites": n_sites,
        "n_frames": n_frames,
        "description": "pathological_short_nonstationary",
        "kappa": kappa,
        "expected_suite_summary": "FAIL",
    }


# ── Registry ─────────────────────────────────────────────────────────

ALL_SYSTEMS = {
    "a": system_a,
    "b": system_b,
    "c": system_c,
    "c2": system_c2,
    "d": system_d,
    "e": system_e,
    "f": system_f,
    "g": system_g,
    "h": system_h,
    "i": system_i,
    "j": system_j,
    "k": system_k,
    "l": system_l,
    "m": system_m,
    "n": system_n,
}


def generate_all() -> dict[str, tuple[list[np.ndarray], dict]]:
    """Generate all 15 synthetic systems.

    Returns
    -------
    dict mapping system name to (site_directions, ground_truth).
    """
    return {name: fn() for name, fn in ALL_SYSTEMS.items()}
