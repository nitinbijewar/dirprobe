"""von Mises–Fisher sampling and analytic eigenvalues.

Wood (1994) rejection sampler for d=3.  Analytic eigenvalue formula for
the second-moment tensor of a vMF distribution centred on the z-axis.
"""

from __future__ import annotations

import numpy as np


def sample_vmf(
    mu: np.ndarray,
    kappa: float,
    n: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample *n* unit vectors from vMF(mu, kappa) in d=3.

    Uses the Wood (1994) rejection algorithm.

    Parameters
    ----------
    mu : (3,) unit vector, distribution mean direction.
    kappa : float >= 0, concentration parameter.
    n : int, number of samples.
    rng : numpy Generator or None.

    Returns
    -------
    (n, 3) array of unit vectors.
    """
    if rng is None:
        rng = np.random.default_rng()

    mu = np.asarray(mu, dtype=float)
    mu = mu / np.linalg.norm(mu)
    d = 3

    if kappa < 1e-12:
        # Uniform on S^2
        samples = rng.normal(0, 1, (n, d))
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        return samples

    # Wood (1994) parameters for d=3 (p = d-1 = 2, m = (p-1)/2 = 0.5)
    # b = (-2*kappa + sqrt(4*kappa^2 + (d-1)^2)) / (d-1)
    b = (-2.0 * kappa + np.sqrt(4.0 * kappa**2 + (d - 1) ** 2)) / (d - 1)
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (d - 1) * np.log(1.0 - x0**2)

    samples = np.empty((n, d))
    idx = 0

    while idx < n:
        # Over-sample to reduce loop iterations
        batch = max(n - idx, 256)
        z = rng.beta((d - 1) / 2.0, (d - 1) / 2.0, size=batch)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        u = rng.uniform(0, 1, size=batch)

        accept = kappa * w + (d - 1) * np.log(1.0 - x0 * w) - c >= np.log(u)
        w_accepted = w[accept]
        n_acc = len(w_accepted)
        if n_acc == 0:
            continue

        take = min(n_acc, n - idx)
        w_use = w_accepted[:take]

        # Random tangent directions
        v = rng.normal(0, 1, (take, d - 1))
        v /= np.linalg.norm(v, axis=1, keepdims=True)

        # Construct samples around z-axis, then rotate to mu
        sqrt_term = np.sqrt(1.0 - w_use**2)
        z_samples = np.column_stack(
            [v[:, 0] * sqrt_term, v[:, 1] * sqrt_term, w_use]
        )
        samples[idx : idx + take] = z_samples
        idx += take

    # Rotate from z-axis to mu
    samples = _rotate_z_to_mu(samples, mu)
    return samples


def _rotate_z_to_mu(samples: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Rotate samples generated around z-axis to be centred on mu."""
    z = np.array([0.0, 0.0, 1.0])
    if np.allclose(mu, z, atol=1e-10):
        return samples
    if np.allclose(mu, -z, atol=1e-10):
        # Reflect through xy-plane
        result = samples.copy()
        result[:, 2] *= -1
        return result

    # Rodrigues' rotation
    v = np.cross(z, mu)
    s = np.linalg.norm(v)
    c = np.dot(z, mu)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    rot = np.eye(3) + vx + vx @ vx * (1.0 - c) / (s**2)
    return samples @ rot.T


def vmf_eigenvalues_3d(kappa: float) -> np.ndarray:
    """Analytic trace-normalised eigenvalues of C = <u u^T> for vMF(z, kappa).

    For d=3, C is diagonal in the {perp, perp, parallel} basis.  The
    parallel eigenvalue is  E[cos^2(theta)] = 1 - 2*L(kappa)/kappa,
    where L(kappa) = coth(kappa) - 1/kappa is the Langevin function.
    Uses 1/tanh for coth to avoid overflow at kappa > 700.

    Parameters
    ----------
    kappa : float >= 0.

    Returns
    -------
    (3,) array, descending order, sum = 1.
    """
    if kappa < 1e-12:
        return np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

    # Langevin function: L(kappa) = coth(kappa) - 1/kappa
    langevin = 1.0 / np.tanh(kappa) - 1.0 / kappa
    # Parallel eigenvalue: E[cos^2(theta)]
    lam_parallel = 1.0 - 2.0 * langevin / kappa
    lam_perp = (1.0 - lam_parallel) / 2.0
    return np.array([lam_parallel, lam_perp, lam_perp])


def vmf_d_dir_3d(kappa: float) -> float:
    """Analytic D_dir for vMF(z, kappa) in d=3.

    Parameters
    ----------
    kappa : float >= 0.

    Returns
    -------
    float, D_dir = 1 / sum(lambda_i^2).
    """
    evals = vmf_eigenvalues_3d(kappa)
    return float(1.0 / np.sum(evals**2))
