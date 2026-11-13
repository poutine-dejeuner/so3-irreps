"""wigner_d.py — Wigner D-matrix decomposition for SO(3) functions.

Approach: use scipy.special.sph_harm for reliable spherical harmonic evaluation,
and construct Wigner D-matrix elements via rotation on the sphere.

Key identity:
    D^l_{m,m'}(\\alpha,\\beta,\\gamma) = e^{-im\\alpha} d^l_{m,m'}(\\beta) e^{-im'\\gamma}

For m'=0:  D^l_{m,0}(\\alpha,\\beta,\\gamma) = sqrt(4π/(2l+1)) * Y_l^m(\\beta, \\alpha)

For general m': we can evaluate by rotating a test function and decomposing.
Practical approach: use the Rodrigues rotation formula to express Wigner D
as an integral, or use numerical rotation of spherical harmonics.

We implement:
    wigner_D_frame(l, R)  — evaluate all (2l+1)² matrix elements for batch of R.

This uses the fact that the Wigner D-matrix is the matrix representation of
SO(3) on the space of degree-l spherical harmonics. So:
    1. Take spherical harmonics Y_l^m evaluated on a grid of points on S²
    2. Apply rotation R to the grid points
    3. The transformed functions expressed in the original basis give D^l(R)
"""

from __future__ import annotations

import numpy as np
from scipy.special import sph_harm_y as _sph_harm_complex  # type: ignore

# ---------------------------------------------------------------------------
# Real spherical harmonics
# ---------------------------------------------------------------------------

def real_sh(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Real spherical harmonic Y_l^m(theta, phi)."""
    y = _sph_harm_complex(l, m, theta, phi)
    if m == 0:
        return y.real.astype(np.float64)
    elif m > 0:
        return (np.sqrt(2.0) * ((-1) ** m) * y.real).astype(np.float64)
    else:
        return (np.sqrt(2.0) * ((-1) ** m) * y.imag).astype(np.float64)


def real_sh_cplx(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Complex spherical harmonic Y_l^m (Condon-Shortley phase)."""
    return _sph_harm_complex(l, m, theta, phi)


# ---------------------------------------------------------------------------
# Wigner D-matrix: evaluate via rotation of spherical harmonics
# ---------------------------------------------------------------------------

def _s2_grid(n_theta: int = 2, n_phi: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gauss-Legendre grid on S² for quadrature."""
    # For exact integration up to degree L, need n_theta >= L+1, n_phi >= 2L+1
    # We use a simple uniform grid for now (not GL, for simplicity)
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TT, PP = np.meshgrid(theta, phi, indexing='ij')
    return TT.ravel(), PP.ravel(), np.ones(n_theta * n_phi) * np.sin(TT.ravel()) * (np.pi/n_theta) * (2*np.pi/n_phi)


def wigner_D_real(l: int, R: np.ndarray) -> np.ndarray:
    """Compute real Wigner D-matrix D^l(R) for each rotation in batch.

    Uses the fact that D^l(R) acts on the (2l+1)-dim space of degree-l
    spherical harmonics. We compute it by:
        1. Evaluating Y_l^m on a quadrature grid on S²
        2. Applying R⁻¹ to the grid points
        3. Evaluating Y_l^{m'} on the rotated points
        4. The matrix of coefficients gives D^l_{m,m'}(R)

    Args:
        l: degree.
        R: (batch, 3, 3) rotation matrices.

    Returns:
        D: (batch, 2l+1, 2l+1) real Wigner D-matrices.
          Rows indexed by m (-l..+l), columns by m' (-l..+l).
    """
    batch = R.shape[0]
    dim = 2 * l + 1

    # Quadrature grid dense enough to resolve degree l
    n_theta = max(l + 2, 6)
    n_phi = max(2 * l + 2, 12)
    theta_grid, phi_grid, _ = _s2_grid(n_theta, n_phi)
    n_grid = len(theta_grid)

    # Cartesian coordinates of grid points
    st = np.sin(theta_grid)
    ct = np.cos(theta_grid)
    sp = np.sin(phi_grid)
    cp = np.cos(phi_grid)
    s2_points = np.stack([st * cp, st * sp, ct], axis=-1)  # (n_grid, 3)

    # Evaluate Y_l^{m'} on the reference grid
    # Y_ref[m', p] = Y_l^{m'}(point_p)
    Y_ref = np.zeros((dim, n_grid), dtype=np.float64)
    for idx, mp in enumerate(range(-l, l + 1)):
        Y_ref[idx] = real_sh(l, mp, theta_grid, phi_grid)

    # Apply inverse rotation to grid points: R^{-1} = R^T
    # rotated_points[b, p, :] = R[b]^T @ s2_points[p]
    rotated = np.einsum('bij,pj->bpi', R, s2_points)  # (batch, n_grid, 3)
    # Note: R^T = R^{-1} for SO(3), but we want R^{-1} on the argument
    # D^l_{m,m'}(R) = integral Y_l^m(R·x) Y_l^{m'}(x) dΩ
    # = integral Y_l^m(y) Y_l^{m'}(R^{-1}·y) dΩ   (change of vars y = R·x)
    # So we evaluate Y^{m'} at R^T @ points:
    # Note: R @ point = rotating the point by R
    # So R^{-1} @ point = R^T @ point
    # We use R^T (transpose) for the inverse rotation.

    # theta, phi of rotated points
    r_norm = np.sqrt(np.sum(rotated**2, axis=-1) + 1e-15)
    rotated = rotated / r_norm[..., None]
    theta_r = np.arccos(np.clip(rotated[:, :, 2], -1, 1))
    phi_r = np.arctan2(rotated[:, :, 1], rotated[:, :, 0])

    # Evaluate Y_l^m at rotated points
    # Y_rot[b, m, p] = Y_l^m(rotated_point_{b,p})
    Y_rot = np.zeros((batch, dim, n_grid), dtype=np.float64)
    for idx, m in enumerate(range(-l, l + 1)):
        Y_rot[:, idx, :] = real_sh(l, m, theta_r, phi_r)

    # D^l_{m,m'}(R) = sum_p Y_l^m(R·point_p) Y_l^{m'}(point_p) * w_p
    # = Y_rot @ Y_ref^T with quadrature weights
    # But we need proper quadrature. For now, use normalized dot product
    # since the grid is uniform and we're doing approximate integration.
    # Better: use the fact that D(R) @ Y_ref basis gives Y evaluated at
    # rotated points. So:
    #   Y_rot[b, :, p] = D(R)[b] @ Y_ref[:, p]
    # Solve in least-squares: D[b] = argmin ||D @ Y_ref - Y_rot[b]||²
    # D[b] = Y_rot[b] @ Y_ref^T @ (Y_ref @ Y_ref^T)^{-1}

    Y_ref_gram = Y_ref @ Y_ref.T  # (dim, dim)
    Y_ref_inv = np.linalg.pinv(Y_ref_gram)

    D = Y_rot @ Y_ref.T @ Y_ref_inv  # (batch, dim, dim)

    return D


def wigner_D_real_m0(l: int, R: np.ndarray) -> np.ndarray:
    """Extract the m'=0 column of Wigner D: D^l_{m,0}(R) = sqrt(4π/(2l+1)) Y_l^m(R·e₁).

    This is proportional to the real spherical harmonic evaluated at the
    north pole rotated by R. Using e₁ (x-axis) instead of e_z:
        R·e₁ has spherical coordinates (θ, φ).
        D^l_{m,0}(R) = sqrt(4π/(2l+1)) Y_l^m(θ, φ).

    Args:
        l: degree.
        R: (batch, 3, 3) rotation matrices.

    Returns:
        D_m0: (batch, 2l+1) — D^l_{m,0}(R) for m = -l..+l.
    """
    # Take column 0 of R (R·e₁) or column 2 (R·e_z)
    # We'll use both and let the caller choose
    col0 = R[:, :, 0]  # (batch, 3)
    theta = np.arccos(np.clip(col0[:, 2], -1, 1))
    phi = np.arctan2(col0[:, 1], col0[:, 0])

    dim = 2 * l + 1
    D = np.zeros((R.shape[0], dim), dtype=np.float64)
    norm = np.sqrt(4 * np.pi / (2 * l + 1))
    for idx, m in enumerate(range(-l, l + 1)):
        D[:, idx] = norm * real_sh(l, m, theta, phi)
    return D


# ---------------------------------------------------------------------------
# Euler angle utilities
# ---------------------------------------------------------------------------

def so3_random(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample n random SO(3) matrices with Haar measure.

    Returns:
        R: (n, 3, 3)
        alpha, beta, gamma: (n,) Euler angles (ZYZ).
    """
    rng = np.random.RandomState(seed)
    cos_beta = 2 * rng.rand(n) - 1
    beta = np.arccos(np.clip(cos_beta, -1, 1))
    alpha = 2 * np.pi * rng.rand(n)
    gamma = 2 * np.pi * rng.rand(n)

    R = euler_to_matrix(alpha, beta, gamma)
    return R, alpha, beta, gamma


def euler_to_matrix(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Convert ZYZ Euler angles to 3×3 rotation matrices.

    R(α,β,γ) = R_z(α) · R_y(β) · R_z(γ)

    Args:
        alpha, beta, gamma: (n,) arrays.

    Returns:
        R: (n, 3, 3)
    """
    n = len(alpha)
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)

    R = np.zeros((n, 3, 3))
    R[:, 0, 0] = ca*cb*cg - sa*sg
    R[:, 1, 0] = sa*cb*cg + ca*sg
    R[:, 2, 0] = -sb*cg
    R[:, 0, 1] = -ca*cb*sg - sa*cg
    R[:, 1, 1] = -sa*cb*sg + ca*cg
    R[:, 2, 1] = sb*sg
    R[:, 0, 2] = ca*sb
    R[:, 1, 2] = sa*sb
    R[:, 2, 2] = cb

    return R


def so3_euler_grid(n_alpha: int = 20, n_beta: int = 10, n_gamma: int = 20):
    """Grid of Euler angles covering SO(3).

    Returns:
        R: (N, 3, 3), alpha: (N,), beta: (N,), gamma: (N,)
    """
    alphas = np.linspace(0, 2*np.pi, n_alpha, endpoint=False)
    betas = np.linspace(0, np.pi, n_beta)
    gammas = np.linspace(0, 2*np.pi, n_gamma, endpoint=False)

    AA, BB, GG = np.meshgrid(alphas, betas, gammas, indexing='ij')
    alpha = AA.ravel()
    beta = BB.ravel()
    gamma = GG.ravel()

    R = euler_to_matrix(alpha, beta, gamma)
    return R, alpha, beta, gamma


def so3_grid_s2_slice(n_alpha: int = 50, n_beta: int = 25):
    """Grid of rotations with gamma=0 (only m'=0 column active).

    Returns SO(3) restricted to the S² slice.
    """
    alphas = np.linspace(0, 2*np.pi, n_alpha, endpoint=False)
    betas = np.linspace(0, np.pi, n_beta)

    AA, BB = np.meshgrid(alphas, betas, indexing='ij')
    alpha = AA.ravel()
    beta = BB.ravel()
    gamma = np.zeros_like(alpha)

    R = euler_to_matrix(alpha, beta, gamma)
    return R, alpha, beta, gamma
