"""
CFM-based NLI computation module.

Extracts the closed-form NLI efficiency functions (SPM, XPM, FWM) from
cfm_o_jlt.ipynb into reusable functions. Includes a high-level wrapper
``compute_edge_nli`` that handles ISRS power evolution, power-profile
fitting, and per-channel NLI computation for a single fiber span.

References
----------
[1] D. Semrau et al., "The Gaussian Noise Model in the Presence of
    Inter-Channel Stimulated Raman Scattering," JLT, 2018.
"""

import jax
import numpy as np
from jax import numpy as jnp
from jax.numpy import (
    exp, sqrt, log, abs, arcsinh, arctan, nan_to_num,
)
from scipy.constants import pi, c

from ong.models import raman_solver
from ong.models.raman_fitting import get_power_profile_fit

# ---------------------------------------------------------------------------
# Binary index tables used by the vmap-based summation over conjugate terms
# ---------------------------------------------------------------------------
_INDICES = (jnp.arange(4)[:, None] >> jnp.arange(2)) & 1
_INDICES_COI = (jnp.arange(16)[:, None] >> jnp.arange(4)) & 1
_INDICES_FWM = (jnp.arange(64)[:, None] >> jnp.arange(6)) & 1


# ---------------------------------------------------------------------------
# Dispersion helpers
# ---------------------------------------------------------------------------
def calc_beta3(lam, S, beta2):
    """Compute beta3 from dispersion slope *S* at wavelength *lam*."""
    return S * lam ** 4 / (4 * pi ** 2 * c ** 2) - beta2 * lam / (pi * c)


def calc_beta4(lam, Shat, beta2, beta3):
    """Compute beta4 from dispersion curvature *Shat* at wavelength *lam*."""
    return lam * (
        -Shat * lam ** 5 - 12 * pi * beta2 * c * lam - 24 * pi ** 2 * beta3 * c ** 2
    ) / (8 * pi ** 3 * c ** 3)


# ---------------------------------------------------------------------------
# SPM efficiency
# ---------------------------------------------------------------------------
@jax.jit
def _eta_GN_SPM(phi_i, B_i, a, a_bar, gamma, Tf, T, L):
    def _fun(x):
        l, l_line = x
        alpha = a + a_bar * l
        alpha_tilde = (alpha * (1 - exp(-alpha * L))) / (
            1 - exp(-alpha * L) - alpha * L * exp(-alpha * L)
        )
        alpha_line = a + a_bar * l_line
        alpha_tilde_line = (alpha_line * (1 - exp(-alpha_line * L))) / (
            1 - exp(-alpha_line * L) - alpha_line * L * exp(-alpha_line * L)
        )
        T_tilde = T * ((-Tf / T) ** l)
        T_tilde_line = T * ((-Tf / T) ** l_line)
        kappa = ((1 - exp(-alpha * L)) ** 2) / (
            1 - exp(-alpha * L) - alpha * L * exp(-alpha * L)
        )
        kappa_line = ((1 - exp(-alpha_line * L)) ** 2) / (
            1 - exp(-alpha_line * L) - alpha_line * L * exp(-alpha_line * L)
        )
        idx = phi_i == 0
        return (
            nan_to_num(
                (16 / 27)
                * gamma ** 2
                / B_i ** 2
                * (
                    (2 * kappa * kappa_line * pi * T_tilde * T_tilde_line)
                    / (phi_i * (alpha_tilde + alpha_tilde_line))
                )
                * (
                    arcsinh(3 * phi_i * B_i ** 2 / (8 * pi * alpha_tilde))
                    + arcsinh(3 * phi_i * B_i ** 2 / (8 * pi * alpha_tilde_line))
                ),
                nan=0,
                posinf=0,
                neginf=0,
            )
            + idx
            * (
                (16 / 27)
                * gamma ** 2
                * (
                    kappa
                    * kappa_line
                    * T_tilde
                    * T_tilde_line
                    * (3 / (4 * alpha_tilde * alpha_tilde_line))
                )
            )
        ).squeeze()

    return jax.vmap(_fun)(_INDICES).sum(axis=0)


# ---------------------------------------------------------------------------
# XPM efficiency
# ---------------------------------------------------------------------------
@jax.jit
def _eta_GN_XPM(Pi, Pk, phi_ik, B_i, B_k, a, a_bar, gamma, Tf, T, L):
    def _fun(x):
        l, l_line = x
        alpha = a + a_bar * l
        alpha_tilde = alpha * (1 - exp(-alpha * L)) / (
            1 - exp(-alpha * L) - alpha * L * exp(-alpha * L)
        )
        alpha_line = a + a_bar * l_line
        alpha_tilde_line = alpha_line * (1 - exp(-alpha_line * L)) / (
            1 - exp(-alpha_line * L) - alpha_line * L * exp(-alpha_line * L)
        )
        kappa = (1 - exp(-alpha * L)) ** 2 / (
            1 - exp(-alpha * L) - alpha * L * exp(-alpha * L)
        )
        kappa_line = (1 - exp(-alpha_line * L)) ** 2 / (
            1 - exp(-alpha_line * L) - alpha_line * L * exp(-alpha_line * L)
        )
        T_tilde = T * ((-Tf / T) ** l)
        T_tilde_line = T * ((-Tf / T) ** l_line)
        return (32 / 27) * jnp.sum(
            nan_to_num(
                (Pk / Pi) ** 2
                * gamma ** 2
                / B_k
                * (
                    kappa
                    * kappa_line
                    * 2
                    * T_tilde
                    * T_tilde_line
                    / (phi_ik * (alpha_tilde + alpha_tilde_line))
                )
                * (
                    arctan(phi_ik * B_i / (2 * alpha_tilde))
                    + arctan(phi_ik * B_i / (2 * alpha_tilde_line))
                ),
                nan=0,
                posinf=0,
                neginf=0,
            ),
            axis=1,
        ).squeeze()

    return jax.vmap(_fun)(_INDICES).sum(axis=0)


# ---------------------------------------------------------------------------
# FWM index generation
# ---------------------------------------------------------------------------
def _FWM_idx(f):
    """Build padded FWM triplet indices and validity mask for all COIs."""
    freqs = f.squeeze()
    f_i = freqs[:, None, None, None]
    f_j = freqs[None, :, None, None]
    f_k = freqs[None, None, :, None]
    f_m = freqs[None, None, None, :]
    cond = (
        ((f_j + f_k - f_m) == f_i)
        & (f_j != f_i)
        & (f_k != f_m)
        & (f_k != f_i)
        & (f_j != f_m)
        & (f_j <= f_k)
    )
    i_idx, j_idx, k_idx, m_idx = jnp.nonzero(cond)
    idx_flat = jnp.stack([i_idx, j_idx, k_idx, m_idx], axis=-1)
    counts = jnp.bincount(i_idx, length=freqs.size)

    n, K, M = counts.size, idx_flat.shape[0], int(counts.max())
    s = jnp.concatenate([jnp.array([0]), jnp.cumsum(counts)[:-1]])
    ch = jnp.repeat(jnp.arange(n), counts)
    r = jnp.arange(K) - s[ch]
    idx_pad = jnp.zeros((n, M, idx_flat.shape[1]), idx_flat.dtype).at[ch, r].set(
        idx_flat
    )
    valid = jnp.zeros((n, M)).at[ch, r].set(True)
    return idx_pad, valid


# ---------------------------------------------------------------------------
# FWM efficiency
# ---------------------------------------------------------------------------
@jax.jit
def _eta_GN_FWM(
    Ptot, P, beta2, beta3, beta4, a, a_bar, f, B, Cr, gamma, L, idx_pad, valid
):
    def _ch(i):
        idx_ch = idx_pad[i]
        valid_ch = valid[i]

        def _eta_per_ch(
            Ptot, P, beta2, beta3, beta4, a, a_bar, f, B, Cr, gamma, L, idx_ch, valid_ch
        ):
            T_tilde = -((Ptot * Cr) / (2 * a)) * f
            T = 1 + T_tilde

            def _eta(idx):
                i, j, k, m = idx

                def _phi(i, j, k, f, beta2, beta3, beta4):
                    phi_jk = -4 * pi ** 2 * (f[j] - f[i]) * (f[k] - f[i]) * (
                        beta2
                        + pi * beta3 * (f[j] + f[k])
                        + (2 / 3)
                        * pi ** 2
                        * beta4
                        * (
                            f[j] ** 2
                            + f[j] * f[k]
                            + f[k] ** 2
                            + 0.5 * (f[j] - f[i]) * (f[k] - f[i])
                        )
                    )
                    dphi_f1 = -4 * pi ** 2 * (f[k] - f[i]) * (
                        beta2
                        + pi * beta3 * (f[j] + f[k] + f[j] - f[i])
                        + (2 / 3)
                        * pi ** 2
                        * beta4
                        * (
                            f[j] ** 2
                            + f[j] * f[k]
                            + f[k] ** 2
                            + 0.5 * (f[j] - f[i]) * (f[k] - f[i])
                            + (f[j] - f[i])
                            * (2 * (f[j] - f[i]) + 1.5 * (f[k] - f[i]) + 3 * f[i])
                        )
                    )
                    dphi_f2 = -4 * pi ** 2 * (f[j] - f[i]) * (
                        beta2
                        + pi * beta3 * (f[j] + f[k] + f[k] - f[i])
                        + (2 / 3)
                        * pi ** 2
                        * beta4
                        * (
                            f[j] ** 2
                            + f[j] * f[k]
                            + f[k] ** 2
                            + 0.5 * (f[j] - f[i]) * (f[k] - f[i])
                            + (f[k] - f[i])
                            * (2 * (f[k] - f[i]) + 1.5 * (f[j] - f[i]) + 3 * f[i])
                        )
                    )
                    return phi_jk, dphi_f1, dphi_f2

                phi_jk, dphi_f1, dphi_f2 = _phi(i, j, k, f, beta2, beta3, beta4)

                def _island(
                    i, j, k, m, a, a_bar, T, T_tilde, B, P, gamma, L, phi_jk, dphi_f1, dphi_f2
                ):
                    Omega = jnp.where(j == k, 1, 2)
                    Bi = B[i]; Bj = B[j]; Bk = B[k]; Bm = B[m]
                    Pi = P[i]; Pj = P[j]; Pk = P[k]; Pm = P[m]
                    gamma_i = gamma[i]
                    f1bound = Bj / 2
                    f2min = -Bk / 2
                    f2max = Bk / 2

                    def _F(a, b, d, f2):
                        term_plus = a + b * f2 + d
                        term_minus = a + b * f2 - d
                        return (
                            (term_plus * arctan(term_plus) - term_minus * arctan(term_minus)) / b
                            - 0.5 / b * (log(1.0 + term_plus ** 2) - log(1.0 + term_minus ** 2))
                        )

                    def _island_COI(x):
                        l_j, l_k, l_j_line, l_k_line = x
                        alpha_j = a[j] / 2 + l_j * a_bar[j]
                        alpha_k = a[k] / 2 + l_k * a_bar[k]
                        alpha_all = alpha_j + alpha_k
                        alpha_j_line = a[j] / 2 + l_j_line * a_bar[j]
                        alpha_k_line = a[k] / 2 + l_k_line * a_bar[k]
                        alpha_all_line = alpha_j_line + alpha_k_line
                        T_tilde_j = T[j] * (-T_tilde[j] / T[j]) ** l_j
                        T_tilde_k = T[k] * (-T_tilde[k] / T[k]) ** l_k
                        T_tilde_all = T_tilde_j * T_tilde_k
                        T_tilde_j_line = T[j] * (-T_tilde[j] / T[j]) ** l_j_line
                        T_tilde_k_line = T[k] * (-T_tilde[k] / T[k]) ** l_k_line
                        T_tilde_all_line = T_tilde_j_line * T_tilde_k_line
                        alpha_tilde_all = (
                            alpha_all
                            * (1 - exp(-alpha_all * L))
                            / (1 - exp(-alpha_all * L) - alpha_all * L * exp(-alpha_all * L))
                        )
                        kappa_all = (
                            (1 - exp(-alpha_all * L)) ** 2
                            / (1 - exp(-alpha_all * L) - alpha_all * L * exp(-alpha_all * L))
                        )
                        alpha_tilde_all_line = (
                            alpha_all_line
                            * (1 - exp(-alpha_all_line * L))
                            / (
                                1
                                - exp(-alpha_all_line * L)
                                - alpha_all_line * L * exp(-alpha_all_line * L)
                            )
                        )
                        kappa_all_line = (
                            (1 - exp(-alpha_all_line * L)) ** 2
                            / (
                                1
                                - exp(-alpha_all_line * L)
                                - alpha_all_line * L * exp(-alpha_all_line * L)
                            )
                        )
                        a1 = phi_jk / alpha_tilde_all
                        a2 = phi_jk / alpha_tilde_all_line
                        b1 = dphi_f2 / alpha_tilde_all
                        b2 = dphi_f2 / alpha_tilde_all_line
                        d1 = dphi_f1 * f1bound / alpha_tilde_all
                        d2 = dphi_f1 * f1bound / alpha_tilde_all_line
                        F1 = _F(a1, b1, d1, f2max) - _F(a1, b1, d1, f2min)
                        F2 = _F(a2, b2, d2, f2max) - _F(a2, b2, d2, f2min)
                        return (
                            T_tilde_all
                            * T_tilde_all_line
                            * kappa_all
                            * kappa_all_line
                            / (alpha_tilde_all + alpha_tilde_all_line)
                            * (F1 + F2)
                            / dphi_f1
                        )

                    def _island_FWM(x):
                        l_j, l_k, l_m, l_j_line, l_k_line, l_m_line = x
                        alpha_j = a[j] / 2 + l_j * a_bar[j]
                        alpha_k = a[k] / 2 + l_k * a_bar[k]
                        alpha_m = a[m] / 2 + l_m * a_bar[m]
                        alpha_i = a[i] / 2
                        alpha_all = alpha_j + alpha_k + alpha_m - alpha_i
                        alpha_j_line = a[j] / 2 + l_j_line * a_bar[j]
                        alpha_k_line = a[k] / 2 + l_k_line * a_bar[k]
                        alpha_m_line = a[m] / 2 + l_m_line * a_bar[m]
                        alpha_i_line = a[i] / 2
                        alpha_all_line = alpha_j_line + alpha_k_line + alpha_m_line - alpha_i_line
                        T_tilde_j = T[j] * (-T_tilde[j] / T[j]) ** l_j
                        T_tilde_k = T[k] * (-T_tilde[k] / T[k]) ** l_k
                        T_tilde_m = T[m] * (-T_tilde[m] / T[m]) ** l_m
                        T_tilde_all = T_tilde_j * T_tilde_k * T_tilde_m
                        T_tilde_j_line = T[j] * (-T_tilde[j] / T[j]) ** l_j_line
                        T_tilde_k_line = T[k] * (-T_tilde[k] / T[k]) ** l_k_line
                        T_tilde_m_line = T[m] * (-T_tilde[m] / T[m]) ** l_m_line
                        T_tilde_all_line = T_tilde_j_line * T_tilde_k_line * T_tilde_m_line
                        alpha_tilde_all = (
                            alpha_all
                            * (1 - exp(-alpha_all * L))
                            / (1 - exp(-alpha_all * L) - alpha_all * L * exp(-alpha_all * L))
                        )
                        kappa_all = (
                            (1 - exp(-alpha_all * L)) ** 2
                            / (1 - exp(-alpha_all * L) - alpha_all * L * exp(-alpha_all * L))
                        )
                        alpha_tilde_all_line = (
                            alpha_all_line
                            * (1 - exp(-alpha_all_line * L))
                            / (
                                1
                                - exp(-alpha_all_line * L)
                                - alpha_all_line * L * exp(-alpha_all_line * L)
                            )
                        )
                        kappa_all_line = (
                            (1 - exp(-alpha_all_line * L)) ** 2
                            / (
                                1
                                - exp(-alpha_all_line * L)
                                - alpha_all_line * L * exp(-alpha_all_line * L)
                            )
                        )
                        a1 = phi_jk / alpha_tilde_all
                        a2 = phi_jk / alpha_tilde_all_line
                        b1 = dphi_f2 / alpha_tilde_all
                        b2 = dphi_f2 / alpha_tilde_all_line
                        d1 = dphi_f1 * f1bound / alpha_tilde_all
                        d2 = dphi_f1 * f1bound / alpha_tilde_all_line
                        F1 = _F(a1, b1, d1, f2max) - _F(a1, b1, d1, f2min)
                        F2 = _F(a2, b2, d2, f2max) - _F(a2, b2, d2, f2min)
                        return (
                            T_tilde_all
                            * T_tilde_all_line
                            * kappa_all
                            * kappa_all_line
                            / (alpha_tilde_all + alpha_tilde_all_line)
                            * (F1 + F2)
                            / dphi_f1
                        )

                    sum_val = jax.lax.cond(
                        m == i,
                        lambda _: jax.vmap(_island_COI)(_INDICES_COI).sum(axis=0),
                        lambda _: jax.vmap(_island_FWM)(_INDICES_FWM).sum(axis=0),
                        operand=None,
                    )

                    eta_jkm = nan_to_num(
                        Omega
                        * (16 / 27)
                        * gamma_i ** 2
                        * (Bi / Pi ** 3)
                        * (Pj * Pk * Pm / (Bj * Bk * Bm))
                        * sum_val,
                        nan=0,
                        posinf=0,
                        neginf=0,
                    )
                    return eta_jkm

                return _island(
                    i, j, k, m, a, a_bar, T, T_tilde, B, P, gamma, L, phi_jk, dphi_f1, dphi_f2
                )

            return jnp.where(valid_ch, jax.vmap(_eta)(idx_ch).squeeze(), 0).sum(axis=0)

        return _eta_per_ch(
            Ptot, P, beta2, beta3, beta4, a, a_bar, f, B, Cr, gamma, L, idx_ch, valid_ch
        )

    return jax.lax.map(_ch, jnp.arange(f.size))


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------
def compute_edge_nli(
    ch_centre_hz,
    ch_bandwidth_hz,
    ch_power_W,
    attenuation_dBm,
    gamma,
    Aeff,
    raman_profile,
    ref_lambda,
    span_length_m,
    num_spans,
    beta2,
    beta3,
    beta4,
    raman_gain_slope,
    o_band_mask=None,
    samples_per_km=2,
):
    """Compute per-channel NLI efficiency for a single homogeneous link.

    Parameters
    ----------
    ch_centre_hz : array (N,)
        Channel centre frequencies relative to ref_lambda [Hz].
    ch_bandwidth_hz : array (N,)
        Channel bandwidths [Hz].
    ch_power_W : array (N,)
        Channel launch powers [W].
    attenuation_dBm : array (N,)
        Per-channel fibre attenuation [dB/m].
    gamma : array (N,)
        Per-channel nonlinear coefficient [1/W/m].
    Aeff : array (N,)
        Per-channel effective area [m^2].
    raman_profile : tuple (freq_offset, gain)
        Raman gain profile.
    ref_lambda : float
        Reference wavelength [m].
    span_length_m : float
        Span length [m].
    num_spans : int
        Number of identical spans.
    beta2, beta3, beta4 : float
        Dispersion coefficients at ref_lambda.
    raman_gain_slope : float
        Raman gain slope [1/W/Hz/m].
    o_band_mask : array (N,) of bool, optional
        True for O-band channels (FWM is computed only for these).
    samples_per_km : int
        Spatial sampling density for ISRS ODE.

    Returns
    -------
    eta_spm : array (N,)
        SPM efficiency per channel.
    eta_xpm : array (N,)
        XPM efficiency per channel.
    eta_fwm : array (N,)
        FWM efficiency per channel (zero for non-O-band channels).
    """
    N = ch_centre_hz.shape[0]
    L = span_length_m

    # --- 1. Solve ISRS ODE for power evolution along the span ----------
    z = jnp.logspace(0, jnp.log10(L), num=int(L / 1e3 * samples_per_km + 1))

    _, power_evo = raman_solver.solve_isrs_evolution(
        ch_centre_i=ch_centre_hz,
        A_eff=Aeff,
        raman_profile=raman_profile,
        length=L,
        attenuation_i=attenuation_dBm,
        ch_power_W_i=ch_power_W,
        ref_lambda=ref_lambda,
        zspan=z,
    )

    # --- 2. Fit power evolution to semi-analytical model ---------------
    length_j = jnp.array([L])
    ch_centre_ij = ch_centre_hz[:, None]
    attenuation_ij = attenuation_dBm[:, None]
    ch_power_W_ij = ch_power_W[:, None]
    raman_gain_slope_j = jnp.array([raman_gain_slope])

    fit_params = get_power_profile_fit(
        length_j=length_j,
        power_evo_j=power_evo[None, :, :],
        ch_centre_ij=ch_centre_ij,
        ch_power_W_ij=ch_power_W_ij,
        attenuation_ij=attenuation_ij,
        raman_gain_slope_j=raman_gain_slope_j,
        zspan=z,
    )[:, :, None]  # shape (N, 3, 1)

    a = fit_params[:, 0]      # shape (N, 1)
    a_bar = fit_params[:, 1]  # shape (N, 1)
    Cr = fit_params[:, 2]     # shape (N, 1)

    # --- 3. Prepare intermediate variables for NLI ---------------------
    j = 0
    P_ij = ch_power_W_ij
    Ptot = jnp.sum(P_ij, axis=0)
    gamma_ij = gamma[:, None]
    if gamma_ij.ndim == 1:
        gamma_ij = jnp.repeat(gamma[:, None], N, axis=1)
    fi = ch_centre_ij
    Bch = ch_bandwidth_hz[:, None]

    a_i = a[:, j : j + 1]
    a_k = jnp.transpose(a[:, j : j + 1])
    a_bar_i = a_bar[:, j : j + 1]
    a_bar_k = jnp.transpose(a_bar[:, j : j + 1])
    f_i = fi[:, j : j + 1]
    f_k = jnp.transpose(fi[:, j : j + 1])
    B_i = Bch[:, j : j + 1]
    B_k = jnp.transpose(Bch[:, j : j + 1])
    Cr_i = Cr[:, j : j + 1]
    Cr_k = jnp.transpose(Cr[:, j : j + 1])
    P_i = P_ij[:, j : j + 1]
    P_k = jnp.transpose(P_ij[:, j : j + 1])

    phi_i = -4 * pi ** 2 * (
        beta2 + pi * beta3 * (f_i + f_i) + 2 * pi ** 2 * beta4 * f_i ** 2
    )

    phi_ik = -4 * pi ** 2 * (f_k - f_i) * (
        beta2
        + pi * beta3 * (f_i + f_k)
        + 2 / 3 * pi ** 2 * beta4 * (f_i ** 2 + f_i * f_k + f_k ** 2)
    )

    Tf_i = -((Ptot[j] * Cr_i) / a_bar_i) * f_i
    Tf_k = -((Ptot[j] * Cr_k) / a_bar_k) * f_k

    T_i = 1 + Tf_i
    T_k = 1 + Tf_k

    # --- 4. SPM ---------------------------------------------------------
    # gamma must be (N,1) to broadcast correctly along the COI axis
    gamma_col = gamma_ij[:, j : j + 1]
    eta_spm = _eta_GN_SPM(
        phi_i, B_i, a_i, a_bar_i, gamma_col, Tf_i, T_i, length_j[j]
    )
    if eta_spm.ndim == 2:
        eta_spm = eta_spm.squeeze()

    # --- 5. XPM ---------------------------------------------------------
    eta_xpm = _eta_GN_XPM(
        P_i, P_k, phi_ik, B_i, B_k, a_k, a_bar_k, gamma_col, Tf_k, T_k, length_j[j]
    )

    # --- 6. FWM (O-band only) ------------------------------------------
    eta_fwm = jnp.zeros(N)
    if o_band_mask is not None and jnp.any(o_band_mask):
        o_idx = jnp.where(o_band_mask)[0]
        f_o = f_i[o_idx, 0]
        idx_pad, valid = _FWM_idx(f_o)
        eta_fwm_o = _eta_GN_FWM(
            Ptot[j],
            P_i[o_idx, 0],
            beta2,
            beta3,
            beta4,
            a_i[o_idx, 0],
            a_bar_i[o_idx, 0],
            f_o,
            B_i[o_idx, 0],
            Cr_i[o_idx, 0],
            gamma_ij[o_idx, j],
            length_j[j],
            idx_pad,
            valid,
        )
        eta_fwm = eta_fwm.at[o_idx].set(eta_fwm_o)

    # --- 7. Scale by num_spans (incoherent accumulation) ----------------
    eta_spm = eta_spm * num_spans
    eta_xpm = eta_xpm * num_spans
    eta_fwm = eta_fwm * num_spans

    return eta_spm, eta_xpm, eta_fwm
