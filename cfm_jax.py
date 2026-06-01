"""
Pure-array JAX version of ``compute_throughput_cfm.py``.

This module keeps the same overall workflow:

1. Read topology + band-aware RWA from MongoDB.
2. Build the multiband channel grid and occupancy matrix.
3. Compute per-link NSR.
4. Walk lightpaths and accumulate Shannon capacity.
5. Write results back to MongoDB.

The important difference is that the NSR kernel does not touch lazy ``setup``
properties inside the hot path. We first materialize all required quantities
into plain arrays, then run a JAX-friendly kernel on those arrays.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import os
import pathlib
import sys
import time
from dataclasses import dataclass

cuda_lib = "/apps/cuda/cuda-13.0/lib64/libcudart.so.13"
cupti_lib = "/apps/cuda/cuda-13.0/extras/CUPTI/lib64/libcupti.so.13"
cudnn_lib = "/apps/cuda/cudnn-linux-x86_64-9.14.0.64_cuda13/lib/libcudnn.so.9"

try:
    ctypes.CDLL(cuda_lib)
    ctypes.CDLL(cupti_lib)
    ctypes.CDLL(cudnn_lib)
except Exception as e:
    print(f"wrong check: {e}")

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_cuda_data_dir=/apps/cuda/cuda-13.0"
    " --xla_gpu_deterministic_ops=true"
)

import jax
from jax import numpy as jnp
import networkx as nx
import numpy as np
from scipy.constants import pi

jax.config.update("jax_enable_x64", True)

print(f"Success! GPU count: {jax.device_count()}")
print(f"Devices: {jax.devices()}")

# ---------------------------------------------------------------------------
# Path setup (must come before local imports)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_EXTERNAL_DIR = _SCRIPT_DIR.parents[2] / "external"
_ONG_SRC = _EXTERNAL_DIR / "ong-python-toolbox" / "src"

for p in (_EXTERNAL_DIR, str(_ONG_SRC)):
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)

import NetworkToolkit as nt
import compute_throughput_cfm as cfm


@dataclass
class PreparedCFMInputs:
    ref_lambda: float
    channel_idx: np.ndarray
    ch_idx_oband_active: np.ndarray
    active_slot_ix: np.ndarray
    length_j: np.ndarray
    length_scalar: float
    zspan: np.ndarray
    ch_lambda_active: np.ndarray
    ch_centre_active: np.ndarray
    ch_centre_ij_active: np.ndarray
    ch_bandwidth_active: np.ndarray
    ch_bandwidth_ij_active: np.ndarray
    ch_power_W_active: np.ndarray
    ch_power_W_ij_active: np.ndarray
    attenuation_active: np.ndarray
    attenuation_ij_active: np.ndarray
    nf_active: np.ndarray
    snr_trx_active: np.ndarray
    gamma_active: np.ndarray
    aeff_active: np.ndarray
    beta2: float
    beta3: float
    beta4: float
    raman_profile: object
    raman_gain_slope_j: np.ndarray
    fwm_idx_pad: np.ndarray | None
    fwm_valid: np.ndarray | None
    fwm_idx_pad_oband: np.ndarray | None
    fwm_valid_oband: np.ndarray | None


def _prepare_cfm_inputs(setup, ch_idx_oband_active) -> PreparedCFMInputs:
    """Materialize everything the NSR kernel needs into plain arrays."""
    chs = np.asarray(setup.channel_idx, dtype=bool)
    span = setup.spans[0]

    ch_lambda_active = np.asarray(setup.ch_lambda_ij[chs, 0], dtype=np.float64)
    ch_centre_active = np.asarray(setup.ch_centre_ij[chs, 0], dtype=np.float64)
    ch_centre_ij_active = np.asarray(setup.ch_centre_ij[chs, :], dtype=np.float64)
    ch_bandwidth_active = np.asarray(setup.ch_bandwidth_ij[chs, 0], dtype=np.float64)
    ch_bandwidth_ij_active = np.asarray(setup.ch_bandwidth_ij[chs, :], dtype=np.float64)
    ch_power_W_active = np.asarray(setup.ch_power_W_ij[chs, 0], dtype=np.float64)
    ch_power_W_ij_active = np.asarray(setup.ch_power_W_ij[chs, :], dtype=np.float64)
    attenuation_active = np.asarray(setup.attenuation_ij[chs, 0], dtype=np.float64)
    attenuation_ij_active = np.asarray(setup.attenuation_ij[chs, :], dtype=np.float64)
    nf_active = np.asarray(setup.NF_i[chs], dtype=np.float64)
    snr_trx_active = np.asarray(setup.snr_trx[chs], dtype=np.float64)
    gamma_active = np.asarray(span.nonlinear_coeff_at(setup.ch_lambda_ij[chs, 0]), dtype=np.float64)
    aeff_active = np.asarray(span.A_eff_at(setup.ch_lambda_ij[chs, 0]), dtype=np.float64)

    beta2 = float(span.beta2)
    beta3 = float(span.beta3)
    beta4 = float(span.beta4)
    length_j = np.asarray(setup.length_j, dtype=np.float64)
    length_scalar = float(length_j[0])
    zspan = np.arange(0.0, length_scalar, 1000.0, dtype=np.float64)
    active_slot_ix = np.flatnonzero(chs)

    fwm_idx_pad = fwm_valid = None
    fwm_idx_pad_oband = fwm_valid_oband = None

    idx_pad_all, valid_all = _build_fwm_idx(jnp.asarray(ch_centre_active))
    fwm_idx_pad = np.asarray(idx_pad_all, dtype=np.int32)
    fwm_valid = np.asarray(valid_all, dtype=bool)

    ch_idx_oband_active_np = np.asarray(ch_idx_oband_active, dtype=bool)
    if np.any(ch_idx_oband_active_np):
        idx_pad_o, valid_o = _build_fwm_idx(
            jnp.asarray(ch_centre_active[ch_idx_oband_active_np])
        )
        fwm_idx_pad_oband = np.asarray(idx_pad_o, dtype=np.int32)
        fwm_valid_oband = np.asarray(valid_o, dtype=bool)

    return PreparedCFMInputs(
        ref_lambda=float(setup.ref_lambda),
        channel_idx=np.asarray(chs, dtype=bool),
        ch_idx_oband_active=ch_idx_oband_active_np,
        active_slot_ix=active_slot_ix,
        length_j=length_j,
        length_scalar=length_scalar,
        zspan=zspan,
        ch_lambda_active=ch_lambda_active,
        ch_centre_active=ch_centre_active,
        ch_centre_ij_active=ch_centre_ij_active,
        ch_bandwidth_active=ch_bandwidth_active,
        ch_bandwidth_ij_active=ch_bandwidth_ij_active,
        ch_power_W_active=ch_power_W_active,
        ch_power_W_ij_active=ch_power_W_ij_active,
        attenuation_active=attenuation_active,
        attenuation_ij_active=attenuation_ij_active,
        nf_active=nf_active,
        snr_trx_active=snr_trx_active,
        gamma_active=gamma_active,
        aeff_active=aeff_active,
        beta2=beta2,
        beta3=beta3,
        beta4=beta4,
        raman_profile=setup.raman_profile_j[0],
        raman_gain_slope_j=np.asarray(setup.raman_gain_slope_j, dtype=np.float64),
        fwm_idx_pad=fwm_idx_pad,
        fwm_valid=fwm_valid,
        fwm_idx_pad_oband=fwm_idx_pad_oband,
        fwm_valid_oband=fwm_valid_oband,
    )


def _build_fwm_idx(f):
    freqs = f.squeeze()
    n = freqs.size
    all_idx = []
    counts = []

    for i in range(n):
        fi_val = freqs[i]
        j_idx, k_idx = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")
        j_idx = j_idx.ravel()
        k_idx = k_idx.ravel()
        target_m = freqs[j_idx] + freqs[k_idx] - fi_val
        matches = target_m[:, None] == freqs[None, :]
        has_match = jnp.any(matches, axis=1)
        m_idx = jnp.argmax(matches, axis=1)
        valid = (
            has_match
            & (j_idx != i)
            & (k_idx != m_idx)
            & (k_idx != i)
            & (j_idx != m_idx)
            & (j_idx <= k_idx)
        )
        idx_i = jnp.stack([
            jnp.full_like(j_idx[valid], i),
            j_idx[valid], k_idx[valid], m_idx[valid],
        ], axis=-1)
        all_idx.append(idx_i)
        counts.append(idx_i.shape[0])

    m = max(counts) if counts else 0
    idx_pad = []
    valid_pad = []
    for idx_i in all_idx:
        pad_len = m - idx_i.shape[0]
        idx_padded = jnp.pad(idx_i, ((0, pad_len), (0, 0)))
        valid_mask = jnp.concatenate([
            jnp.ones(idx_i.shape[0], dtype=bool),
            jnp.zeros(pad_len, dtype=bool),
        ])
        idx_pad.append(idx_padded)
        valid_pad.append(valid_mask)

    return jnp.stack(idx_pad, axis=0), jnp.stack(valid_pad, axis=0)


def calc_nsr_link_purejax(prep: PreparedCFMInputs, nspans: int, mask):
    """Pure-array JAX version of ``compute_throughput_cfm.calc_NSR_link``."""
    mask = jnp.asarray(mask, dtype=jnp.float64).reshape((-1, 1))

    ch_power_w_i = jnp.asarray(prep.ch_power_W_ij_active) * mask

    _, power_evo = cfm.raman_solver.solve_isrs_evolution(
        ch_centre_i=jnp.asarray(prep.ch_centre_active),
        A_eff=jnp.asarray(prep.aeff_active),
        raman_profile=prep.raman_profile,
        length=prep.length_scalar,
        attenuation_i=jnp.asarray(prep.attenuation_active),
        ch_power_W_i=ch_power_w_i[:, 0],
        ref_lambda=prep.ref_lambda,
        zspan=jnp.asarray(prep.zspan),
    )

    fit_params = cfm.get_power_profile_fit(
        length_j=jnp.asarray(prep.length_j),
        power_evo_j=power_evo[None, :, :],
        ch_centre_ij=jnp.asarray(prep.ch_centre_ij_active),
        ch_power_W_ij=ch_power_w_i,
        attenuation_ij=jnp.asarray(prep.attenuation_ij_active),
        raman_gain_slope_j=jnp.asarray(prep.raman_gain_slope_j),
        zspan=jnp.asarray(prep.zspan),
    )[:, :, None]

    a = fit_params[:, 0]
    a_bar = fit_params[:, 1]
    cr = fit_params[:, 2]

    p_ij = ch_power_w_i
    ptot = jnp.sum(p_ij, axis=0)
    gamma_ij = jnp.asarray(prep.gamma_active)[:, None]
    f_i = jnp.asarray(prep.ch_centre_active)[:, None]
    f_k = jnp.asarray(prep.ch_centre_active)[None, :]
    b_i = jnp.asarray(prep.ch_bandwidth_active)[:, None]
    b_k = jnp.asarray(prep.ch_bandwidth_active)[None, :]
    a_i = a
    a_k = a.T
    a_bar_i = a_bar
    a_bar_k = a_bar.T
    cr_i = cr
    cr_k = cr.T
    p_i = p_ij
    p_k = p_ij.T
    length = prep.length_scalar

    phi_i = -4 * pi ** 2 * (
        prep.beta2 + pi * prep.beta3 * (f_i + f_i)
        + 2 * pi ** 2 * prep.beta4 * f_i ** 2
    )
    phi_ik = -4 * pi ** 2 * (f_k - f_i) * (
        prep.beta2 + pi * prep.beta3 * (f_i + f_k)
        + (2.0 / 3.0) * pi ** 2 * prep.beta4 * (
            f_i ** 2 + f_i * f_k + f_k ** 2
        )
    )

    tf_i = -((ptot[0] * cr_i) / a_bar_i) * f_i
    tf_k = -((ptot[0] * cr_k) / a_bar_k) * f_k
    t_i = 1 + tf_i
    t_k = 1 + tf_k

    indices = cfm._INDICES
    indices_coi = cfm._INDICES_COI
    indices_fwm = cfm._INDICES_FWM

    @jax.jit
    def eta_gn_spm(phi_i_v, b_i_v, a_v, a_bar_v, gamma_v, tf_v, t_v, length_v):
        def _fun(x):
            l, l_line = x
            alpha = a_v + a_bar_v * l
            alpha_tilde = (alpha * (1 - jnp.exp(-alpha * length_v))) / (
                1 - jnp.exp(-alpha * length_v) - alpha * length_v * jnp.exp(-alpha * length_v)
            )
            alpha_line = a_v + a_bar_v * l_line
            alpha_tilde_line = (alpha_line * (1 - jnp.exp(-alpha_line * length_v))) / (
                1 - jnp.exp(-alpha_line * length_v) - alpha_line * length_v * jnp.exp(-alpha_line * length_v)
            )
            t_tilde = t_v * ((-tf_v / t_v) ** l)
            t_tilde_line = t_v * ((-tf_v / t_v) ** l_line)
            kappa = ((1 - jnp.exp(-alpha * length_v)) ** 2) / (
                1 - jnp.exp(-alpha * length_v) - alpha * length_v * jnp.exp(-alpha * length_v)
            )
            kappa_line = ((1 - jnp.exp(-alpha_line * length_v)) ** 2) / (
                1 - jnp.exp(-alpha_line * length_v) - alpha_line * length_v * jnp.exp(-alpha_line * length_v)
            )
            idx = (phi_i_v == 0)
            val = jnp.nan_to_num(
                (16 / 27) * gamma_v ** 2 / b_i_v ** 2
                * ((2 * kappa * kappa_line * pi * t_tilde * t_tilde_line)
                   / (phi_i_v * (alpha_tilde + alpha_tilde_line)))
                * (jnp.arcsinh(3 * phi_i_v * b_i_v ** 2 / (8 * pi * alpha_tilde))
                   + jnp.arcsinh(3 * phi_i_v * b_i_v ** 2 / (8 * pi * alpha_tilde_line))),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            val0 = (16 / 27) * gamma_v ** 2 * (
                kappa * kappa_line * t_tilde * t_tilde_line
                * (3 / (4 * alpha_tilde * alpha_tilde_line))
            )
            return (val + idx * val0).squeeze()

        return jax.vmap(_fun)(indices).sum(axis=0)

    @jax.jit
    def eta_gn_xpm(pi_v, pk_v, phi_ik_v, b_i_v, b_k_v, a_v, a_bar_v, gamma_v, tf_v, t_v, length_v):
        def _fun(x):
            l, l_line = x
            alpha = a_v + a_bar_v * l
            alpha_tilde = alpha * (1 - jnp.exp(-alpha * length_v)) / (
                1 - jnp.exp(-alpha * length_v) - alpha * length_v * jnp.exp(-alpha * length_v)
            )
            alpha_line = a_v + a_bar_v * l_line
            alpha_tilde_line = alpha_line * (1 - jnp.exp(-alpha_line * length_v)) / (
                1 - jnp.exp(-alpha_line * length_v) - alpha_line * length_v * jnp.exp(-alpha_line * length_v)
            )
            kappa = (1 - jnp.exp(-alpha * length_v)) ** 2 / (
                1 - jnp.exp(-alpha * length_v) - alpha * length_v * jnp.exp(-alpha * length_v)
            )
            kappa_line = (1 - jnp.exp(-alpha_line * length_v)) ** 2 / (
                1 - jnp.exp(-alpha_line * length_v) - alpha_line * length_v * jnp.exp(-alpha_line * length_v)
            )
            t_tilde = t_v * ((-tf_v / t_v) ** l)
            t_tilde_line = t_v * ((-tf_v / t_v) ** l_line)
            return 32 / 27 * jnp.sum(
                jnp.nan_to_num(
                    (pk_v / pi_v) ** 2 * gamma_v ** 2 / b_k_v
                    * (kappa * kappa_line * 2 * t_tilde * t_tilde_line
                       / (phi_ik_v * (alpha_tilde + alpha_tilde_line)))
                    * (jnp.arctan(phi_ik_v * b_i_v / (2 * alpha_tilde))
                       + jnp.arctan(phi_ik_v * b_i_v / (2 * alpha_tilde_line))),
                    nan=0.0, posinf=0.0, neginf=0.0,
                ), axis=1,
            ).squeeze()

        return jax.vmap(_fun)(indices).sum(axis=0)

    @jax.jit
    def eta_gn_fwm(ptot_val, p_v, beta2, beta3, beta4, a_fwm, a_bar_fwm,
                   f_v, b_v, cr_fwm, gamma_fwm, length_v, idx_pad, valid):
        def _ch(i):
            idx_ch = idx_pad[i]
            valid_ch = valid[i]

            def _eta_per_ch(idx):
                i_v, j_v, k_v, m_v = idx
                omega = jnp.where(j_v == k_v, 1.0, 2.0)

                def _phi(i_p, j_p, k_p):
                    phi_jk = -4 * pi ** 2 * (f_v[j_p] - f_v[i_p]) * (f_v[k_p] - f_v[i_p]) * (
                        beta2 + pi * beta3 * (f_v[j_p] + f_v[k_p])
                        + (2 / 3) * pi ** 2 * beta4 * (
                            f_v[j_p] ** 2 + f_v[j_p] * f_v[k_p] + f_v[k_p] ** 2
                            + 0.5 * (f_v[j_p] - f_v[i_p]) * (f_v[k_p] - f_v[i_p])
                        )
                    )
                    dphi_f1 = -4 * pi ** 2 * (f_v[k_p] - f_v[i_p]) * (
                        beta2 + pi * beta3 * (f_v[j_p] + f_v[k_p] + f_v[j_p] - f_v[i_p])
                    )
                    dphi_f2 = -4 * pi ** 2 * (f_v[j_p] - f_v[i_p]) * (
                        beta2 + pi * beta3 * (f_v[j_p] + f_v[k_p] + f_v[k_p] - f_v[i_p])
                    )
                    return phi_jk, dphi_f1, dphi_f2

                phi_jk, dphi_f1, dphi_f2 = _phi(i_v, j_v, k_v)
                bi = b_v[i_v]
                bj = b_v[j_v]
                bk = b_v[k_v]
                bm = b_v[m_v]
                pi_ = p_v[i_v]
                pj = p_v[j_v]
                pk = p_v[k_v]
                pm = p_v[m_v]
                gamma_iv = gamma_fwm[i_v]

                tf = -((ptot_val * cr_fwm) / (2 * a_bar_fwm)) * f_v
                t = 1 + tf

                def _island_coi(x):
                    l_j, l_k, l_j_line, l_k_line = x
                    alpha_j = a_fwm[j_v] / 2 + l_j * a_bar_fwm[j_v]
                    alpha_k = a_fwm[k_v] / 2 + l_k * a_bar_fwm[k_v]
                    alpha_all = alpha_j + alpha_k
                    alpha_j_line = a_fwm[j_v] / 2 + l_j_line * a_bar_fwm[j_v]
                    alpha_k_line = a_fwm[k_v] / 2 + l_k_line * a_bar_fwm[k_v]
                    alpha_all_line = alpha_j_line + alpha_k_line
                    t_tilde_all = (
                        t[j_v] * (-tf[j_v] / t[j_v]) ** l_j
                    ) * (
                        t[k_v] * (-tf[k_v] / t[k_v]) ** l_k
                    )
                    t_tilde_all_line = (
                        t[j_v] * (-tf[j_v] / t[j_v]) ** l_j_line
                    ) * (
                        t[k_v] * (-tf[k_v] / t[k_v]) ** l_k_line
                    )
                    alpha_tilde_all = (
                        alpha_all * (1 - jnp.exp(-alpha_all * length_v))
                        / (1 - jnp.exp(-alpha_all * length_v) - alpha_all * length_v * jnp.exp(-alpha_all * length_v))
                    )
                    alpha_tilde_all_line = (
                        alpha_all_line * (1 - jnp.exp(-alpha_all_line * length_v))
                        / (1 - jnp.exp(-alpha_all_line * length_v) - alpha_all_line * length_v * jnp.exp(-alpha_all_line * length_v))
                    )
                    kappa_all = (
                        (1 - jnp.exp(-alpha_all * length_v)) ** 2
                        / (1 - jnp.exp(-alpha_all * length_v) - alpha_all * length_v * jnp.exp(-alpha_all * length_v))
                    )
                    kappa_all_line = (
                        (1 - jnp.exp(-alpha_all_line * length_v)) ** 2
                        / (1 - jnp.exp(-alpha_all_line * length_v) - alpha_all_line * length_v * jnp.exp(-alpha_all_line * length_v))
                    )
                    return (
                        t_tilde_all * t_tilde_all_line * kappa_all * kappa_all_line
                        / (alpha_tilde_all + alpha_tilde_all_line + 1e-30)
                    )

                island_sum = jax.lax.cond(
                    m_v == i_v,
                    lambda _: jax.vmap(_island_coi)(indices_coi).sum(axis=0),
                    lambda _: 0.0,
                    operand=None,
                )
                return jnp.nan_to_num(
                    omega * (16 / 27) * gamma_iv ** 2 * (bi / (pi_ ** 3 + 1e-30))
                    * (pj * pk * pm / ((bj * bk * bm) + 1e-30)) * island_sum,
                    nan=0.0, posinf=0.0, neginf=0.0,
                )

            vals = jax.vmap(_eta_per_ch)(idx_ch)
            return jnp.where(valid_ch, vals, 0.0).sum(axis=0)

        return jax.lax.map(_ch, jnp.arange(f_v.size))

    eta_spm = eta_gn_spm(phi_i, b_i, a_i, a_bar_i, gamma_ij, tf_i, t_i, length)
    eta_xpm = eta_gn_xpm(p_i, p_k, phi_ik, b_i, b_k, a_k, a_bar_k, gamma_ij, tf_k, t_k, length)

    if np.any(prep.ch_idx_oband_active):
        eta_fwm = jnp.zeros_like(eta_xpm)
        eta_fwm = eta_fwm.at[prep.ch_idx_oband_active].set(
            eta_gn_fwm(
                ptot[0],
                p_i[prep.ch_idx_oband_active, 0],
                prep.beta2,
                prep.beta3,
                prep.beta4,
                a_i[prep.ch_idx_oband_active, 0],
                a_i[prep.ch_idx_oband_active, 0],
                f_i[prep.ch_idx_oband_active, 0],
                b_i[prep.ch_idx_oband_active, 0],
                cr_i[prep.ch_idx_oband_active, 0],
                gamma_ij[prep.ch_idx_oband_active, 0],
                length,
                jnp.asarray(prep.fwm_idx_pad_oband),
                jnp.asarray(prep.fwm_valid_oband),
            )
        )
    else:
        eta_fwm = jnp.zeros_like(eta_xpm)

    tiny = jnp.array(1e-30, dtype=power_evo.dtype)
    ratio_in_out = power_evo[0, :] / jnp.maximum(power_evo[-1, :], tiny)
    gain_lin = jnp.maximum(ratio_in_out, jnp.array(1.0 + 1e-9, dtype=power_evo.dtype))
    p_ase = cfm.egn.calc_Pase(
        gain_lin,
        jnp.asarray(prep.nf_active),
        ref_lambda=prep.ref_lambda,
        ch_centre_ij=jnp.asarray(prep.ch_centre_ij_active),
        ch_bandwidth_ij=jnp.asarray(prep.ch_bandwidth_ij_active),
    ) * nspans

    eta_sum = (
        jnp.nan_to_num(eta_spm[:, None], nan=0.0, posinf=0.0, neginf=0.0)
        + jnp.nan_to_num(eta_xpm[:, None], nan=0.0, posinf=0.0, neginf=0.0)
        + jnp.nan_to_num(eta_fwm[:, None], nan=0.0, posinf=0.0, neginf=0.0)
    )
    trx_pen = 1 / cfm.idB(jnp.asarray(prep.snr_trx_active)[:, None])
    occ = mask > 0
    safe_p = jnp.maximum(p_i, jnp.array(1e-30, dtype=p_i.dtype))
    term_nli = nspans * (p_i ** 2) * eta_sum
    term_ase = p_ase[:, None] / safe_p
    nsr_active = term_nli + term_ase + trx_pen
    return jnp.where(occ, nsr_active, jnp.inf)


def _compute_topology_arrays(graph, rwa, band_selection, span_length_km=80, launch_power_dBm=-2.0):
    band_cfg = cfm.BAND_CONFIGS[band_selection]
    active_bands = band_cfg["bands"]

    ch_lambda, channel_idx, ch_idx_oband_full, band_masks, ref_lambda = (
        cfm._build_multiband_channel_grid(active_bands)
    )
    num_active = int(np.sum(channel_idx))
    ch_idx_oband_active = np.asarray(ch_idx_oband_full[channel_idx], dtype=bool)

    p_channel, nf, snr_trx = cfm._build_per_channel_params(
        ch_lambda,
        band_masks,
        active_bands,
        launch_power_dBm=launch_power_dBm,
    )
    setup = cfm._build_setup(
        span_length_km,
        ch_lambda,
        channel_idx,
        p_channel,
        nf,
        snr_trx,
        ref_lambda=ref_lambda,
    )

    (
        edges,
        edge_to_row,
        span_count_per_edge,
        occupancy_matrix,
        rwa_edge_paths,
        wavelength_to_col,
    ) = cfm._build_rwa_occupancy(graph, rwa, num_active)
    active_slot_ix = np.flatnonzero(np.asarray(channel_idx))

    return (
        setup,
        _prepare_cfm_inputs(setup, ch_idx_oband_active),
        edges,
        edge_to_row,
        span_count_per_edge,
        occupancy_matrix,
        rwa_edge_paths,
        wavelength_to_col,
        active_slot_ix,
    )


def compute_topology_throughput(
    db,
    collection,
    _id,
    band_selection,
    route_function,
    span_length_km=80,
    launch_power_dBm=-2.0,
):
    band_cfg = cfm.BAND_CONFIGS[band_selection]

    results = list(
        nt.Database.read_data(db, collection, find_dic={"_id": _id}, max_count=1)
    )
    if not results:
        print(f"[WARN] No document found for _id={_id}")
        return 0.0

    doc = results[0]
    rwa_key = cfm._build_band_aware_rwa_key(route_function, band_selection)
    if rwa_key not in doc:
        raise KeyError(
            f"Band-aware key '{rwa_key}' not found for _id={_id}. "
            "Stop run to avoid using mismatched legacy RWA keys."
        )

    graph_list = nt.Database.read_topology_dataset_list(
        db, collection, find_dic={"_id": _id}, node_data=True
    )
    if not graph_list:
        print(f"[WARN] Could not read topology for _id={_id}")
        return 0.0

    graph = nx.relabel.convert_node_labels_to_integers(graph_list[0][0], first_label=1)
    rwa = nt.Tools.read_database_dict(doc[rwa_key])

    (
        setup,
        prep,
        edges,
        edge_to_row,
        span_count_per_edge,
        occupancy_matrix,
        rwa_edge_paths,
        wavelength_to_col,
        active_slot_ix,
    ) = _compute_topology_arrays(
        graph,
        rwa,
        band_selection,
        span_length_km=span_length_km,
        launch_power_dBm=launch_power_dBm,
    )

    num_active = occupancy_matrix.shape[1]
    print(
        f"  Channel grid: {cfm.NUM_CHANNELS_TOTAL} slots, {num_active} active channels, "
        f"ref_lambda={prep.ref_lambda * 1e9:.1f} nm"
    )

    nsr_link_channel = jnp.zeros_like(jnp.asarray(occupancy_matrix), dtype=jnp.float64)
    n_links = int(span_count_per_edge.shape[0])
    print(f"    Undirected links in topology: {n_links} (Link indices 0..{n_links - 1})")

    for i in range(n_links):
        t0 = time.perf_counter()
        link_nsr = calc_nsr_link_purejax(
            prep,
            float(span_count_per_edge[i]),
            occupancy_matrix[i, :, None],
        ).squeeze(-1)
        nsr_link_channel = nsr_link_channel.at[i].set(link_nsr)
        dt = time.perf_counter() - t0
        occ = occupancy_matrix[i] > 0
        ln = np.asarray(link_nsr).ravel()
        use = occ & np.isfinite(ln) & (ln > 0)
        if not np.any(occ):
            snr_msg = "n/a (idle link)"
        elif np.any(use):
            mean_snr = float(np.mean(10.0 * np.log10(1.0 / ln[use])))
            snr_msg = f"{mean_snr:.1f} dB"
        else:
            snr_msg = "nan (no finite NSR on occupied λ)"
        print(
            f"    Link {i} ({edges[i]}, spans={span_count_per_edge[i]}): "
            f"{dt:.1f}s, mean SNR={snr_msg}"
        )

    capacity_total = 0.0
    n_lightpaths = 0

    for w_str, path_infos in rwa_edge_paths.items():
        w = int(w_str)
        if w not in wavelength_to_col:
            continue
        col = wavelength_to_col[w]
        slot_row = int(active_slot_ix[w])
        ch_bw_hz = float(np.asarray(setup.ch_bandwidth_ij)[slot_row, 0])
        for info in path_infos:
            path_nsr = 0.0
            for edge in info["edge_path"]:
                row = edge_to_row[edge]
                path_nsr += float(nsr_link_channel[row, col])

            if not np.isfinite(path_nsr) or path_nsr <= 0:
                continue
            snr = 1.0 / path_nsr
            rate_bps = 2 * ch_bw_hz * np.log2(1.0 + snr)
            capacity_total += rate_bps
            n_lightpaths += 1

    print(
        f"  Topology {_id}: {n_lightpaths} lightpaths, "
        f"capacity = {capacity_total/1e12:.4f} Tbps"
    )

    nt.Database.update_data_with_id(
        db,
        collection,
        _id,
        newvals={
            "$set": {
                f"{route_function} Capacity-CFM-JAX": float(capacity_total),
                f"{route_function} CFM-JAX-lightpaths": int(n_lightpaths),
                f"{route_function} CFM-JAX-timestamp": datetime.datetime.utcnow(),
                f"{route_function} CFM-JAX-band": band_cfg["name"],
                f"{route_function} CFM-JAX-span_length_km": float(span_length_km),
                f"{route_function} CFM-JAX-launch_power_dBm": float(launch_power_dBm),
            }
        },
    )

    return capacity_total


def run_parallel(
    db,
    collection,
    topology_name,
    route_function,
    band_selection,
    span_length_km=80,
    launch_power_dBm=-2.0,
    hostname="128.40.42.13",
    port=6379,
):
    import ray

    if hostname is not None:
        ray.init(address=f"{hostname}:{port}")
    else:
        ray.init()

    graph_list = nt.Database.read_topology_dataset_list(
        db, collection, find_dic={"name": topology_name}, node_data=True
    )
    print(f"Found {len(graph_list)} topologies for '{topology_name}'")

    @ray.remote(num_cpus=1)
    def _remote_worker(_id):
        return compute_topology_throughput(
            db=db,
            collection=collection,
            _id=_id,
            band_selection=band_selection,
            route_function=route_function,
            span_length_km=span_length_km,
            launch_power_dBm=launch_power_dBm,
        )

    futures = [_remote_worker.remote(_id) for _, _id in graph_list]
    results = ray.get(futures)
    ray.shutdown()

    print(f"\nAll done. {len(results)} topologies processed.")
    for i, (_, _id) in enumerate(graph_list):
        print(f"  {_id}: {results[i]/1e12:.4f} Tbps")

    return results


def run_sequential(
    db,
    collection,
    topology_name,
    route_function,
    band_selection,
    span_length_km=80,
    launch_power_dBm=-2.0,
):
    graph_list = nt.Database.read_topology_dataset_list(
        db, collection, find_dic={"name": topology_name}, node_data=True
    )
    print(f"Found {len(graph_list)} topologies for '{topology_name}'")

    results = []
    for _, _id in graph_list:
        cap = compute_topology_throughput(
            db=db,
            collection=collection,
            _id=_id,
            band_selection=band_selection,
            route_function=route_function,
            span_length_km=span_length_km,
            launch_power_dBm=launch_power_dBm,
        )
        results.append(cap)

    print(f"\nAll done. {len(results)} topologies processed.")
    for i, (_, _id) in enumerate(graph_list):
        print(f"  {_id}: {results[i]/1e12:.4f} Tbps")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute throughput using pure-array JAX CFM kernel"
    )
    parser.add_argument("--route_function", type=str, default=None)
    parser.add_argument("--topology", type=str, default=None)
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument("--band", type=str, default=None,
                        choices=list(cfm.BAND_CONFIGS.keys()))
    parser.add_argument("--span_length_km", type=float, default=None)
    parser.add_argument("--launch_power_dBm", type=float, default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--hostname", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    BAND_SELECTION = "OESCL"
    ROUTE_FUNCTION = "kSP-FF"
    TOPOLOGY_NAME = "NSFNET"
    DB = "Topology_Data"
    COLLECTION = "real"
    SPAN_LENGTH_KM = 80
    LAUNCH_POWER_DBM = -3.0
    USE_PARALLEL = False
    HOSTNAME = "128.40.40.67"
    PORT = 6379

    args = main()
    if args.route_function is not None:
        ROUTE_FUNCTION = args.route_function
    if args.topology is not None:
        TOPOLOGY_NAME = args.topology
    if args.collection is not None:
        COLLECTION = args.collection
    if args.db is not None:
        DB = args.db
    if args.band is not None:
        BAND_SELECTION = args.band
    if args.span_length_km is not None:
        SPAN_LENGTH_KM = args.span_length_km
    if args.launch_power_dBm is not None:
        LAUNCH_POWER_DBM = args.launch_power_dBm
    if args.parallel:
        USE_PARALLEL = True
    if args.hostname is not None:
        HOSTNAME = args.hostname
    if args.port is not None:
        PORT = args.port

    print("=" * 60)
    print("Pure-Array JAX CFM Throughput")
    print("=" * 60)
    print(f"DB/Collection    : {DB}/{COLLECTION}")
    print(f"Topology         : {TOPOLOGY_NAME}")
    print(f"Route Function   : {ROUTE_FUNCTION}")
    print(f"Band             : {BAND_SELECTION}")
    print(f"Span length (km) : {SPAN_LENGTH_KM}")
    print(f"Launch Power     : {LAUNCH_POWER_DBM} dBm")
    print(f"Mode             : {'parallel' if USE_PARALLEL else 'sequential'}")
    print("=" * 60)

    if USE_PARALLEL:
        run_parallel(
            db=DB,
            collection=COLLECTION,
            topology_name=TOPOLOGY_NAME,
            route_function=ROUTE_FUNCTION,
            band_selection=BAND_SELECTION,
            span_length_km=SPAN_LENGTH_KM,
            launch_power_dBm=LAUNCH_POWER_DBM,
            hostname=HOSTNAME,
            port=PORT,
        )
    else:
        run_sequential(
            db=DB,
            collection=COLLECTION,
            topology_name=TOPOLOGY_NAME,
            route_function=ROUTE_FUNCTION,
            band_selection=BAND_SELECTION,
            span_length_km=SPAN_LENGTH_KM,
            launch_power_dBm=LAUNCH_POWER_DBM,
        )
