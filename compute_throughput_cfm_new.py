
from __future__ import annotations

import os
# os.environ['JAX_ENABLE_X64'] = 'True'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.3'

import pathlib
import sys

# ---------------------------------------------------------------------------
# Path setup  (must come before ong / gpu_init imports)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_EXTERNAL_DIR = _SCRIPT_DIR.parents[2] / "external"
_ONG_SRC = _EXTERNAL_DIR / "ong-python-toolbox" / "src"

for p in (_EXTERNAL_DIR, str(_ONG_SRC)):
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)

# GPU initialisation (as in cfm_network1.ipynb)
from ong.utils import gpu_init

import jax
print(f"Success! GPU count: {jax.device_count()}")
print(f"Devices: {jax.devices()}")

import argparse
import datetime
import time
from dataclasses import field

import networkx as nx
import numpy as np
from jax import numpy as jnp
from jax.numpy import exp, log, arcsinh, arctan, nan_to_num
from scipy.constants import c, pi

import NetworkToolkit as nt
from ong import models
from ong.measurements import load_corning_smf_28_raman
from ong.models import raman_solver, FibreSpanSetupAdvanced, egn
from ong.models.raman_fitting import get_power_profile_fit_new
from ong.utils import struct
from sibase import Value
import data as fibre_data

dBm = lambda x: 10 * np.log10(x / 1e-3)
idBm = lambda x: 10 ** (x / 10) * 1e-3
dB = lambda x: 10 * np.log10(x)
idB = lambda x: 10 ** (x / 10)

# ===========================================================================
# BANDS / BAND_CONFIGS  (unchanged from compute_throughput_cfm.py)
# ===========================================================================
BANDS = {
    'O': {'TransceiverSNR': 20,   'noise_figure': 5,   'p_launch': 23},
    'E': {'TransceiverSNR': 18.5, 'noise_figure': 6.5, 'p_launch': 15.57},
    'S': {'TransceiverSNR': 20,   'noise_figure': 7,   'p_launch': 14.03},
    'C': {'TransceiverSNR': 23.5, 'noise_figure': 5,   'p_launch': 20.69},
    'L': {'TransceiverSNR': 22,   'noise_figure': 6,   'p_launch': 21.9},
}

BAND_CONFIGS = {
    "O":    {"name": "O band",    "bands": ["O"],                     "description": "O band only"},
    "E":    {"name": "E band",    "bands": ["E"],                     "description": "E band only"},
    "S":    {"name": "S band",    "bands": ["S"],                     "description": "S band only"},
    "C":    {"name": "C band",    "bands": ["C"],                     "description": "C band only"},
    "L":    {"name": "L band",    "bands": ["L"],                     "description": "L band only"},
    "CL":   {"name": "CL band",   "bands": ["C", "L"],                "description": "C+L bands"},
    "SCL":  {"name": "SCL band",  "bands": ["S", "C", "L"],           "description": "S+C+L bands"},
    "ESCL": {"name": "ESCL band", "bands": ["E", "S", "C", "L"],      "description": "E+S+C+L bands"},
    "OESCL": {"name": "OESCL band", "bands": ["O", "E", "S", "C", "L"], "description": "All bands (O+E+S+C+L)"},
}

# ===========================================================================
# Multi-band channel grid with gaps  (unchanged)
# ===========================================================================
CH_SPACING_HZ = 100e9
CH_BW_HZ = 96e9
NUM_CHANNELS_TOTAL = 555
REF_LAMBDA_M = 1419.4e-9

_BAND_BOUNDS = {
    'O': (1260e-9, 1358e-9),
    'E': (1405e-9, 1464e-9),
    'S': (1470e-9, 1526e-9),
    'C': (1530e-9, 1566e-9),
    'L': (1573e-9, 1625e-9),
}


def _build_multiband_channel_grid(active_bands=("O", "E", "S", "C", "L")):
    """Build the full channel grid with band gaps."""
    chs = jnp.arange(NUM_CHANNELS_TOTAL) - (NUM_CHANNELS_TOTAL - 1) / 2
    ch_lambda = c / (chs * CH_SPACING_HZ + c / REF_LAMBDA_M)

    band_masks = {b: (ch_lambda >= lo) & (ch_lambda <= hi)
                  for b, (lo, hi) in _BAND_BOUNDS.items()}

    channel_idx = jnp.zeros(NUM_CHANNELS_TOTAL, dtype=bool)
    for b in active_bands:
        channel_idx = channel_idx | band_masks[b]

    ch_idx_oband = band_masks.get('O', jnp.zeros(NUM_CHANNELS_TOTAL, dtype=bool))
    return (np.asarray(ch_lambda), np.asarray(channel_idx),
            np.asarray(ch_idx_oband), band_masks, REF_LAMBDA_M)


# ===========================================================================
# Setup classes  (unchanged)
# ===========================================================================
@struct.setupclass
class SystemRequiredSetup:
    name: str
    symbol_rate: Value
    seq_length: int
    delta_g: float


@struct.setupclass
class SystemSetup(models.GNSetup, SystemRequiredSetup):
    description: str = ''
    rng_seed: int = field(default_factory=lambda: time.time_ns())
    roll_off: float = 0.01
    samples_per_symbol = 2
    modulation_name: str = 'gaussian'
    num_polarisation = 2
    PMD_parameter = 0
    DGD_parameter = 0
    polarisation_rotation = False

    @property
    def prng_key(self):
        return jax.random.PRNGKey(self.rng_seed)

    @property
    def sample_rate(self):
        return self.samples_per_symbol * self.symbol_rate


@struct.setupclass
class DynPowerSystemSetup(SystemSetup):
    ch_power_dBm: struct.ArrayLike
    noise_figure: struct.ArrayLike
    snr_trx: struct.ArrayLike
    channel_idx: struct.ArrayLike

    @property
    def beta4_j(self) -> jnp.ndarray:
        return jnp.array([span.beta4 for span in self.spans])


# ===========================================================================
# Per-channel params + setup builders  (unchanged)
# ===========================================================================
def _build_per_channel_params(ch_lambda, band_masks, active_bands,
                              launch_power_dBm=-2.0):
    """Set per-channel launch power, NF, and transceiver SNR by band."""
    n = len(ch_lambda)
    P_channel = -jnp.inf * jnp.ones(n)
    nf = jnp.zeros(n)
    snr_trx = jnp.zeros(n)

    band_power = (launch_power_dBm if isinstance(launch_power_dBm, dict)
                  else {b: launch_power_dBm for b in active_bands})

    for b in active_bands:
        mask = band_masks[b]
        P_channel = P_channel.at[mask].set(band_power.get(b, -2.0))
        nf = nf.at[mask].set(BANDS[b]['noise_figure'])
        snr_trx = snr_trx.at[mask].set(BANDS[b]['TransceiverSNR'])

    return P_channel, nf, snr_trx


def _build_setup(span_length_km, ch_lambda, channel_idx, P_channel, nf,
                 snr_trx, ref_lambda, fibre_type='min', n2_m2_per_w=30.6e-21):
    """Build DynPowerSystemSetup matching cfm_network1.ipynb."""
    lambda_active = ch_lambda[channel_idx]
    lambda_upper = float(max(lambda_active))
    lambda_lower = float(min(lambda_active))
    dispersion_fit_wavelength = jnp.array([lambda_lower, ref_lambda, lambda_upper])

    ch_lambda_jnp = jnp.asarray(ch_lambda, dtype=jnp.float64)
    aeff_m2 = jnp.array(fibre_data.A_eff_at(fibre_type, ch_lambda_jnp)) * 1e-12
    gamma_per_m = (2 * jnp.pi) / ch_lambda_jnp * (n2_m2_per_w / aeff_m2)

    fibre_span = FibreSpanSetupAdvanced(
        length=f'{span_length_km}km',
        dispersion_order=2,
        dispersion_ref_lambda=ref_lambda,
        dispersion_fit_points=dispersion_fit_wavelength,
        attenuation_profile=fibre_data.fit_attenuation(fibre_type),
        raman_profile=load_corning_smf_28_raman(),
        dispersion_profile=fibre_data.get_dispersion(fibre_type),
        effective_area_profile=(ch_lambda_jnp, aeff_m2),
        nonlinear_coeff_profile=(ch_lambda_jnp, gamma_per_m),
    )

    setup = DynPowerSystemSetup(
        spans=[fibre_span],
        delta_g=1e-7,
        seq_length=2 ** 16,
        symbol_rate=f'{CH_BW_HZ * 1e-9}GBaud',
        num_channels=NUM_CHANNELS_TOTAL,
        ch_bandwidth=f'{CH_BW_HZ * 1e-9}GHz',
        ch_spacing=f'{CH_SPACING_HZ * 1e-9}GHz',
        ref_lambda=ref_lambda,
        ch_power_dBm=P_channel,
        name="",
        description="CFM throughput computation",
        roll_off=0.0001,
        noise_figure=nf,
        modulation_name='16QAM',
        snr_trx=snr_trx,
        channel_idx=channel_idx,
    )
    return setup


# ===========================================================================
# calc_NSR_link — per-link NSR, math/parameters from cfm_network1.ipynb.
# (ASE gain uses the linear ratio power_evo[0] / power_evo[-1].)
# ===========================================================================
_INDICES = (jnp.arange(4)[:, None] >> jnp.arange(2)) & 1
_INDICES_COI = (jnp.arange(16)[:, None] >> jnp.arange(4)) & 1
_INDICES_FWM = (jnp.arange(64)[:, None] >> jnp.arange(6)) & 1


def calc_NSR_link(setup, Nspans, mask, ch_idx_oband=None):
    """Per-channel NSR for one link (notebook GN-NLI + ISRS ASE + TRX penalty).

    Returns ``(nsr, fit_params, eta_spm, eta_xpm, eta_fwm, nsr_ase)``; idle
    wavelengths (mask == 0) get ``nsr = inf`` and ``nsr_ase = nan``.
    """
    j = 0
    chs = setup.channel_idx
    gamma_i = jnp.array(setup.spans[j].nonlinear_coeff_at(setup.ch_lambda_ij[chs, j]))
    beta2_j = jnp.array([setup.spans[j].beta2])
    beta3_j = jnp.array([setup.spans[j].beta3])
    beta4_j = jnp.array([setup.spans[j].beta4])
    Aeff_i = jnp.array(setup.spans[j].A_eff_at(setup.ch_lambda_ij[chs, j]))

    l = setup.length_j[j]
    z = jnp.arange(0, l, 1000)
    ch_power_W_i = setup.ch_power_W_ij[chs, :] * mask

    _, power_evo = raman_solver.solve_isrs_evolution(
        ch_centre_i=setup.ch_centre_ij[chs, j],
        A_eff=Aeff_i,
        raman_profile=setup.raman_profile_j[j],
        length=setup.length_j[j],
        attenuation_i=setup.attenuation_ij[chs, j],
        ch_power_W_i=ch_power_W_i[:, j],
        ref_lambda=setup.ref_lambda,
        zspan=z,
    )

    # --- ISRS power-profile fit (ONG get_power_profile_fit, as in notebook) ---
    fit_params = get_power_profile_fit_new(
        length_j=setup.length_j,
        power_evo_j=power_evo[None, :, :],
        ch_centre_ij=setup.ch_centre_ij[chs],
        ch_power_W_ij=ch_power_W_i,
        attenuation_ij=setup.attenuation_ij[chs],
        raman_gain_slope_j=setup.raman_gain_slope_j,
        zspan=z,
    )[:, :, None]

    a = fit_params[:, 0]
    a_bar = fit_params[:, 1]
    Cr = fit_params[:, 2]

    P_ij = ch_power_W_i
    Ptot = jnp.sum(P_ij, axis=0)
    gamma_ij = gamma_i[:, None]
    if gamma_ij.ndim == 1:
        gamma_ij = jnp.repeat(gamma_ij[None], NUM_CHANNELS_TOTAL, axis=1)
    fi = setup.ch_centre_ij
    Bch = setup.ch_bandwidth_ij

    a_i = a[:, j:j+1]
    a_k = jnp.transpose(a[:, j:j+1])
    a_bar_i = a_bar[:, j:j+1]
    a_bar_k = jnp.transpose(a_bar[:, j:j+1])
    f_i = fi[chs, j:j+1]
    f_k = jnp.transpose(fi[chs, j:j+1])
    B_i = Bch[chs, j:j+1]
    B_k = jnp.transpose(Bch[chs, j:j+1])
    Cr_i = Cr[:, j:j+1]
    Cr_k = jnp.transpose(Cr[:, j:j+1])
    P_i = P_ij[:, j:j+1]
    P_k = jnp.transpose(P_ij[:, j:j+1])

    phi_i = -4 * pi ** 2 * (
        beta2_j[j] + pi * beta3_j[j] * (f_i + f_i)
        + 2 * pi ** 2 * beta4_j[j] * f_i ** 2
    )
    phi_ik = -4 * pi ** 2 * (f_k - f_i) * (
        beta2_j[j] + pi * beta3_j[j] * (f_i + f_k)
        + 2 / 3 * pi ** 2 * beta4_j[j] * (f_i ** 2 + f_i * f_k + f_k ** 2)
    )

    Tf_i = -((Ptot[j] * Cr_i) / a_bar_i) * f_i
    Tf_k = -((Ptot[j] * Cr_k) / a_bar_k) * f_k
    T_i = 1 + Tf_i
    T_k = 1 + Tf_k

    # --- SPM ---
    @jax.jit
    def _eta_GN_SPM(phi_i, B_i, a, a_bar, gamma, Tf, T, L):
        def _fun(x):
            l, l_line = x
            alpha = a + a_bar * l
            alpha_tilde = (alpha * (1 - exp(-alpha * L))) / (
                1 - exp(-alpha * L) - alpha * L * exp(-alpha * L))
            alpha_line = a + a_bar * l_line
            alpha_tilde_line = (alpha_line * (1 - exp(-alpha_line * L))) / (
                1 - exp(-alpha_line * L) - alpha_line * L * exp(-alpha_line * L))
            T_tilde = T * ((-Tf / T) ** l)
            T_tilde_line = T * ((-Tf / T) ** l_line)
            kappa = ((1 - exp(-alpha * L)) ** 2) / (
                1 - exp(-alpha * L) - alpha * L * exp(-alpha * L))
            kappa_line = ((1 - exp(-alpha_line * L)) ** 2) / (
                1 - exp(-alpha_line * L) - alpha_line * L * exp(-alpha_line * L))
            idx = (phi_i == 0)
            return (nan_to_num(
                (16 / 27) * gamma ** 2 / B_i ** 2
                * ((2 * kappa * kappa_line * pi * T_tilde * T_tilde_line)
                   / (phi_i * (alpha_tilde + alpha_tilde_line)))
                * (arcsinh(3 * phi_i * B_i ** 2 / (8 * pi * alpha_tilde))
                   + arcsinh(3 * phi_i * B_i ** 2 / (8 * pi * alpha_tilde_line))),
                nan=0, posinf=0, neginf=0,
            ) + idx * ((16 / 27) * gamma ** 2 * (
                kappa * kappa_line * T_tilde * T_tilde_line
                * (3 / (4 * alpha_tilde * alpha_tilde_line))))).squeeze()
        return jax.vmap(_fun)(_INDICES).sum(axis=0)

    # --- XPM ---
    @jax.jit
    def _eta_GN_XPM(Pi, Pk, phi_ik, B_i, B_k, a, a_bar, gamma, Tf, T, L):
        def _fun(x):
            l, l_line = x
            alpha = a + a_bar * l
            alpha_tilde = alpha * (1 - exp(-alpha * L)) / (
                1 - exp(-alpha * L) - alpha * L * exp(-alpha * L))
            alpha_line = a + a_bar * l_line
            alpha_tilde_line = alpha_line * (1 - exp(-alpha_line * L)) / (
                1 - exp(-alpha_line * L) - alpha_line * L * exp(-alpha_line * L))
            kappa = (1 - exp(-alpha * L)) ** 2 / (
                1 - exp(-alpha * L) - alpha * L * exp(-alpha * L))
            kappa_line = (1 - exp(-alpha_line * L)) ** 2 / (
                1 - exp(-alpha_line * L) - alpha_line * L * exp(-alpha_line * L))
            T_tilde = T * ((-Tf / T) ** l)
            T_tilde_line = T * ((-Tf / T) ** l_line)
            return 32 / 27 * jnp.sum(
                nan_to_num(
                    (Pk / Pi) ** 2 * gamma ** 2 / B_k
                    * (kappa * kappa_line * 2 * T_tilde * T_tilde_line
                       / (phi_ik * (alpha_tilde + alpha_tilde_line)))
                    * (arctan(phi_ik * B_i / (2 * alpha_tilde))
                       + arctan(phi_ik * B_i / (2 * alpha_tilde_line))),
                    nan=0, posinf=0, neginf=0,
                ), axis=1,
            ).squeeze()
        return jax.vmap(_fun)(_INDICES).sum(axis=0)

    # --- FWM index builder ---
    def _FWM_idx(f):
        freqs = np.asarray(f).squeeze()
        n = len(freqs)
        ch_spacing = np.median(np.abs(np.diff(freqs)))
        grid = np.rint((freqs - freqs[0]) / ch_spacing).astype(np.int64)
        pos = {g: idx for idx, g in enumerate(grid)}

        all_idx = []
        counts = []
        for i in range(n):
            idx_i = []
            for jj in range(n):
                for k in range(n):
                    target = grid[jj] + grid[k] - grid[i]
                    m = pos.get(target)
                    if m is None:
                        continue
                    if jj != i and k != m and k != i and jj != m and jj <= k:
                        idx_i.append([i, jj, k, m])
            idx_i = np.asarray(idx_i, dtype=np.int32)
            all_idx.append(idx_i)
            counts.append(idx_i.shape[0])

        M = max(counts) if counts else 0
        idx_pad = np.zeros((n, M, 4), dtype=np.int32)
        valid = np.zeros((n, M), dtype=bool)
        for i, idx_i in enumerate(all_idx):
            idx_pad[i, :idx_i.shape[0]] = idx_i
            valid[i, :idx_i.shape[0]] = True
        return jnp.asarray(idx_pad), jnp.asarray(valid)

    # --- FWM ---
    @jax.jit
    def _eta_GN_FWM(Ptot_val, P, beta2, beta3, beta4, a_fwm, a_bar_fwm,
                    f, B, Cr_fwm, gamma_fwm, L_val, idx_pad, valid):
        def _ch(i):
            idx_ch = idx_pad[i]
            valid_ch = valid[i]

            def _eta_per_ch(Ptot_val, P, beta2, beta3, beta4, a_fwm, a_bar_fwm,
                            f, B, Cr_fwm, gamma_fwm, L_val, idx_ch, valid_ch):
                T_tilde = -((Ptot_val * Cr_fwm) / (2 * a_fwm)) * f
                T_local = 1 + T_tilde

                def _eta(idx):
                    i_v, j_v, k_v, m_v = idx

                    def _phi(i_p, j_p, k_p, f_arr, b2, b3, b4):
                        phi_jk = -4 * pi ** 2 * (f_arr[j_p] - f_arr[i_p]) * (f_arr[k_p] - f_arr[i_p]) * (
                            b2 + pi * b3 * (f_arr[j_p] + f_arr[k_p])
                            + (2 / 3) * pi ** 2 * b4 * (
                                f_arr[j_p] ** 2 + f_arr[j_p] * f_arr[k_p] + f_arr[k_p] ** 2
                                + 0.5 * (f_arr[j_p] - f_arr[i_p]) * (f_arr[k_p] - f_arr[i_p])))
                        dphi_f1 = -4 * pi ** 2 * (f_arr[k_p] - f_arr[i_p]) * (
                            b2 + pi * b3 * (f_arr[j_p] + f_arr[k_p] + f_arr[j_p] - f_arr[i_p])
                            + (2 / 3) * pi ** 2 * b4 * (
                                f_arr[j_p] ** 2 + f_arr[j_p] * f_arr[k_p] + f_arr[k_p] ** 2
                                + 0.5 * (f_arr[j_p] - f_arr[i_p]) * (f_arr[k_p] - f_arr[i_p])
                                + (f_arr[j_p] - f_arr[i_p]) * (
                                    2 * (f_arr[j_p] - f_arr[i_p])
                                    + 1.5 * (f_arr[k_p] - f_arr[i_p]) + 3 * f_arr[i_p])))
                        dphi_f2 = -4 * pi ** 2 * (f_arr[j_p] - f_arr[i_p]) * (
                            b2 + pi * b3 * (f_arr[j_p] + f_arr[k_p] + f_arr[k_p] - f_arr[i_p])
                            + (2 / 3) * pi ** 2 * b4 * (
                                f_arr[j_p] ** 2 + f_arr[j_p] * f_arr[k_p] + f_arr[k_p] ** 2
                                + 0.5 * (f_arr[j_p] - f_arr[i_p]) * (f_arr[k_p] - f_arr[i_p])
                                + (f_arr[k_p] - f_arr[i_p]) * (
                                    2 * (f_arr[k_p] - f_arr[i_p])
                                    + 1.5 * (f_arr[j_p] - f_arr[i_p]) + 3 * f_arr[i_p])))
                        return phi_jk, dphi_f1, dphi_f2

                    phi_jk, dphi_f1, dphi_f2 = _phi(i_v, j_v, k_v, f, beta2, beta3, beta4)

                    def _island(i_v, j_v, k_v, m_v, a_loc, a_bar_loc, T_loc, T_tilde_loc,
                                B_loc, P_loc, gamma_loc, L_loc, phi_jk, dphi_f1, dphi_f2):
                        Omega = jnp.where(j_v == k_v, 1, 2)
                        Bi = B_loc[i_v]; Bj = B_loc[j_v]; Bk = B_loc[k_v]; Bm = B_loc[m_v]
                        Pi = P_loc[i_v]; Pj = P_loc[j_v]; Pk = P_loc[k_v]; Pm = P_loc[m_v]
                        gamma_iv = gamma_loc[i_v]
                        f1bound = Bj / 2
                        f2min = -Bk / 2
                        f2max = Bk / 2

                        def _F(a_v, b_v, d_v, f2):
                            tp = a_v + b_v * f2 + d_v
                            tm = a_v + b_v * f2 - d_v
                            return ((tp * arctan(tp) - tm * arctan(tm)) / b_v
                                    - 0.5 / b_v * (log(1.0 + tp ** 2) - log(1.0 + tm ** 2)))

                        def _island_COI(x):
                            l_j, l_k, l_j_line, l_k_line = x
                            alpha_j = a_loc[j_v] / 2 + l_j * a_bar_loc[j_v]
                            alpha_k = a_loc[k_v] / 2 + l_k * a_bar_loc[k_v]
                            alpha_all = alpha_j + alpha_k
                            alpha_j_line = a_loc[j_v] / 2 + l_j_line * a_bar_loc[j_v]
                            alpha_k_line = a_loc[k_v] / 2 + l_k_line * a_bar_loc[k_v]
                            alpha_all_line = alpha_j_line + alpha_k_line
                            T_tilde_j = T_loc[j_v] * (-T_tilde_loc[j_v] / T_loc[j_v]) ** l_j
                            T_tilde_k = T_loc[k_v] * (-T_tilde_loc[k_v] / T_loc[k_v]) ** l_k
                            T_tilde_all = T_tilde_j * T_tilde_k
                            T_tilde_j_line = T_loc[j_v] * (-T_tilde_loc[j_v] / T_loc[j_v]) ** l_j_line
                            T_tilde_k_line = T_loc[k_v] * (-T_tilde_loc[k_v] / T_loc[k_v]) ** l_k_line
                            T_tilde_all_line = T_tilde_j_line * T_tilde_k_line
                            alpha_tilde_all = (alpha_all * (1 - exp(-alpha_all * L_loc))
                                / (1 - exp(-alpha_all * L_loc) - alpha_all * L_loc * exp(-alpha_all * L_loc)))
                            kappa_all = ((1 - exp(-alpha_all * L_loc)) ** 2
                                / (1 - exp(-alpha_all * L_loc) - alpha_all * L_loc * exp(-alpha_all * L_loc)))
                            alpha_tilde_all_line = (alpha_all_line * (1 - exp(-alpha_all_line * L_loc))
                                / (1 - exp(-alpha_all_line * L_loc) - alpha_all_line * L_loc * exp(-alpha_all_line * L_loc)))
                            kappa_all_line = ((1 - exp(-alpha_all_line * L_loc)) ** 2
                                / (1 - exp(-alpha_all_line * L_loc) - alpha_all_line * L_loc * exp(-alpha_all_line * L_loc)))
                            a1 = phi_jk / alpha_tilde_all
                            a2 = phi_jk / alpha_tilde_all_line
                            b1 = dphi_f2 / alpha_tilde_all
                            b2_v = dphi_f2 / alpha_tilde_all_line
                            d1 = dphi_f1 * f1bound / alpha_tilde_all
                            d2 = dphi_f1 * f1bound / alpha_tilde_all_line
                            F1 = _F(a1, b1, d1, f2max) - _F(a1, b1, d1, f2min)
                            F2 = _F(a2, b2_v, d2, f2max) - _F(a2, b2_v, d2, f2min)
                            return (T_tilde_all * T_tilde_all_line * kappa_all * kappa_all_line
                                    / (alpha_tilde_all + alpha_tilde_all_line) * (F1 + F2) / dphi_f1)

                        def _island_FWM(x):
                            l_j, l_k, l_m, l_j_line, l_k_line, l_m_line = x
                            alpha_j = a_loc[j_v] / 2 + l_j * a_bar_loc[j_v]
                            alpha_k = a_loc[k_v] / 2 + l_k * a_bar_loc[k_v]
                            alpha_m = a_loc[m_v] / 2 + l_m * a_bar_loc[m_v]
                            alpha_i = a_loc[i_v] / 2
                            alpha_all = alpha_j + alpha_k + alpha_m - alpha_i
                            alpha_j_line = a_loc[j_v] / 2 + l_j_line * a_bar_loc[j_v]
                            alpha_k_line = a_loc[k_v] / 2 + l_k_line * a_bar_loc[k_v]
                            alpha_m_line = a_loc[m_v] / 2 + l_m_line * a_bar_loc[m_v]
                            alpha_i_line = a_loc[i_v] / 2
                            alpha_all_line = alpha_j_line + alpha_k_line + alpha_m_line - alpha_i_line
                            T_tilde_j = T_loc[j_v] * (-T_tilde_loc[j_v] / T_loc[j_v]) ** l_j
                            T_tilde_k = T_loc[k_v] * (-T_tilde_loc[k_v] / T_loc[k_v]) ** l_k
                            T_tilde_m = T_loc[m_v] * (-T_tilde_loc[m_v] / T_loc[m_v]) ** l_m
                            T_tilde_all = T_tilde_j * T_tilde_k * T_tilde_m
                            T_tilde_j_line = T_loc[j_v] * (-T_tilde_loc[j_v] / T_loc[j_v]) ** l_j_line
                            T_tilde_k_line = T_loc[k_v] * (-T_tilde_loc[k_v] / T_loc[k_v]) ** l_k_line
                            T_tilde_m_line = T_loc[m_v] * (-T_tilde_loc[m_v] / T_loc[m_v]) ** l_m_line
                            T_tilde_all_line = T_tilde_j_line * T_tilde_k_line * T_tilde_m_line
                            alpha_tilde_all = (alpha_all * (1 - exp(-alpha_all * L_loc))
                                / (1 - exp(-alpha_all * L_loc) - alpha_all * L_loc * exp(-alpha_all * L_loc)))
                            kappa_all = ((1 - exp(-alpha_all * L_loc)) ** 2
                                / (1 - exp(-alpha_all * L_loc) - alpha_all * L_loc * exp(-alpha_all * L_loc)))
                            alpha_tilde_all_line = (alpha_all_line * (1 - exp(-alpha_all_line * L_loc))
                                / (1 - exp(-alpha_all_line * L_loc) - alpha_all_line * L_loc * exp(-alpha_all_line * L_loc)))
                            kappa_all_line = ((1 - exp(-alpha_all_line * L_loc)) ** 2
                                / (1 - exp(-alpha_all_line * L_loc) - alpha_all_line * L_loc * exp(-alpha_all_line * L_loc)))
                            a1 = phi_jk / alpha_tilde_all
                            a2 = phi_jk / alpha_tilde_all_line
                            b1 = dphi_f2 / alpha_tilde_all
                            b2_v = dphi_f2 / alpha_tilde_all_line
                            d1 = dphi_f1 * f1bound / alpha_tilde_all
                            d2 = dphi_f1 * f1bound / alpha_tilde_all_line
                            F1 = _F(a1, b1, d1, f2max) - _F(a1, b1, d1, f2min)
                            F2 = _F(a2, b2_v, d2, f2max) - _F(a2, b2_v, d2, f2min)
                            return (T_tilde_all * T_tilde_all_line * kappa_all * kappa_all_line
                                    / (alpha_tilde_all + alpha_tilde_all_line) * (F1 + F2) / dphi_f1)

                        island_sum = jax.lax.cond(
                            m_v == i_v,
                            lambda _: jax.vmap(_island_COI)(_INDICES_COI).sum(axis=0),
                            lambda _: jax.vmap(_island_FWM)(_INDICES_FWM).sum(axis=0),
                            operand=None,
                        )
                        return nan_to_num(
                            Omega * (16 / 27) * gamma_iv ** 2 * (Bi / Pi ** 3)
                            * (Pj * Pk * Pm / (Bj * Bk * Bm)) * island_sum,
                            nan=0, posinf=0, neginf=0,
                        )

                    return _island(i_v, j_v, k_v, m_v, a_fwm, a_bar_fwm, T_local, T_tilde,
                                   B, P, gamma_fwm, L_val, phi_jk, dphi_f1, dphi_f2)

                return jnp.where(valid_ch, jax.vmap(_eta)(idx_ch).squeeze(), 0).sum(axis=0)

            return _eta_per_ch(Ptot_val, P, beta2, beta3, beta4, a_fwm, a_bar_fwm,
                               f, B, Cr_fwm, gamma_fwm, L_val, idx_ch, valid_ch)

        return jax.lax.map(_ch, jnp.arange(f.size))

    _eta_SPM = _eta_GN_SPM(phi_i, B_i, a_i, a_bar_i, gamma_ij, Tf_i, T_i, setup.length_j[j])
    _eta_XPM = _eta_GN_XPM(P_i, P_k, phi_ik, B_i, B_k, a_k, a_bar_k, gamma_ij, Tf_k, T_k, setup.length_j[j])

    if ch_idx_oband is not None and jnp.any(ch_idx_oband):
        idx_pad, valid = _FWM_idx(f_i[ch_idx_oband])
        _eta_FWM = jnp.zeros_like(_eta_XPM)
        _eta_FWM = _eta_FWM.at[ch_idx_oband].set(
            _eta_GN_FWM(Ptot, P_i[ch_idx_oband], beta2_j[j], beta3_j[j], beta4_j[j],
                        a_i[ch_idx_oband], a_bar_i[ch_idx_oband], f_i[ch_idx_oband],
                        B_i[ch_idx_oband], Cr_i[ch_idx_oband],
                        gamma_ij[ch_idx_oband, j], setup.length_j[j], idx_pad, valid))
    elif ch_idx_oband is None:
        idx_pad, valid = _FWM_idx(f_i)
        _eta_FWM = _eta_GN_FWM(Ptot, P_i, beta2_j[j], beta3_j[j], beta4_j[j],
                               a_i, a_bar_i, f_i, B_i, Cr_i,
                               gamma_ij[:, j], setup.length_j[j], idx_pad, valid)
    else:
        _eta_FWM = jnp.zeros_like(_eta_XPM)

    # --- ASE from ISRS power evolution (linear on/off gain P0 / Pend) ---
    gain = power_evo[0, :] / power_evo[-1, :]
    p_ASE = egn.calc_Pase(
        gain,
        setup.NF_i[chs],
        ref_lambda=setup.ref_lambda,
        ch_centre_ij=setup.ch_centre_ij[chs, :],
        ch_bandwidth_ij=setup.ch_bandwidth_ij[chs, :],
    ) * Nspans

    # --- NSR = GN-NLI + ASE + transceiver penalty (notebook return form) ---
    eta_sum = _eta_SPM[:, None] + _eta_XPM[:, None] + _eta_FWM[:, None]
    safe_P = jnp.maximum(P_i, jnp.array(1e-30, dtype=P_i.dtype))
    term_ase = p_ASE[:, None] / safe_P
    nsr_active = (Nspans * P_i ** 3 * eta_sum / safe_P
                  + term_ase
                  + 1 / idB(setup.snr_trx[chs, None]))
    occ = jnp.asarray(mask, dtype=P_i.dtype).reshape(P_i.shape[0], 1) > 0
    nsr = jnp.where(occ, nsr_active, jnp.inf)
    nsr_ase = jnp.where(occ, term_ase, jnp.nan)
    eta_spm = jnp.reshape(jnp.asarray(_eta_SPM, dtype=jnp.float64), (-1,))
    eta_xpm = jnp.reshape(jnp.asarray(_eta_XPM, dtype=jnp.float64), (-1,))
    eta_fwm = jnp.reshape(jnp.asarray(_eta_FWM, dtype=jnp.float64), (-1,))
    return nsr, fit_params, eta_spm, eta_xpm, eta_fwm, nsr_ase


# ===========================================================================
# RWA occupancy + helpers  (unchanged from compute_throughput_cfm.py)
# ===========================================================================
def _build_rwa_occupancy(graph, rwa, num_active_channels):
    """Build edge list, occupancy matrix, and edge-path info from graph + RWA."""
    def canon_edge(u, v):
        return (min(u, v), max(u, v))

    _min_effective_spans = 1e-15
    edge_span_map = {}
    for s, d in graph.edges():
        n_spans = float(graph[s][d].get("weight", 1))
        if not np.isfinite(n_spans) or n_spans <= 0.0:
            n_spans = _min_effective_spans
        edge_span_map[canon_edge(s, d)] = n_spans

    edges = sorted(edge_span_map.keys())
    edge_to_row = {edge: i for i, edge in enumerate(edges)}
    span_count_per_edge = np.array([edge_span_map[e] for e in edges], dtype=np.float64)

    wavelengths_sorted = sorted(
        int(w) for w in rwa.keys() if int(w) < num_active_channels
    )
    wavelength_to_col = {w: i for i, w in enumerate(wavelengths_sorted)}
    if set(wavelengths_sorted) != set(range(num_active_channels)):
        print(
            "[WARN] RWA wavelength keys are not exactly 0..N-1 for N="
            f"{num_active_channels} (found {len(wavelengths_sorted)} keys). "
            "Columns follow logical index w (Mongo OESCL = 0..N-1)."
        )
    occupancy_matrix = np.zeros((len(edges), num_active_channels), dtype=int)
    rwa_edge_paths = {}

    for w_str, paths in rwa.items():
        w = int(w_str)
        if w >= num_active_channels:
            continue
        rwa_edge_paths[w_str] = []
        for node_path in paths:
            if not node_path or len(node_path) < 2:
                continue
            node_path_int = [int(n) for n in node_path]
            edge_path = [canon_edge(node_path_int[i], node_path_int[i + 1])
                         for i in range(len(node_path_int) - 1)]
            rwa_edge_paths[w_str].append({
                "node_path": node_path_int,
                "edge_path": edge_path,
            })
            for edge in edge_path:
                if edge in edge_to_row:
                    occupancy_matrix[edge_to_row[edge], w] = 1

    return (edges, edge_to_row, span_count_per_edge,
            occupancy_matrix, rwa_edge_paths, wavelength_to_col)


def _build_band_aware_rwa_key(route_function, band_selection):
    """Return an RWA key like 'kSP-FF SCL RWA'."""
    band_label = str(band_selection).strip().upper()
    return f"{route_function} {band_label} RWA"


# ---------------------------------------------------------------------------
# Per-topology throughput computation  (unchanged pipeline)
# ---------------------------------------------------------------------------
def compute_topology_throughput(
    db,
    collection,
    _id,
    band_selection,
    route_function,
    span_length_km=80,
    launch_power_dBm=-2.0,
    save_snr_to_disk=True,
):
    """Compute throughput for one topology document stored in MongoDB."""
    band_cfg = BAND_CONFIGS[band_selection]
    active_bands = band_cfg["bands"]

    # --- Read topology and RWA from MongoDB ---
    results = list(
        nt.Database.read_data(db, collection, find_dic={"_id": _id}, max_count=1)
    )
    if not results:
        print(f"[WARN] No document found for _id={_id}")
        return 0.0

    doc = results[0]
    rwa_key = _build_band_aware_rwa_key(route_function, band_selection)
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

    graph = graph_list[0][0]
    graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)

    rwa_raw = doc[rwa_key]
    rwa = nt.Tools.read_database_dict(rwa_raw)

    # --- Build multi-band channel grid ---
    ch_lambda, channel_idx, ch_idx_oband_full, band_masks, ref_lambda = (
        _build_multiband_channel_grid(active_bands)
    )
    num_active = int(np.sum(channel_idx))
    ch_idx_oband_active = ch_idx_oband_full[channel_idx]

    print(f"  Channel grid: {NUM_CHANNELS_TOTAL} slots, {num_active} active channels, "
          f"ref_lambda={ref_lambda*1e9:.1f} nm")

    # --- Per-channel parameters ---
    P_channel, nf, snr_trx = _build_per_channel_params(
        ch_lambda, band_masks, active_bands, launch_power_dBm=launch_power_dBm,
    )

    # --- Build setup object ---
    setup = _build_setup(
        span_length_km, ch_lambda, channel_idx, P_channel, nf, snr_trx,
        ref_lambda=ref_lambda,
    )

    # --- Build occupancy matrix ---
    (edges, edge_to_row, span_count_per_edge,
     occupancy_matrix, rwa_edge_paths, wavelength_to_col) = (
        _build_rwa_occupancy(graph, rwa, num_active)
    )
    active_slot_ix = np.flatnonzero(np.asarray(channel_idx))

    # --- Compute per-link NSR with occupancy ---
    nsr_link_channel = jnp.zeros_like(jnp.asarray(occupancy_matrix), dtype=jnp.float64)
    fit_params_per_link = {}
    eta_spm_per_link: dict[int, np.ndarray] = {}
    eta_xpm_per_link: dict[int, np.ndarray] = {}
    eta_fwm_per_link: dict[int, np.ndarray] = {}
    nsr_ase_per_link: dict[int, np.ndarray] = {}

    n_links = int(span_count_per_edge.shape[0])
    print(f"    Undirected links in topology: {n_links} (Link indices 0..{n_links - 1})")

    for i in range(n_links):
        t0 = time.perf_counter()
        link_nsr, link_fit_params, eta_spm, eta_xpm, eta_fwm, link_nsr_ase = calc_NSR_link(
            setup,
            float(span_count_per_edge[i]),
            occupancy_matrix[i, :, None],
            ch_idx_oband_active,
        )
        link_nsr = link_nsr.squeeze(-1)
        nsr_ase_per_link[i] = np.asarray(link_nsr_ase.squeeze(-1), dtype=np.float64)
        fit_params_per_link[i] = np.asarray(link_fit_params.squeeze(-1))
        eta_spm_per_link[i] = np.asarray(eta_spm, dtype=np.float64).reshape(-1)
        eta_xpm_per_link[i] = np.asarray(eta_xpm, dtype=np.float64).reshape(-1)
        eta_fwm_per_link[i] = np.asarray(eta_fwm, dtype=np.float64).reshape(-1)
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
        print(f"    Link {i} ({edges[i]}, spans={span_count_per_edge[i]}): "
              f"{dt:.1f}s, mean SNR={snr_msg}")

    # --- Save per-link per-channel SNR + fit params + occupancy to disk ---
    if save_snr_to_disk:
        snr_dir = _SCRIPT_DIR / "data" / "snr"
        snr_dir.mkdir(parents=True, exist_ok=True)
        ch_lambda_active = np.asarray(ch_lambda)[np.asarray(channel_idx)]
        ch_freq_active = c / ch_lambda_active
        nsr_np = np.asarray(nsr_link_channel)

        for i in range(n_links):
            occ = occupancy_matrix[i] > 0
            ln = nsr_np[i]
            snr_dB = np.full_like(ln, np.nan)
            valid = occ & np.isfinite(ln) & (ln > 0)
            snr_dB[valid] = 10.0 * np.log10(1.0 / ln[valid])

            ln_ase = nsr_ase_per_link[i]
            snr_ase_dB = np.full_like(ln_ase, np.nan)
            valid_ase = occ & np.isfinite(ln_ase) & (ln_ase > 0)
            snr_ase_dB[valid_ase] = 10.0 * np.log10(1.0 / ln_ase[valid_ase])

            fp = fit_params_per_link[i]
            u, v = edges[i]
            fname = f"{_id}_{band_selection}_{route_function}_link{i}_{u}-{v}.npz"
            np.savez_compressed(
                snr_dir / fname,
                channel_index=np.arange(num_active),
                wavelength_m=ch_lambda_active,
                frequency_hz=ch_freq_active,
                occupancy=occ.astype(int),
                snr_dB=snr_dB,
                nsr_linear=ln,
                nsr_ase_linear=ln_ase,
                snr_ase_dB=snr_ase_dB,
                eta_spm_linear=eta_spm_per_link[i],
                eta_xpm_linear=eta_xpm_per_link[i],
                eta_fwm_linear=eta_fwm_per_link[i],
                edge=np.array([u, v]),
                spans=float(span_count_per_edge[i]),
                a=fp[:, 0],
                a_bar=fp[:, 1],
                Cr=fp[:, 2],
            )

        occ_fname = f"{_id}_{band_selection}_{route_function}_occupancy.npz"
        np.savez_compressed(
            snr_dir / occ_fname,
            occupancy_matrix=occupancy_matrix,
            edges=np.array(edges),
            spans=np.asarray(span_count_per_edge),
            channel_index=np.arange(num_active),
            wavelength_m=ch_lambda_active,
            frequency_hz=ch_freq_active,
        )
        print(f"  Saved per-link SNR/fit-params + occupancy to {snr_dir}")

    # --- Walk RWA lightpaths and compute capacity ---
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
            edge_path = info["edge_path"]
            path_nsr = 0.0
            for edge in edge_path:
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

    # --- Write results back to MongoDB ---
    nt.Database.update_data_with_id(
        db,
        collection,
        _id,
        newvals={
            "$set": {
                f"{route_function} Capacity-CFM": float(capacity_total),
                f"{route_function} CFM-lightpaths": int(n_lightpaths),
                f"{route_function} CFM-timestamp": datetime.datetime.utcnow(),
                f"{route_function} CFM-band": band_cfg["name"],
                f"{route_function} CFM-span_length_km": float(span_length_km),
                f"{route_function} CFM-launch_power_dBm": (
                    {b: float(p) for b, p in launch_power_dBm.items()}
                    if isinstance(launch_power_dBm, dict) else float(launch_power_dBm)
                ),
            }
        },
    )

    return capacity_total


# ---------------------------------------------------------------------------
# Ray-parallel wrapper
# ---------------------------------------------------------------------------
def run_parallel(
    db,
    collection,
    topology_name,
    route_function,
    band_selection,
    span_length_km=80,
    launch_power_dBm=-2.0,
    save_snr_to_disk=True,
    hostname="128.40.42.13",
    port=6379,
):
    """Launch Ray tasks, one per topology document."""
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
            save_snr_to_disk=save_snr_to_disk,
        )

    futures = [_remote_worker.remote(_id) for _, _id in graph_list]
    results = ray.get(futures)

    ray.shutdown()
    print(f"\nAll done. {len(results)} topologies processed.")
    for i, (_, _id) in enumerate(graph_list):
        print(f"  {_id}: {results[i]/1e12:.4f} Tbps")

    return results


# ---------------------------------------------------------------------------
# Sequential (non-Ray) execution for testing
# ---------------------------------------------------------------------------
def run_sequential(
    db,
    collection,
    topology_name,
    route_function,
    band_selection,
    span_length_km=80,
    launch_power_dBm=-2.0,
    save_snr_to_disk=True,
):
    """Process topologies one by one (no Ray needed)."""
    graph_list = nt.Database.read_topology_dataset_list(
        db, collection, find_dic={"name": topology_name}, node_data=True
    )
    print(f"Found {len(graph_list)} topologies for '{topology_name}'")

    results = []
    for graph, _id in graph_list:
        cap = compute_topology_throughput(
            db=db,
            collection=collection,
            _id=_id,
            band_selection=band_selection,
            route_function=route_function,
            span_length_km=span_length_km,
            launch_power_dBm=launch_power_dBm,
            save_snr_to_disk=save_snr_to_disk,
        )
        results.append(cap)

    print(f"\nAll done. {len(results)} topologies processed.")
    for i, (_, _id) in enumerate(graph_list):
        print(f"  {_id}: {results[i]/1e12:.4f} Tbps")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute throughput using CFM NLI model (multi-band)"
    )
    parser.add_argument("--route_function", type=str, default=None,
                        help="RWA key prefix (ILP-connections | kSP-FF | FF-kSP)")
    parser.add_argument("--topology", type=str, default=None,
                        help="Topology name in MongoDB")
    parser.add_argument("--collection", type=str, default=None,
                        help="MongoDB collection")
    parser.add_argument("--db", type=str, default=None,
                        help="MongoDB database name")
    parser.add_argument("--band", type=str, default=None,
                        choices=list(BAND_CONFIGS.keys()),
                        help="Band selection (C | CL | SCL | SCLO)")
    parser.add_argument("--span_length_km", type=float, default=None)
    parser.add_argument("--launch_power_dBm", type=float, default=None)
    parser.add_argument("--parallel", action="store_true",
                        help="Use Ray for parallel execution")
    parser.add_argument("--hostname", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # ================================================================
    #  USER CONFIGURATION
    # ================================================================
    SAVE_SNR_TO_DISK = True
    BAND_SELECTION = "OESCL"
    ROUTE_FUNCTION = "kSP-FF"
    TOPOLOGY_NAME = "NSFNET"

    DB = "Topology_Data"
    COLLECTION = "real"

    SPAN_LENGTH_KM = 80
    # Per-band launch power [dBm] in order O, E, S, C, L.
    # Use a single float (e.g. -2.0) for uniform launch across all bands,
    # or a list/dict to set per-band values.
    LAUNCH_POWER_DBM = [-1.63, -2.63, -2.25, -4.76, -4.54]
    # LAUNCH_POWER_DBM = [-2, -2, -2, -2, -2]


    USE_PARALLEL = False
    HOSTNAME = "128.40.40.67"
    PORT = 6379

    # ================================================================
    #  Command-line overrides
    # ================================================================
    cli = main()
    band       = cli.band            or BAND_SELECTION
    route_fn   = cli.route_function  or ROUTE_FUNCTION
    topo       = cli.topology        or TOPOLOGY_NAME
    collection = cli.collection      or COLLECTION
    db         = cli.db              or DB
    span_km    = cli.span_length_km  if cli.span_length_km  is not None else SPAN_LENGTH_KM
    lp_dBm     = cli.launch_power_dBm if cli.launch_power_dBm is not None else LAUNCH_POWER_DBM
    # Normalise list/tuple -> dict keyed by band order O,E,S,C,L
    if isinstance(lp_dBm, (list, tuple)):
        _ORDER = ["O", "E", "S", "C", "L"]
        if len(lp_dBm) != len(_ORDER):
            raise ValueError(f"LAUNCH_POWER_DBM list must have {len(_ORDER)} entries (O,E,S,C,L), got {len(lp_dBm)}")
        lp_dBm = {b: float(p) for b, p in zip(_ORDER, lp_dBm)}
    parallel   = cli.parallel        or USE_PARALLEL
    hostname   = cli.hostname        or HOSTNAME
    port       = cli.port            if cli.port is not None else PORT

    if band not in BAND_CONFIGS:
        raise ValueError(f"Invalid band: {band}. Options: {list(BAND_CONFIGS.keys())}")

    band_cfg = BAND_CONFIGS[band]
    ch_lambda, channel_idx, _, _, ref_lambda_main = _build_multiband_channel_grid(band_cfg["bands"])
    num_active = int(np.sum(channel_idx))

    # ================================================================
    #  Print configuration summary
    # ================================================================
    print("=" * 60)
    print("CFM-based Throughput Computation (multi-band, notebook math)")
    print("=" * 60)
    print(f"Route function   : {route_fn}")
    print(f"Topology         : {topo}")
    print(f"DB / Collection  : {db} / {collection}")
    print(f"Band selection   : {band_cfg['name']} ({band_cfg['description']})")
    print(f"Active bands     : {band_cfg['bands']}")
    print(f"Active channels  : {num_active}")
    print(f"Channel spacing  : {CH_SPACING_HZ/1e9:.0f} GHz")
    print(f"Channel bandwidth: {CH_BW_HZ/1e9:.0f} GHz")
    print(f"Ref lambda (grid): {REF_LAMBDA_M*1e9:.1f} nm")
    print(f"Ref lambda (disp): {ref_lambda_main*1e9:.1f} nm")
    print(f"Span length      : {span_km} km")
    if isinstance(lp_dBm, dict):
        print("Launch power     : per-band " + ", ".join(f"{b}={lp_dBm[b]:.2f}" for b in lp_dBm) + " dBm")
    else:
        print(f"Launch power     : {lp_dBm} dBm (uniform)")
    for b in band_cfg["bands"]:
        print(f"  {b}-band: NF={BANDS[b]['noise_figure']} dB, "
              f"TRX SNR={BANDS[b]['TransceiverSNR']} dB")
    print(f"Parallel (Ray)   : {parallel}")
    print(f"Save SNR to disk : {SAVE_SNR_TO_DISK}")
    print("=" * 60)

    # ================================================================
    #  Run
    # ================================================================
    if parallel:
        run_parallel(
            db=db,
            collection=collection,
            topology_name=topo,
            route_function=route_fn,
            band_selection=band,
            span_length_km=span_km,
            launch_power_dBm=lp_dBm,
            save_snr_to_disk=SAVE_SNR_TO_DISK,
            hostname=hostname,
            port=port,
        )
    else:
        run_sequential(
            db=db,
            collection=collection,
            topology_name=topo,
            route_function=route_fn,
            band_selection=band,
            span_length_km=span_km,
            launch_power_dBm=lp_dBm,
            save_snr_to_disk=SAVE_SNR_TO_DISK,
        )
