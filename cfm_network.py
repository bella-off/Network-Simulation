"""
Reference: interactive workflow in ``cfm_network.ipynb`` (there is no second code path).

The same physical model is implemented for MongoDB / CLI in
``compute_throughput_cfm.py``:

- ``calc_NSR_link``: per-link NSR = span-scaled GN NLI (SPM+XPM+FWM on O) +
  ASE from ``egn.calc_Pase`` with **linear** fibre on/off gain
  ``P(0)/P(L)``, plus transceiver penalty ``1/idB(snr_trx)``.
- Throughput: for each RWA lightpath, ``path_nsr = sum_e NSR_e``,
  ``snr = 1/path_nsr``, ``rate = 2 * ch_bandwidth * log2(1 + snr)``.

If an older notebook cell still uses ``gain = 10*log10(P0/Pend)`` for
``calc_Pase``, that overstates SNR vs the library example
``ong/examples/egn_example.py`` (linear ratio).  ``compute_throughput_cfm.py``
uses the linear gain; totals can be **lower** than a notebook run with the
dB mistake.

The notebook’s ``rwa_throughput`` uses ``path_nsr <= 0 -> rate_bps = inf``;
``compute_throughput_cfm.py`` **skips** non-finite / non-positive ``path_nsr``
instead (finite, conservative total).
"""
