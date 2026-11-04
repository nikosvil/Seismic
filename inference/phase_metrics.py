# Title: phase_metrics.py
# Author: N. D. Vilanakis
#
# A progressive multi-output 1-D convolutional network for low frequency
# seismic trace reconstruction
# N. D. Vilanakis, N. Economou, A. Vafidis
# Computers and Geosciences
#
# User guide
#   Purpose : Computes the timing and phase diagnostics reported in Sections
#             2.8, 4.3 and 5 of the paper: the cross-correlation lag as a
#             fraction of the band period, the envelope-weighted instantaneous
#             phase difference, the single scalar gain, the RRMSE before and
#             after that rescaling, and the coda spectrum.
#   Inputs  : the predictions_real_data_*.mat files written by the
#             infer_real_data_*.m scripts. Set DATA_ROOT below, or in the
#             environment, to the folder holding the "Ricker Final" and
#             "UHRS Final" trees.
#   Outputs : one table per model, printed to the console.
#   Needs   : Python 3 with numpy, h5py and scipy.

"""Phase metrics for the cross-physics records. See phase_metrics.md.

The question behind these numbers is whether the mid-band error of Tables 3
and 4 is a timing error or a gain error. If FWI normalizes amplitude before
inversion, only the former matters.

Per band, all against the TRUE trace of the same band, on the coda window of
Tables 3 and 4 (0.005 to 0.010 s):

  lag       position of the cross-correlation maximum, parabolic sub-sample
            refinement, in ms and as a fraction of the band period
  dphi      mean |instantaneous phase difference| via the Hilbert transform,
            weighted by the true envelope, in degrees
  gain      the scalar g minimizing ||g*pred - true||
  RRMSE     as published, and again after that single rescaling
  fpeak     amplitude-spectrum peak, Hann taper, zero-padded, interpolated
  centroid  amplitude-weighted mean frequency

The spectral measures are taken on the coda, not the full trace: on the full
trace the primary arrival and the near-DC content dominate and the peak is
meaningless (it returns 0 Hz for the Ricker 500 Hz band).

The relW and ampR vectors stored in the .mat files are stale, written before
the RRMSE denominator became the amplitude range and before AmpRatio became a
ratio of maxima. Everything here is recomputed from Y_true and YPred.

Needs numpy, h5py, scipy. The .mat files are MATLAB v7.3, so h5py, and the
arrays come back transposed.
"""
import numpy as np, h5py, io, sys, os
from scipy.signal import hilbert

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
# Default assumes this repository sits inside the Publication folder that
# also holds the "Ricker Final" and "UHRS Final" trees.
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.dirname(os.path.dirname(BASE)))

CASES = [
    ("UHRS progressive, matched RF 13.90 ms",
     os.path.join("UHRS Final", "Progressive_matchedRF",
                  "predictions_real_data_UHS.mat")),
    ("UHRS multi-head,  matched RF 13.90 ms",
     os.path.join("UHRS Final", "Multihead_matchedRF",
                  "predictions_real_data_UHS_multihead.mat")),
    ("Ricker progressive, 7.96 ms",
     os.path.join("Ricker Final", "Progressive",
                  "predictions_real_data_Ricker.mat")),
]


def arr(f, k):
    a = np.array(f[k])
    if a.ndim == 2 and min(a.shape) == 1:
        return a.ravel()
    return a.T if a.ndim == 2 else a.ravel()


def _parabolic(y, k):
    """Sub-sample peak position from the three samples around index k."""
    if 0 < k < len(y) - 1:
        y0, y1, y2 = y[k - 1], y[k], y[k + 1]
        den = y0 - 2 * y1 + y2
        if den:
            return k + 0.5 * (y0 - y2) / den
    return float(k)


def lag_ms(yt, yp, dt):
    """Cross-correlation lag; positive means the prediction is late."""
    c = np.correlate(yp - yp.mean(), yt - yt.mean(), mode="full")
    k = _parabolic(np.abs(c), int(np.argmax(np.abs(c))))
    return (k - (len(yt) - 1)) * dt * 1e3


def dphi_deg(yt, yp):
    """Envelope-weighted mean |instantaneous phase difference|, degrees."""
    at, ap = hilbert(yt), hilbert(yp)
    d = np.angle(ap * np.conj(at))
    w = np.abs(at)
    return np.degrees(np.sum(np.abs(d) * w) / (w.sum() + 1e-30))


def spectrum(x, dt, npad=8192):
    """Peak frequency and amplitude-weighted centroid."""
    x = x - x.mean()
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)), npad))
    df = np.fft.rfftfreq(npad, dt)[1]
    peak = _parabolic(X, int(np.argmax(X))) * df
    f = np.fft.rfftfreq(npad, dt)
    return peak, float((f * X).sum() / (X.sum() + 1e-30))


for label, rel in CASES:
    path = os.path.join(DATA_ROOT, rel)
    if not os.path.isfile(path):
        print("SKIP %-40s (not found: %s)" % (label, path))
        continue
    with h5py.File(path, "r") as f:
        Yt, Yp = arr(f, "Y_true"), arr(f, "YPred")
        t, hz, tw = arr(f, "t"), arr(f, "targetHz").astype(int), arr(f, "tWin")
    dt = float(t[1] - t[0])
    win = np.where((t >= tw[0]) & (t <= tw[1]))[0]

    print("=" * 118)
    print("%s   coda window %.3f-%.3f s, %d samples, dt = %.0e s"
          % (label, tw[0], tw[1], len(win), dt))
    print("=" * 118)
    print("%-7s %-9s %-9s %-8s %-8s %-9s %-9s %-10s %-10s %-9s %-9s"
          % ("band", "lag(ms)", "lag/T", "dphi", "gain", "RRMSE", "RRMSE_g",
             "fpeak_t", "fpeak_p", "cent_t", "cent_p"))
    print("-" * 118)
    rows = []
    for i, f_ in enumerate(hz):
        yt, yp = Yt[win, i], Yp[win, i]
        lg = lag_ms(yt, yp, dt)
        dp = dphi_deg(yt, yp)
        g = float(np.dot(yt, yp) / (np.dot(yp, yp) + 1e-30))
        rng = yt.max() - yt.min() + 1e-12
        r0 = 100 * np.sqrt(np.mean((yp - yt) ** 2)) / rng
        r1 = 100 * np.sqrt(np.mean((g * yp - yt) ** 2)) / rng
        (pt, ct), (pp, cp) = spectrum(yt, dt), spectrum(yp, dt)
        rows.append((f_, lg, lg * f_ / 1000.0, dp, g, r0, r1, pt, pp, ct, cp))
        print("%-7d %-9.3f %-9.3f %-8.1f %-8.3f %-9.1f %-9.1f %-10.0f %-10.0f %-9.0f %-9.0f"
              % rows[-1])
    a = np.array([r[1:] for r in rows])
    print("-" * 118)
    print("%-7s %-9.3f %-9.3f %-8.1f %-8.3f %-9.1f %-9.1f"
          % ("mean", a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(),
             a[:, 3].mean(), a[:, 4].mean(), a[:, 5].mean()))
    print("   max |lag| %.3f ms = %.3f of a period   worst dphi %.1f deg"
          % (np.abs(a[:, 0]).max(), np.abs(a[:, 1]).max(), a[:, 2].max()))
    pk = 100 * (a[:, 7] - a[:, 6]) / a[:, 6]
    cn = 100 * (a[:, 9] - a[:, 8]) / a[:, 8]
    print("   peak frequency: mean |diff| %.1f %%, max %.1f %%"
          % (np.abs(pk).mean(), np.abs(pk).max()))
    print("   centroid      : mean diff %+.1f %%, max %+.1f %%  (prediction high "
          "in %d of %d bands)" % (cn.mean(), cn.max(), (cn > 0).sum(), len(cn)))
    print()
