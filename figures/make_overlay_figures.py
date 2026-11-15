# Title: make_overlay_figures.py
# Author: N. D. Vilanakis
#
# A progressive multi-output 1-D convolutional network for lower-frequency
# trace reconstruction of UHRS seismic data
# N. D. Vilanakis, N. Economou, A. Vafidis
# Computers and Geosciences
#
# User guide
#   Purpose : Draws Figs. 2 and 3 of the paper, in which one test trace is
#             reconstructed by the progressive and the multi-head model and the
#             difference of each from the truth is shaded. Fig. 2 is the Ricker
#             family at the 7.96 ms head receptive field; Fig. 3 is the UHRS
#             family with both models at the matched field of 13.90 ms.
#   Inputs  : predictions_progressive_test.mat and predictions_multihead_test.mat
#             for each case, written by the infer_*_multioutput.m scripts in
#             inference/. Set DATA_ROOT below to the folder that holds the
#             "Ricker Final" and "UHRS Final" trees.
#   Outputs : Figure_testset_Ricker.png and Figure_testset_UHRS.png, plus a
#             table of the per-band mean absolute difference printed to the
#             console. Set the environment variable FIG_OUT to write them
#             somewhere other than this folder.
#   Needs   : Python 3 with numpy, h5py and matplotlib.
#
#   Note on the metric. The number annotated on each panel is the mean absolute
#   difference over the displayed window, 0 to 0.015 s, normalized by the
#   maximum of the true trace. It is NOT the RRMSE of Tables 3 and 4, which is
#   taken over the coda window alone and normalized by the amplitude range.

"""Merged in-domain comparison figures: one per wavelet, both models on the
same test trace, with the deviation of each from the truth."""
import os, numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set FIG_FORMAT to pdf or eps for vector output, which publishers ask for with
# line art. fonttype 42 embeds TrueType subsets rather than Type 3, which some
# production systems reject.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

BASE = os.path.dirname(os.path.abspath(__file__))

# Folder holding "Ricker Final" and "UHRS Final". The default assumes this
# repository sits inside the Publication folder that also holds those trees;
# set it explicitly otherwise.
DATA_ROOT = os.environ.get(
    "DATA_ROOT", os.path.dirname(os.path.dirname(BASE)))
OUT = os.environ.get("FIG_OUT", BASE)
FMT = os.environ.get("FIG_FORMAT", "png").lower()

TRACE = 1451            # same index in both wavelet families
TMAX  = 0.015           # displayed time range, seconds

# name -> (progressive folder, multi-head folder, note appended to the title)
# The UHRS case uses the matched-receptive-field models, which is what Fig. 3
# of the paper shows; the 7.96 ms pair gives visibly different panels.
CASES = {
    "Ricker": (os.path.join("Ricker Final", "Progressive"),
               os.path.join("Ricker Final", "Multihead"),
               ""),
    "UHRS":   (os.path.join("UHRS Final", "Progressive_matchedRF"),
               os.path.join("UHRS Final", "Multihead_matchedRF"),
               ", matched receptive field (d = [4 4])"),
}


def load(path, idx):
    """Return t, input trace, true bands, predicted bands for one test trace."""
    with h5py.File(path, "r") as f:
        t  = np.array(f["t"]).squeeze()
        hz = np.array(f["targetHz"]).squeeze()
        # arrays are [nt x N x nStages] in MATLAB -> h5py gives them reversed
        Yp = f["YPred_test"]
        Yt = f["Y_test"] if "Y_test" in f else None
        X  = f["X_test"]
        # h5py order is (nStages, N, nt); a single trace is already [stage, time]
        if Yp.ndim != 3:
            raise RuntimeError("unexpected YPred_test shape %s" % (Yp.shape,))
        P = np.array(Yp[:, idx, :])
        T = np.array(Yt[:, idx, :]) if Yt is not None else None
        x = np.array(X[idx, :]).squeeze()
    return t, x, T, P, hz


for name, (dp_, dm_, rf_note) in CASES.items():
    fp = os.path.join(DATA_ROOT, dp_, "predictions_progressive_test.mat")
    fm = os.path.join(DATA_ROOT, dm_, "predictions_multihead_test.mat")
    if not (os.path.isfile(fp) and os.path.isfile(fm)):
        print("SKIP %-7s (missing: %s)" % (name, ", ".join(
            p for p in (fp, fm) if not os.path.isfile(p))))
        continue
    t, x, Yt, Pp, hz = load(fp, TRACE)
    _, _, _,  Pm, _  = load(fm, TRACE)
    n = len(hz)

    m = t <= TMAX
    fig, ax = plt.subplots(n + 1, 1, figsize=(9.0, 1.05 * (n + 1)), sharex=True)

    ax[0].plot(t[m], x[m], color="#1f4e9c", lw=0.9)
    ax[0].set_ylabel("6400 Hz\ninput", fontsize=7)
    ax[0].tick_params(labelsize=6); ax[0].grid(alpha=.25, lw=.4)
    # Three lines rather than two: the one-line form overflows the canvas at
    # this width once the receptive-field note is present.
    ax[0].set_title("%s, test trace %d%s\ntrue (black), progressive (red dashed), "
                    "multi-head (blue dash-dot);\nshaded: |predicted - true| as %% "
                    "of trace maximum (right axis)" % (name, TRACE, rf_note),
                    fontsize=9)

    # One denominator for both the panel annotation and the printed table: the
    # maximum of the true trace over the displayed window.
    dens = [np.max(np.abs(Yt[i][m])) or 1.0 for i in range(n)]

    for i in range(n):
        a = ax[i + 1]
        yt, yp, ym = Yt[i][m], Pp[i][m], Pm[i][m]
        den = dens[i]
        dp, dm = 100 * np.abs(yp - yt) / den, 100 * np.abs(ym - yt) / den

        b = a.twinx()
        b.fill_between(t[m], 0, dm, color="#2b7bba", alpha=.25, lw=0)
        b.fill_between(t[m], 0, dp, color="#c0392b", alpha=.25, lw=0)
        b.set_ylim(0, max(20, np.percentile(np.r_[dp, dm], 99.5) * 1.25))
        b.set_ylabel("%", fontsize=6, color="#555555")
        b.tick_params(labelsize=5, colors="#555555")

        a.plot(t[m], yt, "k-",  lw=0.9)
        a.plot(t[m], yp, "--",  color="#c0392b", lw=0.9)
        a.plot(t[m], ym, "-.",  color="#2b7bba", lw=0.9)
        a.set_ylabel("%g Hz" % hz[i], fontsize=7)
        a.tick_params(labelsize=6); a.grid(alpha=.25, lw=.4)
        a.set_zorder(b.get_zorder() + 1); a.patch.set_visible(False)
        a.text(0.995, 0.92, "mean |diff|  prog %.1f %%   MH %.1f %%"
               % (dp.mean(), dm.mean()), transform=a.transAxes,
               ha="right", va="top", fontsize=6, color="#333333")

    ax[-1].set_xlabel("Time (s)", fontsize=8)
    ax[-1].set_xlim(0, TMAX)
    fig.tight_layout(h_pad=0.15)

    if not os.path.isdir(OUT):
        os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, "Figure_testset_%s.%s" % (name, FMT))
    fig.savefig(dst, dpi=300); plt.close(fig)
    print("written:", dst)
    print("   band    prog mean|d|%   MH mean|d|%")
    for i in range(n):
        print("   %5g %13.1f %13.1f" % (hz[i],
              (100 * np.abs(Pp[i][m] - Yt[i][m]) / dens[i]).mean(),
              (100 * np.abs(Pm[i][m] - Yt[i][m]) / dens[i]).mean()))
    print()
