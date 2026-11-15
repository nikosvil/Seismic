# Title: make_fd_figures.py
# Author: N. D. Vilanakis
#
# A progressive multi-output 1-D convolutional network for lower-frequency
# trace reconstruction of UHRS seismic data
# N. D. Vilanakis, N. Economou, A. Vafidis
# Computers and Geosciences
#
# User guide
#   Purpose : Draws Figs. 5, 6 and 7 of the paper, the cross-physics
#             reconstructions of the finite difference records. One panel per
#             band, the true trace in black and the prediction in red dashed,
#             with the absolute difference shaded against a right-hand axis in
#             per cent of the trace maximum. The coda evaluation window is
#             shaded grey behind the traces.
#   Inputs  : the predictions_real_data_*.mat files written by the
#             infer_real_data_*.m scripts in inference/. Set DATA_ROOT below to
#             the folder that holds the "Ricker Final" and "UHRS Final" trees.
#   Outputs : Figure_fd_Ricker_progressive.png, Figure_fd_UHRS_progressive.png
#             and Figure_fd_UHRS_multihead.png, plus the per-band annotation
#             values printed to the console. Set the environment variable
#             FIG_OUT to write them somewhere other than this folder.
#   Needs   : Python 3 with numpy, h5py and matplotlib.
#
#   Note on the two metrics. The number annotated on each panel is the mean
#   absolute difference over the whole displayed window, 0 to 0.015 s,
#   normalized by the maximum of the true trace. It is NOT the RRMSE of
#   Tables 3 and 4, which is taken over the coda window alone and normalized
#   by the amplitude range. The two are different by construction.

"""Cross-physics figures: one finite difference record per model, every band
compared against the truth, with the deviation shaded on a second axis."""
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

TMAX  = 0.015        # displayed time range, seconds
CODA  = (0.005, 0.010)

# label -> (folder, candidate file names, output file)
CASES = [
    ("Ricker, progressive model",
     os.path.join("Ricker Final", "Progressive"),
     ("predictions_real_data_Ricker.mat",),
     "Figure_fd_Ricker_progressive"),
    ("Progressive model, matched receptive field (d = [4 4])",
     os.path.join("UHRS Final", "Progressive_matchedRF"),
     ("predictions_real_data_UHRS.mat", "predictions_real_data_UHS.mat"),
     "Figure_fd_UHRS_progressive"),
    ("Multi-head ablation, matched receptive field (d = [4 4])",
     os.path.join("UHRS Final", "Multihead_matchedRF"),
     ("predictions_real_data_UHRS_multihead.mat",
      "predictions_real_data_UHS_multihead.mat"),
     "Figure_fd_UHRS_multihead"),
]


def resolve(folder, names):
    """First of `names` that exists in `folder`, else None.

    Both spellings are accepted because the records were first written under
    the earlier UHS name and the scripts now write UHRS.
    """
    for n in names:
        p = os.path.join(DATA_ROOT, folder, n)
        if os.path.isfile(p):
            return p
    return None


def load(path):
    """Return t, input trace, true bands, predicted bands, band frequencies.

    MATLAB writes these as [nt x nStages]; h5py hands them back transposed.
    """
    with h5py.File(path, "r") as f:
        def a(k):
            v = np.array(f[k])
            if v.ndim == 2 and min(v.shape) == 1:
                return v.ravel()
            return v.T if v.ndim == 2 else v.ravel()
        return a("t"), a("X_in"), a("Y_true"), a("YPred"), a("targetHz").astype(int)


for label, folder, names, outfile in CASES:
    path = resolve(folder, names)
    if path is None:
        print("SKIP  %-54s (no %s in %s)"
              % (label, " or ".join(names), os.path.join(DATA_ROOT, folder)))
        continue

    t, x, Yt, Yp, hz = load(path)
    m = t <= TMAX
    n = len(hz)

    fig, ax = plt.subplots(n + 1, 1, figsize=(9.0, 1.05 * (n + 1)), sharex=True)

    ax[0].plot(t[m], x[m], color="#1f4e9c", lw=0.9)
    ax[0].set_ylabel("6400 Hz\ninput", fontsize=7)
    ax[0].tick_params(labelsize=6)
    ax[0].grid(alpha=.25, lw=.4)
    ax[0].axvspan(CODA[0], CODA[1], color="#000000", alpha=.07, lw=0)

    # Three lines rather than two: the one-line form overflows the canvas at
    # this width and the ends of the title are cut off.
    ax[0].set_title(
        "%s\ntrue (black), predicted (red dashed), absolute difference in per "
        "cent of trace maximum (blue, right axis);\nshaded band is the coda "
        "evaluation window, %.3f to %.3f s" % (label, CODA[0], CODA[1]),
        fontsize=9)

    stats = []
    for i in range(n):
        a_ = ax[i + 1]
        yt, yp = Yt[m, i], Yp[m, i]
        den = np.max(np.abs(yt)) or 1.0
        d = 100 * np.abs(yp - yt) / den
        stats.append((hz[i], d.mean(), d.max()))

        b = a_.twinx()
        b.fill_between(t[m], 0, d, color="#2b7bba", alpha=.45, lw=0)
        b.set_ylim(0, max(20, d.max() * 1.25))
        b.set_ylabel("%", fontsize=6, color="#2b7bba")
        b.tick_params(labelsize=5, colors="#2b7bba")

        a_.axvspan(CODA[0], CODA[1], color="#000000", alpha=.07, lw=0)
        a_.plot(t[m], yt, "k-", lw=0.9)
        a_.plot(t[m], yp, "--", color="#c0392b", lw=1.0)
        a_.set_ylabel("%g Hz" % hz[i], fontsize=7)
        a_.tick_params(labelsize=6)
        a_.grid(alpha=.25, lw=.4)
        a_.set_zorder(b.get_zorder() + 1)
        a_.patch.set_visible(False)
        a_.text(0.995, 0.92, "mean |diff| = %.1f %%   max = %.1f %%"
                % (d.mean(), d.max()), transform=a_.transAxes,
                ha="right", va="top", fontsize=6, color="#2b7bba")

    ax[-1].set_xlabel("Time (s)", fontsize=8)
    ax[-1].set_xlim(0, TMAX)
    fig.tight_layout(h_pad=0.15)

    if not os.path.isdir(OUT):
        os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, "%s.%s" % (outfile, FMT))
    fig.savefig(dst, dpi=300)
    plt.close(fig)

    print("written:", dst)
    print("   from:", path)
    print("   %-8s %-14s %s" % ("band", "mean |diff| %", "max |diff| %"))
    for f_, mn, mx in stats:
        print("   %-8g %-14.1f %.1f" % (f_, mn, mx))
    print()
