# A progressive multi-output 1-D convolutional network for lower-frequency trace reconstruction of UHRS seismic data

This repository contains the code for the paper:

> N. Vilanakis, N. Economou and A. Vafidis, *A progressive multi-output 1-D convolutional network for lower-frequency trace reconstruction of UHRS seismic data*,
> submitted to Computers & Geosciences.

The network reconstructs ten low frequency traces, from 5500 Hz down to 500 Hz,
from a single high frequency trace recorded at 6400 Hz. A shared encoder feeds
ten convolutional heads. Every head below the first also receives a compact
summary of the predictions already obtained at the higher frequencies, which we
call inter-stage conditioning.

For the software and hardware needed, please read [REQUIREMENTS.md](REQUIREMENTS.md).

## What is here

The repository contains **code only**. The trained networks and the generated
datasets are not uploaded as they are too large for a git repository: one
trained model is about 55 MB and one dataset is about 3.5 GB.

The datasets are produced by the generator scripts included here, and every run uses a fixed random seed,
`rng(42, 'twister')`. Running the generator and then the training script reproduces the models of the paper.

## The six models

The paper compares six trained models, each with its own training script.

| Training script | Wavelet | Conditioning | Head receptive field | Where it appears in the paper |
| --- | --- | --- | --- | --- |
| `UHRS/Progressive/progressive` | UHRS | yes | 7.96 ms | Tables 2, 4 |
| `UHRS/Progressive/progressive_matchedRF` | UHRS | yes | 13.90 ms | Tables 1, 2, 3, 4; Figs. 3, 6 |
| `UHRS/Multihead/multihead` | UHRS | no | 7.96 ms | Tables 2, 4 |
| `UHRS/Multihead/multihead_matchedRF` | UHRS | no | 13.90 ms | Tables 2, 3, 4; Figs. 3, 7 |
| `Ricker/Progressive` | Ricker | yes | 7.96 ms | Table 1; Figs. 2, 5 |
| `Ricker/Multihead` | Ricker | no | 7.96 ms | Fig. 2, Section 3.2 |

The four UHRS models form the two-by-two design of Section 3.2: conditioning
present or absent, receptive field 7.96 or 13.90 ms.

"Conditioning" means the quota gates, `C_quota = 8` with `detachUpTo = 3`. Where
it is absent, `C_quota = 0` and there is no gradient detachment. The wider
receptive field is obtained by dilating the two head convolutions by a factor of
4, which does not change the number of parameters.

## Repository layout

```
common/            reflectivity and wavelet generators, train/val/test split
  reflectivity_generator_Ricker.m          Ricker family
  reflectivity_generator_UHRS_symmetric.m  UHRS family
  split_train_val_test_Ricker.m
  split_train_val_test_UHRS.m
Ricker/            the two Ricker models
UHRS/              the four UHRS models of the two-by-two design
inference/         in-domain and finite difference inference scripts,
                   and phase_metrics.py for the timing diagnostics
robustness/        the wavelet, noise and out-of-distribution test suite
ablations/         Pack 1, the robustness comparison of the progressive model
                   against the multi-head baseline behind Section 3.2, and the
                   four further sweeps of Section 3.3 (Packs 2 to 5),
                   summarised in the supplementary material of the paper
figures/           scripts that draw Figs. 2, 3, 5, 6 and 7
```

The two split scripts are not interchangeable: each one knows the variable
names its own generator writes, and each reads the file its own generator
produces, `dataset_multi_freq_Ricker.mat` or `dataset_multi_freq_UHRS.mat`.

## How to reproduce a result

The steps below are for one model. Repeat them for the model you need.

**1. Generate the wavelets and the training data.**
Run the generator in `common/` for the wavelet family you want. This writes the
reflectivity series, convolves them with the band wavelets and saves the result.

**2. Split into training, validation and test sets.**
Run `common/split_train_val_test_Ricker.m` or `common/split_train_val_test_UHRS.m`,
depending on the family. The split is by reflectivity group, so no reflectivity
series appears in more than one set. The result is 22 500 training, 3 000
validation and 4 500 test traces.

**3. Train.**
Run the `train_*.m` script inside the model folder. It writes the trained
network as a `.mat` file and a text file with the training history, which
contains the per-stage validation loss at every fifth epoch.

**4. Evaluate on the test set.**
Run the matching `infer_*_multioutput.m` script in `inference/`. This produces
the per-band RMSE and the per-trace correlations of Tables 1 and 2.

**5. Evaluate on the finite difference record.**
Run the matching `infer_real_data_*.m` script. This produces Tables 3 and 4 and
Figs. 5, 6 and 7. The record itself, `basepingkmat_unno.mat`, is a `.mat` file
and is therefore not in this repository. How it was made is described in the
next section.

Please note the normalization rule described in Section 2.4 of the paper: at
deployment the whole record must be scaled by **one common factor** for all
eleven traces. If each trace is normalized separately, the amplitude relations
between the bands are destroyed and the predicted amplitudes will be wrong. The
inference scripts already do this correctly.

**6. Phase diagnostics (optional).**
`inference/phase_metrics.py` recomputes the lag, the instantaneous phase
difference and the RRMSE before and after a single scalar rescaling, which
are the quantities quoted in Sections 2.8, 4.3 and 5. It reads the
`predictions_real_data_*.mat` files written in step 5.

**7. Robustness tests (optional).**
Run the scripts in `robustness/`. They cover the wavelet perturbations, the
noise levels and the out-of-distribution reflectivity of Section 3.4, and they
write a summary text file with the composite robustness index.

**8. Figures.**
`figures/make_overlay_figures.py` draws Figs. 2 and 3, the in-domain test-trace
overlays. `figures/make_fd_figures.py` draws Figs. 5, 6 and 7, the cross-physics
reconstructions. Both read the `.mat` files written by the inference scripts, so
run step 4 or step 5 first; set `DATA_ROOT` at the top of each script to the
folder that holds the `Ricker Final` and `UHRS Final` trees. `FIG_OUT` chooses
where the figures are written and `FIG_FORMAT` chooses the format: `png` by
default, or `pdf` and `eps` for vector output, which is what a publisher asks
for with line art.

The inference scripts themselves also write a plain figure of predictions
against ground truth, as a quick check while running. Those are not the figures
of the paper.

Fig. 1 is a drawn diagram and Fig. 4 is produced by the finite difference
forward code, so neither is generated by this repository.

## The finite difference record

The cross-physics test of Section 4 uses a record produced by a 1-D acoustic
finite difference simulation, with fourth-order staggered spatial derivatives
and fourth-order Adams-Bashforth time integration.

That forward code is not redistributed here. It is the work of Florian Wittkamp
and is released under the GNU General Public License v3. It is described in -and is also available from-
the reference cited in Section 4.1 of the paper:

> Bohlen, T. and Wittkamp, F., 2016. Three-dimensional viscoelastic time-domain
> finite-difference seismic modelling using the staggered Adams-Bashforth time
> integrator. Geophysical Journal International, 204 (3), 1781-1788.

The simulation used for the paper has a source and a receiver at the same grid
point, a pressure-free boundary at both ends of the 20 m model, dt = 5.5339e-07 s
and a total time of 0.018 s, resampled afterwards to dt = 1e-5 s and zero-padded
to 2001 samples. The velocity model is the one shown in Fig. 4 of the paper.
Anyone who wishes to repeat the test can reproduce the record from that
description together with the reference above.

## A note on the receptive field

The receptive field is the length of input trace that one output sample can
see. It follows from the kernel lengths:

- encoder: three convolutions of length 200, so 1 + 3 x 199 = 598 samples
- head: two convolutions of length 100, so 1 + 2 x 99 = 199 samples
- encoder and head together: 796 samples, which is 7.96 ms at dt = 10 us
- with the head convolutions dilated by 4: 1390 samples, which is 13.90 ms

The UHRS wavelet at 500 Hz is 1000 samples long, that is 10.0 ms. A receptive
field of 7.96 ms therefore does not cover it, and 13.90 ms does. This is the
main result of Section 3.2.

## Citation

If you use this code, please cite the paper:

```
@article{vilanakis2026progressive,
  author  = {Vilanakis, Nikos and Economou, Nikos and Vafidis, Antonis},
  title   = {A progressive multi-output 1-D convolutional network for low
             frequency seismic trace reconstruction},
  journal = {Computers \& Geosciences},
  year    = {2026},
  note    = {submitted}
}
```

## Acknowledgements

The training runs were performed on the ARIS national HPC facility, with
computational time granted by the Greek Research and Technology Network
(GRNET) under project ID OperALFR.

## License

MIT. See [LICENSE](LICENSE).
