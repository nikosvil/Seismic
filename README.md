# Progressive multi-output 1-D CNN for low frequency seismic trace reconstruction

This repository contains the code for the paper:

> N. Vilanakis, N. Economou and A. Vafidis, *A progressive multi-output 1-D
> convolutional network for low frequency seismic trace reconstruction*,
> submitted to Computers & Geosciences.

The network reconstructs ten low frequency traces, from 5500 Hz down to 500 Hz,
from a single high frequency trace recorded at 6400 Hz. A shared encoder feeds
ten convolutional heads. Every head below the first also receives a compact
summary of the predictions already obtained at the higher frequencies, which we
call inter-stage conditioning.

For the software and hardware needed, please read [REQUIREMENTS.md](REQUIREMENTS.md).

## What is here, and what is not

The repository contains **code only**. The trained networks and the generated
datasets are not uploaded, because they are too large for a git repository: one
trained model is about 55 MB and one dataset is about 3.5 GB.

This is not a loss of reproducibility. The datasets are produced by the
generator scripts included here, and every run uses a fixed random seed,
`rng(42, 'twister')`. Running the generator and then the training script
reproduces the models of the paper.

## The seven models

The paper compares seven trained models. Each one has its own folder.

| Folder | Wavelet | Conditioning | Head receptive field | Where it appears in the paper |
| --- | --- | --- | --- | --- |
| `UHRS/Progressive/progressive` | UHRS, symmetric | yes | 7.96 ms | Tables 2, 3, 5 |
| `UHRS/Progressive/progressive_matchedRF` | UHRS, symmetric | yes | 13.90 ms | Tables 1, 2, 4, 5; Figs. 3, 6 |
| `UHRS/Multihead/multihead` | UHRS, symmetric | no | 7.96 ms | Tables 2, 5 |
| `UHRS/Multihead/multihead_matchedRF` | UHRS, symmetric | no | 13.90 ms | Tables 2, 4, 5; Figs. 3, 7 |
| `UHRS/Initial_wavelet` | UHRS, initial | yes | 7.96 ms | Table 3 |
| `Ricker/Progressive` | Ricker | yes | 7.96 ms | Table 1; Figs. 2, 5 |
| `Ricker/Multihead` | Ricker | no | 7.96 ms | Fig. 2, Section 3.2 |

The four UHRS models with the symmetric wavelet form the two-by-two design of
Section 3.2: conditioning present or absent, receptive field 7.96 or 13.90 ms.

"Conditioning" means the quota gates, `C_quota = 8` with `detachUpTo = 3`. Where
it is absent, `C_quota = 0` and there is no gradient detachment. The wider
receptive field is obtained by dilating the two head convolutions by a factor of
4, which does not change the number of parameters.

## Repository layout

```
common/            wavelet and reflectivity generators, train/val/test split
Ricker/            the two Ricker models
UHRS/              the five UHRS models
inference/         in-domain and finite difference inference scripts
robustness/        the wavelet, noise and out-of-distribution test suite
ablations/         the four further sweeps of Section 3.3
fd_modelling/      1-D finite difference forward code and velocity model
figures/           scripts that draw the figures of the paper
```

## How to reproduce a result

The steps below are for one model. Repeat them for the model you need.

**1. Generate the wavelets and the training data.**
Run the generator in `common/` for the wavelet family you want. This writes the
reflectivity series, convolves them with the band wavelets and saves the result.

**2. Split into training, validation and test sets.**
Run `common/split_train_val_test.m`. The split is by reflectivity group, so no
reflectivity series appears in more than one set. The result is 22 500 training,
3 000 validation and 4 500 test traces.

**3. Train.**
Run the `train_*.m` script inside the model folder. It writes the trained
network as a `.mat` file and a text file with the training history, which
contains the per-stage validation loss at every fifth epoch.

**4. Evaluate on the test set.**
Run the matching `infer_*_multioutput.m` script in `inference/`. This produces
the per-band RMSE and the per-trace correlations of Tables 1 and 2.

**5. Evaluate on the finite difference record.**
Run the matching `infer_real_data_*.m` script. This produces Tables 4 and 5 and
Figs. 5, 6 and 7.

Please note the normalization rule described in Section 2.4 of the paper: at
deployment the whole record must be scaled by **one common factor** for all
eleven traces. If each trace is normalized separately, the amplitude relations
between the bands are destroyed and the predicted amplitudes will be wrong. The
inference scripts already do this correctly.

**6. Robustness tests (optional).**
Run the scripts in `robustness/`. They cover the wavelet perturbations, the
noise levels and the out-of-distribution reflectivity of Section 3.4, and they
write a summary text file with the composite robustness index.

**7. Figures.**
`figures/make_overlay_figures.py` draws Figs. 2 and 3. The finite difference
figures are drawn by the inference scripts.

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
