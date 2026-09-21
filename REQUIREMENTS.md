# Requirements

## MATLAB

The training, inference and robustness code is written in MATLAB.

- **Tested on MATLAB R2025b**, on Linux (the GPU runs were performed on the ARIS
  national HPC facility) and on Windows 11 for the smaller runs.
- Earlier releases have not been tested. The newest functions used are
  `convolution1dLayer` and `preluLayer`, so a release older than R2022b will
  not run the code at all. Between R2022b and R2025b the code is expected to
  work, but we have not verified it.

### Toolboxes

| Toolbox | Needed for | Required? |
| --- | --- | --- |
| Deep Learning Toolbox | `dlnetwork`, `dlarray`, `dlfeval`, `dlgradient`, `adamupdate`, `convolution1dLayer`, `batchNormalizationLayer`, `preluLayer`, `dropoutLayer` | **Yes.** Nothing runs without it. |
| Parallel Computing Toolbox | `gpuArray`, `canUseGPU` | Optional. The scripts check for a GPU and fall back to the CPU if none is found. Training on the CPU is possible but very slow. |
| Signal Processing Toolbox | `hilbert`, used by the wavelet perturbation tests | Only for the robustness suite. Training and inference do not need it. |

No other toolbox is used. All remaining functions are part of base MATLAB.

## Python

Python is used only to draw two of the figures. It is not needed to train or
to evaluate the models.

- Python 3.9 or newer
- `numpy`
- `h5py` (to read MATLAB v7.3 `.mat` files)
- `matplotlib`

```
pip install numpy h5py matplotlib
```

## Hardware and running time

The models in the paper were trained on one NVIDIA A100-SXM4-80GB GPU.

- One training run takes about 16 to 18 hours for 300 epochs, with 22 500
  training traces and a mini-batch size of 256.
- Peak GPU memory is well below 80 GB. A card with 16 GB is enough if the
  mini-batch size is reduced, but note that the mini-batch size changes the
  trained model, so any comparison must keep it fixed. This is explained in
  Section 2.6 of the paper.
- Inference and the robustness suite run in minutes and do not need a GPU.

## Disk space

The generated dataset is large, because ten target bands are stored for every
trace at 2001 samples in single precision:

- `dataset_train_val_test_UHS.mat` is about 3.5 GB
- the equivalent Ricker dataset is of the same order

Please make sure the disk has room before running the generator.
