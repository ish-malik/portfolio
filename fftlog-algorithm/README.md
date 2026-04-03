# FFTLog Algorithm

Python implementation of the FFTLog algorithm for computing Fourier/Hankel transforms of logarithmically-spaced data.

## Background

Standard FFT assumes uniformly-spaced data. Many physics problems (cosmological power spectra, radial correlation functions, optics) naturally live on logarithmic grids spanning many decades. FFTLog handles this by working in log-space: it applies a standard DFT to log-resampled data, then uses a multiplier function to account for the change of variables and the properties of the target transform (Hankel transforms involve Bessel functions).

## What the notebook covers

1. **Log-spaced data generation** — generates `r_n = r_0 · exp(nL/N)` and adds Gaussian noise
2. **Forward DFT and inverse DFT** — applies FFT in log-space and validates reconstruction
3. **Multiplier function** — computes `u_m` using Bessel and Gamma functions to bias-correct the transform coefficients
4. **Hankel transform** — uses the modified coefficients to evaluate the full transform

## How to run

```bash
pip install numpy scipy matplotlib jupyter
jupyter notebook FFT_log_v1.ipynb
```

## Key result

`comparison_plot_FFTLog.png` — shows the FFTLog output vs the reference, demonstrating accurate reconstruction of the transform on a log grid.
