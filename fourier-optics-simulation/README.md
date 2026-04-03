# Fourier Optics Simulation — Space vs Ground Telescope Imaging

Simulates and compares diffraction-limited (space-based) and seeing-limited (ground-based) telescope imaging using real Hubble Space Telescope data of the M51 galaxy.

## What it does

1. **Loads real HST data** — FITS file from the Hubble archive (M51, F814W band, 12200×8600 px)
2. **Crops a 2048×2048 ROI** — sized as a power of 2 to optimize FFT convolution
3. **Builds two PSFs:**
   - Space: Airy pattern from a circular aperture (2.4 m HST primary, λ = 0.814 µm)
   - Ground: Gaussian seeing disk (FWHM = 1.2 arcsec)
4. **Simulates imaging** via FFT-based 2D convolution: `I_obs = IFFT(FFT(I) × FFT(PSF))`
5. **Quantifies resolution loss** with radially-averaged power spectra — directly shows how atmospheric seeing suppresses high spatial frequencies

## Key results

| | Space (Airy) | Ground (Gaussian, 1.2") |
|---|---|---|
| PSF type | Diffraction-limited | Seeing-limited |
| Detail preserved | Spiral arm structure visible | Blurred, fine structure lost |

Figures in [`figures/`](./figures/):

| File | Description |
|---|---|
| `fig_1_originalM51.png` | Original Hubble image (log display) |
| `fig_2_roi.png` | Cropped 2048×2048 ROI |
| `fig_3_psf.png` | Space PSF (Airy) vs Ground PSF (Gaussian) |
| `fig_4_simulated.png` | Simulated space vs ground images |
| `fig_5_powerplot.png` | Radially-averaged power spectra comparison |

## How to run

Requires MATLAB with the Aerospace Toolbox (for `fitsread`).

```matlab
% Place opticsproject.m and h_m51_i_s05_drz_sci.fits in the same folder
run('opticsproject.m')
```

## Physics background

The observed image is modeled as a convolution of the true image with the telescope's PSF:

```
I_obs(x,y) = I_true(x,y) * PSF(x,y)
```

In the Fourier domain this becomes a multiplication, making FFT-based convolution efficient for large images. The space PSF is the Fraunhofer diffraction pattern of a circular aperture (Airy function involving a first-order Bessel function). The ground PSF approximates atmospheric seeing as a Gaussian blur.

## Data

The FITS file is not included in this repo due to its size (~325 MB). Download it from the Hubble Legacy Archive:

- Dataset: M51 galaxy, F814W filter (~0.814 µm)
- Filename: `h_m51_i_s05_drz_sci.fits`
- Place it in the same folder as `opticsproject.m` before running
