# Helmholtz Resonator Analysis

Real-time acoustic analysis tool for measuring Helmholtz resonance. Records live microphone input, computes a spectrogram and power spectral density (PSD), and identifies the peak resonant frequency. Built for an acoustics course project.

## What it does

1. **Records live audio** from microphone input (auto-detects a supported sample rate)
2. **Computes a spectrogram** using short-time FFT (window = 2^10, nfft = 2^14)
3. **Interactive time window selection** — click two points on the spectrogram to isolate the resonance event
4. **Plots PSD** of the selected segment and reports the peak frequency in Hz

## How to run

**Python version:**

```bash
pip install numpy scipy matplotlib sounddevice
python acoustic_demo.py
```

When prompted, blow across or tap the resonator opening, then press Enter to stop recording. Click two points on the spectrogram to select your window.

**MATLAB version:**

```matlab
run('Acoustic_demo.m')
```

## Physics background

A Helmholtz resonator is a cavity with a narrow neck. Its resonant frequency is:

```
f = (c / 2π) · sqrt(A / (V · L_eff))
```

where `c` is the speed of sound, `A` is the neck cross-section, `V` is the cavity volume, and `L_eff` is the effective neck length (includes an end correction). This project measures `f` experimentally via spectral analysis and compares it to the theoretical prediction.

## Signal processing notes

- Spectrogram uses `nperseg = 2^10`, `noverlap = 2^10 - 2^8`, `nfft = 2^14` for high frequency resolution
- PSD is smoothed over 5 Hz before peak detection
- Window and nfft sizes are powers of 2 for FFT efficiency
