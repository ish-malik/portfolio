"""
Acoustic Demo - Helmholtz Resonance Analysis
A demo of Helmholtz resonance for acoustic class

This script records sounds from a microphone and subsequently plots
the spectrogram and spectrum, allowing analysis of resonant frequencies.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import uniform_filter1d
import sounddevice as sd


def powerval_gt(v):
    """
    Find the smallest power of 2 that is >= v.
    Equivalent to 2^ceil(log2(v))
    """
    ind = 0
    pow_val = 2 ** ind
    while pow_val < v:
        ind += 1
        pow_val = 2 ** ind
    return pow_val


def powerval_lt(v):
    """
    Find the largest power of 2 that is < v.
    """
    ind = 0
    pow_val = 1
    while 2 ** ind < v:
        pow_val = 2 ** ind
        ind += 1
    return pow_val


def get_supported_samplerate():
    """
    Find a supported sample rate for the default input device.
    Returns the first working sample rate from common options.
    """
    # Common sample rates to try (in order of preference)
    candidates = [44100, 48000, 22050, 16000, 8000, 96000]

    device_info = sd.query_devices(kind='input')
    default_sr = device_info['default_samplerate']

    # Try default first
    candidates.insert(0, int(default_sr))

    for sr in candidates:
        try:
            sd.check_input_settings(samplerate=sr)
            return sr
        except Exception:
            continue

    raise RuntimeError("Could not find a supported sample rate for your microphone")


def acoustic_demo():
    """
    Record sound from microphone and analyze with spectrogram and spectrum.

    Returns:
        tuple: (peak_frequency_hz, peak_power_db)
    """
    # Get a supported sample rate
    fs = get_supported_samplerate()
    print(f"Using sample rate: {fs} Hz")

    # Configuration
    flim = (0, 4000)  # Frequency axis limits for display (Hz)
    fsm = 5  # Smooth PSD over this many Hz

    # Record signal
    print("Recording... Press Enter to stop.")
    recording = []

    def callback(indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        recording.append(indata.copy())

    # Start recording stream
    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        input()  # Wait for Enter key

    # Concatenate recorded chunks
    sig = np.concatenate(recording, axis=0).flatten()
    taxis = np.arange(len(sig)) / fs

    print(f"Recorded {len(sig) / fs:.2f} seconds of audio")

    # Compute spectrogram
    # Parameters: window=2^10, overlap=2^10-2^8, nfft=2^14
    nperseg = 2 ** 10
    noverlap = 2 ** 10 - 2 ** 8
    nfft = 2 ** 14

    f, t, Sxx = signal.spectrogram(sig, fs=fs, nperseg=nperseg,
                                   noverlap=noverlap, nfft=nfft)

    # Filter to frequency limits
    freq_mask = (f >= flim[0]) & (f <= flim[1])
    f = f[freq_mask]
    Sxx = Sxx[freq_mask, :]

    # Plot initial spectrogram
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    fig1.patch.set_facecolor('white')

    # Convert to dB
    Sxx_dB = 10 * np.log10(Sxx + 1e-12)  # Add small value to avoid log(0)

    im = ax1.imshow(Sxx_dB, aspect='auto', origin='lower',
                    extent=[taxis[0], taxis[-1], f[0], f[-1]],
                    cmap='viridis')
    ax1.set_xlabel('Time (s)', fontsize=14)
    ax1.set_ylabel('Frequency (Hz)', fontsize=14)
    ax1.set_title('Spectrogram of Recorded Signal', fontsize=18)
    ax1.tick_params(labelsize=12)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax1, label='Power (dB)')
    plt.tight_layout()
    plt.show(block=False)

    # Select time window by mouse clicks
    print("\nClick two points on the spectrogram to select a time window...")
    points = plt.ginput(2, timeout=0)

    if len(points) < 2:
        print("Selection cancelled. Using full signal.")
        sig2 = sig
    else:
        xa, _ = points[0]
        xb, _ = points[1]

        # Ensure xa < xb
        if xa > xb:
            xa, xb = xb, xa

        # Extract selected time window
        idx_start = max(0, int(xa * fs))
        idx_end = min(len(sig), int(xb * fs))
        sig2 = sig[idx_start:idx_end]
        print(f"Selected time window: {xa:.2f}s to {xb:.2f}s")

    plt.close(fig1)

    # Compute spectrogram of selected segment
    f2, t2, Sxx2 = signal.spectrogram(sig2, fs=fs, nperseg=nperseg,
                                      noverlap=noverlap, nfft=nfft)

    freq_mask2 = (f2 >= flim[0]) & (f2 <= flim[1])
    f2 = f2[freq_mask2]
    Sxx2 = Sxx2[freq_mask2, :]

    # Create figure with spectrogram and spectrum
    fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 10))
    fig2.patch.set_facecolor('white')

    # Plot spectrogram of selected segment
    Sxx2_dB = 10 * np.log10(Sxx2 + 1e-12)
    im2 = ax2.imshow(Sxx2_dB, aspect='auto', origin='lower',
                     extent=[0, len(sig2) / fs, f2[0], f2[-1]],
                     cmap='viridis', vmin=-80, vmax=-40)
    ax2.set_xlabel('Time (s)', fontsize=14)
    ax2.set_ylabel('Frequency (Hz)', fontsize=14)
    ax2.set_title('Spectrogram of Selected Signal', fontsize=18)
    ax2.tick_params(labelsize=12)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(im2, ax=ax2, label='Power (dB)')

    # Compute and plot spectrum (PSD)
    # FFT of the selected signal
    fft_result = np.fft.fftshift(np.fft.fft(sig2))
    psd_dB = 20 * np.log10(np.abs(fft_result) + 1e-12)

    # Smooth the PSD
    smooth_samples = int(len(sig2) / fs * fsm)
    if smooth_samples > 1:
        psd_dB = uniform_filter1d(psd_dB, size=smooth_samples, mode='reflect')

    # Create frequency axis
    faxis2 = np.linspace(-fs / 2, fs / 2, len(psd_dB))

    # Normalize PSD
    psd_dB_normalized = psd_dB - np.max(psd_dB)

    # Plot spectrum
    ax3.plot(faxis2, psd_dB_normalized, linewidth=1)
    ax3.set_xlim(-flim[1], flim[1])
    ax3.set_xlabel('Frequency (Hz)', fontsize=14)
    ax3.set_ylabel('Normalized PSD (dB)', fontsize=14)
    ax3.set_title('Power Spectral Density', fontsize=18)
    ax3.tick_params(labelsize=12)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Find peak frequency
    peak_idx = np.argmax(psd_dB_normalized)
    peak_freq = abs(faxis2[peak_idx])
    peak_power = psd_dB_normalized[peak_idx]

    print(f"\n{peak_freq:.2f} Hz maximum frequency, and {peak_power:.2f} dB re 1 unit maximum value")

    return peak_freq, peak_power


if __name__ == "__main__":
    print("=" * 60)
    print("Acoustic Demo - Helmholtz Resonance Analysis")
    print("=" * 60)
    print()

    # Print available devices for debugging
    print("Available audio devices:")
    print(sd.query_devices())
    print()

    try:
        result = acoustic_demo()
        print(f"\nAnalysis complete. Peak frequency: {result[0]:.2f} Hz")
    except KeyboardInterrupt:
        print("\nRecording cancelled.")
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have a microphone connected and sounddevice installed.")
        print("Install with: pip install sounddevice numpy scipy matplotlib")
