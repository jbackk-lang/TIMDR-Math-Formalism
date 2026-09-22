# tests/test_envelope_demodulation.py
"""Testy timdr_formalism.envelope_demodulation -- wydzielonego rdzenia
demodulacji obwiedniowej (patrz modul docstring po kontekst
historyczny/PREREG)."""
from __future__ import annotations

import numpy as np
import pytest

from timdr_formalism.envelope_demodulation import (
    bandpass_fft,
    hilbert_envelope,
    decimate_simple,
    kurtosis_excess,
    select_resonance_band,
    envelope_spectrum,
    envelope_spectrum_peak,
)


def test_bandpass_fft_isolates_target_frequency():
    fs = 1000.0
    n = 2000
    t = np.arange(n) / fs
    signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 300 * t)
    filtered = bandpass_fft(signal, fs, 40, 60)
    # energia przefiltrowanego sygnalu powinna byc bliska energii
    # samej skladowej 50Hz, nie calego sygnalu (ktory ma tez 300Hz)
    pure_50 = np.sin(2 * np.pi * 50 * t)
    assert np.corrcoef(filtered, pure_50)[0, 1] > 0.99


def test_bandpass_fft_removes_offband_frequency():
    fs = 1000.0
    n = 2000
    t = np.arange(n) / fs
    signal = np.sin(2 * np.pi * 300 * t)  # tylko poza pasmem
    filtered = bandpass_fft(signal, fs, 40, 60)
    assert np.std(filtered) < 0.01 * np.std(signal)


def test_hilbert_envelope_constant_amplitude_am_signal():
    fs = 2000.0
    n = 4000
    t = np.arange(n) / fs
    carrier = np.sin(2 * np.pi * 200 * t)
    envelope = hilbert_envelope(carrier)
    # obwiednia stalej-amplitudy nosnej powinna byc bliska stalej 1.0
    # (poza efektami brzegowymi)
    interior = envelope[50:-50]
    assert np.median(interior) == pytest.approx(1.0, abs=0.05)


def test_hilbert_envelope_tracks_am_modulation():
    fs = 2000.0
    n = 4000
    t = np.arange(n) / fs
    mod = 1.0 + 0.5 * np.sin(2 * np.pi * 5 * t)  # wolna modulacja amplitudy
    carrier = mod * np.sin(2 * np.pi * 200 * t)
    envelope = hilbert_envelope(carrier)
    interior = slice(100, -100)
    assert np.corrcoef(envelope[interior], mod[interior])[0, 1] > 0.9


def test_decimate_simple_reduces_length_and_preserves_mean():
    signal = np.arange(1000, dtype=float)
    decimated = decimate_simple(signal, 10)
    assert len(decimated) == 100
    assert np.mean(decimated) == pytest.approx(np.mean(signal), rel=1e-6)


def test_kurtosis_excess_gaussian_near_zero():
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 1, 200000)
    k = kurtosis_excess(signal)
    assert abs(k) < 0.1


def test_kurtosis_excess_impulsive_signal_positive():
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 1, 10000)
    # dodaj rzadkie duze impulsy -- impulsywny sygnal ma dodatnia
    # kurtoze nadmiarowa (grubsze ogony niz Gauss)
    idx = rng.choice(10000, size=20, replace=False)
    signal[idx] += rng.normal(0, 15, 20)
    k = kurtosis_excess(signal)
    assert k > 1.0


def test_select_resonance_band_picks_band_containing_impulsive_content():
    fs = 5000.0
    n = 10000
    rng = np.random.default_rng(0)
    t = np.arange(n) / fs
    # sygnal impulsywny skoncentrowany w pasmie 2000-2500Hz
    carrier = np.sin(2 * np.pi * 2200 * t)
    envelope_mod = np.zeros(n)
    for k in range(0, n, 500):
        envelope_mod[k : k + 20] = 5.0
    impulsive = carrier * (1 + envelope_mod) + rng.normal(0, 0.3, n)
    candidates = [(500, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3000)]
    band, kurt = select_resonance_band(impulsive, fs, candidates)
    assert band == (2000, 2500)


def test_envelope_spectrum_peak_detects_periodic_impulses():
    fs = 5000.0
    n = 10000
    rng = np.random.default_rng(1)
    t = np.arange(n) / fs
    f_target = 80.0
    f_res = 1800.0
    period = fs / f_target
    signal = rng.normal(0, 1, n)
    k = 0
    while k * period < n:
        idx = int(k * period)
        local_t = (np.arange(n) - idx) / fs
        mask = (local_t >= 0) & (local_t < 0.01)
        signal[mask] += 5.0 * np.exp(-local_t[mask] / 0.001) * np.sin(2 * np.pi * f_res * local_t[mask])
        k += 1

    resonance_band = (1500.0, 2000.0)
    peaks_signal = envelope_spectrum_peak(signal, fs, resonance_band, {"target": f_target})

    noise_only = rng.normal(0, 1, n)
    peaks_noise = envelope_spectrum_peak(noise_only, fs, resonance_band, {"target": f_target})

    assert peaks_signal["target"] > peaks_noise["target"]


def test_envelope_spectrum_peak_nan_when_no_bins_in_window():
    fs = 100.0
    n = 200
    signal = np.random.default_rng(0).normal(0, 1, n)
    peaks = envelope_spectrum_peak(signal, fs, (10, 20), {"way_too_high": 1e6}, window_hz=0.001)
    assert np.isnan(peaks["way_too_high"])
