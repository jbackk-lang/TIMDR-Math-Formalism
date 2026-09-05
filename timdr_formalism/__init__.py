"""TIMDR-Math-Formalism — detektor matematycznej sensowności.

Testuje, czy zaproponowana struktura (wzór, "rezonans", konstrukcja
kategorii, geometryczny pattern) ma realne, statystycznie istotne
poparcie w danych, czy jest tylko ładnie wyglądającym dopasowaniem bez
dowodu (numerologią). Patrz docs/PROTOCOL.md.
"""

from .pipeline import (
    Hypothesis,
    Preregistration,
    TestResult,
    ControlResult,
    mann_whitney_test,
    run_controls,
    bonferroni_correct,
    format_report,
    sieve_of_eratosthenes,
    random_background,
    ar1_noise,
    rank_biserial_effect_size,
    effect_size_label,
    calibrate_resonance_threshold,
)
from .chronosignal import (
    tempo,
    drift,
    anomalia_flags,
    defekt_flags,
    skret_flags,
)
from .calibration import (
    CalibrationResult,
    calibrate_resonance_K,
    theoretical_independence_baseline_rate,
    compute_anomaly_flags,
    load_parameter_windows_from_csv,
    load_krakow_weather_window,
    P_ANOMALY_2SIGMA_NORMAL,
)

__all__ = [
    "Hypothesis",
    "Preregistration",
    "TestResult",
    "ControlResult",
    "mann_whitney_test",
    "run_controls",
    "bonferroni_correct",
    "format_report",
    "sieve_of_eratosthenes",
    "random_background",
    "ar1_noise",
    "rank_biserial_effect_size",
    "effect_size_label",
    "calibrate_resonance_threshold",
    "tempo",
    "drift",
    "anomalia_flags",
    "defekt_flags",
    "skret_flags",
    "CalibrationResult",
    "calibrate_resonance_K",
    "theoretical_independence_baseline_rate",
    "compute_anomaly_flags",
    "load_parameter_windows_from_csv",
    "load_krakow_weather_window",
    "P_ANOMALY_2SIGMA_NORMAL",
]
