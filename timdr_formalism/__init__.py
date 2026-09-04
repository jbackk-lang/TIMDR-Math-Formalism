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
)
from .chronosignal import (
    tempo,
    drift,
    anomalia_flags,
    defekt_flags,
    skret_flags,
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
    "tempo",
    "drift",
    "anomalia_flags",
    "defekt_flags",
    "skret_flags",
]
