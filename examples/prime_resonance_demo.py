"""
examples/prime_resonance_demo.py

Przykład użycia pełnego protokołu (docs/PROTOCOL.md) na hipotezie w
stylu opisanym w zgłoszeniu: "mam ciąg, który ma specjalny rezonans z
liczbami pierwszymi".

WAŻNE: ten skrypt NIE zakłada z góry wyniku. Metryka użyta tutaj
(fracja elementów, których suma cyfr jest parzysta) jest celowo prostym,
dobrze zdefiniowanym przykładem — typu naiwnej numerologicznej hipotezy,
jaką ktoś mógłby zaproponować. Uruchom skrypt i przeczytaj wydrukowany
raport, zamiast zakładać wynik z tego komentarza (dokładnie to, przed
czym ostrzega krok 6 protokołu i skill: nie interpretuj wyniku, zanim go
nie zobaczysz).

Uruchomienie:
    python examples/prime_resonance_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from timdr_formalism.pipeline import (  # noqa: E402
    Hypothesis,
    Preregistration,
    mann_whitney_test,
    run_controls,
    format_report,
    sieve_of_eratosthenes,
    random_background,
)


def digit_sum_even_fraction(values: np.ndarray) -> float:
    """Metryka testowana: frakcja elementów, których suma cyfr
    (dziesiętnie) jest parzysta. Prosta, jednoznaczna, łatwa do
    ręcznego zweryfikowania — dobra pierwsza metryka do przetestowania
    naiwnej hipotezy "X ma specjalny związek z liczbami pierwszymi"."""
    values = np.asarray(values, dtype=np.int64)
    digit_sums = np.array([sum(int(d) for d in str(int(v))) for v in values])
    return float(np.mean(digit_sums % 2 == 0))


def _inject_even_digit_sum(window_size: int, seed) -> np.ndarray:
    """Kontrola pozytywna: liczby SKONSTRUOWANE tak, żeby mieć NAPRAWDĘ
    parzystą sumę cyfr — nie tylko parzystą wartość.

    POPRAWKA (znaleziona samowalidacją tego repo — niezależna
    reimplementacja tego samego protokołu w JS, uruchomiona w
    przeglądarce, żeby faktycznie WYKONAĆ kontrolę, a nie tylko ją
    przeczytać). Pierwsza wersja wymuszała tylko parzystość SAMEJ
    LICZBY (`rng.integers(1, 500, size=window_size) * 2`), po cichu
    zakładając, że to przełoży się na parzystą sumę cyfr. To błędne
    założenie: parzystość liczby zależy wyłącznie od ostatniej cyfry, a
    suma cyfr od WSZYSTKICH cyfr — związek między nimi jest słaby.
    Realny test na tym wstrzykiwaczu: kontrola pozytywna dawała
    p≈0.22 (metryka NIE wykrywała efektu, który miała wykryć), więc
    bramka `run_controls` poprawnie by to odrzuciła jako niewiarygodną
    kalibrację. Naprawiono przez jawne skonstruowanie liczby z trzech
    cyfr (a, b, c), gdzie c jest dobierane tak, żeby a+b+c było
    parzyste — to GWARANTUJE efekt zamiast na niego liczyć."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 10, size=window_size)
    b = rng.integers(0, 10, size=window_size)
    ab_even = (a + b) % 2 == 0
    even_digits = np.array([0, 2, 4, 6, 8])
    odd_digits = np.array([1, 3, 5, 7, 9])
    c = np.where(
        ab_even,
        even_digits[rng.integers(0, 5, size=window_size)],
        odd_digits[rng.integers(0, 5, size=window_size)],
    )
    return 100 * a + 10 * b + c


def _random_control(window_size: int, seed) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(1, 1000, size=window_size)


def main() -> None:
    # --- Krok 1: pre-rejestracja ---
    hypothesis = Hypothesis(
        name="prime_digit_sum_parity_demo",
        description=(
            "Liczby pierwsze mają inną frakcję elementów o parzystej "
            "sumie cyfr niż losowe tło liczb całkowitych w tym samym "
            "zakresie."
        ),
        effect_description=(
            "różnica we frakcji liczb o parzystej sumie cyfr między "
            "liczbami pierwszymi <= N a losowym tłem z tego samego zakresu"
        ),
    )
    params = {"n_max": 100_000, "seed": 42}
    prereg = Preregistration.create(hypothesis, params)
    print(f"Pre-rejestracja: {prereg.fingerprint[:16]}... (krok 1 — zamrożone PRZED wynikiem)\n")

    # --- Krok 5: kontrola pozytywna/negatywna, PRZED danymi głównymi ---
    controls = run_controls(
        metric_fn=digit_sum_even_fraction,
        positive_injector=_inject_even_digit_sum,
        negative_generator_a=_random_control,
        negative_generator_b=_random_control,
        n_windows=25,
        window_size=200,
        seed=params["seed"],
    )

    if not controls.passed:
        print(format_report(hypothesis, controls, main_result=None))
        return

    # --- Krok 2+3: dane główne + metryka ---
    primes = sieve_of_eratosthenes(params["n_max"])
    background = random_background(
        n=len(primes), low=2, high=params["n_max"], seed=params["seed"]
    )

    # Metryka liczona per-okno (żeby test U miał więcej niż 1 punkt na
    # grupę) — dziel obie próbki na okna o stałym rozmiarze.
    window = 200
    n_windows = min(len(primes), len(background)) // window

    prime_windows = [
        digit_sum_even_fraction(primes[i * window : (i + 1) * window])
        for i in range(n_windows)
    ]
    background_windows = [
        digit_sum_even_fraction(background[i * window : (i + 1) * window])
        for i in range(n_windows)
    ]

    # --- Krok 4: test statystyczny ---
    main_result = mann_whitney_test(prime_windows, background_windows)

    # --- Krok 6: raport / werdykt ---
    print(format_report(hypothesis, controls, main_result, n_comparisons=1))


if __name__ == "__main__":
    main()
