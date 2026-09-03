"""
Testy dla examples/prime_resonance_demo.py.

Regresja konkretnego błędu znalezionego samowalidacją (niezależna
reimplementacja protokołu w JS, uruchomiona w przeglądarce): pierwsza
wersja `_inject_even_digit_sum` (wtedy `_inject_even_bias`) wymuszała
tylko parzystość SAMEJ LICZBY (`* 2`), zakładając błędnie, że to da
parzystą sumę cyfr — realny test pokazał, że kontrola pozytywna miała
wtedy p≈0.22 (nie wykrywała efektu). Test niżej sprawdza wprost, na
twardych liczbach, że naprawiony wstrzykiwacz faktycznie gwarantuje
parzystą sumę cyfr — żeby ten dokładny błąd nie wrócił niezauważony.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

import prime_resonance_demo as demo  # noqa: E402


def _digit_sum(n: int) -> int:
    return sum(int(d) for d in str(int(n)))


def test_inject_even_digit_sum_guarantees_even_sum_for_every_value():
    values = demo._inject_even_digit_sum(500, seed=0)
    bad = [int(v) for v in values if _digit_sum(v) % 2 != 0]
    assert bad == [], f"wartości z nieparzystą sumą cyfr (nie powinno ich być): {bad[:10]}"


def test_inject_even_digit_sum_matches_metric_perfectly():
    values = demo._inject_even_digit_sum(500, seed=1)
    assert demo.digit_sum_even_fraction(values) == pytest.approx(1.0)


def test_old_even_number_trick_is_not_a_reliable_proxy():
    """Dokumentuje SAM błąd, nie tylko naprawę: mnożenie przez 2
    (parzysta LICZBA) nie daje wiarygodnie parzystej SUMY CYFR — to był
    źródłowy błąd w pierwszej wersji wstrzykiwacza."""
    rng = np.random.default_rng(0)
    naive_even_numbers = rng.integers(1, 500, size=2000) * 2
    fraction = demo.digit_sum_even_fraction(naive_even_numbers)
    # Rzeczywista frakcja jest bliska 0.5 (czyli praktycznie tłu) —
    # zdecydowanie nie bliska 1.0, jak naiwnie można by założyć.
    assert 0.3 < fraction < 0.7


def test_digit_sum_even_fraction_hand_computed():
    # 2->2 (parzysta), 11->1+1=2 (parzysta), 19->1+9=10 (parzysta),
    # 13->1+3=4 (parzysta), 21->2+1=3 (nieparzysta)
    values = np.array([2, 11, 19, 13, 21])
    assert demo.digit_sum_even_fraction(values) == pytest.approx(4 / 5)
