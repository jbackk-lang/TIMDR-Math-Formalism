"""
Testy timdr_formalism.pipeline.paired_wilcoxon_test /
_wilcoxon_signed_rank_z_p_numpy.

Kontekst (patrz obszerna uwaga w pipeline.py nad tą sekcją kodu):
podczas testowania mostu "membrana" na realnych danych PROTECT-90
(TIMDR-Grid-Monitor/real_protect90_membrane_bridge.py) znaleziono, że
`scipy.stats.wilcoxon(alternative="two-sided", method="approx")` zwraca
`zstatistic` przepuszczony przez `z = -abs(z)` -- dla testu
dwustronnego znak jest WIĘC ZAWSZE ujemny, niezależnie od faktycznego
kierunku efektu. `test_paired_wilcoxon_positive_shift_gives_positive_z`
i `test_paired_wilcoxon_negative_shift_gives_negative_z` poniżej to
DOKŁADNIE testy regresyjne na ten bug -- gdyby ktoś kiedyś przypadkiem
cofnął naprawę (np. zastąpił `_wilcoxon_signed_rank_z_p_numpy` gołym
`scipy.stats.wilcoxon(...).zstatistic`), test (b) zacząłby fałszywie
przechodzić (bo scipy zawsze daje ujemny znak), ale test (a) MUSIAŁBY
zacząć failować (dodatni, jednoznaczny efekt dawałby ujemny z).

Zasada projektowa (ta sama co test_pipeline.py): unikać asercji
opartych na "typowym" zachowaniu losowego seeda -- dane tu mają dużą,
jednoznaczną separację (stałe przesunięcie o +-2.0), więc wynik jest
zdeterminowany przez konstrukcję, nie przez szczęśliwe losowanie.
"""
import math

import numpy as np
import pytest

from timdr_formalism import pipeline
from timdr_formalism.pipeline import (
    TestResult,
    paired_wilcoxon_test,
    _wilcoxon_signed_rank_z_p_numpy,
)

try:
    from scipy import stats as scipy_stats
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    scipy_stats = None
    _HAS_SCIPY = False


# ---------------------------------------------------------------------
# (a)/(b) Testy regresyjne na bug scipy z=-abs(z) -- odtwarzają
# dokładnie odkrycie z sesji PROTECT-90.
# ---------------------------------------------------------------------

def test_paired_wilcoxon_positive_shift_gives_positive_z_and_r():
    # event = baseline + 2.0 -- jednoznacznie dodatni, stały efekt.
    rng = np.random.default_rng(0)
    baseline = rng.normal(loc=0.0, scale=1.0, size=40)
    event = baseline + 2.0

    result = paired_wilcoxon_test(event, baseline)

    assert result.inconclusive is False
    assert result.statistic > 0  # zstatistic MUSI byc dodatni (kierunek efektu)
    assert result.effect_size_r > 0
    assert result.pvalue < 1e-6  # separacja calkowita (kazda para rozna)
    assert result.median_test > result.median_background


def test_paired_wilcoxon_negative_shift_gives_negative_z_and_r():
    # event = baseline - 2.0 -- odwrotny przypadek, znak MUSI sie odwrocic.
    rng = np.random.default_rng(0)
    baseline = rng.normal(loc=0.0, scale=1.0, size=40)
    event = baseline - 2.0

    result = paired_wilcoxon_test(event, baseline)

    assert result.inconclusive is False
    assert result.statistic < 0
    assert result.effect_size_r < 0
    assert result.pvalue < 1e-6
    assert result.median_test < result.median_background


# ---------------------------------------------------------------------
# (c) Zgodnosc p-wartosci z bezposrednim wywolaniem scipy.stats.wilcoxon
# (to jest ten sam cross-check, ktory paired_wilcoxon_test robi
# WEWNETRZNIE -- tu sprawdzamy go niezaleznie, z zewnatrz).
# ---------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_SCIPY, reason="scipy niedostepne w tym srodowisku")
def test_paired_wilcoxon_pvalue_matches_scipy_directly():
    rng = np.random.default_rng(7)
    baseline = rng.normal(loc=5.0, scale=2.0, size=60)
    event = baseline + rng.normal(loc=1.0, scale=2.0, size=60)

    result = paired_wilcoxon_test(event, baseline)

    diff = event - baseline
    nz = diff != 0
    scipy_res = scipy_stats.wilcoxon(
        event[nz], baseline[nz],
        alternative="two-sided", method="approx", zero_method="wilcox",
        correction=False,
    )
    assert result.pvalue == pytest.approx(float(scipy_res.pvalue), abs=1e-9)
    # I dokladnie ten bug: scipy zawsze daje ujemny zstatistic dla
    # two-sided, niezaleznie od faktycznego kierunku -- nasz wynik NIE
    # jest zniewolony tym znakiem.
    assert float(scipy_res.zstatistic) < 0


@pytest.mark.skipif(not _HAS_SCIPY, reason="scipy niedostepne w tym srodowisku")
def test_scipy_wilcoxon_zstatistic_is_always_negative_two_sided():
    # Sanity check samego bugu w scipy (nie naszego kodu) -- dokumentuje
    # dlaczego cala ta ostroznosc jest potrzebna. Dwa przeciwne kierunki
    # efektu, OBA dostaja ujemny zstatistic ze scipy wprost.
    rng = np.random.default_rng(1)
    baseline = rng.normal(loc=0.0, scale=1.0, size=30)
    up = baseline + 3.0
    down = baseline - 3.0

    res_up = scipy_stats.wilcoxon(up, baseline, alternative="two-sided", method="approx")
    res_down = scipy_stats.wilcoxon(down, baseline, alternative="two-sided", method="approx")

    assert float(res_up.zstatistic) < 0
    assert float(res_down.zstatistic) < 0


# ---------------------------------------------------------------------
# (d) Przypadki brzegowe -- za malo par / wszystkie remisy.
# ---------------------------------------------------------------------

def test_paired_wilcoxon_too_few_nonzero_pairs_is_inconclusive():
    # Prog z oryginalnej implementacji PROTECT-90: n_nonzero < 10.
    baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    event = baseline + 1.0  # 5 par niezerowych, < 10
    result = paired_wilcoxon_test(event, baseline)

    assert result.inconclusive is True
    assert math.isnan(result.pvalue)
    assert math.isnan(result.statistic)
    assert math.isnan(result.effect_size_r)
    assert result.n_nonzero == 5
    assert "za mało" in result.reason


def test_paired_wilcoxon_all_ties_is_inconclusive():
    baseline = np.arange(20, dtype=float)
    event = baseline.copy()  # roznica == 0 wszedzie -> 0 par niezerowych
    result = paired_wilcoxon_test(event, baseline)

    assert result.inconclusive is True
    assert result.n_nonzero == 0
    assert math.isnan(result.pvalue)


def test_paired_wilcoxon_min_nonzero_threshold_is_configurable():
    baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    event = baseline + 1.0  # 5 par niezerowych
    # Domyslny prog (10) -> inconclusive.
    assert paired_wilcoxon_test(event, baseline).inconclusive is True
    # Obnizony prog (<=5) -> policzalny wynik.
    result = paired_wilcoxon_test(event, baseline, min_nonzero=5)
    assert result.inconclusive is False


def test_paired_wilcoxon_nan_pairs_are_dropped_and_counted():
    rng = np.random.default_rng(2)
    baseline = rng.normal(loc=0.0, scale=1.0, size=30)
    event = baseline + 2.0
    event_with_nan = event.copy()
    event_with_nan[:5] = np.nan

    result = paired_wilcoxon_test(event_with_nan, baseline)
    assert result.n_nan_dropped == 5
    assert result.n_test == 25
    assert result.n_background == 25


def test_paired_wilcoxon_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        paired_wilcoxon_test([1.0, 2.0, 3.0], [1.0, 2.0])


def test_paired_wilcoxon_rejects_one_sided_alternative():
    with pytest.raises(ValueError):
        paired_wilcoxon_test([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], alternative="greater")


# ---------------------------------------------------------------------
# Cross-check bramka: rozbieznosc wlasnej implementacji ze scipy MUSI
# podniesc RuntimeError, nie po cichu przejsc dalej.
# ---------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_SCIPY, reason="scipy niedostepne w tym srodowisku")
def test_paired_wilcoxon_raises_on_cross_check_mismatch(monkeypatch):
    # Zepsuj wlasna implementacje tak, zeby dawala inna p-wartosc niz
    # scipy -- bramka cross-check MUSI to zlapac.
    def _broken(d_nonzero):
        return 1.0, 0.5  # zawsze ta sama, bledna p-wartosc
    monkeypatch.setattr(pipeline, "_wilcoxon_signed_rank_z_p_numpy", _broken)

    rng = np.random.default_rng(0)
    baseline = rng.normal(loc=0.0, scale=1.0, size=40)
    event = baseline + 2.0

    with pytest.raises(RuntimeError):
        paired_wilcoxon_test(event, baseline)


# ---------------------------------------------------------------------
# Funkcja pomocnicza niskiego poziomu -- test bezposredni, bez scipy.
# ---------------------------------------------------------------------

def test_wilcoxon_signed_rank_z_p_numpy_hand_computed_all_positive():
    # Wszystkie roznice dodatnie -> r_plus = suma WSZYSTKICH rang = n(n+1)/2
    # -> z musi byc dodatnie i maksymalne dla danego n.
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z, p = _wilcoxon_signed_rank_z_p_numpy(d)
    assert z > 0
    assert p < 0.10  # n=5 to mala proba, ale kierunek/znak jest jednoznaczny


def test_wilcoxon_signed_rank_z_p_numpy_empty_input():
    z, p = _wilcoxon_signed_rank_z_p_numpy(np.array([]))
    assert math.isnan(z)
    assert p == 1.0


def test_wilcoxon_signed_rank_z_p_numpy_symmetric_sign_flip():
    d = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -1.5, 2.5])
    z_pos, p_pos = _wilcoxon_signed_rank_z_p_numpy(d)
    z_neg, p_neg = _wilcoxon_signed_rank_z_p_numpy(-d)
    assert z_pos == pytest.approx(-z_neg)
    assert p_pos == pytest.approx(p_neg)


# ---------------------------------------------------------------------
# TestResult.verdict() dla inconclusive -- nie udaje wyniku, ktorego nie ma.
# ---------------------------------------------------------------------

def test_inconclusive_verdict_does_not_claim_no_effect():
    result = TestResult(
        statistic=float("nan"), pvalue=float("nan"), n_test=5, n_background=5,
        median_test=1.0, median_background=2.0, alternative="two-sided",
        effect_size_r=float("nan"), n_nonzero=3, inconclusive=True,
        reason="za mało niezerowych par (3 < 10)",
    )
    verdict = result.verdict()
    assert "NIEROZSTRZYGNIĘTY" in verdict
    assert "Brak istotnego efektu" not in verdict
    assert "za mało niezerowych par" in verdict
