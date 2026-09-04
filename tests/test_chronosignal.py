"""
Testy timdr_formalism.chronosignal.

Ta sama zasada projektowa co tests/test_pipeline.py: unikac asercji
opartych na "typowym" zachowaniu losowego seeda. Kazdy test tutaj albo
uzywa recznie skonstruowanych, dokladnie obliczalnych serii (podane w
komentarzach obliczenia progow/statystyk na papierze), albo sprawdza
przypadki brzegowe wprost.

UWAGA: ten plik zostal napisany w sesji bez dostepu do sandboxa bash
(kazda liczba w komentarzach ponizej zostala przesledzona recznie PRZED
uruchomieniem), ale zostal odtad faktycznie uruchomiony przez
uzytkownika (`pytest tests/test_chronosignal.py -v`) -- wszystkie 16
testow PRZESZLO, potwierdzajac reczne wyliczenia.
"""
import numpy as np
import pytest

from timdr_formalism.chronosignal import (
    tempo,
    drift,
    anomalia_flags,
    defekt_flags,
    skret_flags,
)


# ---------------------------------------------------------------------
# tempo()
# ---------------------------------------------------------------------

def test_tempo_basic():
    # timestamps = [0, 60, 120, 185] -> odstepy [60, 60, 65]
    result = tempo([0, 60, 120, 185])
    np.testing.assert_array_equal(result, np.array([60.0, 60.0, 65.0]))


def test_tempo_requires_at_least_two_points():
    with pytest.raises(ValueError):
        tempo([0])


def test_tempo_rejects_non_increasing_timestamps():
    with pytest.raises(ValueError):
        tempo([0, 60, 60, 120])  # zdublowany znacznik -> diff=0, nie > 0
    with pytest.raises(ValueError):
        tempo([0, 60, 30])  # cofniecie w czasie


# ---------------------------------------------------------------------
# drift()
# ---------------------------------------------------------------------

def test_drift_basic():
    # timestamps = [0, 60, 125, 180] -> tempo = [60, 65, 55]
    # nominal=60 -> drift = [0, 5, -5]
    result = drift([0, 60, 125, 180], nominal_interval=60.0)
    np.testing.assert_array_equal(result, np.array([0.0, 5.0, -5.0]))


def test_drift_rejects_nonpositive_nominal():
    with pytest.raises(ValueError):
        drift([0, 60, 120], nominal_interval=0.0)
    with pytest.raises(ValueError):
        drift([0, 60, 120], nominal_interval=-1.0)


# ---------------------------------------------------------------------
# anomalia_flags() -- zgodnosc z anomaly_flags() w
# examples/real_weather_resonance_validation.py
# ---------------------------------------------------------------------

def test_anomalia_flags_hand_constructed():
    # 9 zer + jeden wyraznie odstajacy element (20).
    # mean = 20/10 = 2.0
    # var  = (9*(0-2)^2 + (20-2)^2) / 10 = (9*4 + 324)/10 = 360/10 = 36.0
    # std  = 6.0 ; próg (k=2) = 12.0
    # |0-2|=2 <= 12 -> brak flagi; |20-2|=18 > 12 -> flaga
    series = [0, 0, 0, 0, 0, 0, 0, 0, 0, 20]
    expected = np.array([False] * 9 + [True])
    np.testing.assert_array_equal(anomalia_flags(series, k=2.0), expected)


def test_anomalia_flags_constant_series_returns_all_false():
    result = anomalia_flags([5, 5, 5, 5], k=2.0)
    np.testing.assert_array_equal(result, np.array([False, False, False, False]))


def test_anomalia_flags_requires_at_least_two_elements():
    with pytest.raises(ValueError):
        anomalia_flags([1])


# ---------------------------------------------------------------------
# defekt_flags()
# ---------------------------------------------------------------------

def test_defekt_flags_hand_constructed():
    # series = [10,10,10,10,50,10,10,10] (n=8)
    # posortowane: [10,10,10,10,10,10,10,50]
    # numpy percentile (linear), n-1=7:
    #   p10: idx=0.7 -> interp(sorted[0]=10, sorted[1]=10) = 10
    #   p90: idx=6.3 -> interp(sorted[6]=10, sorted[7]=50)
    #        = 10 + 0.3*(50-10) = 22
    # spread = 22-10 = 12 ; prog (factor=0.3) = 3.6
    # diffs = [0,0,0,40,-40,0,0] ; |diff|>3.6 tylko na indeksach 3,4
    series = [10, 10, 10, 10, 50, 10, 10, 10]
    expected = np.array([False, False, False, True, True, False, False])
    np.testing.assert_array_equal(defekt_flags(series, factor=0.3), expected)


def test_defekt_flags_constant_series_returns_all_false():
    result = defekt_flags([5, 5, 5, 5], factor=0.3)
    np.testing.assert_array_equal(result, np.array([False, False, False]))


def test_defekt_flags_requires_at_least_two_elements():
    with pytest.raises(ValueError):
        defekt_flags([1])


# ---------------------------------------------------------------------
# skret_flags()
# ---------------------------------------------------------------------

def test_skret_flags_hand_constructed():
    # series = [0, 10, 15, 5, -5] -> d = diff = [10, 5, -10, -10]
    # sign(d) = [+1, +1, -1, -1]
    #   flip miedzy d[0]/d[1]: +,+ -> brak
    #   flip miedzy d[1]/d[2]: +,- -> TAK
    #   flip miedzy d[2]/d[3]: -,- -> brak
    # magnitude = |d[1:]-d[:-1]| = [|5-10|, |-10-5|, |-10-(-10)|] = [5, 15, 0]
    # mean(d) = -1.25 ; odchylenia: 11.25, 6.25, -8.75, -8.75
    #   kwadraty: 126.5625, 39.0625, 76.5625, 76.5625 -> suma=318.75
    #   var(ddof=0) = 318.75/4 = 79.6875 -> std ~= 8.9271
    # prog (factor=1.5) ~= 13.39
    # magnitude > prog: [5>13.39 False, 15>13.39 True, 0>13.39 False]
    # wynik = sign_flip AND magnitude>prog = [False, True, False]
    series = [0, 10, 15, 5, -5]
    result = skret_flags(series, factor=1.5)
    np.testing.assert_array_equal(result, np.array([False, True, False]))


def test_skret_flags_constant_slope_returns_all_false():
    # d = diff([0,1,2,3,4]) = [1,1,1,1] -> std(d)=0 -> brak flag (guard)
    result = skret_flags([0, 1, 2, 3, 4], factor=1.5)
    np.testing.assert_array_equal(result, np.array([False, False, False]))


def test_skret_flags_requires_at_least_three_elements():
    with pytest.raises(ValueError):
        skret_flags([1, 2])


def test_skret_flags_flat_then_moving_is_not_a_flip():
    # d = diff([5,5,5,8]) = [0,0,3] -> sign(d)=[0,0,+1]
    # sign_flip wymaga OBU sasiadow niezerowych -> zawsze False tutaj,
    # niezaleznie od magnitude/progu (plaski odcinek nie jest "zwrotem").
    result = skret_flags([5, 5, 5, 8], factor=0.01)
    np.testing.assert_array_equal(result, np.array([False, False]))


# ---------------------------------------------------------------------
# Integracja: tempo() -> defekt_flags() daje ten sam wynik co bezposrednio
# na tej samej serii liczbowej.
# ---------------------------------------------------------------------

def test_integration_tempo_then_defekt_matches_direct_series():
    # timestamps skonstruowane tak, ze tempo(timestamps) == dokladnie
    # seria z test_defekt_flags_hand_constructed: [10,10,10,10,50,10,10,10]
    timestamps = np.cumsum([0, 10, 10, 10, 10, 50, 10, 10, 10]).astype(float)
    computed_tempo = tempo(timestamps)
    np.testing.assert_array_equal(
        computed_tempo, np.array([10.0, 10.0, 10.0, 10.0, 50.0, 10.0, 10.0, 10.0])
    )
    expected = np.array([False, False, False, True, True, False, False])
    np.testing.assert_array_equal(defekt_flags(computed_tempo, factor=0.3), expected)
