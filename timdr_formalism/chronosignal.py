"""
timdr_formalism/chronosignal.py

Czas jako sygnal: instancja generycznego obiektu x:T->R z gale
zi sygnalowej TIMDR (Axioms_S_TIMDR_Signal.md) na konkretny przypadek
"T = kolejne znaczniki czasu (timestamps)".

Nie wprowadza nowej matematyki ani nowych aksjomatow -- to jest wpisanie
sie w istniejaca definicje sygnalu z Chronoprocesu Xi=(T,x,Gamma,phi)
(patrz GIA-TIMDR/SKILL_timdr-signal-framework.md, dyskusja "naprawa M/S"),
z dwiema rozdzielonymi seriami:

    tempo(t)  = t[i+1] - t[i]                        -- proces odstepow
    drift(t)  = tempo_zmierzone(t) - tempo_nominalne  -- wymaga zegara
                                                          referencyjnego

Detektory anomalia/defekt/skret ponizej sa samodzielna implementacja
definicji z sekcji 1 skilla timdr-signal-framework (ten konkretny kod
detektorow NIE istnial dotad w tym repo -- pipeline.py to warstwa
PROTOKOLU testowania numerologii/formalizmu, nie warstwa detektorow),
zastosowana tu do sygnalu tempa/driftu.

Konwencje (`m, s = series.mean(), series.std()` dla anomalii; p90-p10
jako skala dla defektu) sa dokladnie tymi samymi, ktorych uzywa
examples/real_weather_resonance_validation.py (anomaly_flags) i skill
paragraf 1 -- nie nowy wybor progu.

UWAGA O WYKONANIU: ten modul zostal napisany w sesji, w ktorej sandbox
bash byl niedostepny (RPC pipe closed) -- matematyka progow zostala
reczne przesledzona przed napisaniem testow. Wszystkie 16 testow w
tests/test_chronosignal.py zostalo odtad faktycznie uruchomionych przez
uzytkownika (`pytest tests/test_chronosignal.py -v`) i PRZESZLO (62/63
w calym repo -- jedyny blad to niezwiazany z tym modulem PermissionError
na lokalnym katalogu tymczasowym Windows w istniejacym wczesniej tescie
test_preregistration_save_and_load_roundtrip). examples/chronosignal_demo.py
(interaktywny skrypt, nie plik testowy) nie zostal jeszcze uruchomiony.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------
# Instancja sygnalu: tempo i drift
# ---------------------------------------------------------------------

def tempo(timestamps: Sequence[float]) -> np.ndarray:
    """tempo(t) = t[i+1] - t[i] -- seria odstepow miedzy kolejnymi
    znacznikami czasu. Dlugosc n-1 dla n znacznikow.

    Wymaga SCISLE rosnacych znacznikow -- nie sortuje ani nie
    deduplikuje sam, zeby blad kolejnosci zdarzen (dane wejsciowe nie w
    kolejnosci chronologicznej, albo zdublowany znacznik) nie zostal po
    cichu zamaskowany posortowaniem w srodku funkcji.
    """
    ts = np.asarray(timestamps, dtype=float)
    if ts.size < 2:
        raise ValueError("tempo() wymaga >= 2 znacznikow czasu")
    if np.any(np.diff(ts) <= 0):
        raise ValueError(
            "timestamps musza byc scisle rosnace -- posortuj/odduplikuj "
            "wejscie PRZED wywolaniem tempo(), nie w srodku"
        )
    return np.diff(ts)


def drift(timestamps: Sequence[float], nominal_interval: float) -> np.ndarray:
    """drift(t) = tempo_zmierzone(t) - tempo_nominalne.

    Wymaga JAWNEGO zewnetrznego zegara referencyjnego
    (`nominal_interval`) -- bez niego drift jest formalnie
    niezdefiniowany (patrz dyskusja "naprawa M/S": tempo i drift to dwie
    OSOBNE wielkosci, tempo nie wymaga referencji, drift wymaga).
    """
    if nominal_interval <= 0:
        raise ValueError("nominal_interval musi byc > 0")
    return tempo(timestamps) - nominal_interval


# ---------------------------------------------------------------------
# Detektory na dowolnym skalarnym sygnale x:T->R (tu: tempo()/drift())
# ---------------------------------------------------------------------

def anomalia_flags(series: Sequence[float], k: float = 2.0) -> np.ndarray:
    """|x - mean| > k*std -- maska tej samej dlugosci co `series`.
    Ta sama definicja co `anomaly_flags` w
    examples/real_weather_resonance_validation.py.
    """
    s = np.asarray(series, dtype=float)
    if s.size < 2:
        raise ValueError("anomalia_flags() wymaga >= 2 elementow")
    m, sd = s.mean(), s.std()
    if sd == 0.0:
        return np.zeros(s.shape, dtype=bool)
    return np.abs(s - m) > k * sd


def defekt_flags(series: Sequence[float], factor: float = 0.3) -> np.ndarray:
    """Skok |diff(series)_i| > factor*(p90-p10) -- maska dlugosci
    len(series)-1; flags[i] mowi o skoku miedzy series[i] i series[i+1].
    """
    s = np.asarray(series, dtype=float)
    if s.size < 2:
        raise ValueError("defekt_flags() wymaga >= 2 elementow")
    p10, p90 = np.percentile(s, [10, 90])
    spread = p90 - p10
    if spread == 0.0:
        return np.zeros(s.size - 1, dtype=bool)
    return np.abs(np.diff(s)) > factor * spread


def skret_flags(series: Sequence[float], factor: float = 1.5) -> np.ndarray:
    """Punkt zwrotny: znak lokalnego nachylenia (diff) zmienia sie miedzy
    dwoma kolejnymi krokami I wielkosc zmiany (|d[i+1]-d[i]|) przekracza
    factor*std(d), gdzie d=diff(series). Maska dlugosci len(series)-2;
    flags[i] odpowiada zwrotowi na styku d[i]/d[i+1] (czyli series[i+2]).

    Wymaga OBU sasiadujacych nachylen niezerowych, zeby plaski odcinek
    (d=0) nastepnie ruszajacy w dowolna strone nie liczyl sie jako
    "zwrot" -- to nie jest w oryginalnej definicji z sekcji 1 skilla,
    ale jest jawnym, udokumentowanym wyborem tej implementacji.
    """
    s = np.asarray(series, dtype=float)
    if s.size < 3:
        raise ValueError("skret_flags() wymaga >= 3 elementow")
    d = np.diff(s)
    sign_d = np.sign(d)
    sign_flip = (sign_d[:-1] != sign_d[1:]) & (sign_d[:-1] != 0) & (sign_d[1:] != 0)
    magnitude = np.abs(d[1:] - d[:-1])
    sd = d.std()
    if sd == 0.0:
        return np.zeros(d.size - 1, dtype=bool)
    return sign_flip & (magnitude > factor * sd)
