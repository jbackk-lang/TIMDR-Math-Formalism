"""
examples/real_weather_resonance_validation.py

Walidacja operatora rezonansu (rezonans = >=K parametrow anomalnych
jednoczesnie) na REALNYCH danych pogodowych, nie syntetycznych. Patrz
docs/REAL_DATA_VALIDATION.md po pelny opis metody i uczciwy werdykt.

Dane: stacja Krakow_Centrum, synoptyk-v2.0-main/krakow_forecast_snapshots.csv,
wiersze source in {IMGW_real_*, web_szukaj_*, OpenMeteo_real_dailymax},
zdeduplikowane per target_date (ostatni wpis wygrywa, jak w
bias_correction.py::_load_pairs), 2026-08-09 .. 2026-09-02 (2026-08-11
brakuje w zrodle - pominiete, nie interpolowane). 24 dni, 3 parametry
(max_temp_c, pressure_hpa, wind_kmh) - kolumny wilgotnosci w tym pliku
nie ma wcale, wiec to zawezony n=3 analog udokumentowanego n=5 systemu.

UWAGA O REPRODUKOWALNOSCI: pierwszy raz ten dokladny test zostal
wykonany jako niezalezna reimplementacja w JavaScript
(mcp__Claude_Browser__javascript_tool), bo Python/bash byly wtedy w tej
sesji niedostepne. Ten skrypt to rownowazna implementacja w Pythonie -
ten sam algorytm, inny RNG (numpy PCG64 zamiast mulberry32), wiec
konkretne wartosci permutacyjne beda inne, ale rozklad wynikow i
werdykt powinny sie zgadzac. Uruchom go, zeby zweryfikowac na wlasnym
interpreterze zamiast ufac liczbom z docs/REAL_DATA_VALIDATION.md.

Uruchomienie:
    python examples/real_weather_resonance_validation.py
"""
from __future__ import annotations

import numpy as np

# Realne dane, stacja Krakow_Centrum (patrz naglowek modulu).
DATES = [
    "08-09", "08-10", "08-12", "08-13", "08-14", "08-15", "08-16", "08-17",
    "08-18", "08-19", "08-20", "08-21", "08-22", "08-23", "08-24", "08-25",
    "08-26", "08-27", "08-28", "08-29", "08-30", "08-31", "09-01", "09-02",
]
TEMP = np.array([
    26.8, 29.2, 21, 18, 23, 31.3, 33.0, 26.0, 19.6, 23.0, 29.0, 25.8, 23.0,
    21.8, 22.7, 16.5, 23.6, 24.9, 28.3, 23.0, 26.4, 27.5, 22.9, 23.8,
])
PRESSURE = np.array([
    1018, 1012, 1027, 1026.9, 1028, 1014, 1013.0, 1010.5, 1009.4, 1010.0,
    1008.7, 1009.6, 1011.2, 1020.6, 1021.6, 1019.8, 1020.2, 1021.4, 1017.9,
    1014.4, 1016.1, 1014.7, 1017.7, 1020.3,
])
WIND = np.array([
    14.4, 14.4, 15, 3, 17, 3, 11.3, 8.1, 14.1, 20.0, 10.7, 6.1, 18.6, 10.9,
    7.4, 9.6, 10.1, 11.4, 13.6, 8.9, 4.0, 5.3, 11.0, 7.0,
])
N = len(DATES)


def anomaly_flags(series: np.ndarray) -> np.ndarray:
    """Zywe (live) progi 2-sigma liczone z tego samego okna, zgodnie z
    analyzer/adaptive_thresholds.py."""
    m, s = series.mean(), series.std()
    return np.abs(series - m) > 2 * s


def rezonans_rate(params_flags: list[np.ndarray], k: int) -> float:
    counts = np.sum(np.vstack(params_flags), axis=0)
    return float(np.mean(counts >= k))


def permutation_test(params_flags: list[np.ndarray], k: int, rng: np.random.Generator,
                      n_perm: int = 5000) -> tuple[float, float]:
    real_rate = rezonans_rate(params_flags, k)
    ge_count = 0
    for _ in range(n_perm):
        shuffled = [rng.permutation(f) for f in params_flags]
        if rezonans_rate(shuffled, k) >= real_rate:
            ge_count += 1
    pvalue = (ge_count + 1) / (n_perm + 1)
    return real_rate, pvalue


def positive_control(params_flags: list[np.ndarray], rng: np.random.Generator,
                      n_perm: int = 5000) -> tuple[float, float]:
    """Sanity check mechaniki testu: wymuszamy pelna wspolbieznosc w 3
    dniach i sprawdzamy, ze permutacyjny test to lapie. Jesli nie lapie,
    caly wynik glowny jest bez znaczenia (test jest slepy)."""
    injected = [f.copy() for f in params_flags]
    for day in (0, 1, 2):
        for f in injected:
            f[day] = True
    return permutation_test(injected, k=3, rng=rng, n_perm=n_perm)


def main() -> None:
    real_flags = [anomaly_flags(TEMP), anomaly_flags(PRESSURE), anomaly_flags(WIND)]
    counts = [int(f.sum()) for f in real_flags]

    print("=== Walidacja operatora rezonansu na realnych danych (Krakow_Centrum) ===\n")
    print(f"Okno: {N} dni ({DATES[0]} .. {DATES[-1]}, rok 2026)")
    print(f"Anomalne dni per parametr: temp={counts[0]}, pressure={counts[1]}, wind={counts[2]}\n")

    rng = np.random.default_rng(2026)
    for k in (2, 3):
        real_rate, pvalue = permutation_test(real_flags, k, rng)
        print(f"K={k}: realna stopa={real_rate:.4f} ({round(real_rate * N)}/{N}), p={pvalue:.4f}")

    rng_pos = np.random.default_rng(7)
    pos_rate, pos_p = positive_control(real_flags, rng_pos)
    print(f"\nKontrola pozytywna (3 wymuszone dni, K=3): stopa={pos_rate:.4f}, p={pos_p:.6f}")

    print(
        "\nWerdykt: jesli p dla realnych danych ~1 ale kontrola pozytywna ma "
        "male p, to NIE jest dowod braku rezonansu - to znaczy, ze w tym "
        "oknie bylo za malo anomalii per parametr, zeby test mial jakakolwiek "
        "moc. Patrz docs/REAL_DATA_VALIDATION.md po pelna interpretacje."
    )


if __name__ == "__main__":
    main()
