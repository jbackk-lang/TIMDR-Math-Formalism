"""
timdr_formalism/calibration.py

Samokalibrujacy mechanizm progu K operatora rezonansu sygnalowego
(R(t) = 1[liczba jednoczesnie anomalnych parametrow >= K], patrz
GIA-TIMDR/docs/theory/Resonance_M_Operator_Empiryczny.md sekcja 1-3,
docs/REAL_DATA_VALIDATION.md, docs/PROTOCOL.md §4b) -- zamienia
jednorazowa, reczna analize z examples/real_weather_resonance_validation.py
w wywolywalna funkcje, ktora dla DOWOLNEGO okna wielu-parametrowych
danych (nie tylko pogodowych):

    1. liczy empiryczna stope rezonansu dla biezacej konfiguracji K/n,
    2. porownuje ja z teoretyczna baza niezaleznosci
       P(Binomial(n, p_anomaly=P(|Z|>2)) >= K) (Resonance_M §2),
    3. uruchamia DOKLADNIE ta sama metodyke testu permutacyjnego, jakiej
       uzywa examples/real_weather_resonance_validation.py (funkcja
       zaladowana STAD, z tego pliku, nie przepisana -- patrz sekcja
       importu nizej) plus kontrole pozytywna (uogolniona na dowolne K),
    4. sprawdza MOC: jesli w oknie jest zero realnych zdarzen rezonansu
       przy biezacym K, test typu "ge" (permutacja >= realna) ma
       MATEMATYCZNIE gwarantowane p=1.0 niezaleznie od danych -- to
       dokladnie sytuacja z realnego okna Krakow_Centrum
       (docs/REAL_DATA_VALIDATION.md: 0/24 zdarzen przy K=2 i K=3).
       Jesli zdarzen >0, ale kontrola pozytywna (najbardziej oczywisty
       mozliwy efekt) i tak nie osiaga istotnosci, test rowniez nie ma
       mocy. W obu przypadkach: KALIBRACJA NIE JEST STOSOWANA -- ten
       modul nigdy nie zglasza pewnej korekty K z niedomocnionego testu
       (dokladnie ta uczciwosc, ktora juz jest w docs/REAL_DATA_VALIDATION.md
       i TestResult.verdict() w pipeline.py).
    5. jesli test MA moc i realna stopa istotnie przekracza baseline
       niezaleznosci, szuka najnizszego wyzszego K, ktore usuwa nadmiar
       (z korekta Bonferroniego za liczbe przetestowanych kandydatow K,
       bonferroni_correct() z pipeline.py -- patrz PROTOCOL.md sekcja
       "Korekta na wielokrotne porownania").

WAZNE O REUZYCIE (punkt 3 zadania): `permutation_test`/`rezonans_rate`
ponizej NIE sa przepisane -- sa zaladowane bezposrednio z pliku
examples/real_weather_resonance_validation.py (`_load_weather_validation_module`
nizej, importlib z jawnej sciezki, zamiast sys.path-hack jak w
tests/test_examples.py, zeby modul biblioteczny nie modyfikowal
globalnego sys.path). Jedno miejsce definicji mechaniki permutacyjnej;
ten plik dodaje tylko decyzje kalibracyjna na jej podstawie.

Logika decyzyjna (`_decide_calibration`) jest CELOWO czysta funkcja bez
efektow ubocznych/losowosci (przyjmuje juz policzone liczby +
`evaluate_candidate` jako wstrzykiwana zaleznosc) -- ta sama zasada
projektowa co reszta repo (patrz naglowek tests/test_pipeline.py):
testy tej funkcji uzywaja recznie dobranych liczb, nie polegaja na
"typowym" zachowaniu konkretnego seeda.
"""
from __future__ import annotations

import csv
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import stats

from .chronosignal import anomalia_flags
from .pipeline import bonferroni_correct


# ---------------------------------------------------------------------
# Zaladowanie metodyki permutacyjnej z examples/real_weather_resonance_validation.py
# ---------------------------------------------------------------------

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
_WEATHER_VALIDATION_PATH = _EXAMPLES_DIR / "real_weather_resonance_validation.py"


def _load_weather_validation_module():
    """Laduje examples/real_weather_resonance_validation.py jako modul,
    bez modyfikowania globalnego sys.path (importlib z jawnej sciezki
    pliku) -- zeby uzyc DOKLADNIE tych samych funkcji
    (`rezonans_rate`, `permutation_test`, `anomaly_flags`,
    `positive_control`), a nie ich kopii."""
    spec = importlib.util.spec_from_file_location(
        "timdr_formalism._real_weather_resonance_validation",
        _WEATHER_VALIDATION_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensywne
        raise ImportError(
            f"Nie mozna zaladowac metodyki permutacyjnej z {_WEATHER_VALIDATION_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_weather = _load_weather_validation_module()


# ---------------------------------------------------------------------
# Teoretyczna baza niezaleznosci (Resonance_M_Operator_Empiryczny.md §2)
# ---------------------------------------------------------------------

# P(|Z|>2) dla standardowego rozkladu normalnego -- 2*(1-Phi(2)), a nie
# zaokraglone na sztywno 0.0455, zeby nie duplikowac przyblizenia, ktore
# scipy juz liczy dokladnie.
P_ANOMALY_2SIGMA_NORMAL: float = float(2.0 * stats.norm.sf(2.0))


def theoretical_independence_baseline_rate(
    n: int, K: int, p_anomaly: float = P_ANOMALY_2SIGMA_NORMAL
) -> float:
    """P(Binomial(n, p_anomaly) >= K) -- teoretyczna baza niezaleznosci
    z Resonance_M_Operator_Empiryczny.md §2. Zaklada niezaleznosc
    parametrow I normalnosc kazdego z osobna -- w realnych danych
    zazwyczaj NIE zachodzi (patrz §2 dokumentu: parametry pogodowe sa
    skorelowane, rozklady nie sa idealnie gaussowskie). Punkt
    odniesienia, nie ground truth -- dlatego krok permutacyjny nizej
    testuje na REALNYCH marginalnych rozkladach, nie na tym zalozeniu.

    Dla n=5, K=3 (udokumentowany system) daje ~0.09%, zgodnie z
    Resonance_M_Operator_Empiryczny.md sekcja 2.
    """
    if n < 1:
        raise ValueError("n musi byc >= 1")
    if not (1 <= K <= n):
        raise ValueError("K musi spelniac 1 <= K <= n")
    if not (0.0 < p_anomaly < 1.0):
        raise ValueError("p_anomaly musi byc w (0, 1)")
    return float(stats.binom.sf(K - 1, n, p_anomaly))


# ---------------------------------------------------------------------
# Generyczny interfejs tablicowy: dowolne parametry -> flagi anomalii
# ---------------------------------------------------------------------

def compute_anomaly_flags(
    parameter_series, k: float = 2.0
) -> list[np.ndarray]:
    """Live progi 2-sigma (|x-mean|>k*std) per parametr -- ta sama
    definicja, ktorej juz uzywaja chronosignal.anomalia_flags (stad
    importowana, nie duplikowana) i
    real_weather_resonance_validation.anomaly_flags. Przyjmuje `dict`
    {nazwa: seria} albo dowolna sekwencje serii -- generyczny interfejs,
    nie zahardkodowany do pogody.
    """
    if isinstance(parameter_series, dict):
        series_list = list(parameter_series.values())
    else:
        series_list = list(parameter_series)
    if len(series_list) < 1:
        raise ValueError("wymagany >=1 parametr")
    return [anomalia_flags(s, k=k) for s in series_list]


# ---------------------------------------------------------------------
# Generyczny loader CSV (nie zahardkodowany do pogody) + wygodny
# wrapper dla formatu krakow_forecast_snapshots.csv
# ---------------------------------------------------------------------

def load_parameter_windows_from_csv(
    csv_path,
    value_columns: Sequence[str],
    *,
    date_column: str = "target_date",
    group_column: Optional[str] = None,
    group_value: Optional[str] = None,
    source_column: Optional[str] = None,
    source_prefixes: Optional[Sequence[str]] = None,
    date_min: Optional[str] = None,
    date_max: Optional[str] = None,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Generyczny loader CSV -> okno wielu-parametrowe. Dziala na
    dowolnym pliku CSV z kolumna daty + N kolumnami liczbowymi +
    opcjonalnymi kolumnami grupujacymi (`group_column`/`group_value`,
    np. stacja) i zrodla (`source_column`/`source_prefixes`, np.
    obserwacje-vs-prognozy) -- nie jest zahardkodowany do pogody.

    Deduplikacja: jesli kilka wierszy (PO przejsciu filtrow
    group/source) ma ta sama wartosc `date_column`, OSTATNI wiersz w
    pliku wygrywa -- ta sama konwencja, jaka opisuje naglowek
    examples/real_weather_resonance_validation.py
    (`bias_correction.py::_load_pairs`, plik spoza tego repo). Wiersze,
    ktorym brakuje ktorejkolwiek z `value_columns`, sa pomijane
    calkowicie (nie interpolowane) -- zeby okno nie mieszalo dni o
    roznej kompletnosci danych (ta sama zasada, co "precip_mm nie
    uzyty" w docs/REAL_DATA_VALIDATION.md).

    Zwraca (posortowane_daty, {kolumna: np.ndarray}) -- kazda tablica
    ma dlugosc len(posortowane_daty), wyrownana do tych samych dat we
    wszystkich kolumnach.
    """
    csv_path = Path(csv_path)
    rows_by_date: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if group_column and group_value is not None:
                if row.get(group_column) != group_value:
                    continue
            if source_column and source_prefixes:
                src = row.get(source_column) or ""
                if not any(src.startswith(p) for p in source_prefixes):
                    continue
            date = row.get(date_column)
            if not date:
                continue
            if date_min and date < date_min:
                continue
            if date_max and date > date_max:
                continue
            if any(not (row.get(c) or "").strip() for c in value_columns):
                continue
            rows_by_date[date] = row  # ostatni wpis wygrywa (kolejnosc pliku)

    dates_sorted = sorted(rows_by_date)
    if not dates_sorted:
        raise ValueError(
            f"Brak wierszy spelniajacych filtry w {csv_path} dla kolumn {value_columns}"
        )
    columns_out = {
        col: np.array(
            [float(rows_by_date[d][col]) for d in dates_sorted], dtype=float
        )
        for col in value_columns
    }
    return dates_sorted, columns_out


# Filtry identyczne jak w examples/real_weather_resonance_validation.py /
# docs/REAL_DATA_VALIDATION.md (stacja Krakow_Centrum, tylko realne
# obserwacje, nie prognozy).
KRAKOW_STATION = "Krakow_Centrum"
KRAKOW_SOURCE_PREFIXES = ("IMGW_real_", "web_szukaj_", "OpenMeteo_real_dailymax")
KRAKOW_COLUMNS = ("max_temp_c", "pressure_hpa", "wind_kmh")


def load_krakow_weather_window(
    csv_path,
    *,
    date_min: Optional[str] = None,
    date_max: Optional[str] = None,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Wygodny wrapper `load_parameter_windows_from_csv()` z dokladnie
    tymi samymi filtrami co docs/REAL_DATA_VALIDATION.md /
    examples/real_weather_resonance_validation.py (stacja
    Krakow_Centrum, source in {IMGW_real_*, web_szukaj_*,
    OpenMeteo_real_dailymax}, kolumny max_temp_c/pressure_hpa/wind_kmh).
    Kolumna wilgotnosci NIE istnieje w tym pliku (patrz
    docs/REAL_DATA_VALIDATION.md) -- to n=3 analog udokumentowanego n=5
    systemu, nie pelna replikacja.
    """
    return load_parameter_windows_from_csv(
        csv_path,
        value_columns=KRAKOW_COLUMNS,
        date_column="target_date",
        group_column="station",
        group_value=KRAKOW_STATION,
        source_column="source",
        source_prefixes=KRAKOW_SOURCE_PREFIXES,
        date_min=date_min,
        date_max=date_max,
    )


# ---------------------------------------------------------------------
# Kontrola pozytywna uogolniona na dowolne K (oryginal w
# real_weather_resonance_validation.py ma K=3 i 3 dni zahardkodowane)
# ---------------------------------------------------------------------

def _positive_control_pvalue(
    params_flags: Sequence[np.ndarray],
    K: int,
    rng: np.random.Generator,
    n_perm: int = 5000,
    n_inject_days: Optional[int] = None,
) -> float:
    """Uogolnienie `real_weather_resonance_validation.positive_control()`
    (tam zahardkodowane K=3, dni 0/1/2) na dowolne K: wymuszamy pelna
    wspolbieznosc we WSZYSTKICH parametrach przez `n_inject_days` dni
    (domyslnie K) i sprawdzamy, czy test permutacyjny to lapie. Uzywa
    DOKLADNIE tej samej funkcji `_weather.permutation_test` co dane
    realne -- generalizacja dotyczy tylko liczby wstrzyknietych dni i
    progu K, nie samego testu statystycznego.

    To jest sanity check MECHANIKI testu (czy metodyka w ogole potrafi
    wykryc oczywisty, wymuszony efekt), NIE prospektywna moc dla
    realnej wielkosci efektu w tych danych (PROTOCOL.md §4b rozroznia
    te dwie rzeczy wprost) -- dlatego w `_decide_calibration` ten wynik
    jest jedna z DWOCH przeslanek `has_power`, nie jedyna: brak
    jakichkolwiek realnych zdarzen rezonansu (zobacz
    `real_event_count == 0` nizej) jest silniejszym, matematycznym
    dowodem braku mocy, niezaleznym od tego testu.
    """
    if n_inject_days is None:
        n_inject_days = K
    injected = [f.copy() for f in params_flags]
    n_inject_days = min(n_inject_days, injected[0].size)
    for day in range(n_inject_days):
        for f in injected:
            f[day] = True
    _, pvalue = _weather.permutation_test(injected, K, rng, n_perm=n_perm)
    return pvalue


# ---------------------------------------------------------------------
# Wynik kalibracji + czysta logika decyzyjna
# ---------------------------------------------------------------------

@dataclass
class CalibrationResult:
    n_params: int
    K: int
    n_timesteps: int
    anomaly_counts: list
    real_event_count: int
    empirical_rate: float
    theoretical_baseline_rate: float
    deviation: float
    permutation_pvalue: float
    positive_control_pvalue: float
    has_power: bool
    # "insufficient_power" | "no_calibration_needed" | "calibration_recommended"
    action: str
    recommended_K: Optional[int]
    n_K_candidates_tested: int
    reason: str

    def report(self) -> str:
        """Czytelny raport tekstowy, w stylu format_report() z
        pipeline.py -- podsumowuje dane wejsciowe, moc i decyzje."""
        lines = [
            "# Raport kalibracji progu K operatora rezonansu",
            "",
            f"n_params={self.n_params}, K (biezace)={self.K}, "
            f"n_timesteps={self.n_timesteps}",
            f"Anomalie per parametr: {self.anomaly_counts}",
            f"Realne zdarzenia rezonansu przy K={self.K}: "
            f"{self.real_event_count}/{self.n_timesteps} "
            f"(stopa={self.empirical_rate:.4g})",
            f"Teoretyczna baza niezaleznosci: {self.theoretical_baseline_rate:.4g}",
            f"Odchylenie (empiryczna - teoretyczna): {self.deviation:.4g}",
            f"p permutacyjne (realne dane): {self.permutation_pvalue:.4g}",
            f"p kontroli pozytywnej (moc): {self.positive_control_pvalue:.4g}",
            f"Moc testu: {'TAK' if self.has_power else 'NIE'}",
            "",
            f"## Decyzja: {self.action}",
            self.reason,
        ]
        if self.action == "calibration_recommended":
            lines.append(f"Rekomendowane K: {self.recommended_K}")
        return "\n".join(lines)


def _decide_calibration(
    n: int,
    K: int,
    real_event_count: int,
    permutation_pvalue: float,
    positive_control_pvalue: float,
    alpha: float,
    evaluate_candidate: Callable[[int], float],
) -> tuple[str, Optional[int], int, str]:
    """Czysta logika decyzyjna (bez losowosci/efektow ubocznych) --
    przyjmuje juz policzone statystyki + `evaluate_candidate(K) ->
    pvalue` jako wstrzykiwana zaleznosc, zeby dalo sie ja testowac
    recznie dobranymi liczbami, tak jak `TestResult.verdict()` i
    `format_report()` w tests/test_pipeline.py.

    Zwraca (action, recommended_K, n_K_candidates_tested, reason).
    """
    has_power = real_event_count > 0 and positive_control_pvalue < alpha

    if not has_power:
        if real_event_count == 0:
            power_reason = (
                f"zero realnych zdarzen rezonansu w tym oknie przy K={K} -- "
                "dla testu permutacyjnego typu 'ge' (permutacja >= realna) "
                "to MATEMATYCZNIE gwarantuje p=1.0 (kazda permutacja ma "
                "stope >= 0), niezaleznie od liczby powtorzen czy ziarna "
                "losowosci -- brak jakiejkolwiek informacji, nie 'brak "
                "efektu' (patrz docs/REAL_DATA_VALIDATION.md, realny "
                "przyklad Krakow_Centrum: 0/24 zdarzen, p=1.0 dla K=2 i K=3)"
            )
        else:
            power_reason = (
                "kontrola pozytywna (wymuszona pelna wspolbieznosc, "
                "najbardziej oczywisty mozliwy efekt) nie osiagnela "
                f"istotnosci (p={positive_control_pvalue:.4g} >= {alpha}) -- "
                "test nie ma mocy nawet do wykrycia tego ekstremalnego "
                "przypadku w tym oknie"
            )
        reason = (
            f"ZA MALA MOC -- KALIBRACJA NIE ZASTOSOWANA. {power_reason}. "
            f"Surowe p permutacyjne dla realnych danych (p={permutation_pvalue:.4g}) "
            "NIE moze byc w tej sytuacji zinterpretowane w zadna strone "
            "(PROTOCOL.md §4b: wynik nieistotny przy braku mocy nie jest "
            "dowodem braku efektu -- i analogicznie nie uzasadnia pewnej "
            "korekty K w druga strone)."
        )
        return "insufficient_power", None, 0, reason

    if permutation_pvalue >= alpha:
        reason = (
            f"Test ma moc (kontrola pozytywna p={positive_control_pvalue:.4g} "
            f"< {alpha}, {real_event_count} realnych zdarzen rezonansu w "
            "oknie), ale realna stopa rezonansu nie odbiega istotnie od "
            f"tego, czego oczekiwac po permutacji (p={permutation_pvalue:.4g} "
            f">= {alpha}) -- brak dowodu na nadmierna wspolbieznosc, K "
            "pozostaje bez zmian."
        )
        return "no_calibration_needed", K, 0, reason

    candidates = list(range(K + 1, n + 1))
    if not candidates:
        reason = (
            f"Realna stopa rezonansu przy K={K} istotnie przekracza baseline "
            f"niezaleznosci (p={permutation_pvalue:.4g} < {alpha}, moc "
            f"potwierdzona kontrola pozytywna p={positive_control_pvalue:.4g}), "
            f"ale K={K} jest juz rowne n={n} -- nie ma wyzszego progu do "
            "zarekomendowania. Nadmiar wspolbieznosci pozostaje "
            "niewyjasniony samym progiem K (mozliwa realna korelacja "
            "miedzy parametrami, nie artefakt zbyt niskiego progu)."
        )
        return "calibration_recommended", K, 0, reason

    n_candidates = len(candidates)
    recommended_K = candidates[-1]
    resolved = False
    for cand in candidates:
        cand_pvalue = evaluate_candidate(cand)
        corrected = bonferroni_correct(cand_pvalue, n_candidates)
        if corrected >= alpha:
            recommended_K = cand
            resolved = True
            break

    if resolved:
        reason = (
            f"Realna stopa rezonansu przy K={K} istotnie przekracza baseline "
            f"niezaleznosci (p={permutation_pvalue:.4g} < {alpha}, moc "
            f"potwierdzona kontrola pozytywna p={positive_control_pvalue:.4g}). "
            f"Rekomendowane K={recommended_K} (po korekcie Bonferroniego za "
            f"{n_candidates} przetestowanych kandydatow K) usuwa nadmiar."
        )
    else:
        reason = (
            f"Realna stopa rezonansu przy K={K} istotnie przekracza baseline "
            f"niezaleznosci (p={permutation_pvalue:.4g} < {alpha}, moc "
            f"potwierdzona kontrola pozytywna p={positive_control_pvalue:.4g}). "
            f"Zaden kandydat K do n={n} nie usuwa nadmiaru po korekcie za "
            f"{n_candidates} porownan -- rekomendowane K={recommended_K} "
            "(maksimum dostepne) jako najlepsze przyblizenie; nadmiar moze "
            "wymagac innej przyczyny niz sam prog K (np. realna korelacja "
            "miedzy parametrami, nie artefakt progu)."
        )
    return "calibration_recommended", recommended_K, n_candidates, reason


def calibrate_resonance_K(
    params_flags: Sequence[np.ndarray],
    K: int,
    *,
    n_perm: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
    p_anomaly: float = P_ANOMALY_2SIGMA_NORMAL,
) -> CalibrationResult:
    """Glowna funkcja modulu: samokalibracja progu K operatora
    rezonansu na REALNYM (lub dowolnym innym) oknie wielu-parametrowych
    flag anomalii.

    params_flags: lista/sekwencja `n` tablic bool tej samej dlugosci
        (jedna per parametr) -- np. wynik `compute_anomaly_flags()`
        albo `chronosignal.anomalia_flags()` zastosowanego recznie do
        kazdego parametru. Generyczny interfejs tablicowy, nie
        zahardkodowany do zadnej domeny.
    K: biezacy prog rezonansu do oceny/kalibracji.
    n_perm, alpha, seed: parametry testu permutacyjnego (patrz
        `_weather.permutation_test`) -- ten sam Davison-Hinkley p =
        (ge_count+1)/(n_perm+1) co w
        examples/real_weather_resonance_validation.py.
    p_anomaly: prawdopodobienstwo anomalii pod zalozeniem niezaleznosci
        + normalnosci (domyslnie P(|Z|>2) dokladnie, nie zaokraglone).

    Zwraca `CalibrationResult` -- NIGDY nie zwraca pewnej rekomendacji
    K z niedomocnionego testu (patrz `_decide_calibration`).
    """
    flags = [np.asarray(f, dtype=bool) for f in params_flags]
    n = len(flags)
    if n < 1:
        raise ValueError("wymagany >=1 parametr")
    lengths = {f.size for f in flags}
    if len(lengths) != 1:
        raise ValueError("wszystkie serie flag musza miec ta sama dlugosc")
    n_timesteps = flags[0].size
    if n_timesteps < 1:
        raise ValueError("okno danych nie moze byc puste")
    if not (1 <= K <= n):
        raise ValueError("K musi spelniac 1 <= K <= n")

    anomaly_counts = [int(f.sum()) for f in flags]

    empirical_rate, permutation_pvalue = _weather.permutation_test(
        flags, K, np.random.default_rng(seed), n_perm=n_perm
    )
    real_event_count = int(round(empirical_rate * n_timesteps))

    theoretical_rate = theoretical_independence_baseline_rate(n, K, p_anomaly)
    deviation = empirical_rate - theoretical_rate

    positive_control_pvalue = _positive_control_pvalue(
        flags, K, np.random.default_rng(seed + 1), n_perm=n_perm
    )

    def _evaluate_candidate(cand_K: int) -> float:
        _, p = _weather.permutation_test(
            flags, cand_K, np.random.default_rng(seed + 100 + cand_K), n_perm=n_perm
        )
        return p

    action, recommended_K, n_tested, reason = _decide_calibration(
        n=n,
        K=K,
        real_event_count=real_event_count,
        permutation_pvalue=permutation_pvalue,
        positive_control_pvalue=positive_control_pvalue,
        alpha=alpha,
        evaluate_candidate=_evaluate_candidate,
    )

    return CalibrationResult(
        n_params=n,
        K=K,
        n_timesteps=n_timesteps,
        anomaly_counts=anomaly_counts,
        real_event_count=real_event_count,
        empirical_rate=empirical_rate,
        theoretical_baseline_rate=theoretical_rate,
        deviation=deviation,
        permutation_pvalue=permutation_pvalue,
        positive_control_pvalue=positive_control_pvalue,
        has_power=(real_event_count > 0 and positive_control_pvalue < alpha),
        action=action,
        recommended_K=recommended_K,
        n_K_candidates_tested=n_tested,
        reason=reason,
    )
