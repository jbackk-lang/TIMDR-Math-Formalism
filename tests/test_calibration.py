"""
Testy timdr_formalism.calibration.

Zasada projektowa (ta sama co tests/test_pipeline.py): unikac asercji
opartych na "typowym" zachowaniu losowego seeda. Trzy warstwy testow:

1. `_decide_calibration` -- czysta logika decyzyjna, testowana WPROST
   recznie dobranymi liczbami (bez RNG, bez permutacji), analogicznie
   do test_format_report_*/test_test_result_verdict_* w
   test_pipeline.py.
2. `calibrate_resonance_K` na DANYCH ENGINEEROWANYCH tak, zeby
   odchylenie realnej stopy rezonansu od oczekiwanej pod niezaleznoscia
   bylo ekstremalne w obie strony (patrz komentarze przy kazdym
   scenariuszu za dokladny rachunek) -- wynik jest wiec zdeterminowany
   przez konstrukcje danych, nie przez to, ktory konkretnie z 5000
   permutacji akurat wypadnie dla danego ziarna RNG.
3. `calibrate_resonance_K` na REALNYCH danych Krakow_Centrum
   (zaimportowanych z examples/real_weather_resonance_validation.py,
   nie przepisanych) -- scenariusz niedomocniony z
   docs/REAL_DATA_VALIDATION.md, gdzie 0 zdarzen rezonansu przy K=3
   MATEMATYCZNIE gwarantuje p=1.0 (patrz `_decide_calibration`
   docstring), wiec wynik jest rowniez zdeterminowany, nie losowy.

UWAGA O WYKONANIU: napisane w sesji bez dostepu do sandboxa
wykonawczego (workspace bash niedostepny -- "RPC pipe closed"); liczby
w komentarzach przy scenariuszach 2 zostaly recznie przesledzone PRZED
napisaniem testow (ten sam kompromis, co juz udokumentowany w
chronosignal.py/test_chronosignal.py), ale NIE zostaly jeszcze
faktycznie uruchomione przez `pytest`. Uruchom `pytest tests/ -v`, zanim
zaufasz temu, ze wszystko przechodzi.
"""
import numpy as np
import pytest

from timdr_formalism.calibration import (
    CalibrationResult,
    calibrate_resonance_K,
    theoretical_independence_baseline_rate,
    compute_anomaly_flags,
    load_parameter_windows_from_csv,
    load_krakow_weather_window,
    P_ANOMALY_2SIGMA_NORMAL,
    _decide_calibration,
    _weather,
)
from timdr_formalism.pipeline import calibrate_resonance_threshold
from timdr_formalism import calibrate_resonance_K as calibrate_resonance_K_public


# ---------------------------------------------------------------------
# theoretical_independence_baseline_rate
# ---------------------------------------------------------------------

def test_calibrate_resonance_K_exported_from_package_root():
    # Wpiecie w __init__.py (punkt 5 zadania): dostepne bez wchodzenia
    # w submodul.
    assert calibrate_resonance_K_public is calibrate_resonance_K


def test_theoretical_baseline_matches_documented_n5_k3():
    # Resonance_M_Operator_Empiryczny.md §2: n=5, K=3 -> ~0.09%.
    rate = theoretical_independence_baseline_rate(5, 3)
    assert 0.0007 < rate < 0.0011


def test_theoretical_baseline_rejects_invalid_K():
    with pytest.raises(ValueError):
        theoretical_independence_baseline_rate(5, 0)
    with pytest.raises(ValueError):
        theoretical_independence_baseline_rate(5, 6)


def test_p_anomaly_2sigma_normal_matches_documented_value():
    # Resonance_M_Operator_Empiryczny.md §2: P(|Z|>2) ~ 0.0455.
    assert P_ANOMALY_2SIGMA_NORMAL == pytest.approx(0.0455, abs=0.0001)


# ---------------------------------------------------------------------
# compute_anomaly_flags -- reuzycie chronosignal.anomalia_flags
# ---------------------------------------------------------------------

def test_compute_anomaly_flags_accepts_dict_and_matches_manual_hand_case():
    # Ten sam recznie policzalny przypadek co
    # test_chronosignal.test_anomalia_flags_hand_constructed.
    series = [0, 0, 0, 0, 0, 0, 0, 0, 0, 20]
    flags = compute_anomaly_flags({"p1": series}, k=2.0)
    expected = np.array([False] * 9 + [True])
    assert len(flags) == 1
    np.testing.assert_array_equal(flags[0], expected)


def test_compute_anomaly_flags_rejects_empty_input():
    with pytest.raises(ValueError):
        compute_anomaly_flags([])


# ---------------------------------------------------------------------
# _decide_calibration -- logika czysta, recznie dobrane liczby
# ---------------------------------------------------------------------

def test_decide_calibration_insufficient_power_zero_real_events():
    # Dokladnie sytuacja Krakow_Centrum z docs/REAL_DATA_VALIDATION.md:
    # 0 zdarzen -> matematycznie p=1.0, has_power musi byc False
    # niezaleznie od kontroli pozytywnej.
    action, recommended_K, n_tested, reason = _decide_calibration(
        n=3, K=3, real_event_count=0,
        permutation_pvalue=1.0, positive_control_pvalue=0.0002,
        alpha=0.05, evaluate_candidate=lambda k: 1.0,
    )
    assert action == "insufficient_power"
    assert recommended_K is None
    assert n_tested == 0
    assert "MATEMATYCZNIE" in reason


def test_decide_calibration_insufficient_power_weak_positive_control():
    # Zdarzenia > 0, ale nawet wymuszona kontrola pozytywna nie
    # osiaga istotnosci -> test strukturalnie nie ma mocy.
    action, recommended_K, n_tested, reason = _decide_calibration(
        n=5, K=3, real_event_count=2,
        permutation_pvalue=0.03, positive_control_pvalue=0.2,
        alpha=0.05, evaluate_candidate=lambda k: 1.0,
    )
    assert action == "insufficient_power"
    assert recommended_K is None


def test_decide_calibration_no_change_when_powered_but_not_significant():
    action, recommended_K, n_tested, reason = _decide_calibration(
        n=5, K=3, real_event_count=4,
        permutation_pvalue=0.4, positive_control_pvalue=0.001,
        alpha=0.05, evaluate_candidate=lambda k: 1.0,
    )
    assert action == "no_calibration_needed"
    assert recommended_K == 3
    assert n_tested == 0


def test_decide_calibration_recommends_first_resolving_K():
    # n=5, K=3 -> kandydaci [4, 5]. K=4 rozwiazuje nadmiar (skorygowane
    # p = min(1, 0.5*2) = 1.0 >= alpha) -> rekomendowane K=4, nie K=5.
    def evaluate(k):
        return {4: 0.5, 5: 0.9}[k]

    action, recommended_K, n_tested, reason = _decide_calibration(
        n=5, K=3, real_event_count=6,
        permutation_pvalue=0.001, positive_control_pvalue=0.0005,
        alpha=0.05, evaluate_candidate=evaluate,
    )
    assert action == "calibration_recommended"
    assert recommended_K == 4
    assert n_tested == 2  # liczba kandydatow w przestrzeni [4,5], nie tylko przetestowanych do stopu


def test_decide_calibration_falls_back_to_max_K_when_no_candidate_resolves():
    def evaluate(k):
        return 0.001  # skorygowane: min(1, 0.001*2)=0.002 < alpha dla obu

    action, recommended_K, n_tested, reason = _decide_calibration(
        n=5, K=3, real_event_count=6,
        permutation_pvalue=0.001, positive_control_pvalue=0.0005,
        alpha=0.05, evaluate_candidate=evaluate,
    )
    assert action == "calibration_recommended"
    assert recommended_K == 5  # maksimum dostepne, n_params
    assert n_tested == 2


def test_decide_calibration_no_candidates_when_K_equals_n():
    action, recommended_K, n_tested, reason = _decide_calibration(
        n=3, K=3, real_event_count=5,
        permutation_pvalue=0.001, positive_control_pvalue=0.0005,
        alpha=0.05, evaluate_candidate=lambda k: 1.0,
    )
    assert action == "calibration_recommended"
    assert recommended_K == 3
    assert n_tested == 0


# ---------------------------------------------------------------------
# calibrate_resonance_K -- scenariusz z REALNYM sygnalem i moca
# (dane inzynierowane, udokumentowane jako syntetyczne, NIE realne
# pomiary -- ale skonstruowane tak, zeby odchylenie bylo ekstremalne w
# obie strony i wiec zdeterminowane niezaleznie od konkretnego ziarna
# permutacji)
# ---------------------------------------------------------------------

def _build_synthetic_powered_dataset():
    """n=5 parametrow, T=100 dni, zbudowane WPROST (bez RNG), zeby:

    - przy K=3: rezonans wystepuje DOKLADNIE na 8 "klastrowych" dniach
      (indeksy 10..17), gdzie parametry 0,1,2 sa jednoczesnie anomalne
      (symulacja realnego, skorelowanego zdarzenia -- np. wspolna
      przyczyna fizyczna wplywajaca na 3 z 5 czujnikow naraz). Wszystkie
      pozostale dni maja co najwyzej 1 anomalny parametr na raz
      (indeksy rozlaczne per parametr: 20-21 tylko param0, 30-31 tylko
      param1, 40-41 tylko param2, 50-59 tylko param3, 60-69 tylko
      param4) -- zaden inny dzien nie osiaga K=3.
      Marginalna liczba anomalii per parametr: 10/100 (10%) dla
      wszystkich piecu.
      Oczekiwana pod niezaleznoscia (shuffle kazdego parametru z
      zachowaniem jego wlasnej liczby 10/100=0.1): P(>=3 z 5, p=0.1)
      = C(5,3)*0.1^3*0.9^2 + C(5,4)*0.1^4*0.9 + 0.1^5
      = 0.0081 + 0.00045 + 0.00001 = 0.00856/dzien
      -> oczekiwane ~0.856 zdarzenia na 100 dni. Realne = 8 -> ~9.3x
      wiecej niz oczekiwane, ogon Poissona(0.856) dla >=8 jest
      pomijalnie maly -> permutacyjne p musi wypasc na (lub bardzo
      blisko) minimum 1/(n_perm+1), niezaleznie od ziarna.

    - przy K=4: te same 8 "klastrowych" dni maja TYLKO 3 anomalne
      parametry (nie 4) -> zaden dzien w calym oknie nie ma >=4
      jednoczesnie anomalnych -> realna stopa = 0/100 DOKLADNIE.
      Zgodnie z `_decide_calibration`/dowodem w calibration.py, stopa
      realna=0 daje MATEMATYCZNIE p=1.0 (kazda permutacja ma stope
      >=0) -- podniesienie progu o 1 usuwa caly nadmiar z K=3, bo
      klaster byl dokladnie 3-way, nie pelnym 5-way rezonansem.
    """
    T = 100
    flags = [np.zeros(T, dtype=bool) for _ in range(5)]
    cluster_days = range(10, 18)  # 8 dni
    for d in cluster_days:
        flags[0][d] = True
        flags[1][d] = True
        flags[2][d] = True
    for d in (20, 21):
        flags[0][d] = True
    for d in (30, 31):
        flags[1][d] = True
    for d in (40, 41):
        flags[2][d] = True
    for d in range(50, 60):
        flags[3][d] = True
    for d in range(60, 70):
        flags[4][d] = True
    return flags


def test_calibrate_resonance_K_recommends_higher_K_on_engineered_overtriggering():
    flags = _build_synthetic_powered_dataset()
    # Sanity check konstrukcji (przed samym testem kalibracji): dokladnie
    # 10 anomalii per parametr, dokladnie 8 dni z >=3 rownoczesnie.
    assert [int(f.sum()) for f in flags] == [10, 10, 10, 10, 10]
    counts = np.sum(np.vstack(flags), axis=0)
    assert int(np.sum(counts >= 3)) == 8
    assert int(np.sum(counts >= 4)) == 0

    result = calibrate_resonance_K(flags, K=3, n_perm=5000, alpha=0.05, seed=0)

    assert result.n_params == 5
    assert result.n_timesteps == 100
    assert result.real_event_count == 8
    assert result.empirical_rate == pytest.approx(0.08)
    # Ekstremalna separacja (8 realnych vs ~0.86 oczekiwane pod
    # niezaleznoscia) -> p musi byc astronomicznie male niezaleznie od
    # konkretnych 5000 permutacji danego ziarna (patrz rachunek w
    # docstringu _build_synthetic_powered_dataset).
    assert result.permutation_pvalue < 0.01
    assert result.has_power is True
    assert result.action == "calibration_recommended"
    assert result.recommended_K == 4
    # Kandydaci dla n=5, K=3 to zakres K+1..n = [4, 5] -> 2 kandydaty
    # w przestrzeni testowanej (korekta Bonferroniego dzieli przez 2),
    # nawet jesli petla zatrzymuje sie juz na pierwszym (K=4).
    assert result.n_K_candidates_tested == 2


def test_calibrate_resonance_threshold_wrapper_matches_direct_call():
    """pipeline.calibrate_resonance_threshold to cienki wrapper --
    sprawdz, ze wynik jest identyczny z bezposrednim wywolaniem
    calibrate_resonance_K przy tych samych argumentach/ziarnie."""
    flags = _build_synthetic_powered_dataset()
    direct = calibrate_resonance_K(flags, K=3, n_perm=2000, seed=1)
    via_pipeline = calibrate_resonance_threshold(flags, K=3, n_perm=2000, seed=1)
    assert via_pipeline == direct


# ---------------------------------------------------------------------
# calibrate_resonance_K -- scenariusz niedomocniony, REALNE dane
# Krakow_Centrum (importowane z examples/real_weather_resonance_validation.py,
# nie przepisane) -- zero zdarzen rezonansu przy K=3, zgodnie z
# docs/REAL_DATA_VALIDATION.md.
# ---------------------------------------------------------------------

def test_calibrate_resonance_K_reports_insufficient_power_on_real_krakow_window():
    real_flags = [
        _weather.anomaly_flags(_weather.TEMP),
        _weather.anomaly_flags(_weather.PRESSURE),
        _weather.anomaly_flags(_weather.WIND),
    ]
    # Sanity check zgodny z docs/REAL_DATA_VALIDATION.md: cisnienie
    # nigdy nie przekracza progu 2-sigma w tym oknie.
    assert int(real_flags[1].sum()) == 0

    result = calibrate_resonance_K(real_flags, K=3, n_perm=2000, alpha=0.05, seed=2026)

    assert result.n_params == 3
    assert result.n_timesteps == 24
    # Zero zdarzen przy K=3 na realnym oknie -> matematycznie p=1.0,
    # niezaleznie od ziarna (patrz docs/REAL_DATA_VALIDATION.md: p=1.0
    # dla K=3 na tych samych danych).
    assert result.real_event_count == 0
    assert result.permutation_pvalue == pytest.approx(1.0)
    assert result.has_power is False
    assert result.action == "insufficient_power"
    assert result.recommended_K is None
    assert "MATEMATYCZNIE" in result.reason


def test_calibrate_resonance_K_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        calibrate_resonance_K([np.array([True, False]), np.array([True])], K=1)


def test_calibrate_resonance_K_rejects_invalid_K():
    flags = [np.array([True, False, True]), np.array([False, False, True])]
    with pytest.raises(ValueError):
        calibrate_resonance_K(flags, K=0)
    with pytest.raises(ValueError):
        calibrate_resonance_K(flags, K=3)


# ---------------------------------------------------------------------
# CSV loader -- generyczny, testowany na syntetycznym pliku tymczasowym
# (nie na zewnetrznym pliku spoza repo, zeby test byl przenosny/nie
# zalezal od tego, czy sibling-repo synoptyk-v2.0-main jest obecny)
# ---------------------------------------------------------------------

_CSV_HEADER = "station,target_date,pull_seq,max_temp_c,pressure_hpa,wind_kmh,source\n"


def test_load_parameter_windows_from_csv_dedup_filters_and_sorts(tmp_path):
    csv_path = tmp_path / "snapshots.csv"
    csv_path.write_text(
        _CSV_HEADER
        # Inna stacja -- musi byc odfiltrowana.
        + "Inna_Stacja,2026-08-09,1,99,999,99,IMGW_real_15:00\n"
        # Prognoza (zly prefiks source) -- odfiltrowana.
        + "Krakow_Centrum,2026-08-10,1,20,1000,10,prognoza\n"
        # Duplikat daty 08-09 -- PIERWSZY wpis (zostanie nadpisany).
        + "Krakow_Centrum,2026-08-09,1,10,1000,5,IMGW_real_08:00\n"
        # Duplikat daty 08-09 -- DRUGI/OSTATNI wpis (ten ma wygrac).
        + "Krakow_Centrum,2026-08-09,2,26.8,1018,14.4,IMGW_real_15:00\n"
        # Brak pressure_hpa -- wiersz pomijany calkowicie.
        + "Krakow_Centrum,2026-08-11,1,25,,12,IMGW_real_15:00\n"
        + "Krakow_Centrum,2026-08-12,1,21,1027,15,web_szukaj_manual\n",
        encoding="utf-8",
    )

    dates, columns = load_parameter_windows_from_csv(
        csv_path,
        value_columns=("max_temp_c", "pressure_hpa", "wind_kmh"),
        date_column="target_date",
        group_column="station",
        group_value="Krakow_Centrum",
        source_column="source",
        source_prefixes=("IMGW_real_", "web_szukaj_"),
    )

    assert dates == ["2026-08-09", "2026-08-12"]
    np.testing.assert_array_equal(columns["max_temp_c"], np.array([26.8, 21.0]))
    np.testing.assert_array_equal(columns["pressure_hpa"], np.array([1018.0, 1027.0]))
    np.testing.assert_array_equal(columns["wind_kmh"], np.array([14.4, 15.0]))


def test_load_parameter_windows_from_csv_raises_when_nothing_matches(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text(_CSV_HEADER, encoding="utf-8")
    with pytest.raises(ValueError):
        load_parameter_windows_from_csv(
            csv_path, value_columns=("max_temp_c",), group_column="station",
            group_value="Krakow_Centrum",
        )


def test_load_krakow_weather_window_wrapper_uses_documented_filters(tmp_path):
    csv_path = tmp_path / "snapshots.csv"
    csv_path.write_text(
        _CSV_HEADER
        + "Krakow_Centrum,2026-08-09,1,26.8,1018,14.4,IMGW_real_15:00\n"
        + "Krakow_Centrum,2026-08-10,1,29.2,1012,14.4,OpenMeteo_real_dailymax\n"
        + "Krakow_Centrum,2026-08-11,1,99,999,99,prognoza\n",
        encoding="utf-8",
    )
    dates, columns = load_krakow_weather_window(csv_path)
    assert dates == ["2026-08-09", "2026-08-10"]
    assert set(columns.keys()) == {"max_temp_c", "pressure_hpa", "wind_kmh"}
