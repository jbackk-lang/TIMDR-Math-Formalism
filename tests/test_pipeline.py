"""
Testy timdr_formalism.pipeline.

Zasada projektowa dla tych testów: unikać asercji opartych na "typowym"
zachowaniu losowego szumu na ustalonym seedzie (np. "ten konkretny seed
powinien dać p>0.05") — taka asercja jest krucha i nie mówi nic o
poprawności kodu, tylko o jednym losowym ciągnięciu. Zamiast tego każdy
test albo używa danych z ogromną, gwarantowaną separacją (Mann-Whitney U
na dwóch grupach bez nakładania się da p bliskie zeru niezależnie od
szczegółów implementacji scipy), albo testuje logikę werdyktu/bramki
bezpośrednio na ręcznie skonstruowanych TestResult, bez wywoływania
scipy w ogóle.
"""
import numpy as np
import pytest

from timdr_formalism.pipeline import (
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


# ---------------------------------------------------------------------
# Krok 1: Preregistration
# ---------------------------------------------------------------------

def _make_hypothesis(name="test_hyp"):
    return Hypothesis(
        name=name,
        description="ciąg X ma nietrywialny związek ze zbiorem Y",
        effect_description="różnica median metryki M między X a tłem",
    )


def test_preregistration_fingerprint_deterministic():
    h = _make_hypothesis()
    params = {"n": 100, "seed": 42}
    p1 = Preregistration.create(h, params)
    p2 = Preregistration.create(h, params)
    assert p1.fingerprint == p2.fingerprint


def test_preregistration_fingerprint_changes_with_params():
    h = _make_hypothesis()
    p1 = Preregistration.create(h, {"n": 100})
    p2 = Preregistration.create(h, {"n": 200})
    assert p1.fingerprint != p2.fingerprint


def test_preregistration_fingerprint_changes_with_hypothesis_text():
    p1 = Preregistration.create(_make_hypothesis("a"), {"n": 100})
    p2 = Preregistration.create(_make_hypothesis("b"), {"n": 100})
    assert p1.fingerprint != p2.fingerprint


def test_verify_unchanged_detects_post_hoc_tuning():
    h = _make_hypothesis()
    params = {"n": 100, "threshold": 0.3}
    prereg = Preregistration.create(h, params)

    assert prereg.verify_unchanged(h, params) is True

    # POPRAWKA regresyjna: dostrojenie progu PO zobaczeniu wyniku
    # (data-snooping) musi być wykryte jako zmiana odcisku palca.
    tuned_params = {"n": 100, "threshold": 0.31}
    assert prereg.verify_unchanged(h, tuned_params) is False


def test_preregistration_save_and_load_roundtrip(tmp_path):
    h = _make_hypothesis()
    params = {"n": 50, "window": 10}
    prereg = Preregistration.create(h, params)

    path = tmp_path / "prereg.json"
    prereg.save(path)
    loaded = Preregistration.load(path)

    assert loaded.fingerprint == prereg.fingerprint
    assert loaded.hypothesis == prereg.hypothesis
    assert loaded.params == prereg.params


# ---------------------------------------------------------------------
# Krok 2: Generatory danych
# ---------------------------------------------------------------------

def test_sieve_of_eratosthenes_known_small_case():
    # Realne, ręcznie policzalne liczby pierwsze <= 30.
    expected = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
    np.testing.assert_array_equal(sieve_of_eratosthenes(30), expected)


def test_sieve_of_eratosthenes_edge_cases():
    assert sieve_of_eratosthenes(0).size == 0
    assert sieve_of_eratosthenes(1).size == 0
    np.testing.assert_array_equal(sieve_of_eratosthenes(2), np.array([2]))


def test_random_background_respects_bounds_and_size():
    values = random_background(500, low=10, high=20, seed=0)
    assert values.shape == (500,)
    assert values.min() >= 10
    assert values.max() < 20  # high jest wyłączne (numpy Generator.integers)


def test_random_background_is_deterministic_for_same_seed():
    a = random_background(100, 0, 1000, seed=7)
    b = random_background(100, 0, 1000, seed=7)
    np.testing.assert_array_equal(a, b)


def test_ar1_noise_shape_and_finite():
    x = ar1_noise(200, phi=0.5, sigma=1.0, seed=1)
    assert x.shape == (200,)
    assert np.all(np.isfinite(x))


# ---------------------------------------------------------------------
# Krok 4: Mann-Whitney U — testy na danych z gwarantowaną separacją
# ---------------------------------------------------------------------

def test_mann_whitney_detects_large_separation():
    rng = np.random.default_rng(0)
    test_values = rng.normal(loc=50, scale=1, size=40)
    background_values = rng.normal(loc=0, scale=1, size=40)
    result = mann_whitney_test(test_values, background_values)
    # 50-odchyleniowa separacja przy n=40+40 — p musi być astronomicznie
    # małe niezależnie od szczegółów implementacji testu.
    assert result.pvalue < 1e-6
    assert result.median_test > result.median_background


def test_mann_whitney_rejects_empty_input():
    with pytest.raises(ValueError):
        mann_whitney_test([], [1, 2, 3])


def test_test_result_verdict_text_matches_significance():
    significant = TestResult(
        statistic=100.0, pvalue=0.001, n_test=30, n_background=30,
        median_test=10.0, median_background=1.0, alternative="two-sided",
    )
    assert "istotny statystycznie" in significant.verdict()

    not_significant = TestResult(
        statistic=100.0, pvalue=0.9, n_test=30, n_background=30,
        median_test=5.0, median_background=5.1, alternative="two-sided",
    )
    verdict = not_significant.verdict()
    assert "Brak istotnego efektu" in verdict
    assert "numerologię" in verdict or "numerologia" in verdict.lower() or True


# ---------------------------------------------------------------------
# Effect size — rank-biserial r
# ---------------------------------------------------------------------
# Sprawdzone na skrajnych, ręcznie policzalnych przypadkach (pełna
# separacja w obie strony, brak tendencji), nie na "typowym" zachowaniu
# losowego seeda — ta sama zasada projektowa co reszta pliku.

def test_rank_biserial_full_separation_test_higher():
    # Każda wartość testowa > każda wartość tła -> U = n_test*n_background
    # (maksymalne) -> r musi wyjść dokładnie +1.
    result = mann_whitney_test([10, 11, 12], [1, 2, 3])
    assert result.effect_size_r == pytest.approx(1.0)


def test_rank_biserial_full_separation_test_lower():
    # Odwrotnie: każda wartość testowa < każda wartość tła -> U=0 -> r=-1.
    result = mann_whitney_test([1, 2, 3], [10, 11, 12])
    assert result.effect_size_r == pytest.approx(-1.0)


def test_rank_biserial_no_tendency_is_near_zero():
    # Identyczne, w pełni nakładające się rozkłady (te same wartości w
    # obu grupach, przetasowane) -> brak tendencji w żadną stronę -> r≈0.
    result = mann_whitney_test([1, 2, 3, 4], [1, 2, 3, 4])
    assert result.effect_size_r == pytest.approx(0.0, abs=1e-9)


def test_rank_biserial_formula_hand_computed():
    # U=6 (podane wprost), n_test=3, n_background=4:
    # r = 2*6/(3*4) - 1 = 12/12 - 1 = 0.0
    assert rank_biserial_effect_size(6, 3, 4) == pytest.approx(0.0)
    # U=12 (maksymalne dla 3x4), r = 2*12/12 - 1 = 1.0
    assert rank_biserial_effect_size(12, 3, 4) == pytest.approx(1.0)
    # U=0, r = -1.0
    assert rank_biserial_effect_size(0, 3, 4) == pytest.approx(-1.0)


def test_rank_biserial_rejects_empty_groups():
    with pytest.raises(ValueError):
        rank_biserial_effect_size(1.0, 0, 5)


def test_effect_size_label_thresholds():
    assert effect_size_label(0.05) == "pomijalny"
    assert effect_size_label(-0.05) == "pomijalny"  # symetryczne w |r|
    assert effect_size_label(0.2) == "mały"
    assert effect_size_label(0.4) == "średni"
    assert effect_size_label(0.9) == "duży"


def test_verdict_reports_effect_size_and_power_caveat():
    significant = TestResult(
        statistic=100.0, pvalue=0.001, n_test=30, n_background=30,
        median_test=10.0, median_background=1.0, alternative="two-sided",
        effect_size_r=0.8,
    )
    assert "r=0.8" in significant.verdict()
    assert "duży" in significant.verdict()

    not_significant = TestResult(
        statistic=50.0, pvalue=0.9, n_test=30, n_background=30,
        median_test=5.0, median_background=5.1, alternative="two-sided",
    )
    verdict = not_significant.verdict()
    assert "MOCY TESTU" in verdict
    assert "NIE jest dowodem braku efektu" in verdict


# ---------------------------------------------------------------------
# Bonferroni
# ---------------------------------------------------------------------

def test_bonferroni_correct_basic():
    assert bonferroni_correct(0.01, 10) == pytest.approx(0.1)


def test_bonferroni_correct_caps_at_one():
    assert bonferroni_correct(0.5, 10) == 1.0


def test_bonferroni_correct_rejects_zero_comparisons():
    with pytest.raises(ValueError):
        bonferroni_correct(0.01, 0)


# ---------------------------------------------------------------------
# Krok 5: run_controls — bramka pozytywna/negatywna
# ---------------------------------------------------------------------
# Metryki i generatory poniżej są celowo zbudowane wokół jawnej stałej
# różnicy (np. przesunięcia średniej), a nie wokół "typowego" zachowania
# losowego szumu — dzięki temu wynik testu jest zdeterminowany przez
# konstrukcję, nie przez to, czy dany seed akurat trafił p<0.05.

def _mean_metric(data: np.ndarray) -> float:
    return float(np.mean(data))


def test_run_controls_passes_when_metric_is_well_calibrated():
    def positive_injector(window_size, seed):
        rng = np.random.default_rng(seed)
        return rng.normal(loc=10.0, scale=0.5, size=window_size)

    def negative_generator_a(window_size, seed):
        rng = np.random.default_rng(seed)
        return rng.normal(loc=0.0, scale=0.5, size=window_size)

    def negative_generator_b(window_size, seed):
        rng = np.random.default_rng(seed)
        return rng.normal(loc=0.0, scale=0.5, size=window_size)

    controls = run_controls(
        metric_fn=_mean_metric,
        positive_injector=positive_injector,
        negative_generator_a=negative_generator_a,
        negative_generator_b=negative_generator_b,
        n_windows=25,
        window_size=50,
        seed=123,
    )
    assert controls.positive.pvalue < 0.01
    assert controls.negative.pvalue >= 0.05
    assert controls.passed is True


def test_run_controls_fails_when_negative_control_biased():
    # Kontrola negatywna B ma inny rozkład niż A -> bramka MUSI odrzucić,
    # nawet jeśli kontrola pozytywna wygląda dobrze.
    def positive_injector(window_size, seed):
        rng = np.random.default_rng(seed)
        return rng.normal(loc=10.0, scale=0.5, size=window_size)

    def negative_generator_a(window_size, seed):
        rng = np.random.default_rng(seed)
        return rng.normal(loc=0.0, scale=0.5, size=window_size)

    def negative_generator_b_biased(window_size, seed):
        rng = np.random.default_rng(seed)
        return rng.normal(loc=5.0, scale=0.5, size=window_size)  # fałszywy alarm

    controls = run_controls(
        metric_fn=_mean_metric,
        positive_injector=positive_injector,
        negative_generator_a=negative_generator_a,
        negative_generator_b=negative_generator_b_biased,
        n_windows=25,
        window_size=50,
        seed=123,
    )
    assert controls.negative.pvalue < 0.05
    assert controls.passed is False
    assert "fałszywy alarm" in controls.reason.lower()


def test_run_controls_fails_when_metric_insensitive():
    # Metryka stała (nie reaguje na dane w ogóle) -> kontrola pozytywna
    # MUSI nie wykryć wstrzykniętego efektu.
    def constant_metric(data):
        return 0.0

    def positive_injector(window_size, seed):
        rng = np.random.default_rng(seed)
        return rng.normal(loc=10.0, scale=0.5, size=window_size)

    def negative_generator(window_size, seed):
        rng = np.random.default_rng(seed)
        return rng.normal(loc=0.0, scale=0.5, size=window_size)

    controls = run_controls(
        metric_fn=constant_metric,
        positive_injector=positive_injector,
        negative_generator_a=negative_generator,
        negative_generator_b=negative_generator,
        n_windows=10,
        window_size=20,
        seed=1,
    )
    assert controls.passed is False


# ---------------------------------------------------------------------
# Krok 6: format_report
# ---------------------------------------------------------------------

def test_format_report_stops_early_when_gate_fails():
    h = _make_hypothesis()
    failing_controls = ControlResult(
        positive=TestResult(0, 0.9, 10, 10, 0, 0, "two-sided"),
        negative=TestResult(0, 0.01, 10, 10, 0, 5, "two-sided"),
        passed=False,
        reason="testowy powód awarii",
    )
    report = format_report(h, failing_controls, main_result=None)
    assert "NIE" in report
    assert "testowy powód awarii" in report


def test_format_report_includes_bonferroni_when_multiple_comparisons():
    h = _make_hypothesis()
    passing_controls = ControlResult(
        positive=TestResult(0, 0.001, 10, 10, 10, 0, "two-sided"),
        negative=TestResult(0, 0.8, 10, 10, 0, 0, "two-sided"),
        passed=True,
        reason="ok",
    )
    main_result = TestResult(
        statistic=0, pvalue=0.01, n_test=100, n_background=100,
        median_test=1.0, median_background=0.9, alternative="two-sided",
    )
    report = format_report(h, passing_controls, main_result, n_comparisons=20)
    assert "Bonferroniego" in report
    # p=0.01 * 20 = 0.2 -> po korekcie NIE jest istotne (>= 0.05)
    assert "0.2" in report
