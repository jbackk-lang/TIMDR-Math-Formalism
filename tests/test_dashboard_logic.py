"""
Testy logiki "Własnego scenariusza" w dashboard.py (bez uruchamiania
GUI — samo `import dashboard` nie tworzy okna Tk, więc działa
headless; jeśli tkinter nie jest zainstalowany w ogóle, testy są
pomijane).
"""
import numpy as np
import pytest

tk = pytest.importorskip("tkinter")

import dashboard  # noqa: E402


# ---------------------------------------------------------------------
# _make_metric_fn — deterministyczne, ręcznie policzalne przypadki
# ---------------------------------------------------------------------

def test_metric_mean():
    fn = dashboard._make_metric_fn("mean", {})
    assert fn(np.array([1, 2, 3, 4])) == pytest.approx(2.5)


def test_metric_median():
    fn = dashboard._make_metric_fn("median", {})
    assert fn(np.array([1, 2, 3, 4])) == pytest.approx(2.5)


def test_metric_frac_modulo():
    fn = dashboard._make_metric_fn("frac_modulo", {"modulus": 2, "remainder": 0})
    # parzyste: 2, 4, 6 -> 3/6
    assert fn(np.array([1, 2, 3, 4, 5, 6])) == pytest.approx(0.5)


def test_metric_frac_threshold():
    fn = dashboard._make_metric_fn("frac_threshold", {"threshold": 3})
    # >3: 4, 5 -> 2/5
    assert fn(np.array([1, 2, 3, 4, 5])) == pytest.approx(0.4)


def test_metric_unknown_raises():
    with pytest.raises(ValueError):
        dashboard._make_metric_fn("nieznana", {})


# ---------------------------------------------------------------------
# _make_source_fn
# ---------------------------------------------------------------------

def test_source_random_bounds():
    fn = dashboard._make_source_fn("random", {"low": 10, "high": 20})
    values = fn(500, 0)
    assert values.min() >= 10
    assert values.max() < 20


def test_source_ar1_shape():
    fn = dashboard._make_source_fn("ar1", {"phi": 0.5, "sigma": 1.0})
    values = fn(200, 1)
    assert values.shape == (200,)
    assert np.all(np.isfinite(values))


def test_source_primes_not_resampleable():
    with pytest.raises(ValueError):
        dashboard._make_source_fn("primes", {})


# ---------------------------------------------------------------------
# _make_positive_injector — sprawdza, że wstrzyknięty efekt faktycznie
# przesuwa metrykę w oczekiwaną stronę (nie testuje istotności
# statystycznej, tylko mechanikę wstrzykiwania).
# ---------------------------------------------------------------------

def test_positive_injector_mean_shift():
    bg_fn = dashboard._make_source_fn("ar1", {"phi": 0.0, "sigma": 0.01})
    injector = dashboard._make_positive_injector(bg_fn, "mean", {}, effect_shift=100.0, bias_strength=0.5)
    injected = injector(50, 0)
    plain = bg_fn(50, 0)
    # Przesunięcie o 100 przy sigma=0.01 musi być ewidentnie widoczne.
    assert np.mean(injected) - np.mean(plain) == pytest.approx(100.0, abs=1e-6)


def test_positive_injector_frac_modulo_forces_condition():
    bg_fn = dashboard._make_source_fn("random", {"low": 1, "high": 1000})
    injector = dashboard._make_positive_injector(
        bg_fn, "frac_modulo", {"modulus": 2, "remainder": 0}, effect_shift=0.0, bias_strength=1.0,
    )
    injected = injector(100, 0)
    # bias_strength=1.0 -> CAŁA próbka wymuszona na warunek (parzyste).
    assert np.all(injected.astype(np.int64) % 2 == 0)


def test_positive_injector_frac_threshold_forces_condition():
    bg_fn = dashboard._make_source_fn("random", {"low": 0, "high": 10})
    injector = dashboard._make_positive_injector(
        bg_fn, "frac_threshold", {"threshold": 5.0}, effect_shift=0.0, bias_strength=1.0,
    )
    injected = injector(100, 0)
    assert np.all(injected > 5.0)


# ---------------------------------------------------------------------
# scenario_custom — pełny przebieg end-to-end, z gwarantowaną separacją
# (duże Δ względem małej sigma tła), żeby wynik był zdeterminowany przez
# konstrukcję, nie przez konkretny seed.
# ---------------------------------------------------------------------

def test_scenario_custom_mean_shift_passes_gate_and_detects_effect():
    params = {
        "seed": 7,
        "n_windows": 20,
        "window_size": 50,
        "name": "test_scenario",
        "description": "grupa testowa ma wyższą średnią niż tło",
        "effect_description": "różnica średnich",
        "test_source": "random",
        "test_params": {"low": 1000, "high": 1010},  # średnia ~1005, znacznie powyżej tła
        "bg_source": "random",
        "bg_params": {"low": 0, "high": 10},  # średnia ~5
        "metric": "mean",
        "metric_params": {},
        "effect_shift": 50.0,  # duże względem std tła (~2.9) -> kontrola powinna przejść
        "bias_strength": 0.5,
    }
    hypothesis, prereg, controls, main_result, test_vals, bg_vals = dashboard.scenario_custom(params)

    assert hypothesis.name == "test_scenario"
    assert prereg.fingerprint  # niepusty fingerprint
    assert controls.passed is True
    assert main_result is not None
    assert len(test_vals) == len(bg_vals) == params["n_windows"]
    # Test [1000,1010) vs tło [0,10) -> ewidentna, ogromna separacja średnich.
    assert main_result.pvalue < 1e-6
    assert main_result.median_test > main_result.median_background


def test_presets_reference_valid_dropdown_values():
    """Regresja: każdy klucz w PRESETS musi mieć odpowiednik w
    _PRESET_KEY_TO_VAR, a test_source/bg_source/metric muszą być
    dokładnie takimi etykietami, jakie GUI wystawia w Comboboxach
    (inaczej _load_preset ustawiłby wartość, której combobox nie zna)."""
    for name, preset in dashboard.PRESETS.items():
        for key in preset:
            assert key in dashboard._PRESET_KEY_TO_VAR, f"{name}: nieznany klucz {key!r}"
        assert preset["test_source"] in dashboard.TEST_SOURCES, name
        assert preset["bg_source"] in dashboard.BG_SOURCES, name
        assert preset["metric"] in dashboard.METRICS, name


def test_presets_run_end_to_end():
    """Każdy gotowy przykład musi dać się przepuścić przez
    scenario_custom() bez wyjątku (kontrola może NIE przejść — to
    dopuszczalny, uczciwy wynik — ale kod nie może się wysypać)."""
    for name, preset in dashboard.PRESETS.items():
        test_params = {}
        if preset["test_source"] == "Liczby pierwsze":
            test_params = {"n_max": preset["test_nmax"]}
        elif preset["test_source"] == "Losowe całkowite":
            test_params = {"low": preset["test_low"], "high": preset["test_high"]}
        elif preset["test_source"] == "Szum AR(1)":
            test_params = {"phi": preset["test_phi"], "sigma": preset["test_sigma"]}

        bg_params = {}
        if preset["bg_source"] == "Losowe całkowite":
            bg_params = {"low": preset["bg_low"], "high": preset["bg_high"]}
        elif preset["bg_source"] == "Szum AR(1)":
            bg_params = {"phi": preset["bg_phi"], "sigma": preset["bg_sigma"]}

        metric_params = {}
        if preset["metric"] == "Frakcja: x mod m == r":
            metric_params = {"modulus": preset["modulus"], "remainder": preset["remainder"]}
        elif preset["metric"] == "Frakcja: x > próg":
            metric_params = {"threshold": preset["threshold"]}

        params = {
            "seed": preset["seed"],
            "n_windows": preset["n_windows"],
            "window_size": preset["window_size"],
            "name": preset["name"],
            "description": preset["description"],
            "effect_description": preset["effect_description"],
            "test_source": dashboard._SOURCE_LABEL_TO_KEY[preset["test_source"]],
            "test_params": test_params,
            "bg_source": dashboard._SOURCE_LABEL_TO_KEY[preset["bg_source"]],
            "bg_params": bg_params,
            "metric": dashboard._METRIC_LABEL_TO_KEY[preset["metric"]],
            "metric_params": metric_params,
            "effect_shift": preset["effect_shift"],
            "bias_strength": preset["bias_strength"],
        }
        result = dashboard.scenario_custom(params)
        hypothesis, prereg, controls, main_result, test_vals, bg_vals = result
        assert hypothesis.name == preset["name"], name
        assert controls.passed is True, f"{name}: kontrola +/- nie przeszła z domyślnymi parametrami"
        assert main_result is not None, name


def test_scenario_custom_raises_when_too_few_windows():
    params = {
        "seed": 1,
        "n_windows": 5,
        "window_size": 10,
        "name": "x",
        "description": "x",
        "effect_description": "x",
        "test_source": "primes",
        "test_params": {"n_max": 3},  # tylko 2 liczby pierwsze (2,3) -> 0 pełnych okien
        "bg_source": "random",
        "bg_params": {"low": 0, "high": 100},
        "metric": "mean",
        "metric_params": {},
        "effect_shift": 50.0,
        "bias_strength": 0.5,
    }
    with pytest.raises(ValueError):
        dashboard.scenario_custom(params)
