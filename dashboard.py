"""
dashboard.py

Proste GUI (Tkinter) do uruchamiania protokołu TIMDR-Math-Formalism
(docs/PROTOCOL.md) bez pisania kodu.

Dwie zakładki:
- "Gotowe scenariusze" — dwa przetestowane, gotowe przykłady (liczby
  pierwsze / regulowane przesunięcie średniej).
- "Własny scenariusz" — budowanie hipotezy z bezpiecznych klocków
  (źródło danych testowych, źródło tła, metryka, siła efektu do
  kontroli) — BEZ wykonywania jakiegokolwiek kodu wklejonego przez
  użytkownika. Wszystkie źródła danych i metryki to z góry
  zdefiniowane, sparametryzowane funkcje (patrz sekcja "Bezpieczne
  klocki" niżej).

W obu przypadkach: klikasz "Uruchom protokół", widzisz pełny raport
(pre-rejestracja, kontrola +/-, test Manna-Whitneya, werdykt) i wykres
porównujący rozkład metryki (test vs tło).
"""
from __future__ import annotations

import sys
import threading
import traceback
from pathlib import Path
from tkinter import (
    Tk, Frame, Label, Button, StringVar, DoubleVar, IntVar,
    ttk, W, E, END, DISABLED, NORMAL,
)
from tkinter.scrolledtext import ScrolledText

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from timdr_formalism.pipeline import (
    Hypothesis,
    Preregistration,
    run_controls,
    mann_whitney_test,
    format_report,
    sieve_of_eratosthenes,
    random_background,
    ar1_noise,
)

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =======================================================================
# Bezpieczne klocki: źródła danych i metryki dla "Własnego scenariusza".
# Zamknięty, z góry zdefiniowany zestaw sparametryzowanych funkcji —
# żadne pole tekstowe w tym trybie nie jest interpretowane jako kod.
# =======================================================================

def _make_source_fn(source: str, sp: dict):
    """Zwraca funkcję (window_size, seed) -> np.ndarray dla źródła
    danych, które MOŻNA resamplować niezależnie (potrzebne do kontroli
    +/- i do tła) — czyli tylko źródła losowe."""
    if source == "random":
        low, high = int(sp["low"]), int(sp["high"])
        return lambda ws, s: random_background(ws, low, high, s)
    if source == "ar1":
        phi, sigma = float(sp["phi"]), float(sp["sigma"])
        return lambda ws, s: ar1_noise(ws, phi, sigma, s)
    raise ValueError(
        f"Źródło '{source}' jest deterministyczne — nie nadaje się do "
        f"resamplingu (rola tła/kontroli), tylko do roli grupy testowej."
    )


def _make_metric_fn(metric: str, mp: dict):
    if metric == "mean":
        return lambda data: float(np.mean(data))
    if metric == "median":
        return lambda data: float(np.median(data))
    if metric == "frac_modulo":
        modulus, remainder = int(mp["modulus"]), int(mp["remainder"])
        if modulus < 1:
            raise ValueError("modulus musi być >= 1")

        def fn(data):
            ints = np.round(np.asarray(data, dtype=float)).astype(np.int64)
            return float(np.mean(ints % modulus == remainder))

        return fn
    if metric == "frac_threshold":
        threshold = float(mp["threshold"])
        return lambda data: float(np.mean(np.asarray(data, dtype=float) > threshold))
    raise ValueError(f"Nieznana metryka '{metric}'")


def _make_positive_injector(bg_source_fn, metric: str, mp: dict, effect_shift: float, bias_strength: float):
    """Kontrola pozytywna: bierze tło i wstrzykuje efekt, który dana
    metryka POWINNA wykryć — mean/median: stałe przesunięcie całej
    próbki; frakcyjne metryki: wymuszenie warunku na losowej podpróbce
    (bias_strength, 0-1) elementów."""

    def injector(window_size, seed):
        bg = bg_source_fn(window_size, seed).astype(float)
        if metric in ("mean", "median"):
            return bg + effect_shift

        rng = np.random.default_rng(seed)
        n_bias = max(1, min(len(bg), int(round(len(bg) * bias_strength))))
        idx = rng.choice(len(bg), size=n_bias, replace=False)
        out = bg.copy()

        if metric == "frac_modulo":
            modulus, remainder = int(mp["modulus"]), int(mp["remainder"])
            base = rng.integers(0, 10_000, size=n_bias)
            out[idx] = (base - (base % modulus) + remainder).astype(float)
        elif metric == "frac_threshold":
            threshold = float(mp["threshold"])
            out[idx] = threshold + rng.uniform(0.5, 5.0, size=n_bias)
        else:
            raise ValueError(f"Nieznana metryka '{metric}'")
        return out

    return injector


def scenario_custom(params: dict):
    """Buduje i uruchamia scenariusz złożony w zakładce "Własny
    scenariusz" z bezpiecznych klocków wybranych w GUI."""
    hypothesis = Hypothesis(
        name=params["name"],
        description=params["description"],
        effect_description=params["effect_description"],
    )
    prereg = Preregistration.create(hypothesis, params)

    metric_fn = _make_metric_fn(params["metric"], params["metric_params"])
    bg_source_fn = _make_source_fn(params["bg_source"], params["bg_params"])
    positive_injector = _make_positive_injector(
        bg_source_fn, params["metric"], params["metric_params"],
        params["effect_shift"], params["bias_strength"],
    )

    controls = run_controls(
        metric_fn=metric_fn,
        positive_injector=positive_injector,
        negative_generator_a=bg_source_fn,
        negative_generator_b=bg_source_fn,
        n_windows=params["n_windows"],
        window_size=params["window_size"],
        seed=params["seed"],
    )

    main_result = None
    test_windows: list = []
    background_windows: list = []

    if controls.passed:
        window = params["window_size"]
        n_windows = params["n_windows"]
        total_n = window * n_windows

        if params["test_source"] == "primes":
            full_test = sieve_of_eratosthenes(params["test_params"]["n_max"]).astype(float)
        else:
            test_source_fn = _make_source_fn(params["test_source"], params["test_params"])
            rng = np.random.default_rng(params["seed"])
            test_seed = int(rng.integers(0, 2**31 - 1))
            full_test = test_source_fn(total_n, test_seed).astype(float)

        full_bg = bg_source_fn(total_n, params["seed"] + 777).astype(float)

        n_w = min(len(full_test), len(full_bg)) // window
        if n_w < 1:
            raise ValueError(
                "Za mało danych na choć jedno okno — zwiększ N max / "
                "zmniejsz rozmiar okna."
            )
        test_windows = [metric_fn(full_test[i * window:(i + 1) * window]) for i in range(n_w)]
        background_windows = [metric_fn(full_bg[i * window:(i + 1) * window]) for i in range(n_w)]
        main_result = mann_whitney_test(test_windows, background_windows)

    return hypothesis, prereg, controls, main_result, test_windows, background_windows


# =======================================================================
# Gotowe scenariusze (jak poprzednio)
# =======================================================================

def _digit_sum_even_fraction(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.int64)
    digit_sums = np.array([sum(int(d) for d in str(int(v))) for v in values])
    return float(np.mean(digit_sums % 2 == 0))


def scenario_prime_resonance(params: dict):
    n_max = params["n_max"]
    seed = params["seed"]
    window = params["window_size"]

    hypothesis = Hypothesis(
        name="prime_digit_sum_parity",
        description=(
            "Liczby pierwsze mają inną frakcję elementów o parzystej "
            "sumie cyfr niż losowe tło liczb całkowitych w tym samym "
            "zakresie."
        ),
        effect_description=(
            "różnica we frakcji liczb o parzystej sumie cyfr między "
            "liczbami pierwszymi <= N a losowym tłem"
        ),
    )
    prereg = Preregistration.create(hypothesis, params)

    def positive_injector(window_size, s):
        rng = np.random.default_rng(s)
        return rng.integers(1, 500, size=window_size) * 2

    def negative_control(window_size, s):
        rng = np.random.default_rng(s)
        return rng.integers(1, 1000, size=window_size)

    controls = run_controls(
        metric_fn=_digit_sum_even_fraction,
        positive_injector=positive_injector,
        negative_generator_a=negative_control,
        negative_generator_b=negative_control,
        n_windows=params["n_windows"],
        window_size=window,
        seed=seed,
    )

    main_result = None
    test_windows: list = []
    background_windows: list = []
    if controls.passed:
        primes = sieve_of_eratosthenes(n_max)
        background = random_background(len(primes), 2, n_max, seed=seed)
        n_windows = min(len(primes), len(background)) // window
        test_windows = [
            _digit_sum_even_fraction(primes[i * window:(i + 1) * window])
            for i in range(n_windows)
        ]
        background_windows = [
            _digit_sum_even_fraction(background[i * window:(i + 1) * window])
            for i in range(n_windows)
        ]
        main_result = mann_whitney_test(test_windows, background_windows)

    return hypothesis, prereg, controls, main_result, test_windows, background_windows


def scenario_mean_shift(params: dict):
    effect_size = params["effect_size"]
    seed = params["seed"]
    window = params["window_size"]
    n_windows = params["n_windows"]

    hypothesis = Hypothesis(
        name="custom_mean_shift",
        description=(
            f"Grupa testowa ma średnią przesuniętą o {effect_size} "
            f"względem tła (odchylenie standardowe tła = 1.0)."
        ),
        effect_description="różnica średnich metryki między grupą testową a tłem",
    )
    prereg = Preregistration.create(hypothesis, params)

    def metric_fn(data):
        return float(np.mean(data))

    def positive_injector(ws, s):
        rng = np.random.default_rng(s)
        return rng.normal(loc=5.0, scale=1.0, size=ws)

    def negative_control(ws, s):
        rng = np.random.default_rng(s)
        return rng.normal(loc=0.0, scale=1.0, size=ws)

    def test_generator(ws, s):
        rng = np.random.default_rng(s)
        return rng.normal(loc=effect_size, scale=1.0, size=ws)

    controls = run_controls(
        metric_fn=metric_fn,
        positive_injector=positive_injector,
        negative_generator_a=negative_control,
        negative_generator_b=negative_control,
        n_windows=n_windows,
        window_size=window,
        seed=seed,
    )

    main_result = None
    test_windows: list = []
    background_windows: list = []
    if controls.passed:
        rng = np.random.default_rng(seed + 1000)
        seeds = rng.integers(0, 2**31 - 1, size=n_windows)
        test_windows = [metric_fn(test_generator(window, int(s))) for s in seeds]
        background_windows = [metric_fn(negative_control(window, int(s) + 1)) for s in seeds]
        main_result = mann_whitney_test(test_windows, background_windows)

    return hypothesis, prereg, controls, main_result, test_windows, background_windows


BUILTIN_SCENARIOS = {
    "Liczby pierwsze — suma cyfr": scenario_prime_resonance,
    "Przesunięcie średniej (syntetyczne, regulowane)": scenario_mean_shift,
}

TEST_SOURCES = ["Liczby pierwsze", "Losowe całkowite", "Szum AR(1)"]
BG_SOURCES = ["Losowe całkowite", "Szum AR(1)"]  # tylko losowe — muszą dać się resamplować
METRICS = ["Średnia", "Mediana", "Frakcja: x mod m == r", "Frakcja: x > próg"]

_SOURCE_LABEL_TO_KEY = {
    "Liczby pierwsze": "primes",
    "Losowe całkowite": "random",
    "Szum AR(1)": "ar1",
}
_METRIC_LABEL_TO_KEY = {
    "Średnia": "mean",
    "Mediana": "median",
    "Frakcja: x mod m == r": "frac_modulo",
    "Frakcja: x > próg": "frac_threshold",
}

# -----------------------------------------------------------------------
# Gotowe przykłady do wczytania w zakładce "Własny scenariusz" — pokazują
# różne kombinacje źródeł/metryk z tych samych bezpiecznych klocków.
# Klucze odpowiadają parametrom przyjmowanym przez scenario_custom()
# (przetłumaczone na wartości pól GUI przez _load_preset).
# -----------------------------------------------------------------------

PRESETS = {
    "1. Liczby pierwsze: średnia vs losowe tło": {
        "seed": 42, "n_windows": 25, "window_size": 200, "alpha": 0.05,
        "name": "prime_mean_vs_random",
        "description": (
            "Średnia wartość liczb pierwszych różni się od średniej "
            "losowych liczb całkowitych w tym samym zakresie."
        ),
        "effect_description": "różnica średnich (mean) między grupą testową a tłem",
        "test_source": "Liczby pierwsze", "test_nmax": 100_000,
        "bg_source": "Losowe całkowite", "bg_low": 0, "bg_high": 100_000,
        "metric": "Średnia",
        "effect_shift": 5_000.0, "bias_strength": 0.5,
    },
    "2. Zero efektu — kalibracja negatywna": {
        "seed": 1, "n_windows": 25, "window_size": 200, "alpha": 0.05,
        "name": "null_calibration",
        "description": (
            "Grupa testowa i tło pochodzą z DOKŁADNIE tego samego "
            "rozkładu — protokół powinien dać werdykt \"brak efektu\"."
        ),
        "effect_description": (
            "różnica średnich między dwiema próbkami z tego samego "
            "rozkładu losowego (oczekiwany brak różnicy)"
        ),
        "test_source": "Losowe całkowite", "test_low": 0, "test_high": 1_000,
        "bg_source": "Losowe całkowite", "bg_low": 0, "bg_high": 1_000,
        "metric": "Średnia",
        "effect_shift": 50.0, "bias_strength": 0.5,
    },
    "3. AR(1) vs biały szum — frakcja ekstremów": {
        "seed": 3, "n_windows": 25, "window_size": 200, "alpha": 0.05,
        "name": "ar1_vs_white_noise_extremes",
        "description": (
            "Silnie skorelowany szum AR(1) (φ=0.8) ma inną frakcję "
            "wartości powyżej progu 1.5 niż nieskorelowany szum (φ≈0)."
        ),
        "effect_description": "frakcja próbek > 1.5: AR(1) φ=0.8 vs AR(1) φ=0 (biały szum)",
        "test_source": "Szum AR(1)", "test_phi": 0.8, "test_sigma": 1.0,
        "bg_source": "Szum AR(1)", "bg_phi": 0.0, "bg_sigma": 1.0,
        "metric": "Frakcja: x > próg", "threshold": 1.5,
        "effect_shift": 0.0, "bias_strength": 0.5,
    },
    "4. Liczby pierwsze mod 4 — reszta 1": {
        "seed": 4, "n_windows": 25, "window_size": 200, "alpha": 0.05,
        "name": "primes_mod4_remainder1",
        "description": (
            "Wśród liczb pierwszych frakcja spełniająca x mod 4 = 1 "
            "różni się od losowego tła w tym samym zakresie."
        ),
        "effect_description": "frakcja liczb ≡ 1 (mod 4): liczby pierwsze vs losowe tło",
        "test_source": "Liczby pierwsze", "test_nmax": 200_000,
        "bg_source": "Losowe całkowite", "bg_low": 3, "bg_high": 200_000,
        "metric": "Frakcja: x mod m == r", "modulus": 4, "remainder": 1,
        "effect_shift": 0.0, "bias_strength": 0.5,
    },
    "5. Mediana: dwa rozłączne zakresy losowych liczb": {
        "seed": 5, "n_windows": 25, "window_size": 200, "alpha": 0.05,
        "name": "median_disjoint_ranges",
        "description": (
            "Mediana grupy testowej (losowe liczby całkowite 500–600) "
            "różni się od mediany tła (losowe liczby całkowite 0–100)."
        ),
        "effect_description": "mediana: losowe liczby [500,600) vs losowe liczby [0,100)",
        "test_source": "Losowe całkowite", "test_low": 500, "test_high": 600,
        "bg_source": "Losowe całkowite", "bg_low": 0, "bg_high": 100,
        "metric": "Mediana",
        "effect_shift": 50.0, "bias_strength": 0.5,
    },
}

_PRESET_KEY_TO_VAR = {
    "seed": "seed_var", "n_windows": "n_windows_var", "window_size": "window_size_var", "alpha": "alpha_var",
    "name": "custom_name_var", "description": "custom_description_var", "effect_description": "custom_effect_description_var",
    "test_source": "custom_test_source_var", "test_nmax": "custom_test_nmax_var",
    "test_low": "custom_test_low_var", "test_high": "custom_test_high_var",
    "test_phi": "custom_test_phi_var", "test_sigma": "custom_test_sigma_var",
    "bg_source": "custom_bg_source_var", "bg_low": "custom_bg_low_var", "bg_high": "custom_bg_high_var",
    "bg_phi": "custom_bg_phi_var", "bg_sigma": "custom_bg_sigma_var",
    "metric": "custom_metric_var", "modulus": "custom_modulus_var", "remainder": "custom_remainder_var",
    "threshold": "custom_threshold_var",
    "effect_shift": "custom_effect_shift_var", "bias_strength": "custom_bias_strength_var",
}


# =======================================================================
# GUI
# =======================================================================

class Dashboard:
    def __init__(self, root: Tk):
        self.root = root
        root.title("TIMDR-Math-Formalism — dashboard")
        root.geometry("1000x820")
        root.minsize(900, 700)

        # Wspólne parametry protokołu (obie zakładki)
        self.seed_var = IntVar(value=42)
        self.n_windows_var = IntVar(value=25)
        self.window_size_var = IntVar(value=200)
        self.alpha_var = DoubleVar(value=0.05)

        # Zakładka: gotowe scenariusze
        self.scenario_var = StringVar(value=list(BUILTIN_SCENARIOS.keys())[0])
        self.n_max_var = IntVar(value=100_000)
        self.effect_size_var = DoubleVar(value=0.5)

        # Zakładka: własny scenariusz
        self.custom_name_var = StringVar(value="prime_mean_vs_random")
        self.custom_description_var = StringVar(
            value="Średnia wartość liczb pierwszych różni się od średniej losowych liczb całkowitych w tym samym zakresie."
        )
        self.custom_effect_description_var = StringVar(
            value="różnica średnich (mean) między grupą testową a tłem"
        )
        self.custom_test_source_var = StringVar(value="Liczby pierwsze")
        self.custom_test_nmax_var = IntVar(value=100_000)
        self.custom_test_low_var = IntVar(value=0)
        self.custom_test_high_var = IntVar(value=100_000)
        self.custom_test_phi_var = DoubleVar(value=0.6)
        self.custom_test_sigma_var = DoubleVar(value=1.0)

        self.custom_bg_source_var = StringVar(value="Losowe całkowite")
        self.custom_bg_low_var = IntVar(value=0)
        self.custom_bg_high_var = IntVar(value=100_000)
        self.custom_bg_phi_var = DoubleVar(value=0.6)
        self.custom_bg_sigma_var = DoubleVar(value=1.0)

        self.custom_metric_var = StringVar(value="Średnia")
        self.custom_modulus_var = IntVar(value=2)
        self.custom_remainder_var = IntVar(value=0)
        self.custom_threshold_var = DoubleVar(value=50_000.0)
        self.custom_effect_shift_var = DoubleVar(value=5_000.0)
        self.custom_bias_strength_var = DoubleVar(value=0.5)
        self.preset_var = StringVar(value="")

        self._build_shared_controls()
        self._build_notebook()
        self._build_output()

    # -- wspólne parametry ------------------------------------------------

    def _build_shared_controls(self):
        top = Frame(self.root)
        top.pack(fill="x", padx=10, pady=(10, 0))

        Label(top, text="Seed:").grid(row=0, column=0, sticky=W)
        ttk.Entry(top, textvariable=self.seed_var, width=10).grid(row=0, column=1, sticky=W, padx=(2, 16))

        Label(top, text="Liczba okien:").grid(row=0, column=2, sticky=W)
        ttk.Entry(top, textvariable=self.n_windows_var, width=10).grid(row=0, column=3, sticky=W, padx=(2, 16))

        Label(top, text="Rozmiar okna:").grid(row=0, column=4, sticky=W)
        ttk.Entry(top, textvariable=self.window_size_var, width=10).grid(row=0, column=5, sticky=W, padx=(2, 16))

        Label(top, text="Alpha:").grid(row=0, column=6, sticky=W)
        ttk.Entry(top, textvariable=self.alpha_var, width=8).grid(row=0, column=7, sticky=W)

        self.status_label = Label(top, text="", fg="#666666")
        self.status_label.grid(row=1, column=0, columnspan=8, sticky=W, pady=(6, 0))

        if not HAS_MATPLOTLIB:
            Label(
                top,
                text="(matplotlib niedostępny — wykres pominięty, raport działa normalnie)",
                fg="#996600",
            ).grid(row=2, column=0, columnspan=8, sticky=W, pady=(2, 0))

    # -- notebook -----------------------------------------------------

    def _build_notebook(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="x", padx=10, pady=10)

        builtin_tab = Frame(self.notebook)
        custom_tab = Frame(self.notebook)
        self.notebook.add(builtin_tab, text="Gotowe scenariusze")
        self.notebook.add(custom_tab, text="Własny scenariusz")

        self._build_builtin_tab(builtin_tab)
        self._build_custom_tab(custom_tab)

    # -- zakładka: gotowe scenariusze ----------------------------------

    def _build_builtin_tab(self, parent: Frame):
        f = Frame(parent)
        f.pack(fill="x", padx=6, pady=6)

        Label(f, text="Scenariusz:").grid(row=0, column=0, sticky=W)
        box = ttk.Combobox(
            f, textvariable=self.scenario_var, values=list(BUILTIN_SCENARIOS.keys()),
            state="readonly", width=45,
        )
        box.grid(row=0, column=1, columnspan=3, sticky=W, padx=5)
        box.bind("<<ComboboxSelected>>", lambda e: self._update_builtin_visibility())

        self.n_max_label = Label(f, text="N max (liczby pierwsze do):")
        self.n_max_label.grid(row=1, column=0, sticky=W, pady=(6, 0))
        self.n_max_entry = ttk.Entry(f, textvariable=self.n_max_var, width=10)
        self.n_max_entry.grid(row=1, column=1, sticky=W, pady=(6, 0))

        self.effect_label = Label(f, text="Wielkość efektu (mean shift):")
        self.effect_label.grid(row=1, column=2, sticky=W, pady=(6, 0))
        self.effect_entry = ttk.Entry(f, textvariable=self.effect_size_var, width=10)
        self.effect_entry.grid(row=1, column=3, sticky=W, pady=(6, 0))

        self.run_button_builtin = Button(f, text="Uruchom protokół", command=self._run_builtin_clicked)
        self.run_button_builtin.grid(row=2, column=0, columnspan=4, sticky=W + E, pady=(12, 0))

        self._update_builtin_visibility()

    def _update_builtin_visibility(self):
        is_prime = self.scenario_var.get() == "Liczby pierwsze — suma cyfr"
        for w in (self.n_max_label, self.n_max_entry):
            (w.grid() if is_prime else w.grid_remove())
        for w in (self.effect_label, self.effect_entry):
            (w.grid_remove() if is_prime else w.grid())

    # -- zakładka: własny scenariusz -----------------------------------

    def _build_custom_tab(self, parent: Frame):
        SUB = {"fg": "#666666", "font": ("TkDefaultFont", 9)}

        presets_frame = Frame(parent)
        presets_frame.pack(fill="x", padx=6, pady=(6, 0))
        Label(presets_frame, text="Wczytaj przykład:").grid(row=0, column=0, sticky=W)
        preset_box = ttk.Combobox(
            presets_frame, textvariable=self.preset_var, values=list(PRESETS.keys()),
            state="readonly", width=45,
        )
        preset_box.grid(row=0, column=1, sticky=W, padx=5)
        preset_box.bind("<<ComboboxSelected>>", lambda e: self._load_preset(self.preset_var.get()))
        Label(
            presets_frame,
            text="wypełnia pola poniżej gotowym przykładem — możesz je potem dowolnie zmienić",
            **SUB,
        ).grid(row=1, column=0, columnspan=2, sticky=W, pady=(2, 0))

        f = Frame(parent)
        f.pack(fill="x", padx=6, pady=6)

        Label(f, text="Nazwa hipotezy:").grid(row=0, column=0, sticky=W)
        ttk.Entry(f, textvariable=self.custom_name_var, width=30).grid(row=0, column=1, columnspan=3, sticky=W + E, padx=5)

        Label(f, text="Opis hipotezy (co twierdzisz):").grid(row=1, column=0, sticky=W, pady=(4, 0))
        ttk.Entry(f, textvariable=self.custom_description_var, width=70).grid(row=1, column=1, columnspan=5, sticky=W + E, padx=5, pady=(4, 0))

        Label(f, text="Mierzony efekt (co dokładnie porównujesz):").grid(row=2, column=0, sticky=W, pady=(4, 0))
        ttk.Entry(f, textvariable=self.custom_effect_description_var, width=70).grid(row=2, column=1, columnspan=5, sticky=W + E, padx=5, pady=(4, 0))

        Label(
            f,
            text="Poniżej: dwie próby danych (testowa i tło) + jedna statystyka (metryka) liczona na obu. "
                 "Wynik = test Manna-Whitneya, czy rozkłady metryki różnią się istotnie.",
            **SUB, wraplength=860, justify="left",
        ).grid(row=3, column=0, columnspan=6, sticky=W, pady=(8, 2))

        # -- dane testowe (Twoja hipoteza) --
        Label(f, text="Dane testowe (Twoja hipoteza):").grid(row=4, column=0, sticky=W, pady=(6, 0))
        test_box = ttk.Combobox(f, textvariable=self.custom_test_source_var, values=TEST_SOURCES, state="readonly", width=22)
        test_box.grid(row=4, column=1, sticky=W, padx=5, pady=(6, 0))
        test_box.bind("<<ComboboxSelected>>", lambda e: self._update_custom_visibility())

        self.ct_nmax_label = Label(f, text="N (górna granica):")
        self.ct_nmax_entry = ttk.Entry(f, textvariable=self.custom_test_nmax_var, width=10)
        self.ct_low_label = Label(f, text="min:")
        self.ct_low_entry = ttk.Entry(f, textvariable=self.custom_test_low_var, width=10)
        self.ct_high_label = Label(f, text="max (wyłącznie):")
        self.ct_high_entry = ttk.Entry(f, textvariable=self.custom_test_high_var, width=10)
        self.ct_phi_label = Label(f, text="autokorelacja φ (0–1):")
        self.ct_phi_entry = ttk.Entry(f, textvariable=self.custom_test_phi_var, width=10)
        self.ct_sigma_label = Label(f, text="odch. std σ:")
        self.ct_sigma_entry = ttk.Entry(f, textvariable=self.custom_test_sigma_var, width=10)
        # Osobny wiersz na parametry źródła (zamiast doklejać do wiersza
        # z comboboxem) — inaczej przy szerszych etykietach pola wychodzą
        # poza prawą krawędź okna.
        for i, w in enumerate((self.ct_nmax_label, self.ct_nmax_entry, self.ct_low_label, self.ct_low_entry,
                                self.ct_high_label, self.ct_high_entry, self.ct_phi_label, self.ct_phi_entry,
                                self.ct_sigma_label, self.ct_sigma_entry)):
            w.grid(row=5, column=1 + (i % 6), sticky=W, padx=(2, 8), pady=(4, 0))

        Label(
            f,
            text="Dane, o których mówi Twoja hipoteza — np. prawdziwe liczby pierwsze do N, "
                 "albo losowa/skorelowana próbka jeśli testujesz coś innego.",
            **SUB, wraplength=860, justify="left",
        ).grid(row=6, column=0, columnspan=6, sticky=W, pady=(2, 0))

        # -- tło / hipoteza zerowa --
        Label(f, text="Tło = hipoteza zerowa (losowe):").grid(row=7, column=0, sticky=W, pady=(10, 0))
        bg_box = ttk.Combobox(f, textvariable=self.custom_bg_source_var, values=BG_SOURCES, state="readonly", width=22)
        bg_box.grid(row=7, column=1, sticky=W, padx=5, pady=(10, 0))
        bg_box.bind("<<ComboboxSelected>>", lambda e: self._update_custom_visibility())

        self.cb_low_label = Label(f, text="min:")
        self.cb_low_entry = ttk.Entry(f, textvariable=self.custom_bg_low_var, width=10)
        self.cb_high_label = Label(f, text="max (wyłącznie):")
        self.cb_high_entry = ttk.Entry(f, textvariable=self.custom_bg_high_var, width=10)
        self.cb_phi_label = Label(f, text="autokorelacja φ (0–1):")
        self.cb_phi_entry = ttk.Entry(f, textvariable=self.custom_bg_phi_var, width=10)
        self.cb_sigma_label = Label(f, text="odch. std σ:")
        self.cb_sigma_entry = ttk.Entry(f, textvariable=self.custom_bg_sigma_var, width=10)
        for i, w in enumerate((self.cb_low_label, self.cb_low_entry, self.cb_high_label, self.cb_high_entry,
                                self.cb_phi_label, self.cb_phi_entry, self.cb_sigma_label, self.cb_sigma_entry)):
            w.grid(row=8, column=1 + (i % 6), sticky=W, padx=(2, 8), pady=(4, 0))

        Label(
            f,
            text="Próbka reprezentująca 'brak efektu' (H0) — musi dać się wylosować wielokrotnie i niezależnie, "
                 "stąd tylko źródła losowe (nie liczby pierwsze — te są ustalone raz na zawsze).",
            **SUB, wraplength=860, justify="left",
        ).grid(row=9, column=0, columnspan=6, sticky=W, pady=(2, 0))

        # -- metryka --
        Label(f, text="Metryka (statystyka testowa):").grid(row=10, column=0, sticky=W, pady=(10, 0))
        metric_box = ttk.Combobox(f, textvariable=self.custom_metric_var, values=METRICS, state="readonly", width=22)
        metric_box.grid(row=10, column=1, sticky=W, padx=5, pady=(10, 0))
        metric_box.bind("<<ComboboxSelected>>", lambda e: self._update_custom_visibility())

        self.cm_modulus_label = Label(f, text="dzielnik m:")
        self.cm_modulus_entry = ttk.Entry(f, textvariable=self.custom_modulus_var, width=8)
        self.cm_remainder_label = Label(f, text="reszta r (x mod m = r):")
        self.cm_remainder_entry = ttk.Entry(f, textvariable=self.custom_remainder_var, width=8)
        self.cm_threshold_label = Label(f, text="próg t (x > t):")
        self.cm_threshold_entry = ttk.Entry(f, textvariable=self.custom_threshold_var, width=10)
        for i, w in enumerate((self.cm_modulus_label, self.cm_modulus_entry, self.cm_remainder_label,
                                self.cm_remainder_entry, self.cm_threshold_label, self.cm_threshold_entry)):
            w.grid(row=11, column=1 + i, sticky=W, padx=(2, 8), pady=(4, 0))

        Label(
            f,
            text="Liczona osobno na próbce testowej i na tle, potem obie serie porównuje test Manna-Whitneya "
                 "(nieparametryczny odpowiednik testu t — nie zakłada rozkładu normalnego).",
            **SUB, wraplength=860, justify="left",
        ).grid(row=12, column=0, columnspan=6, sticky=W, pady=(2, 0))

        # -- siła efektu do kontroli bramkowej --
        Label(f, text="Kontrola samosprawdzająca — wstrzyknięty efekt:").grid(row=13, column=0, columnspan=2, sticky=W, pady=(10, 0))
        self.ce_shift_label = Label(f, text="przesunięcie średniej Δ:")
        self.ce_shift_entry = ttk.Entry(f, textvariable=self.custom_effect_shift_var, width=10)
        self.ce_bias_label = Label(f, text="udział próby z wymuszonym warunkiem (0–1):")
        self.ce_bias_entry = ttk.Entry(f, textvariable=self.custom_bias_strength_var, width=10)
        for i, w in enumerate((self.ce_shift_label, self.ce_shift_entry, self.ce_bias_label, self.ce_bias_entry)):
            w.grid(row=14, column=1 + i, sticky=W, padx=(2, 8), pady=(4, 0))

        Label(
            f,
            text="To NIE jest część Twojej hipotezy — to sztucznie dodany efekt, który metryka MUSI wykryć "
                 "(kontrola pozytywna), zanim zaufamy testowi na właściwych danych powyżej. Przy metryce "
                 "'Średnia'/'Mediana' Δ powinno być rzędu odchylenia standardowego tła; przy metrykach "
                 "frakcyjnych to udział próby (0–1) sztucznie ustawiony na warunek.",
            **SUB, wraplength=860, justify="left",
        ).grid(row=15, column=0, columnspan=6, sticky=W, pady=(2, 0))

        self.run_button_custom = Button(f, text="Uruchom własny scenariusz", command=self._run_custom_clicked)
        self.run_button_custom.grid(row=16, column=0, columnspan=6, sticky=W + E, pady=(14, 0))

        self._update_custom_visibility()

    def _load_preset(self, preset_name: str):
        preset = PRESETS.get(preset_name)
        if preset is None:
            return
        for key, value in preset.items():
            var_name = _PRESET_KEY_TO_VAR.get(key)
            if var_name is None:
                continue
            getattr(self, var_name).set(value)
        self._update_custom_visibility()

    def _update_custom_visibility(self):
        test_source = self.custom_test_source_var.get()
        bg_source = self.custom_bg_source_var.get()
        metric = self.custom_metric_var.get()

        def show(widgets, condition):
            for w in widgets:
                (w.grid() if condition else w.grid_remove())

        show((self.ct_nmax_label, self.ct_nmax_entry), test_source == "Liczby pierwsze")
        show((self.ct_low_label, self.ct_low_entry, self.ct_high_label, self.ct_high_entry), test_source == "Losowe całkowite")
        show((self.ct_phi_label, self.ct_phi_entry, self.ct_sigma_label, self.ct_sigma_entry), test_source == "Szum AR(1)")

        show((self.cb_low_label, self.cb_low_entry, self.cb_high_label, self.cb_high_entry), bg_source == "Losowe całkowite")
        show((self.cb_phi_label, self.cb_phi_entry, self.cb_sigma_label, self.cb_sigma_entry), bg_source == "Szum AR(1)")

        is_frac = metric in ("Frakcja: x mod m == r", "Frakcja: x > próg")
        show((self.cm_modulus_label, self.cm_modulus_entry, self.cm_remainder_label, self.cm_remainder_entry),
             metric == "Frakcja: x mod m == r")
        show((self.cm_threshold_label, self.cm_threshold_entry), metric == "Frakcja: x > próg")

        show((self.ce_shift_label, self.ce_shift_entry), not is_frac)
        show((self.ce_bias_label, self.ce_bias_entry), is_frac)

    # -- wyjście (raport + wykres) --------------------------------------

    def _build_output(self):
        bottom = Frame(self.root)
        bottom.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.report_text = ScrolledText(bottom, height=14, wrap="word", font=("Consolas", 10))
        self.report_text.pack(fill="both", expand=True, side="top")

        self.chart_frame = Frame(bottom, height=200)
        self.chart_frame.pack(fill="x", side="bottom", pady=(8, 0))
        self.chart_frame.pack_propagate(False)

    # -- uruchamianie ---------------------------------------------------

    def _run_builtin_clicked(self):
        self._start_run(self._resolve_builtin_scenario)

    def _run_custom_clicked(self):
        self._start_run(self._resolve_custom_scenario)

    def _start_run(self, resolver):
        self.run_button_builtin.config(state=DISABLED)
        self.run_button_custom.config(state=DISABLED)
        self.status_label.config(text="Liczę...")
        self.report_text.delete("1.0", END)
        threading.Thread(target=self._run_protocol, args=(resolver,), daemon=True).start()

    def _shared_params(self) -> dict:
        return {
            "seed": int(self.seed_var.get()),
            "n_windows": int(self.n_windows_var.get()),
            "window_size": int(self.window_size_var.get()),
        }

    def _resolve_builtin_scenario(self):
        params = self._shared_params()
        params["n_max"] = int(self.n_max_var.get())
        params["effect_size"] = float(self.effect_size_var.get())
        scenario_fn = BUILTIN_SCENARIOS[self.scenario_var.get()]
        return scenario_fn(params)

    def _resolve_custom_scenario(self):
        params = self._shared_params()

        test_key = _SOURCE_LABEL_TO_KEY[self.custom_test_source_var.get()]
        bg_key = _SOURCE_LABEL_TO_KEY[self.custom_bg_source_var.get()]
        metric_key = _METRIC_LABEL_TO_KEY[self.custom_metric_var.get()]

        test_params = {}
        if test_key == "primes":
            test_params = {"n_max": int(self.custom_test_nmax_var.get())}
        elif test_key == "random":
            test_params = {"low": int(self.custom_test_low_var.get()), "high": int(self.custom_test_high_var.get())}
        elif test_key == "ar1":
            test_params = {"phi": float(self.custom_test_phi_var.get()), "sigma": float(self.custom_test_sigma_var.get())}

        bg_params = {}
        if bg_key == "random":
            bg_params = {"low": int(self.custom_bg_low_var.get()), "high": int(self.custom_bg_high_var.get())}
        elif bg_key == "ar1":
            bg_params = {"phi": float(self.custom_bg_phi_var.get()), "sigma": float(self.custom_bg_sigma_var.get())}

        metric_params = {}
        if metric_key == "frac_modulo":
            metric_params = {"modulus": int(self.custom_modulus_var.get()), "remainder": int(self.custom_remainder_var.get())}
        elif metric_key == "frac_threshold":
            metric_params = {"threshold": float(self.custom_threshold_var.get())}

        params.update({
            "name": self.custom_name_var.get().strip() or "wlasny_scenariusz",
            "description": self.custom_description_var.get().strip() or "(brak opisu)",
            "effect_description": self.custom_effect_description_var.get().strip() or "(brak opisu efektu)",
            "test_source": test_key,
            "test_params": test_params,
            "bg_source": bg_key,
            "bg_params": bg_params,
            "metric": metric_key,
            "metric_params": metric_params,
            "effect_shift": float(self.custom_effect_shift_var.get()),
            "bias_strength": float(self.custom_bias_strength_var.get()),
        })
        return scenario_custom(params)

    def _run_protocol(self, resolver):
        try:
            hypothesis, prereg, controls, main_result, test_vals, bg_vals = resolver()
            report = format_report(
                hypothesis, controls, main_result,
                n_comparisons=1, alpha=float(self.alpha_var.get()),
            )
            header = f"Pre-rejestracja (krok 1): {prereg.fingerprint[:16]}...\n\n"
            self.root.after(0, self._show_result, header + report, test_vals, bg_vals)
        except Exception:
            tb = traceback.format_exc()
            self.root.after(0, self._show_error, tb)

    def _show_result(self, report: str, test_vals, bg_vals):
        self.report_text.insert(END, report)
        self.status_label.config(text="Gotowe.")
        self.run_button_builtin.config(state=NORMAL)
        self.run_button_custom.config(state=NORMAL)
        self._draw_chart(test_vals, bg_vals)

    def _show_error(self, tb: str):
        self.report_text.insert(END, f"Błąd:\n{tb}")
        self.status_label.config(text="Błąd — patrz szczegóły powyżej.")
        self.run_button_builtin.config(state=NORMAL)
        self.run_button_custom.config(state=NORMAL)

    def _draw_chart(self, test_vals, bg_vals):
        for child in self.chart_frame.winfo_children():
            child.destroy()
        if not HAS_MATPLOTLIB or not test_vals or not bg_vals:
            return
        fig = Figure(figsize=(8, 2.0), dpi=100)
        ax = fig.add_subplot(111)
        ax.boxplot([test_vals, bg_vals], labels=["test", "tło"], vert=False)
        ax.set_title("Rozkład wartości metryki: test vs tło", fontsize=10)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


def main():
    root = Tk()
    Dashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
