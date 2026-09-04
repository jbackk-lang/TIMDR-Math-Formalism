# TIMDR-Math-Formalism

Detektor matematycznej sensowności: czy wzór / "rezonans" / struktura,
którą ktoś wymyślił, to realna matematyka, czy tylko ładnie wyglądający
pattern bez dowodu (numerologia).

Wejściem nie jest sygnał czasowy, tylko **hipoteza**: precyzyjny opis
struktury + konkretne, testowalne twierdzenie o niej. Pipeline stosuje
do niej ten sam protokół, który reszta ekosystemu TIMDR stosuje do
pytania "czy ten sygnał ma moc predykcyjną": pre-rejestracja, kontrola
pozytywna/negatywna, prawdziwy test statystyczny (Mann-Whitney U),
uczciwie raportowany wynik negatywny.

📝 Pełny opis protokołu i uzasadnienie każdego kroku: [docs/PROTOCOL.md](docs/PROTOCOL.md)

![Diagram protokołu: hipoteza → dane testowe/tło → bramka kontroli +/- → test Manna-Whitneya lub stop → werdykt](docs/diagram.svg)

## 🔧 Instalacja

```
pip install -r requirements.txt
```

Wymaga `numpy` i `scipy` (Python 3.9+).

## 🚀 Szybki start

```python
from timdr_formalism import Hypothesis, Preregistration, run_controls, mann_whitney_test, format_report

hypothesis = Hypothesis(
    name="moja_struktura",
    description="ciąg X ma nietrywialny związek ze zbiorem Y",
    effect_description="różnica median metryki M między X a losowym tłem",
)

# Krok 1: zamroź hipotezę PRZED dotknięciem danych
prereg = Preregistration.create(hypothesis, params={"n": 1000, "seed": 0})

# Krok 5: kontrola +/- jako bramka, PRZED testem głównym
controls = run_controls(
    metric_fn=moja_metryka,
    positive_injector=wstrzyknij_znany_efekt,
    negative_generator_a=czyste_tlo,
    negative_generator_b=czyste_tlo,
)

if controls.passed:
    # Krok 4: prawdziwy test statystyczny
    wynik = mann_whitney_test(wartosci_testowe, wartosci_tla)
    print(format_report(hypothesis, controls, wynik))
else:
    print(format_report(hypothesis, controls, main_result=None))
```

Pełny, uruchamialny przykład (hipoteza "ciąg ma specjalny rezonans z
liczbami pierwszymi", na prawdziwych liczbach pierwszych z sita
Eratostenesa):

```
python examples/prime_resonance_demo.py
```

## 🖥️ Dashboard

GUI (Tkinter) do uruchamiania protokołu bez pisania kodu, dwie zakładki:

- **Gotowe scenariusze** — dwa przetestowane przykłady (liczby pierwsze
  / regulowane przesunięcie średniej).
- **Własny scenariusz** — budujesz hipotezę z bezpiecznych klocków:
  wybierasz dane testowe (liczby pierwsze / losowe całkowite / szum
  AR(1)), tło reprezentujące hipotezę zerową (losowe całkowite / AR(1)
  — musi dać się wielokrotnie wylosować), metrykę (średnia / mediana /
  frakcja spełniająca warunek modulo / frakcja powyżej progu) i siłę
  efektu do samosprawdzającej kontroli pozytywnej. Zero wykonywania
  dowolnego kodu — wszystkie pola są opisane po polsku bez żargonu
  TIMDR, więc nie trzeba znać reszty ekosystemu, żeby wiedzieć, co
  wpisać. Zakładka ma też combobox "Wczytaj przykład" z 5 gotowymi,
  zweryfikowanymi konfiguracjami (liczby pierwsze vs losowe tło,
  kalibracja zerowa/negatywna, AR(1) vs biały szum, liczby pierwsze
  mod 4, mediana dwóch rozłącznych zakresów) — wybierasz z listy, pola
  się same wypełniają, możesz od razu uruchomić albo dostroić.

Klikasz "Uruchom protokół" — widzisz pełny raport (pre-rejestracja,
kontrola +/-, test Manna-Whitneya, werdykt) i wykres porównujący
rozkład metryki (test vs tło) na żywo.

```
python dashboard.py
```

Na Windows wystarczy dwuklik na `run.bat` — zainstaluje zależności
(`numpy`, `scipy`, `matplotlib`) i od razu odpali dashboard.

## 🧩 Moduł — skrót

| Funkcja/klasa | Krok protokołu | Co robi |
|---|---|---|
| `Hypothesis`, `Preregistration` | 1 | Zamraża definicję struktury + parametrów przed testem (SHA-256 fingerprint, wykrywa dostrajanie po fakcie) |
| `sieve_of_eratosthenes`, `random_background`, `ar1_noise` | 2 | Dokładne liczby pierwsze, losowe tło, szum AR(1) (nie biały — patrz PROTOCOL.md) |
| `mann_whitney_test` | 4 | Prawdziwy test istotności (Mann-Whitney U), nie porównanie "na oko" |
| `run_controls` | 5 | Kontrola pozytywna + negatywna jako bramka przed testem głównym |
| `bonferroni_correct` | — | Korekta na wielokrotne porównania (look-elsewhere effect) |
| `format_report` | 6 | Czytelny raport/werdykt, w tym uczciwy wynik negatywny |
| `chronosignal.tempo`, `chronosignal.drift` | — (instancja sygnału) | Odstępy/drift z sekwencji znaczników czasu — patrz §⏱️ niżej |
| `chronosignal.anomalia_flags/defekt_flags/skret_flags` | — (detektory) | Definicje §1 skilla, zastosowane do tempa/driftu |

## 🧪 Testy

```
pip install pytest
pytest tests/ -v
```

Testy używają danych z gwarantowaną separacją (nie losowych progów
zależnych od konkretnego seeda) albo ręcznie skonstruowanych wyników —
każda asercja jest zdeterminowana przez konstrukcję testu, nie przez to,
czy akurat trafił się "dobry" losowy ciąg. `tests/test_dashboard_logic.py`
sprawdza logikę "Własnego scenariusza" (metryki, źródła danych,
wstrzykiwanie efektu) bez otwierania okna GUI.

## ⏱️ Czas jako sygnał — `timdr_formalism.chronosignal`

Instancja generycznego sygnału `x:T→ℝᵈ` gałęzi M/S na konkretny
przypadek "T = kolejne znaczniki czasu": nie nowa matematyka, tylko
poprawne wpisanie się w istniejącą definicję sygnału (część
Chronoprocesu `Ξ=(T,x,Γ,φ)` opisanego w
`GIA-TIMDR/SKILL_timdr-signal-framework.md`).

Dwie rozdzielone, osobno zdefiniowane serie:

- **`tempo(t)`** = `t[i+1]-t[i]` — sam odstęp, nie wymaga niczego poza
  znacznikami czasu.
- **`drift(t)`** = `tempo_zmierzone(t) - tempo_nominalne` — wymaga
  jawnego zegara referencyjnego (`nominal_interval`); bez niego drift
  jest formalnie niezdefiniowany.

Plus samodzielna implementacja `anomalia_flags`/`defekt_flags`/
`skret_flags` (definicje z sekcji 1 skilla `timdr-signal-framework`,
zastosowane tu konkretnie do tempa/driftu — `rezonans` nie ma sensu na
pojedynczym sygnale skalarnym, więc nie jest tu zaimplementowany).

```python
from timdr_formalism.chronosignal import tempo, drift, defekt_flags

ts = [0, 60, 120, 185, 190, 250]        # znaczniki czasu (dowolna jednostka)
print(tempo(ts))                          # [60, 60, 65, 5, 60]
print(drift(ts, nominal_interval=60.0))   # [0, 0, 5, -55, 0]
print(defekt_flags(tempo(ts), factor=0.3))
```

Pełny, uruchamialny przykład, przepuszczający hipotezę o tempie przez
cały protokół repo (pre-rejestracja, kontrola +/-, Mann-Whitney,
werdykt) — analogicznie do `prime_resonance_demo.py`, ale dla tempa
zamiast liczb pierwszych:

```
python examples/chronosignal_demo.py
```

**✅ Zweryfikowane.** `chronosignal.py` i `tests/test_chronosignal.py`
zostały napisane w sesji bez dostępu do sandboxa bash (matematyka progów
była ręcznie prześledzona przed uruchomieniem), ale wszystkie 16 testów
zostało odtąd faktycznie uruchomionych przez użytkownika (`pytest
tests/test_chronosignal.py -v`) i **przeszło**. `examples/chronosignal_demo.py`
(interaktywny skrypt, nie plik testowy) nie był jeszcze uruchomiony —
uruchom go samodzielnie (`python examples/chronosignal_demo.py`), jeśli
chcesz zobaczyć werdykt na konkretnych liczbach.

## 🌦️ Walidacja na realnych danych

Poza syntetycznymi przykładami, protokół został raz zastosowany do
prawdziwej hipotezy na prawdziwych danych: czy operator rezonansu
(≥K parametrów anomalnych jednocześnie) na realnych danych pogodowych
(stacja Krakow_Centrum, 24 dni) przekracza to, czego oczekiwałbyś z
samego przypadku. Wynik — łącznie z jego ograniczeniami (za krótkie
okno, brak danych o wilgotności) i uczciwym werdyktem — jest w
[docs/REAL_DATA_VALIDATION.md](docs/REAL_DATA_VALIDATION.md);
odtwarzalny skrypt: [examples/real_weather_resonance_validation.py](examples/real_weather_resonance_validation.py).

## ⚠️ Czego ten pipeline NIE robi

Nie dowodzi twierdzeń i nie zastępuje sprawdzenia, czy dana dziedzina ma
już własny, ugruntowany model statystyczny (homemade metryka, która nic
nie znajduje, jest dowodem przeciwko TEJ metryce, nie automatycznie
przeciwko zjawisku — patrz PROTOCOL.md). Nie automatyzuje też
replikacji na niezależnym zbiorze danych — pojedynczy pozytywny wynik
nigdy nie jest, sam w sobie, dowodem.

## 📄 Licencja

MIT — patrz [LICENSE](LICENSE).
