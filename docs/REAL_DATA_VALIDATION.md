# Walidacja na realnych danych — rezonans jako operator (Krakow_Centrum)

Ten dokument stosuje protokół z [PROTOCOL.md](PROTOCOL.md) do jednej konkretnej
hipotezy spoza numerologii: **czy realna współbieżność "rezonansu"
(≥K parametrów anomalnych jednocześnie) przekracza to, czego oczekiwałbyś
z samego przypadku, biorąc pod uwagę realne tempo anomalii każdego
parametru z osobna.**

To bezpośrednia empiryczna weryfikacja teoretycznego oszacowania z
`GIA-TIMDR/docs/theory/` (rozkład dwumianowy przy założeniu niezależności
i normalności, n=5 parametrów, P(≥3 anomalne)≈0.09%) — na realnych danych
zamiast na założeniu gaussowskim.

## Dane

Źródło: `synoptyk-v2.0-main/krakow_forecast_snapshots.csv`, stacja
`Krakow_Centrum`, wiersze `source` ∈ {`IMGW_real_*`, `web_szukaj_*`,
`OpenMeteo_real_dailymax`} (realne obserwacje, nie prognozy).
Deduplikacja: przy kilku wpisach na tę samą `target_date`, ostatni wpis
wygrywa (ta sama konwencja co `bias_correction.py::_load_pairs`).

Zakres: 2026-08-09 .. 2026-09-02, **24 dni** (2026-08-11 brakuje w
źródle — pominięty, nie interpolowany).

Dostępne parametry: `max_temp_c`, `pressure_hpa`, `wind_kmh` — obecne
dla wszystkich 24 dni. `precip_mm` też istnieje w danych, ale tylko dla
części dni (starsze wpisy `IMGW_real`/`web_szukaj` go nie mają) — **nie
użyty**, żeby nie mieszać okien o różnej długości. Kolumny wilgotności
w tym pliku **nie ma wcale**.

**To jest realne ograniczenie, nie pominięcie:** dokumentowany system
używa n=5 parametrów (temp, ciśnienie, wilgotność, wiatr, opad); ten
test używa n=3 (temp, ciśnienie, wiatr). Wynik poniżej dotyczy więc
zawężonego, 3-parametrowego analogu, nie pełnego systemu.

## Metoda

1. Progi anomalii liczone **live z tego samego okna** (zgodnie z
   `adaptive_thresholds.py`): `anomalia(t) = |x(t) - mean| > 2*std`.
2. `rezonans(t) = 1[liczba anomalnych parametrów w dniu t >= K]`, dla
   K=2 i K=3 (K=3 przy n=3 oznacza: wszystkie trzy naraz).
3. Model null: permutacja — każdy parametr tasowany **niezależnie**
   (własne realne wartości i własny realny rozkład anomalii w czasie,
   tylko zerwane wyrównanie dat między parametrami), 5000 powtórzeń,
   p-wartość Davison–Hinkley `(liczba_permutacji_>=_realna + 1)/(5000+1)`.
4. Kontrola pozytywna (sanity check mechaniki testu, nie hipotezy):
   sztucznie wymuszona pełna współbieżność w 3 dniach — test **musi**
   to wykryć, inaczej wynik główny jest bez znaczenia.

Uruchomione jako niezależna reimplementacja w JavaScript
(`mcp__Claude_Browser__javascript_tool`), bo Python/bash były w tej
sesji niedostępne — dokładnie ten sam kompromis co przy samowalidacji
opisanej w `tests/test_examples.py`: inny RNG, ten sam algorytm.
Równoważny skrypt Python do odtworzenia tego na realnym interpreterze:
[`examples/real_weather_resonance_validation.py`](../examples/real_weather_resonance_validation.py).

## Wynik (realny, nieedytowany)

| Parametr | Liczba dni anomalnych (z 24) |
|---|---|
| temp (max_temp_c) | 2 |
| pressure_hpa | 0 |
| wind_kmh | 1 |

Ciśnienie w tym oknie **nigdy** nie przekroczyło progu 2σ.

| K | realna stopa rezonansu | średnia null | p (permutacja) |
|---|---|---|---|
| 2 | 0/24 = 0.0 | 0.0034 | 1.0 |
| 3 | 0/24 = 0.0 | 0.0 | 1.0 |

Kontrola pozytywna (3 sztucznie wymuszone dni pełnej współbieżności):
stopa 3/24 = 0.125, p ≈ 0.0002 — **wykryte poprawnie**.

## Werdykt (uczciwy)

To **nie jest** dowód, że rezonans nie przekracza przypadku — to jest
**brak wystarczających danych, żeby to w ogóle przetestować** w tym
oknie. Realna stopa anomalii per parametr (temp 8.3%, wiatr 4.2%,
ciśnienie 0%) jest w tym 24-dniowym oknie znacznie niższa niż zakładane
teoretycznie ~4.55% przy normalności — więc przy K≥2 zdarzeń
współbieżnych jest zero w obu grupach (realnej i null), test nie ma
mocy statystycznej w żadną stronę (p=1 dla obu K nie znaczy "brak
efektu", znaczy "nie było czego mierzyć").

Kontrola pozytywna potwierdza, że sama metodyka (progi + permutacja)
**działa poprawnie** i wykrywa realną współbieżność, gdy ta faktycznie
występuje (p≈0.0002) — więc wynik zerowy powyżej to ograniczenie danych
(za krótkie / za spokojne pogodowo okno), nie błąd testu.

**Co byłoby potrzebne, żeby to faktycznie przetestować:** dłuższe okno
(więcej dni ekstremalnej pogody, żeby każdy parametr miał realistyczną
liczbę własnych anomalii do porównania) i/lub dane z kolumną
wilgotności, żeby dojść do udokumentowanych n=5 parametrów zamiast n=3.

Ten wynik — łącznie z jego niejednoznacznością — trafia wprost do
`GIA-TIMDR/docs/theory/Resonance_M_Operator_Empiryczny.md`, zamiast
teoretycznego oszacowania podawanego jako gotowy fakt.
