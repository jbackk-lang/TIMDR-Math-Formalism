# DRAFT: sprzężenie helikalne K ↔ Θ_bif

> **Zamknięte na tym etapie (2026-09-15), uczciwie niedomknięte:**
> konstrukcja matematyczna (sekcje 1-4) jest wewnętrznie spójna i
> zweryfikowana numerycznie — pomysł sam w sobie jest sensowny. Droga
> do sprawdzenia go na realnych danych jest zablokowana: jedyny
> zbudowany estymator `Re(λ)/Im(λ)` (sekcja 6) zawodzi katastroficznie
> przy realistycznym poziomie szumu, z rozpoznaną przyczyną
> strukturalną (nie tylko brakiem dostrojenia). Nie promowane do
> "modułu", nie testowane na realnych danych, nie kontynuowane bez
> osobnej decyzji o dalszej pracy nad estymatorem.

> **Status: NOWA KONSTRUKCJA, nie wyprowadzenie z istniejącego,
> połączonego formalizmu.** `gs_matrix.py` (macierz `K`, układ
> `dV/dt=KV`) i `theta_bifurcation.py` (`S_down`, `S_up`, `λ_down`,
> `λ_up`, `N(t)`) są dziś w kodzie DWOMA ROZŁĄCZONYMI obiektami — patrz
> `theta_bifurcation.py`, sekcja nagłówka "UWAGA O POMINIĘTYM det(K)":
> sprawdzone bezpośrednio różniczkami skończonymi, że w istniejącej
> pętli SG-Coupling (`G=R0+α·S²`) `G` jest JEDNOKIERUNKOWYM odczytem z
> `S`, nigdy nie wpływa z powrotem na dynamikę `S` — Jakobian ma kolumnę
> `G_old` tożsamościowo zerową. `K` z `gs_matrix.py` jest osobnym
> obiektem, nigdy numerycznie niepołączonym z tą pętlą. Ten dokument
> PROPONUJE nowe, jawnie oznaczone połączenie — nie odkrywa istniejącego.
> Zero kodu na tym etapie, zgodnie z ustaleniem.

## 0. Problem: dwie niespójne, wcześniej istniejące idee "sprzężenia G,S"

1. `gs_matrix.py`: abstrakcyjny układ liniowy `dV/dt=KV`, `V=[G,S]ᵀ`
   (nazwane tak w docstringu), testowany wyłącznie na dowolnych
   macierzach `K` pod kątem zachowania normy — nigdy nie podstawiono
   tam konkretnych, powiązanych z realną symulacją wartości.
2. Test-owa pętla SG-Coupling (`run_sg_coupling_simulation_v2`,
   `tests/test_sg_coupling_full_operator.py`): `G=R0+α·S²` (ALGEBRAICZNE,
   jednokierunkowe), `Q(R)=1-L0/(L0+2πR)`, `Θ_bif` aplikowany do `S` gdy
   `Q>Q_crit`. To NIE jest postaci `dV/dt=KV` w ogóle.

Ta konstrukcja wybiera **(1)** jako podstawę (bo to jedyna postać, która
w ogóle daje strukturę wartości własnych/rotacji) i buduje na niej nowy
most do `λ_down`/`λ_up`/`N(t)` z Θ_bif — świadomie NIE próbując pogodzić
tego z algebraicznym `G=R0+α·S²` z (2). To rozwiązuje niespójność (1) vs
(2) przez zastąpienie (2), nie przez połączenie obu.

## 1. Nowa, jawna decyzja definicyjna: `S_down`/`S_up` jako baza własna `K_sym`

`K_sym` (symetryczna część `K`) jest rzeczywistą macierzą symetryczną →
ma ortogonalną bazę własną z rzeczywistymi wartościami własnymi. **Nowa
konstrukcja (nie wymuszona przez nic istniejącego)**: wybieramy układ
współrzędnych tak, żeby ta baza własna POKRYWAŁA SIĘ z kierunkami
`S_down` i `S_up` z Θ_bif, i identyfikujemy wartości własne wprost z
istniejącymi stałymi czasowymi Θ_bif:

```
μ_down = -λ_down   (kierunek "S_down", zanikający — μ_down < 0)
μ_up   = +λ_up     (kierunek "S_up", rosnący — μ_up > 0)
```

W tej bazie `K_sym = diag(μ_down, μ_up)`. Dodajemy nowy parametr
sprzężenia rotacyjnego `ω₀` (odpowiednik `K_anti` w tej bazie):

```
K = [ μ_down   ω₀  ]  =  [ -λ_down    ω₀  ]
    [ -ω₀    μ_up  ]     [   -ω₀    λ_up  ]
```

**`ω₀` jest CAŁKOWICIE NOWYM parametrem** — nieobecnym w obecnym kodzie
(`α` z pętli SG-Coupling jest strukturalnie czymś innym: bezwymiarowym
współczynnikiem algebraicznego `G=R0+α·S²`, nie tempem rotacji). `ω₀`
przejmuje KONCEPCYJNĄ rolę "siły sprzężenia G↔S", ale jako tempo
(jednostka `1/czas`, ta sama co `λ_down`/`λ_up` — to dobra własność:
wszystkie trzy stałe żyją w tej samej skali), nie jako współczynnik
kwadratowy.

## 2. Warunek na pojawienie się spirali (główny wynik)

Wielomian charakterystyczny `K`: `λ² - (μ_down+μ_up)λ + (μ_down·μ_up+ω₀²) = 0`.

```
Re(λ) = (μ_down + μ_up) / 2 = (λ_up - λ_down) / 2      [DOKŁADNE, niezależne od ω₀]

próg:  ω₀_crit = |μ_down - μ_up| / 2 = (λ_down + λ_up) / 2

jeśli ω₀ > ω₀_crit:  Im(λ) = ω = sqrt(ω₀² - ω₀_crit²)   [SPIRALA — genuine rotacja]
jeśli ω₀ ≤ ω₀_crit:  Im(λ) = 0                          [BRAK rotacji — dzisiejszy Θ_bif]
```

**Zweryfikowane numerycznie** (nie tylko odręcznie) na `λ_down=2.0,
λ_up=1.0` (domyślne wartości `theta_bifurcation.py`): dla `ω₀∈{0, 0.5,
1.0, 1.5, 2.0, 3.0, 5.0}` obliczone wartości własne `K` dają `Re(λ)=-0.5`
identycznie dla KAŻDEGO `ω₀` (zgodnie z przewidywaniem — niezmiennik
śladu), próg przejścia real→complex dokładnie przy `ω₀=1.5=ω₀_crit`
przewidzianym wzorem, i `Im(λ)` powyżej progu zgadza się z
`sqrt(ω₀²-1.5²)` co do 4 miejsc po przecinku dla wszystkich testowanych
`ω₀`.

**Odczyt**: dzisiejszy Θ_bif (dwa niezależne, czysto rzeczywiste kanały
— `exp(-λ_down·τ)` i `tanh(λ_up·τ)`, zero rotacji) odpowiada dokładnie
reżimowi `ω₀≤ω₀_crit` tej konstrukcji — czyli SPECJALNEMU PRZYPADKOWI
braku sprzężenia rotacyjnego, nie ogólnemu przypadkowi. "Dynamika
helikalna" pojawia się dopiero, gdy nowo wprowadzone `ω₀` przekroczy
próg wyznaczony przez SUMĘ dotychczasowych stałych czasowych.

**Ciekawa (nie zweryfikowana empirycznie, tylko wewnętrznie spójna)
zgodność**: znak `Re(λ)` — `(λ_up-λ_down)/2` — pokrywa się kierunkowo z
istniejącym już w kodzie kryterium "miękka/twarda strefa"
(`docs/SG_COUPLING_PHASE_DIAGRAM.md`: `β_min≥0.5`→miękka,
`β_min<0.5`→twarda, próg na równowadze wag kanałów). To NIE jest dowód
niczego — to sygnał, że nowa konstrukcja nie jest sprzeczna z istniejącym
językiem "miękki/twardy", tylko obserwacja spójności.

## 3. N(t) w nowej konstrukcji: przewidywanie podwójnej częstości

Rozwiązania rzeczywiste dla pary wartości własnych zespolonych
`λ=Re(λ)±iω` mają postać:

```
S_down(t) = e^{Re(λ)t} [A_d cos(ωt) + B_d sin(ωt)]
S_up(t)   = e^{Re(λ)t} [A_u cos(ωt) + B_u sin(ωt)]
```

(stałe `A_d,B_d,A_u,B_u` z warunku początkowego `V(0)`). Iloczyn (przez
tożsamości `cos²`, `sin²`, `sin·cos`):

```
N(t) = S_down(t)·S_up(t) = e^{2Re(λ)t} · [ c₀ + c₁·cos(2ωt) + c₂·sin(2ωt) ]
```

**Zweryfikowane numerycznie**: symulacja pełnej trajektorii
`V(t)=exp(Kt)V(0)` (SciPy `expm`, `λ_down=2, λ_up=1, ω₀=3` →
`Re(λ)=-0.5, ω=Im(λ)=2.598`), `N(t)` odtrendowane przez
`exp(-2·Re(λ)·t)`, dopasowane regresją liniową do
`[1, cos(2ωt), sin(2ωt)]` na horyzoncie `t∈[0,20]`: **R²=1.000000**
(dopasowanie dokładne, nie przybliżone — potwierdza tożsamość
algebraiczną, nie tylko jej sens jakościowy).

**Odczyt**: to jest KONKRETNA, falsyfikowalna różnica względem
dzisiejszego Θ_bif. Obecne `N(t)=S_down·S_up` (dwa czysto rzeczywiste,
monotoniczne czynniki) jest ZAWSZE monotoniczne/bezoscylacyjne. Nowa
konstrukcja przewiduje: jeśli sprzężenie helikalne (`ω₀>ω₀_crit`)
istnieje, `N(t)` powinno oscylować z częstością `2ω` na obwiedni
`exp(2·Re(λ)·t)` — czyli PODWÓJNA częstość względem samej spirali
`(G,S)`. To jest warunek współzależności między gałęziami z punktu 2
Twojej prośby (`ω, Re(λ), Im(λ), N(t), S↓, S↑` — teraz wszystkie
policzalne z tych samych trzech parametrów `λ_down, λ_up, ω₀`).

## 4. Co jest tu jawnie NOWĄ konstrukcją (nie wyprowadzeniem)

- Utożsamienie `S_down`/`S_up` z bazą własną `K_sym` — wybór, nie
  konieczność.
- `ω₀` jako nowy parametr sprzężenia — nic w istniejącym kodzie go nie
  definiuje ani nie ogranicza. Zero empirycznego zakotwiczenia na tym
  etapie.
- Rozdzielenie Q/`Q_crit`/`β(t)` — CAŁKOWICIE pominięte w tej wersji.
  Prawdziwy Θ_bif miesza `S_down`/`S_up` wagą `β(Q)` zależną od
  przekroczenia progu; tu oba kanały są "zawsze w pełni aktywne" jako
  para modów liniowego układu. Pogodzenie z `β(t)` nie zrobione.
- Liniowość: prawdziwy `S_up` w Θ_bif ma **nasycenie przez `tanh`**
  (nieliniowe, konieczne żeby uniknąć przepełnienia — patrz historia
  błędu w nagłówku `theta_bifurcation.py`). Ta konstrukcja jest liniowa
  i ważna tylko lokalnie/dla wczesnych `τ` (małe wychylenie od zera,
  gdzie `tanh(x)≈x`) — NIE odtwarza sufitu `S_up_max`. Pełna,
  nieliniowa wersja (np. oscylator typu Stuart-Landau z nasyceniem
  amplitudy) to osobna, większa praca, nie zrobiona tu.
- `α` z istniejącej pętli SG-Coupling nie ma tu żadnej roli — ta
  konstrukcja ZASTĘPUJE algebraiczne `G=R0+α·S²`, nie rozszerza go.

## 5. Otwarte przed jakimkolwiek kodem/twierdzeniem "moduł"

1. Jak (jeśli w ogóle) szacować `ω₀` NIEZALEŻNIE z realnych danych —
   analogicznie do tego, jak `Δt`/`Δf` były liczone niezależnie przez
   `temporal_spread`/`fft_modalities`, nie definiowane przez to, co mają
   potwierdzać (pułapka tautologii złapana wcześniej w tej samej
   rozmowie przy próbie `Δf` z wartości własnej wprost).
2. Pogodzenie z `β(Q)` i nieliniowym nasyceniem `S_up` — bez tego
   przewidywanie "N(t) oscyluje z `2ω`" jest twierdzeniem o UPROSZCZONYM,
   liniowym modelu, nie o prawdziwym `theta_bifurcation()`.
3. Własna pre-rejestracja + kontrola pozytywna (syntetyczny sygnał z
   ZNANYM `ω₀`, sprawdzenie czy estymacja z punktu 1 go odtwarza) —
   dokładnie ten sam rygor co `PREREG_MS_K_EVENTS.md` — zanim cokolwiek
   tu nazwie się "modułem" albo zostanie użyte do realnych danych.

Do tego momentu: to jest matematyka nowej, jawnie oznaczonej konstrukcji
— zweryfikowana wewnętrznie (algebra + numeryka się zgadzają), ale bez
JAKIEJKOLWIEK empirycznej treści.

## 6. Kontrola pozytywna estymatora Re(λ)/Im(λ) — ZNALEZIONY BLOKER (przed dotknięciem realnych danych)

Punkt otwarty 1 z sekcji 5 (jak niezależnie szacować `Re(λ)`, `Im(λ)` z
pojedynczego, skalarnego `x(t)`) przetestowany metodą standardową
(DMD-styl: zanurzenie opóźniające `V[i]=[x[i],x[i+τ]]`, dopasowanie
najmniejszych kwadratów jednokrokowego propagatora `A` (`V[i+1]≈A·V[i]`),
`K=log(A)/dt_eff`) — PRZED jakimkolwiek dotknięciem realnych danych, na
syntetycznym sygnale z ZNANYM `K` (dokładnie ten sam wzorzec kontroli
pozytywnej co w `PREREG_MS_K_EVENTS.md`).

**Bez szumu**: metoda jest dokładna — odtwarza `Re(λ)`, `Im(λ)` co do 4
miejsc po przecinku, dla obu reżimów (spiralnego `ω₀=3.0` i
rzeczywistego `ω₀=0.5`).

**Z szumem — całkowita awaria, nawet przy bardzo małym poziomie**: przy
`noise_sigma=0.01` (1% amplitudy sygnału ~1.0) na oknie `N=64` próbek
(ten sam rozmiar okna co w `PREREG_MS_K_EVENTS.md`, licząc OD początku
epizodu bifurkacji):

| reżim | Re(λ) prawdziwe | Re(λ) fit (szum 0.01) | Im(λ) prawdziwe | Im(λ) fit (szum 0.01) |
|---|---|---|---|---|
| spiralny (`ω₀=3.0`) | −0.500 | −0.590 (τ=8, ~18% błędu — brzeg akceptowalności) | 2.598 | 2.712 |
| rzeczywisty (`ω₀=0.5`) | −1.914 | −13.07 (τ=8, ~7× za duże) | 0.000 | 39.27 (**artefakt**, `=5π/dt_eff`) |
| brak rotacji (`ω₀=0.0`) | −2.000 | −9.34 (τ=8) | 0.000 | 39.27 (**artefakt**) |

Przy `noise_sigma=0.05` (5%) nawet reżim spiralny łamie się (`Re(λ)`
błędne o czynnik ~6×).

**Diagnoza (nie tylko obserwacja awarii — zrozumiany mechanizm)**: dla
reżimu z DWOMA różnymi wartościami rzeczywistymi (`ω₀≤ω₀_crit`),
trajektoria `[x(t),x(t+τ)]` asymptotycznie kolabuje na JEDNĄ linię
(dominujący mod `e^{μ_max·t}` przytłacza drugi), więc `X0` w regresji
najmniejszych kwadratów jest niemal osobliwe niezależnie od szumu —
widoczne wprost we wskaźniku uwarunkowania (`cond(X0)~1e14-1e17` nawet
przy ZEROWYM szumie dla `ω₀=0.0`). To NIE jest błąd implementacji — to
strukturalna nieidentyfikowalność: pojedyncza skalarna obserwacja
sumy dwóch realnych modów o różnym tempie nie niesie dość informacji,
żeby odtworzyć OBA tempa, gdy jeden już zdominował.

**Wniosek — bloker, nie szczegół do poprawki w locie**: naiwny
DMD/estymator zanurzeniowy NIE nadaje się do użycia na jakichkolwiek
realnych danych M/S (szum rzędu kilku-kilkunastu % jest tam normą, nie
wyjątkiem — patrz np. `noise_sigma` w kontrolach pozytywnych
`PREREG_MS_K_EVENTS.md`, `0.05`). To jest osobny, nietrywialny problem
identyfikacji systemu (potencjalne kierunki: regresja całkowitych
najmniejszych kwadratów zamiast OLS, metody widmowe zamiast
czasowo-różniczkowych, filtracja przed zanurzeniem, dłuższe uśrednianie)
— NIEZWERYFIKOWANY, niezrobiony tutaj. Zanim jakikolwiek pomiar
`Re(λ)`/`Im(λ)` z realnych danych ma sens, ten estymator musi najpierw
przejść WŁASNĄ kontrolę pozytywną przy realistycznym poziomie szumu —
czego obecna wersja NIE przechodzi.
