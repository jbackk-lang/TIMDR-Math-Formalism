# Pre-rejestracja: estymator Re(λ)/Im(λ) przez dopasowanie kształtu

> Status: ZAMROŻONE przed oficjalnym uruchomieniem testu poniżej,
> 2026-09-15. Kontynuacja `GS_Matrix_Helical_Coupling_DRAFT.md` sekcji
> 6-7 (bloker znaleziony, potem wstępnie odblokowany propozycją
> dopasowania kształtu zamiast różnicowania).
>
> **Uczciwe zastrzeżenie o statusie tej pre-rejestracji**: progi w
> sekcji 4 poniżej są INFORMOWANE przez wstępną eksplorację już
> przeprowadzoną w sekcji 7 draftu (widziałem tam rzędy wielkości
> błędu przed napisaniem tego dokumentu) — to NIE jest pre-rejestracja
> "przed dotknięciem tej metody w ogóle". Jest to jednak pre-rejestracja
> **przed dotknięciem REALNYCH DANYCH**, co jest tu właściwą linią
> obrony przed numerologią (dokładnie ten sam wzorzec co
> `PREREG_MS_K_EVENTS.md`, gdzie parametry mostu Fouriera pochodziły ze
> znanej matematyki, nie z dostrajania pod docelowy wynik). Test
> poniżej używa NOWYCH, większych próbek (`N_trials=30` zamiast 6-10) i
> NOWEJ, szerszej siatki poziomów szumu niż cokolwiek widziane wcześniej
> — to nie jest powtórzenie już zobaczonego wyniku.

## 1. Cel

Sprawdzić, czy estymator z sekcji 7 (`GS_Matrix_Helical_Coupling_DRAFT.md`)
— dopasowanie kształtu `x(t)=[exp(Kt)·V0]₀` do całego okna metodą
nieliniowych najmniejszych kwadratów, zamiast różnicowania — jest
wystarczająco dokładny i niezawodny, żeby przejść do użycia na realnych
danych. To test WŁASNOŚCI ESTYMATORA (czy poprawnie odzyskuje znane
parametry), NIE test tego, czy realne dane M/S faktycznie mają ten
kształt (to osobne, późniejsze pytanie — patrz sekcja 6).

## 2. Syntetyczny wzorzec (ten sam co w sekcji 6-7 draftu, teraz zamrożony formalnie)

`x(t) = [exp(Kt)·V0]₀`, `K = [[μ_down, ω₀],[-ω₀, μ_up]]`,
`μ_down=-λ_down=-2.0`, `μ_up=+λ_up=+1.0` (wartości domyślne
`theta_bifurcation.py` — niezmienione).

Cztery reżimy `ω₀`, testowane RAZEM (nie osobno wybierane po wyniku):
- `ω₀=3.0` — głęboko spiralny (`Im(λ)_true=2.598`)
- `ω₀=1.5` — dokładnie na progu (`ω₀_crit=1.5`, `Im(λ)_true=0`,
  przypadek graniczny — patrz sekcja 4, kryterium C)
- `ω₀=0.5` — rzeczywisty, brak rotacji (`Im(λ)_true=0`)
- `ω₀=0.0` — brak sprzężenia w ogóle (`Im(λ)_true=0`)

`V0=(v1,v2)`, losowane NIEZALEŻNIE na każde powtórzenie:
`v1,v2 ~ Uniform(0.5, 1.5)` (nie stały warunek początkowy — żeby nie
testować tylko jednego przypadku).

`N=64` próbek (zgodność z resztą tej sesji — `PREREG_MS_K_EVENTS.md`),
`dt=0.01` (jednostka umowna).

## 3. Poziom szumu — siatka, nie pojedynczy punkt

`noise_sigma ∈ {0.0, 0.02, 0.05, 0.10, 0.20}` (addytywny szum gaussowski
na obserwację `x(t)`, względem amplitudy sygnału rzędu `O(1)`) — siatka,
żeby zobaczyć krzywą degradacji dokładności, nie punktowy wynik na
jednym, wygodnym poziomie szumu.

`N_trials=30` powtórzeń na każdą kombinację `(ω₀, noise_sigma)` — więcej
niż 6-10 z wcześniejszej eksploracji, dla stabilniejszych median/
proporcji.

## 4. Procedura estymacji — bez zmian względem sekcji 7 draftu

Nieliniowe najmniejsze kwadraty (`scipy.optimize.least_squares`,
metoda `lm`) dopasowujące `(μ_down, μ_up, ω₀, v1, v2)` do CAŁEGO okna
`x_obs(t)` naraz. Wielostartowość: siatka `ω₀_guess ∈ {0.1, 1.0, 2.0, 5.0}`
(4 starty — NIE informowane prawdziwym `ω₀`), `μ_down_guess=-1.0`,
`μ_up_guess=1.0` (stałe, niezależne od prawdziwych wartości),
`v1_guess=x_obs[0]`, `v2_guess=0.0`. Wybierane dopasowanie o
najmniejszym koszcie reszt (`res.cost`) spośród 4 startów. Zero zmian
tych wartości po zobaczeniu wyników testu poniżej.

## 5. Kryteria sukcesu (ustalone przed uruchomieniem, nie zmieniane po wyniku)

**Kryterium A (dokładność w reżimie spiralnym)**: dla `ω₀=3.0`, przy
`noise_sigma<=0.10`: mediana błędu względnego
`|Re_fit-Re_true|/|Re_true| < 25%` ORAZ `|Im_fit-Im_true|/|Im_true| < 25%`
po `N_trials=30` powtórzeniach.

**Kryterium B (klasyfikacja obecności rotacji)**: próg klasyfikacji
`Im_classify_threshold = 0.3` (separacja o rząd wielkości od typowego
`Im(λ)_true~2.6` w reżimie spiralnym). Przy `noise_sigma<=0.10`:
- `ω₀=3.0`: `Im_fit > 0.3` w `>=80%` powtórzeń (czułość)
- `ω₀=0.0` i `ω₀=0.5`: `Im_fit <= 0.3` w `>=80%` powtórzeń (swoistość)

**Kryterium C (przypadek graniczny `ω₀=1.5`)**: NIE jest częścią
kryterium sukcesu/porażki — dokładnie na progu, drobny szum może
przełączać klasyfikację w obie strony. Raportowane jako informacja
diagnostyczna (jaki % powtórzeń wykrywa rotację blisko granicy), nie
oceniane jako "zaliczone/niezaliczone".

**Ogólny wynik testu**: SUKCES tylko jeśli A i B oba przejdą. Częściowy
sukces (np. tylko B) raportowany uczciwie jako częściowy, nie zaokrąglany
w górę.

## 6. Ograniczenia (jawnie, przed uruchomieniem)

- Ten test mówi WYŁĄCZNIE o tym, czy estymator poprawnie odzyskuje
  parametry ZNANEGO układu z tej konkretnej rodziny (`exp(Kt)V0`). Nie
  mówi NIC o tym, czy jakiekolwiek realne okno M/S faktycznie ma ten
  kształt — to osobne pytanie, do sprawdzenia dopiero PO przejściu tego
  testu, analogicznie do filtru kształtu zastosowanego wcześniej przy
  moście Fouriera (`PREREG_MS_K_EVENTS.md` sekcja 12).
- Nieliniowości pominięte w konstrukcji (nasycenie `tanh`, waga `β(Q)`
  — `GS_Matrix_Helical_Coupling_DRAFT.md` sekcja 4) NIE są tu testowane
  — to ograniczenie MODELU, nie tego testu ESTYMATORA.
- Zakłada się `dt` znane dokładnie (stała częstotliwość próbkowania) —
  rozsądne dla realnych danych (patrz też niezmienniczość `ratio`
  względem `dt` zweryfikowana przy moście Fouriera, `PREREG_MS_K_EVENTS.md`
  sekcja 10 — analogiczna niezmienniczość TU nie jest jeszcze
  zweryfikowana, warto sprawdzić przy okazji).
- Wielostartowość (4 starty) redukuje, ale nie eliminuje ryzyka minimów
  lokalnych (obserwowane wcześniej przy pojedynczym starcie) — jeśli
  `N_trials=30` pokaże istotny odsetek "uciekających" dopasowań
  (duże odchylenia mediany od trzpienia rozkładu), zostanie to
  zgłoszone jako osobne ograniczenie, nie ukryte w samej medianie.

## 7. Co dalej, zależnie od wyniku

- **A+B przechodzą**: estymator wystarczająco solidny, żeby przejść do
  kroku "realne dane" — te same domeny co w `PREREG_MS_K_EVENTS.md`
  (sejsmika, łożyska, BTC) dla porównywalności, zastosować estymator do
  tych samych okien zdarzeń, zgłosić rozkład `Re(λ)_obs`/`Im(λ)_obs`,
  sprawdzić czy jest tam stabilna struktura (analogiczne pytanie "wąski
  czy rozjechany rozkład" co przy moście Fouriera).
- **A lub B nie przechodzą**: uczciwy raport, zero retuningu progów po
  wyniku, zamknięcie tej gałęzi na tym etapie — analogicznie do losu
  mostu Fouriera na realnych danych.
