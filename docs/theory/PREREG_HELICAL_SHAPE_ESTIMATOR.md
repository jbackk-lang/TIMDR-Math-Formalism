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

## 8. Wynik oficjalnego testu (2026-09-15, PO zamrożeniu powyższego)

Uruchomiono dokładnie wg specyfikacji: 4 reżimy × 5 poziomów szumu ×
30 powtórzeń (V0 losowane NIEZALEŻNIE każdorazowo z `Uniform(0.5,1.5)`,
NIE stałe jak we wstępnej eksploracji sekcji 7 draftu — to jest ważna
różnica, patrz niżej), 4-punktowa wielostartowość bez zmian.

**Kryterium A (dokładność, `ω₀=3.0`) — NIE PRZECHODZI.** Błąd `Re(λ)`
rośnie szybko wraz z szumem i przekracza próg 25% już przy najmniejszym
nieszumowym poziomie testowanym: `noise=0.02`→30.2%, `noise=0.05`→66.6%,
`noise=0.10`→**99.6%** (błąd rzędu samej wartości). `Im(λ)` natomiast
zachowuje się dobrze przez cały zakres: `noise=0.02`→1.8%,
`noise=0.05`→4.1%, `noise=0.10`→8.4% — WSZYSTKIE poniżej progu 25%.
**To jest asymetryczny wynik: `Im(λ)` (częstość rotacji) jest solidnie
odzyskiwane, `Re(λ)` (tempo obwiedni) NIE jest.**

**Kryterium B (klasyfikacja obecności rotacji) — PRZECHODZI.** Czułość
(`ω₀=3.0`, `Im_fit>0.3`): `100%, 96.7%, 93.3%` dla `noise=0.02/0.05/0.10`
— powyżej progu 80% na całym zakresie. Swoistość (`ω₀=0.0` i `ω₀=0.5`,
`Im_fit<=0.3`): `100%/90%/96.7%` (`ω₀=0.0`) i `100%/100%/100%`
(`ω₀=0.5`) dla `noise=0.02/0.05/0.10` — również powyżej 80% wszędzie.
Binarny detektor "czy jest genuine rotacja" jest solidny w całym
testowanym zakresie szumu, mimo że dokładna wartość `Re(λ)` nie jest.

**Ogólny wynik: CZĘŚCIOWY SUKCES (tylko B), zgodnie z regułą z sekcji 5
— nie zaokrąglone w górę do "sukces".** Zero progów zmienionych po
zobaczeniu wyniku.

**Dlaczego wynik jest gorszy niż wstępna eksploracja sekcji 7 draftu**:
tamten test używał JEDNEGO, STAŁEGO warunku początkowego (`V0=(1.0,0.3)`)
i tylko 6-10 powtórzeń. Ten test losuje `V0` niezależnie na każde
powtórzenie i używa 30 powtórzeń — bardziej reprezentatywny, i wychodzi
na jaw, że wcześniejszy dobry wynik dla `Re(λ)` był częściowo artefaktem
jednego, wygodnego `V0`, nie ogólną własnością metody. Dokładnie to, po
co jest formalna pre-rejestracja z większą próbą — złapać to, zanim
trafi do realnych danych.

**Odnotowany artefakt numeryczny (nie błąd wyniku)**: dla `ω₀=1.5`
(dokładnie na progu) `Im(λ)_true` z `np.linalg.eigvals` wychodzi jako
`~1.3e-8` zamiast dokładnego zera (szum zmiennoprzecinkowy przy
podwójnym pierwiastku), co przy liczeniu błędu względnego
(`|Im_fit-Im_true|/|Im_true|`) daje pozornie duże/dziwne liczby przy
dzieleniu przez niemal-zero. Sekcja 5 (kryterium C) z góry wyłączała
`ω₀=1.5` z oceny sukcesu/porażki dokładnie z tego powodu — artefakt nie
wpływa na werdykt A/B powyżej.

## 9. Wniosek i zawężenie zakresu (wzorem mostu Fouriera)

Ten estymator NIE nadaje się do ilościowego pomiaru `Re(λ)` na realnych
danych — błąd rzędu dziesiątek-do-100%+ nawet przy umiarkowanym szumie.
NADAJE SIĘ natomiast jako binarny detektor "czy w tym oknie jest
w ogóle genuine rotacja (`Im(λ)≠0`), czy nie" — to zweryfikowane
solidnie w całym testowanym zakresie szumu.

Jeśli ma być kontynuowany krok "realne dane", to WYŁĄCZNIE w tym
zawężonym zakresie: pytanie "ile realnych okien M/S klasyfikuje się
jako 'ma rotację' vs 'nie ma'", NIE "jaka jest wartość `Re(λ)`/`Im(λ)`
tych okien". To jest analogiczne zawężenie do tego, jakie spotkało most
Fouriera (`RESULT_FOURIER_BRIDGE_SCOPE.md`) — konstrukcja przeżywa, ale
w węższym, uczciwie odnotowanym zakresie niż pierwotnie zakładano.

## 10. Zastosowanie do realnych danych (binarny detektor, zgodnie z zawężeniem)

Te same okna zdarzeń co `PREREG_MS_K_EVENTS.md` (sekcje 9 i 11 —
sejsmika Ridgecrest 2019, łożyska CWRU, BTC/USD godzinowy), dla
porównywalności. Klasyfikacja wg zamrożonego progu `Im(λ)_fit>0.3`
(sekcja 4-5 powyżej), procedura dopasowania BEZ ZMIAN.

**Znaleziona i naprawiona usterka implementacyjna (przy pierwszym
uruchomieniu na realnych danych, PRZED zapisaniem wyniku końcowego)**:
pierwsza próba dała przepełnienia (`RuntimeWarning: overflow in exp`) —
optymalizator (`method='lm'`, bez granic) w trakcie poszukiwań
odwiedzał wartości `μ·t` na tyle duże (t do 63), że `exp()` przepełniał
zakres `double`. Dodano: (a) granice parametrów
(`|μ|,ω₀<=10`, dobrane tak, żeby `exp(10·63)` nie przepełniało),
`method='trf'` (obsługuje granice, `'lm'` nie); (b) normalizację
amplitudy okna (`(x-mean)/std`) PRZED dopasowaniem — nieobecną w
oryginalnej pre-rejestracji, bo syntetyczny test miał już amplitudę
rzędu `O(1)`, a sejsmika ma amplitudy rzędu `O(1e4)`. **To jest
udokumentowana poprawka numeryczna, zastosowana JEDNOLICIE do
wszystkich domen przed policzeniem końcowych liczb — nie dobrana pod
konkretny wynik którejkolwiek domeny.** Wynik NAIWNY (przed poprawką,
zanieczyszczony przepełnieniami) i POPRAWIONY różnią się istotnie
(np. sejsmika CLC: 45.5%→0.0%) — POPRAWIONY jest tym, który się liczy;
naiwny odnotowany tu wyłącznie dla przejrzystości procesu.

**Wynik (wersja poprawiona, N_fit_ok=N_okien wszędzie — zero
nieudanych dopasowań)**:

| domena | N okien | N z rotacją | % z rotacją |
|---|---|---|---|
| sejsmika CLC | 11 | 0 | 0.0% |
| sejsmika RIO | 27 | 2 | 7.4% |
| łożyska normal | 4 | 2 | 50.0% |
| łożyska ball_fault | 4 | 4 | 100.0% |
| łożyska outer_race | 1 | 1 | 100.0% |
| łożyska inner_race | 0 | — | brak okien (jak w PREREG_MS_K_EVENTS) |
| BTC/USD godzinowy | 0 | — | brak okien (jak w PREREG_MS_K_EVENTS) |

**Odczyt, ostrożnie**: sejsmika niemal nigdy nie klasyfikuje się jako
"ma genuine rotację" (0-7%), łożyska niemal zawsze (50-100%) — kierunek
fizycznie sensowny (wibracje łożysk są z natury wielocyklowe/
oscylacyjne w oknie 64 próbek, sejsmiczne wstąpienie fazy P jest
bardziej impulsywne, mniej oscylacyjne w tej samej skali okna — zgodne
z ustaleniem z `RESULT_FOURIER_BRIDGE_SCOPE.md`). **To jest opisowa
obserwacja na małej próbie (N=1-27 na domenę), NIE test istotności
statystycznej** — brak porównania z tłem/oknami losowymi, brak
Manna-Whitneya, brak korekty Bonferroniego — żaden z wymogów protokołu
anty-numerologicznego (`SKILL_timdr-signal-framework.md` §2) nie został
tu zastosowany. Jeśli ten wynik ma być podstawą jakiegokolwiek
twierdzenia silniejszego niż "opisowo widać taki wzorzec", potrzebny
osobny, pełny test z kontrolami — nie zrobiony tutaj.
