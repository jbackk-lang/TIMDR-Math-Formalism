# Protokół: numerologia vs. realna matematyka

TIMDR-Math-Formalism nie jest detektorem sygnału czasowego — jest
detektorem **matematycznej sensowności**. Wejściem nie jest szereg
czasowy, tylko **hipoteza**: opis struktury (ciąg, wzór, "rezonans",
konstrukcja kategorii, geometryczny pattern) plus konkretne twierdzenie
o niej ("ta sekwencja ma specjalny rezonans z liczbami pierwszymi", "ta
transformacja zawsze zmniejsza energię", "ta kategoria ma wyjątkową
własność"). Hipoteza musi być sformułowana precyzyjnie — "czuję, że tu
coś jest" nie kwalifikuje się.

Ten dokument jest zastosowaniem protokołu z `timdr-signal-framework`
(sekcje §9 i §13, patrz niżej) do statycznych struktur matematycznych
zamiast do sygnałów czasowych — sam mechanizm (pre-rejestracja, kontrola
+/-, prawdziwy test statystyczny, honest negative result) jest
identyczny.

## Sześć kroków

### 1. Pre-rejestracja (`Hypothesis` + `Preregistration`)

Zapisz definicję struktury/wzoru PRZED dotknięciem jakichkolwiek danych.
Żadnego dostrajania po fakcie — to jest odpowiednik data-snooping.
`Preregistration.create()` liczy odcisk SHA-256 hipotezy + parametrów;
`verify_unchanged()` łapie każdą zmianę definicji po zobaczeniu wyniku.

*Odpowiednik skilla:* §9 krok 1 ("Pre-register the feature definition
before touching real data"), §13 krok 1 ("Define the exact objects and
exact mapping BEFORE running anything").

### 2. Generowanie danych testowych (`sieve_of_eratosthenes`,
`random_background`, `ar1_noise`)

Syntetyczne: liczby pierwsze (dokładne, sito Eratostenesa — nie ręcznie
przepisana tabela, patrz przestroga niżej), losowe tło, szum AR(1).
Realne: katalogi/zbiory/grafy — dostarczane przez wywołującego jako
zwykłe `np.ndarray`, pipeline nie zakłada źródła danych.

*Odpowiednik skilla:* §9 krok 2 ("Compare a pre-event window against
random background windows"). AR(1), nie biały szum, dla kontroli
negatywnej — §9 krok 4: biały szum bywa zbyt gładki, żeby w ogóle
wyzwolić testowaną metrykę, co robi z porównania biały-vs-biały
zdegenerowaną (0 vs 0) kontrolę.

### 3. Mierzalny efekt (`metric_fn`, dostarczana przez wywołującego)

Np. "moja struktura daje niższy `tension_zscore` niż tło" albo "moja
transformacja ma mniejszy błąd niż baseline". Pipeline nie narzuca
konkretnej metryki — narzuca tylko, że musi być jedna, ustalona z góry
funkcja `ndarray -> float`, ta sama w kontrolach i w teście głównym.

### 4. Prawdziwy test statystyczny (`mann_whitney_test`)

Mann-Whitney U, nie porównanie percentylowe "na oko".

*Odpowiednik skilla:* §13 krok 3 protokołu ("Use a real significance
test (Mann-Whitney U), not just a percentile comparison — a percentile
against one background distribution is weaker evidence than a proper
two-sample test").

#### 4a. Effect size (`rank_biserial_effect_size`, `TestResult.effect_size_r`)

p-wartość odpowiada na pytanie "czy jest efekt", nie "jak duży jest
efekt" — dwa wyniki o p=0.001 mogą mieć zupełnie różny rozmiar różnicy
między grupami. `mann_whitney_test` liczy dodatkowo rank-biserial
correlation:

```
r = 2*U/(n_test*n_background) - 1
```

gdzie `U` to statystyka Manna-Whitneya dla grupy testowej. `r ∈ [-1, 1]`:
`r>0` — grupa testowa tendencyjnie wyżej niż tło, `r<0` — niżej,
`|r|=1` — pełna separacja rang (żadnego nakładania), `r=0` — brak
tendencji w żadną stronę. Progi umowne do etykietowania
(`effect_size_label`, Cohen-style): `<0.1` pomijalny, `0.1–0.3` mały,
`0.3–0.5` średni, `≥0.5` duży.

Rozróżnienie "istotny, ale mały" vs "istotny i duży" ma znaczenie
praktyczne: przy dużym `n` nawet trywialna, nieistotna z punktu widzenia
zastosowania różnica da p bardzo małe — `r` mówi, czy warto się tym
przejmować, `p` tylko, czy różnica jest odtwarzalna, a nie przypadkowa.

#### 4b. Moc testu (power) — czego p-wartość NIE mówi

**Wynik nieistotny (p ≥ α) NIE jest automatycznie dowodem braku
efektu — może być brakiem mocy statystycznej.** Moc testu to
prawdopodobieństwo wykrycia efektu danej wielkości, GDYBY istniał, przy
danym `n_test`, `n_background` i teście. Niska moc (za mało obserwacji,
za mało zmienności/zdarzeń w danych) daje p wysokie niezależnie od tego,
czy efekt realnie istnieje.

Realny przykład tego dokładnego problemu: walidacja operatora rezonansu
na prawdziwych danych pogodowych (`docs/REAL_DATA_VALIDATION.md`) dała
p=1.0 dla obu testowanych progów — ale nie dlatego, że rezonans nie
przekracza przypadku, tylko dlatego, że w 24-dniowym oknie było zero
zdarzeń współbieżnych do porównania w OBU grupach (realnej i null).
Test nie miał żadnej mocy statystycznej w tym oknie — kontrola
pozytywna (sztucznie wymuszona współbieżność) wykazała, że sama
metodyka działa (p≈0.0002), więc zerowy wynik główny to ograniczenie
danych, nie mechaniki testu.

**Jak sprawdzić moc, zamiast zgadywać:**

- **NIE licz "mocy obserwowanej" (post-hoc/retrospective power) z
  samego wyniku testu.** To zdeterminowana funkcja p-wartości — nie
  wnosi żadnej nowej informacji i jest odradzane w literaturze
  statystycznej właśnie dlatego, że tworzy złudzenie niezależnego
  potwierdzenia tam, gdzie go nie ma.
- **Licz moc prospektywnie, symulacją, PRZED interpretacją wyniku
  nieistotnego jako "brak efektu":** wygeneruj wiele syntetycznych
  powtórzeń ze ZNANYM, założonym efektem (dokładnie to, co robi
  `positive_injector` w `run_controls`, tylko z realnym `n_test`/
  `n_background` z Twojego przypadku, nie z domyślnych parametrów
  kontroli) i sprawdź, jaki odsetek powtórzeń wykrywa efekt (p<α). Niski
  odsetek = niska moc = wynik nieistotny nie mówi nic pewnego o tym, czy
  efekt istnieje.
- Praktyczny sygnał ostrzegawczy bez pełnej symulacji: jeśli jedna z
  grup ma zerową lub bliską zeru wariancję/liczbę zdarzeń (jak
  ciśnienie w przykładzie z Krakowa — 0 anomalii na 24 dni), test
  strukturalnie nie ma jak wykryć różnicy, niezależnie od tego, czy
  realny efekt istnieje.

#### 4c. Efekt jako operator (formalizacja)

Do tej pory "efekt" był nieformalnie opisywany przez pola `TestResult`
(`median_test`, `median_background`, `effect_size_r`), bez jednego
nazwanego obiektu, który je zbiera. Formalnie:

```
E(x_test, x_bg) := (median(x_test) − median(x_bg),  r(x_test, x_bg))  ∈  𝓔
𝓔 := ℝ × [−1, 1]          (przestrzeń efektów)
e₀ := (0, 0) ∈ 𝓔          (wyróżniony element "brak efektu")
```

gdzie `x_test = {metric_fn(w) : w ∈ okna_testowe}`,
`x_bg = {metric_fn(w) : w ∈ okna_tła}` — próbki metryki policzonej na
każdym oknie (`P_k` z §4c), `r` to `rank_biserial_effect_size` z §4a.

To NIE jest nowy rachunek — `E(x_test, x_bg)` to dokładnie para
`(main_result.median_test − main_result.median_background,
main_result.effect_size_r)`, którą `TestResult` już zwraca. Formalizacja
tu polega na nazwaniu tego, co kod już liczy, jako jednego operatora
`E`, i na oddzieleniu go od testu istotności:

- `E` opisuje **co i jak bardzo się różni** (kierunek przez pierwszą
  współrzędną, skalowana wielkość przez drugą) — nie mówi, czy różnica
  jest wiarygodna.
- `mann_whitney_test` (p-wartość) opisuje **czy różnicę można odróżnić
  od przypadku** przy danym `n_test`, `n_background` — nie mówi nic o
  jej wielkości.

`TestResult` łączy oba w jeden obiekt (`p`, `E`) — a nie tylko `p` — co
jest właśnie treścią §4a: "istotny, ale mały" ma `p<α`, `|E₂|` blisko 0;
"istotny i duży" ma `p<α`, `|E₂|` blisko 1.

#### 4d. Generatory kontroli jako rozkłady, moc kontroli

`positive_injector`, `negative_generator_a`, `negative_generator_b` są
w kodzie funkcjami Python `(window_size, seed) -> ndarray`. Formalnie
każda z nich jest **próbnikiem** (samplerem) z pewnego rozkładu
prawdopodobieństwa na przestrzeni okien:

```
D_pos, D_A, D_B  — miary probabilistyczne na ℝ^window_size
positive_injector(window_size, seed) ~ D_pos
negative_generator_a(window_size, seed) ~ D_A
negative_generator_b(window_size, seed) ~ D_B     (D_B = D_A z konstrukcji)
```

`metric_fn` przenosi je (pushforward) na rozkłady wartości metryki:
`F_pos = metric_fn_*(D_pos)`, `F_A = metric_fn_*(D_A)`,
`F_B = metric_fn_*(D_B)`.

- **Kontrola pozytywna** testuje `H₀: F_pos =_d F_A` (równość
  rozkładów) i chce ją ODRZUCIĆ (`p_pos < α`) — bo `D_pos` z konstrukcji
  ma różnić się od `D_A`.
- **Kontrola negatywna** testuje `H₀: F_A =_d F_B` i chce ją
  ZACHOWAĆ (`p_neg ≥ α`) — bo `D_A = D_B` z konstrukcji (ten sam
  generator, inny seed), więc jedyna różnica, jaką test mógłby złapać,
  byłaby fałszywym alarmem.

**Moc kontroli** (odrębna od mocy testu głównego w §4b): tutaj, w
przeciwieństwie do danych realnych, `D_pos` i `D_A` są ZNANE z
konstrukcji — więc moc kontroli pozytywnej DA SIĘ policzyć dokładnie,
metodą Monte Carlo, zamiast tylko szacować:

```
power_pos(n_windows) ≈ (1/R) · Σ_{i=1}^{R} 𝟙[p_pos^(i) < α]
```

gdzie `p_pos^(i)` to p-wartość z `i`-tego niezależnego powtórzenia
`run_controls` przy ustalonym `n_windows` (różne seedy). Praktyczna
konsekwencja: jeśli kontrola pozytywna regularnie nie przechodzi przy
rozsądnym `n_windows` (np. 30), są dwie różne przyczyny do rozróżnienia
— (a) wstrzyknięty efekt jest za słaby (np. `effect_shift`/
`bias_strength` w `dashboard.py` ustawione za nisko) — wtedy zwiększenie
`n_windows` PODNOSI moc i problem znika; albo (b) `metric_fn` jest
strukturalnie nieczuła na tego rodzaju efekt — wtedy zwiększanie
`n_windows` w nieskończoność tego nie naprawi, tylko coraz dokładniej
to udowodni (moc rośnie do sufitu niższego niż 1, nie do 1).

#### 4e. Okno jako operator (formalizacja)

Krok 2/5 dzieli dane na okna o rozmiarze `window_size` — dotąd tylko
parametr w kodzie, nie nazwany obiekt matematyczny.

Klasyczny operator przesuwanego okna o promieniu `k` na sygnale
`x: T → ℝ`:

```
W_k(x)(t) = (x(t-k), ..., x(t+k))    ∈ ℝ^(2k+1)
```

Funkcjonał `φ` (mediana, kwantyl `q`) liczony na oknie to złożenie
`φ ∘ W_k : (T→ℝ) → (T→ℝ)`, `(φ∘W_k)(x)(t) = φ(W_k(x)(t))` — dokładnie
to, co robi `metric_fn` w `run_controls`/`mann_whitney_test`, tylko
zapisane jako złożenie dwóch operatorów zamiast pojedynczej funkcji
Python.

**Ważne zastrzeżenie, żeby `W_k` nie było mylone z tym, co faktycznie
robi kod:** obecna implementacja (`examples/prime_resonance_demo.py`,
`dashboard.py::scenario_custom`) dzieli dane na okna **rozłączne**
(partycja, nie przesuwane/nachodzące) — każdy punkt danych trafia do
dokładnie jednego okna, nie do `window_size` różnych okien jak przy
klasycznym sliding window. To świadomy wybór, nie uproszczenie: test
Manna-Whitneya zakłada niezależne obserwacje w każdej grupie —
nachodzące na siebie okna dałyby silnie skorelowane próbki (sąsiednie
okna dzielą większość punktów) i złamałyby to założenie. Formalnie
używany operator to więc **partycja** `P_k(x) = (W'_1, ..., W'_m)` z
`W'_i` rozłącznymi blokami rozmiaru `k`, nie `W_k` w klasycznym sensie
— `W_k` powyżej jest podane jako punkt odniesienia z literatury, nie
jako opis kodu.

### 5. Kontrola pozytywna i negatywna (`run_controls`)

Uruchamiane RAZEM, PRZED testem na danych głównych, jako bramka:

- **Pozytywna** — przypadek ze sztucznie wstrzykniętym efektem, który
  metryka POWINNA wykryć (p małe). Jeśli nie wykrywa, metryka jest za
  mało czuła — wynik na danych głównych byłby niediagnostyczny.
- **Negatywna** — dwie NIEZALEŻNE próbki czystego tła bez efektu (p
  duże / nieistotne). Jeśli metryka znajduje "efekt" między dwoma
  próbkami szumu, jest przeczulona — nie da się jej ufać.

Główny test uruchamiasz TYLKO jeśli obie kontrole przejdą.

*Odpowiednik skilla:* §9 krok 4 ("Run a synthetic self-test with BOTH a
positive and a negative control before running on real data, and gate
the real run on both passing").

### 6. Werdykt (`TestResult.verdict` / `format_report`)

Wynik negatywny (p wysokie) jest PEŁNĄ odpowiedzią, nie "prawie
działa". Jeśli p ≈ 0.9–0.99 → brak efektu, koniec, raportuj to wprost.

*Odpowiednik skilla:* §9 krok 5 ("A negative result is a valid, complete
answer — report it as such"), §13 krok 4 ("Report the actual result,
including 'no effect', without narrative softening").

## Formalna przestrzeń TIMDR-Math

Sekcje 4a-4e opisują poszczególne kroki jako osobne wzory. Ten dokument
zbiera je w jedną przestrzeń, żeby było widać, że sześć kroków to
złożenie operatorów na wspólnych obiektach, nie sześć niezależnych
sztuczek.

**Przestrzeń hipotez `H`.** Krotka `(opis_struktury, opis_efektu, θ)`,
gdzie pierwsze dwa to teksty (pola `Hypothesis.description`,
`Hypothesis.effect_description`), a `θ ∈ Θ` to parametry (`dict`, np.
`{"n_max": 100000, "seed": 42}`). Pre-rejestracja to funkcja
`fp: H → {0,1}^256` (SHA-256 z serializacji JSON) — `fp(h₁)=fp(h₂) ⟹
h₁=h₂` z zaniedbywalnym prawdopodobieństwem kolizji. To jest założenie
kryptograficzne (własność SHA-256), nie twierdzenie matematyczne
udowodnione w tym repo — warto to rozróżnienie zachować, żeby nie
przypisać `fp` mocniejszej gwarancji, niż faktycznie ma.

**Przestrzeń metryk `M`.** Zbiór funkcji `𝓜: 𝒳 → ℝ`, gdzie `𝒳` to
przestrzeń możliwych okien danych (np. `ℝ^window_size` albo
`ℕ^window_size` dla danych całkowitych, jak liczby pierwsze).
`metric_fn` dostarczana przez wywołującego jest elementem `M` —
pipeline nie narzuca, który, tylko wymaga, żeby był jeden, ustalony,
ten sam w kontrolach i w teście głównym (to jest dokładnie treść kroku
3, tu tylko nazwana jako "ten sam element `M` używany dwa razy").

**Operator testowy jako funkcjonał.** `mann_whitney_test` jest
funkcjonałem

```
T: M × 𝒫(𝒳) × 𝒫(𝒳) → 𝕋
T(𝓜, X_test, X_bg) = TestResult(...)
```

gdzie `𝒫(𝒳)` to zbiory skończonych próbek okien, a `𝕋` to przestrzeń
`TestResult` (statystyka, p, mediany, `E ∈ 𝓔` z §4c). `T` bierze
metrykę — element `M` — i dwie próbki, zwraca jeden obiekt łączący test
istotności i efekt.

**Cały protokół jako złożenie.** Sześć kroków (§1-§6) to w tym
zapisie:

```
Report = format_report ∘ ⟨Gate, T⟩ ∘ Generate ∘ Preregister
```

- `Preregister: H × Θ → Preregistration` (krok 1, `fp` powyżej)
- `Generate: Preregistration → (X_test, X_bg, X_pos, X_A, X_B)` (krok 2,
  generatory z §4d — tu jako próbki wyciągnięte z `D_pos, D_A, D_B`)
- `⟨Gate, T⟩`: `Gate: (X_pos, X_A, X_B) → ControlResult` (krok 5,
  `run_controls`) jest **strażnikiem** dla `T` na danych głównych — `T`
  na `(X_test, X_bg)` liczy się TYLKO gdy `Gate(...).passed = 1`
  (dokładnie dlatego `format_report` w kodzie zwraca wcześnie, gdy
  bramka nie przeszła — to nie jest wygoda API, to część definicji
  złożenia: `T` na danych głównych jest częściową funkcją, zdefiniowaną
  tylko na obrazie `Gate⁻¹(1)`)
- `format_report: (ControlResult, Optional[𝕋]) → Report` (krok 6)

To złożenie NIE jest nowym mechanizmem — to jest dokładnie to, co robi
`main()` w `examples/prime_resonance_demo.py`, zapisane jako jeden wzór
zamiast sekwencji wywołań funkcji.

## Dwa zabezpieczenia spoza sześciu kroków — też wbudowane

**Korekta na wielokrotne porównania** (`bonferroni_correct`). Jeśli
hipoteza powstała z przeszukania wielu kandydatów (pozycji w ciągu,
progów, wariantów wzoru) zamiast z jednej, z góry ustalonej definicji,
surowe p jest niewiarygodne — musi być skorygowane za liczbę
przeszukanych wariantów. Patrz skill §13 case study 1: "lokalnie dobre"
30-cyfrowe okno o p≈0.012 wyparowało do p≈1.0 po korekcie Bonferroniego
za ~2000 przeszukanych okien — podręcznikowy artefakt wielokrotnych
porównań, nie efekt.

**Replikacja na niezależnym zbiorze** — NIE jest zautomatyzowana w
kodzie (wymaga świadomej decyzji wywołującego, żeby uruchomić protokół
drugi raz na innych danych), ale jest częścią protokołu: pojedynczy,
pozornie pozytywny wynik to nie dowód. Patrz skill §9 krok 6: sygnał
"działający" na Bitcoinie w jednym repo tego ekosystemu zmienił znak po
replikacji na złocie — klasyczny overfitting, nie realna struktura.

## Coś, czego ten pipeline NIE robi za ciebie

**Sprawdzenie, czy istnieje już ugruntowany, dedykowany model
statystyczny dla tej dziedziny** — zanim uznasz negatywny wynik
homemade metryki za ostateczny. Patrz skill §13 case study 4,
kontynuacja: metryka ad hoc (kształt przerwy vs `log(x)`) dała wynik
negatywny dla liczb pierwszych, ale po przejściu na prawdziwy,
ugruntowany model tej dziedziny (model Craméra/koniektura Gallaghera:
`x_n = gap_n / log(p_n)` zbiega do Exponential(1)) znalazła się
subtelna, ale realna, statystycznie istotna struktura (korelacja
szeregowa między kolejnymi przerwami, r=-0.0568, p≈4.4e-57).

**Lekcja ogólna, do zastosowania ręcznie przy każdym użyciu tego
pipeline'u**: negatywny wynik na homemade metryce jest dowodem przeciwko
TEJ metryce, nie automatycznie przeciwko zjawisku, które miała wykryć.
Zanim ogłosisz "brak struktury", zapytaj: czy ta dziedzina ma już
własny, ugruntowany test (jak `tension_zscore` dla napięć
kosmologicznych, skill §18) — i jeśli tak, uruchom `mann_whitney_test`
na TYM modelu, zanim potraktujesz wynik jako ostateczny.

## LLM jako warstwa "Matematyka / Formalizm"

Ten pipeline dostarcza mechanikę testu, nie osąd matematyczny. Rola
LLM-a (lub człowieka) przy formułowaniu `Hypothesis` i interpretacji
wyniku:

- sprawdzić, czy struktura jest dobrze zdefiniowana (brak sprzeczności,
  brak niejasności) — patrz skill §15 dla przykładu odwrotnego: poprawnie
  wyglądające słownictwo teorii kategorii bez spełnionych definicji
  (brak morfizmów identycznościowych, symbole używane jednocześnie jako
  morfizmy i funktory, itd.) — struktura może być źle zdefiniowana
  jeszcze PRZED jakimkolwiek testem statystycznym;
- porównać z istniejącą matematyką (czy to już znane, czy nowe, czy bez
  sensu);
- pomóc zbudować `metric_fn`, `positive_injector`,
  `negative_generator_a/b` — czyli podjąć decyzje z kroków 2-3 powyżej.

LLM nie "udowadnia" — pomaga odsiać numerologię od rzeczy, które warto
dalej badać, i pomaga złożyć poprawnie skalibrowany test. Ostateczny
werdykt zawsze pochodzi z `mann_whitney_test` + kontroli, nie z oceny
"na oko".

## Przykład

`examples/prime_resonance_demo.py` implementuje dokładnie scenariusz
"ktoś mówi: mam ciąg, który ma specjalny rezonans z liczbami
pierwszymi" — pre-rejestracja, kontrola +/-, test na prawdziwych liczbach
pierwszych (sito Eratostenesa) vs losowe tło, werdykt. Uruchom go
lokalnie i przeczytaj wydrukowany raport — ten dokument celowo NIE
podaje z góry, jaki wynik dostaniesz (to byłoby dokładnie tym
narracyjnym miękczeniem wyniku, przed którym ostrzega krok 6).
