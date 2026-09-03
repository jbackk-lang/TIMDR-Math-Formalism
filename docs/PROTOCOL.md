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
