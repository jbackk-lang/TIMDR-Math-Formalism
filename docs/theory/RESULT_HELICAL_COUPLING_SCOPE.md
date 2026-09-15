# Wynik: zakres sprzężenia helikalnego K↔Θ_bif

> Streszczenie `GS_Matrix_Helical_Coupling_DRAFT.md` +
> `PREREG_HELICAL_SHAPE_ESTIMATOR.md` (pełna metodologia i kroki
> pośrednie tam). Ten plik to sam wynik i wniosek. Ten wątek pozostaje
> częścią GS-Matrix — **eksploracyjny, nieaksjomatyzowany toy model**,
> nie podnoszony tu do statusu aksjomatu (w przeciwieństwie do mostu
> Fouriera, który ma twardą matematyczną podstawę — Δt·Δf=1/4π to
> ustalona tożsamość analizy Fouriera; to sprzężenie to nowa,
> zbudowana w tej sesji konstrukcja).

## Wniosek

**Konstrukcja matematyczna sprzężenia K↔Θ_bif jest wewnętrznie spójna
i zweryfikowana numerycznie, ale nie ma jeszcze ilościowej treści
empirycznej — ma za to ograniczoną, zweryfikowaną treść JAKOŚCIOWĄ.**

Nowy, jawnie oznaczony most (K i Θ_bif były wcześniej rozłącznymi
obiektami w kodzie) daje: próg `ω₀_crit=(λ_down+λ_up)/2` na pojawienie
się genuine rotacji, `Re(λ)=(λ_up-λ_down)/2`, i przewidywanie, że
`N(t)=S_down·S_up` powinno oscylować z częstością `2ω` — wszystko
zweryfikowane algebraicznie i numerycznie (R²=1.000000 dla przewidywania
N(t)).

Droga do realnych danych wymagała estymatora `Re(λ)/Im(λ)` z pojedynczego
`x(t)`. Naiwny estymator (różnicowanie/DMD) zawiódł katastrofalnie przy
szumie (błędy 700-3000%). Estymator przez dopasowanie znanego kształtu
(nieliniowe najmniejsze kwadraty względem całego okna) jest znacznie
lepszy, ale formalny test na większej próbie (N_trials=30, losowane
warunki początkowe) pokazał wynik ASYMETRYCZNY: dokładna wartość
`Re(λ)` NIE jest wiarygodnie odzyskiwana (błąd 30-100%+ już przy
umiarkowanym szumie), ale BINARNA klasyfikacja "czy jest genuine
rotacja" JEST wiarygodna (czułość/swoistość 90-100% w całym testowanym
zakresie szumu).

## Zawężenie (wzorem mostu Fouriera)

Ta konstrukcja **nie nadaje się** do ilościowych twierdzeń o `Re(λ)`
na realnych danych. **Nadaje się** jako binarny detektor obecności
genuine rotacji w oknie sygnału.

Zastosowany w tym zawężonym zakresie do tych samych okien zdarzeń co
`PREREG_MS_K_EVENTS.md` (sejsmika Ridgecrest 2019, łożyska CWRU,
BTC/USD godzinowy):

| domena | N okien | % z wykrytą rotacją |
|---|---|---|
| sejsmika CLC | 11 | 0.0% |
| sejsmika RIO | 27 | 7.4% |
| łożyska normal | 4 | 50.0% |
| łożyska ball_fault | 4 | 100.0% |
| łożyska outer_race | 1 | 100.0% |

Kierunek fizycznie sensowny (łożyska wielocyklowe w oknie 64 próbek →
częściej "rotacja"; sejsmika bardziej impulsywna → rzadziej) — ale to
**opisowa obserwacja na małej próbie, nie test istotności
statystycznej**. Brak porównania z tłem, brak Manna-Whitneya, brak
korekty Bonferroniego — protokół anty-numerologiczny
(`SKILL_timdr-signal-framework.md` §2) nie został tu w pełni
zastosowany.

## Co to oznacza dla dalszej pracy

- Konstrukcja pozostaje we właściwym miejscu ekosystemu: GS-Matrix jako
  świadomie oznaczona eksploracja/heurystyka (`gs_matrix.py`,
  `theta_bifurcation.py`), NIE promowana do `Axioms_*` — inaczej niż
  most Fouriera, który miał gotową matematykę PRZED tą sesją.
- Jeśli ktoś zechce twierdzić więcej niż "opisowo widać taki wzorzec
  rotacji na kilku realnych domenach": potrzebny pełny test z
  kontrolami (tło, Mann-Whitney, Bonferroni) — nie zrobiony.
- Jeśli ktoś zechce ilościowych `Re(λ)`/`Im(λ)` na realnych danych:
  potrzebny lepszy estymator niż ten z sekcji 7 draftu — otwarte,
  nieprzydzielone.
- **Future work, świadomie odłożone**: katalog naturalnie
  helikalnych/oscylacyjnych zdarzeń (analogicznie do "opcji 1B" z
  mostu Fouriera) mógłby dać czystszy test niż okna wyznaczone progiem
  `anomalia_flags`.
