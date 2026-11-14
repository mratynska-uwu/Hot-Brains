# Hot Brain: możliwe optymalizacje i krytyczna ocena projektu

Dokument zbiera wnioski z 9 prac z folderu `papers/`. Szczegółowe podsumowania są w [`docs/papers/`](papers/). Plik `s00521-024-10240-z (1).pdf` to bajtowy duplikat, więc pominęliśmy go. Stan projektu oceniamy na listopad 2026, po dodaniu sesji walidacyjnej i testowej.

## Status decyzji zespołu (14.11.2026)

Aktualnym celem jest **optymalizacja treningu**. Po przeglądzie dokumentu zespół podjął następujące decyzje:

| Punkt | Decyzja | Realizacja w kodzie |
|---|---|---|
| 2.2.2 „Etykiety pochodzą z ocen tłumu” i #2 (samoocena SAM) | **odrzucone** | – |
| #26 pomiar timingu fotodiodą | **odrzucone** | – |
| #3 kontrola confoundu wizualnego | **wstrzymane** | – |
| #20 RF / gradient boosting | **wstrzymane** | (random forest dodał wcześniej Szymon; boosting nie dodany) |
| #1 test permutacyjny | przyjęte | raport po treningu: CV + `permutation_test_score`; walidacja: p-value permutacji etykiet |
| #5 theta 4–8 Hz | przyjęte | `bands_hz.theta`, konfiguracja 1 |
| #6 ablacja pasm | przyjęte | raport: CV bez każdego pasma i tylko z nim |
| #7 kalibracja dla celu 2 | przyjęte | sesja **Calibration**: bloki 20 s, okna 2 s / krok 1 s, uczenie blokami; modele „stanu” sterują testem |
| #8 dłuższe okno i silniejsze wygładzanie | przyjęte | test z modelami stanu dekoduje okna 2 s; `feedback.smoothing` 0,1 |
| #9 odrzucanie epok ±150 µV | przyjęte | `eeg.artifact_max_abs_uv`, powody odrzuceń w raporcie |
| #10 CAR | przyjęte | `eeg.reference: "car"` (domyślnie wyłączone) |
| #11 selekcja cech i strojenie C w zagnieżdżonej CV | przyjęte | model `logreg_select` |
| #12 Welch | przyjęte | `eeg.psd_method: "welch"` (domyślnie) |
| #13 asymetria alfa i indeks β/α | przyjęte | rodziny `asymmetry` (valence) i `beta_alpha_ratio` (arousal) w konfiguracji 1 |
| #14 Hjorth, Katz/Petrosian, wariancja | przyjęte | rodziny `hjorth`, `katz_fd`, `petrosian_fd`, `variance` (do włączenia w profilu) |
| #15 `class_weight="balanced"` | przyjęte | logreg, SVM, SVM liniowy |
| #16 środkowy tercyl arousal | przyjęte | `targets.arousal.exclude_middle = 1/3` |
| #17 próg mediany tylko z treningu | przyjęte | progi liczone ze zbioru treningowego (`rethreshold`) |
| #18 regresja dla płaszczyzny V-A | przyjęte | model `ridge` (wartość ciągła w [−1, 1]) |
| #19 baseline −500…−200 ms | przyjęte | `profiles.*.baseline_ms` |
| #21 podejrzane etykiety | przyjęte | raport: prawdopodobieństwo poza foldem < 0,2 (cleanlab, jeśli zainstalowany) |
| #22 spoczynek z otwartymi oczami | przyjęte | sesja **Resting baseline** (90 s), zapis i mediana mocy pasm; normalizacja cech tym odniesieniem jeszcze nie jest stosowana |
| #23 reguła zatrzymania | przyjęte | GUI **Stopping rule** (domyślnie 65% balanced accuracy z 40 predykcji) |
| #24 czytelny feedback | przyjęte | kolor kropki zależny od odległości, sygnał dźwiękowy przy celu, próbna sesja testowa |
| #4 cechy ERP, #25 logistyka | bez decyzji | rodzina `erp` dostępna w profilach |

Dodatkowo, na prośbę zespołu, cechy są definiowane **osobno dla valence i arousal** w dwóch wybieralnych konfiguracjach. Konfiguracja 2 ma okno 450 ms; arousal korzysta w niej z alfy potylicznej (PO7, Oz, PO8), a valence z bety i gammy na elektrodach czołowych i centralnych (Fz, C3, Cz, C4). Po każdym treningu powstaje raport jakości z parametrami każdego modelu, osobno dla valence i arousal.

## 0. Cele projektu (punkt odniesienia)

1. **Cel 1:** wytrenować klasyfikator valence–arousal dla danego uczestnika na 80% zbioru treningowego i sprawdzić go na pozostałych 20%.
2. **Cel 2:** sesja testowa. Pokazujemy obrazy testowe o znanej valence i arousal, a uczestnik próbuje dostosować do nich własny stan emocjonalny, widząc na żywo valence i arousal odczytane z EEG.

## 1. Źródła w skrócie

| Plik | Praca | Najważniejsza lekcja dla nas |
|---|---|---|
| `s41598-024-75263-x` | Fiorini i in. 2024, *Sci Rep*: obrazy OASIS po 1 s, 64 kanały | Najbliższy nam paradygmat i ten sam filtr 1–48 Hz + notch 50 Hz. Najwięcej informacji niesie klaster **centralny** w oknie **LPP 400–500 ms**. Etykiety są bardzo zaszumione. Wynik przy podziale według osób jest słaby (F1 ≈ 0,42). |
| `brainsci-14-00364` | Lim i in. 2024, przegląd 77 badań | Arousal najlepiej widać w **potylicznej alfie**, valence w **beta/gamma** nad skroniami. Okno 700 ms niesie głównie informację ERP. Etykiety własne uczestnika są lepsze od ocen tłumu. Ostrzega przed confoundem kategorii obrazu. |
| `fnhum-16-1051463` | Garg i in. 2022, *Front Hum Neurosci*: Emotiv, obrazy OASIS | Przy obrazach i tanim headsecie valence jest prawie nie do zdekodowania (R² = 0,04, arousal 0,24). Najlepsze były **1–2 cechy**. Najlepsze elektrody (F7, T7, T8, F8) są poza naszym montażem. |
| `sensors-23-00915` | Yuvaraj i in. 2023, *Sensors*: porównanie metod ekstrakcji cech | **Fractal dimension** to najstabilniejszy zestaw cech, co wspiera nasz wybór. Warto dodać Katz/Petrosian i Hjortha, CAR oraz odrzucanie epok przy ±100 µV. Wyniki 80–97% są zawyżone przez wyciek danych. |
| `3647817.3647826` | Chen 2023, ACM ICBBS: DEAP | Realistyczny poziom to **61–65%** przy binarnym valence/arousal. Proponuje dodać theta, wybór klasyfikatora nie ma znaczenia, a próg mediany liczyć tylko na treningu. |
| `s00521-024-10240-z` | Castiblanco Jimenez i in. 2024, *Neural Comput Appl*: VR | Theta była najsilniejsza. Indeksy valence (asymetria alfa) i arousal (β/α) liczone na okienkach 2 s. Wynik 91–97% jest niewiarygodny (jeden podział, brak CV). |
| `brewe-2024` | Brewe i in. 2024, *J Autism Dev Disord*: BCI z feedbackiem | Regresja logistyczna per osoba, kalibracja do progu trafności i okno 3 s. Feedback był dla uczestników **niezrozumiały**, a awarie czepka kosztowały uczestników. |
| `sensors-21-03414` | Galvão i in. 2021, *Sensors*: regresja valence/arousal | Ciągły odczyt (regresja) i okna 4 s. Wynik 1-NN z PCC ≈ 0,8 to typowy efekt wycieku danych. |
| `Praca_magisterska-2` | Chałupnik 2020, PWr: NeuroSky, 1 kanał | Wynik 99,8% to podręcznikowy przykład wycieku danych. Do samoregulacji wygładzać predykcje z 3–10 s. Model z krótkich epok bodźcowych może się nie przenieść na podtrzymywany stan. |

**Wspólny wniosek:** każda praca z trafnością powyżej 85% miała wyciek danych: losowy podział na nakładających się oknach albo jedno nagranie na klasę. Rzetelne badania subject-dependent z obrazami lub filmami raportują **ok. 55–65%** w zadaniu binarnym (Chen, Lim, Fiorini, Garg). Takiego poziomu powinniśmy się spodziewać.

## 2. Krytyczna ocena aktualnego stanu

### 2.1 Co jest zrobione dobrze (zgodnie z literaturą)

- **Model per uczestnik** i **test-before-train**: każda predykcja jest robiona przed uczeniem na danej próbie. To dokładnie ten rodzaj walidacji, którego brak zawyżał wyniki w Chałupnik, Galvão, Yuvaraj i Castiblanco.
- **Podział 80/20 według obrazów** (stratyfikowany, z seedem). Żaden obraz walidacyjny nie pojawia się w treningu, a okna z jednej próby nigdy nie trafiają do obu zbiorów, czego wymagają Lim i Chałupnik.
- **Walidacja raportuje balanced accuracy i baseline klasy większościowej.** Jest to konieczne, bo valence w części walidacyjnej ma 29 obrazów pozytywnych na 15 negatywnych (66% za samo „zawsze positive”).
- **Preprocessing 1–48 Hz + notch 50 Hz** pokrywa się z Fiorini i in. (2024).
- **Cechy: log band power + Higuchi FD.** Moc pasma to w praktyce differential entropy, najczęstsza cecha w przeglądzie Lim. FD była najstabilniejszym zestawem w Yuvaraj.
- **Regresja logistyczna / SVM:** wybór klasyfikatora w literaturze nie ma dużego znaczenia (Chen). Regresja logistyczna sprawdziła się w BCI z feedbackiem (Brewe).
- **Symulator EEG z kontrolą 0 µV** sprawdza, czy w pipeline nie ma wycieku etykiet. Przy 0 µV trafność zostaje na poziomie przypadku.
- **Dużo zapisanych metadanych** (config, podział, predykcje każdej próby, surowe epoki) umożliwia późniejszą analizę offline.

### 2.2 Problemy krytyczne (ryzyko, że wynik nic nie znaczy)

1. **Valence = kategoria obrazu, czyli confound wizualny (Lim).** Etykieta valence pokrywa się w 100% z folderem joy/sadness. Klasyfikator może rozróżniać cechy wizualne (kolorystykę, jasność, twarze, zwierzęta) zamiast stanu emocjonalnego. Trafność powyżej przypadku **nie dowodzi** dekodowania emocji.
2. **Etykiety pochodzą z ocen tłumu, nie od uczestnika (Lim, Fiorini, Garg).** U Fiorini nawet przy skrajnych obrazach uczestnicy często oceniali je jako 3–5. U Garg różnica między normą a samooceną to MSE ok. 1,3–1,4. Uczymy się więc „średniego Amerykanina z bazy”, a nie reakcji badanej osoby.
3. **Arousal jest słabo zdefiniowany.** `aroused_mean` mieści się w wąskim zakresie 1,5–3,9 i dzielimy go na medianie 2,35, więc wiele obrazów leży tuż przy granicy klas. Arousal jest też skorelowany z valence (joy jest średnio bardziej pobudzające), więc obie trafności nie są niezależne. Mediana jest liczona na całym zbiorze (train + walidacja). To tylko oceny, nie EEG, więc nie ma wycieku EEG, ale Chen zaleca liczyć próg tylko na treningu.
4. **Cel 2: przesunięcie rozkładu danych (Chałupnik, Galvão, Castiblanco, Brewe).** Model uczy się na **wywołanej odpowiedzi** w oknie 700 ms od onsetu obrazu. W sesji testowej dekodujemy **podtrzymywany, samodzielnie wywołany stan** z przesuwanego okna o tej samej długości, bez związku z onsetem. Cechy ERP i krótkie okno FFT praktycznie nie przenoszą się na taki stan. Odczyt na płaszczyźnie może być w dużej mierze szumem. **To największe ryzyko projektu.**
5. **Wartość „EEG” na płaszczyźnie to pewność klasyfikatora (2·P−1), a nie skalibrowana ocena.** Nie należy jej interpretować jako „uczestnik ma valence 0,6”. Cel (target) jest w innej skali: znormalizowana ocena obrazu. Odległość między nimi jest umowna, a próg „✓ on target” = 0,35 jest arbitralny.
6. **Mała próba walidacyjna.** 44 obrazy po jednym razie dają 95% przedział ufności rzędu ±15 punktów procentowych. Bez testu permutacyjnego nie da się odróżnić 60% od przypadku (Chen, Lim, Garg).

### 2.3 Problemy istotne (obniżają jakość sygnału)

7. **Okno 700 ms jest krótkie dla cech spektralnych:** rozdzielczość ok. 1,4 Hz. Wszystkie prace spektralne używały ≥ 1–4 s, a przegląd Lim wskazuje, że 700 ms niesie głównie informację ERP. Domyślnie **nie mamy cech ERP**, choć rodzina `erp` istnieje w configu. Fiorini wskazuje LPP 400–500 ms nad kanałami centralnymi.
8. **Brak theta (4–8 Hz).** Castiblanco: najsilniejsze pasmo. Chen: parietalna theta najlepsza dla arousal. Lim: rekomenduje wszystkie pasma.
9. **Gamma 30–45 Hz przy suchych elektrodach to w dużej mierze EMG** (ostrzegają Garg, Lim, Chen i Galvão). Ranga gammy w literaturze może wynikać z napięcia mięśni twarzy, co przy emocjach jest szczególnie prawdopodobne.
10. **Prymitywne odrzucanie artefaktów.** Odrzucamy tylko przy międzyszczytowej powyżej 500 µV albo poniżej 0,05 µV. Yuvaraj odrzuca przy ±100 µV, a Fiorini używa Autoreject (ok. 5% epok). Brak CAR (stosują go Garg, Yuvaraj i Fiorini). Mrugnięcia nie są usuwane, co przy Fz jest istotne.
11. **Za dużo cech na tak mało prób.** 32 cechy przy ok. 170 próbach treningowych, a Garg najlepsze wyniki miał z 1–2 cech. Nie ma selekcji cech ani strojenia regularyzacji (C).
12. **Montaż.** Brak elektrod skroniowych i czołowo-bocznych, najważniejszych u Garg i Lim. Nie da się policzyć klasycznej frontalnej asymetrii alfa (F3/F4), na której opierają się indeksy valence w Castiblanco i Galvão. Zostają asymetrie C4−C3 i P4−P3.
13. **Brak niezależnej weryfikacji timingu.** Onset jest liczony programowo; README to opisuje, ale do tej pory nie zrobiono pomiaru fotodiodą.

### 2.4 Ocena ogólna

Infrastruktura jest w dobrym stanie: walidacja bez wycieku, sesje, zapis, symulacja i konfigurowalność. **Słabością jest projekt eksperymentu, a nie kod:**
- etykiety (oceny tłumu, confound z kategorią obrazu),
- definicja arousal,
- przeniesienie modelu z odpowiedzi na bodziec na stan podtrzymywany w celu 2.

Realistyczny wynik celu 1 to **55–65% balanced accuracy**. Wynik celu 2 bez dodatkowej kalibracji należy traktować jako eksploracyjny.

## 3. Lista optymalizacji

Priorytety: **P1** = przed zbieraniem danych uczestników, **P2** = warto zrobić w trakcie hackathonu, **P3** = jeśli zostanie czas. „Koszt” to szacowany nakład pracy.

### P1: wiarygodność wyniku

| # | Zmiana | Uzasadnienie (źródło) | Koszt | Jak w naszym kodzie |
|---|---|---|---|---|
| 1 | **Test permutacyjny** w walidacji (np. 1000 permutacji etykiet, p-value) | Chen, Lim, Garg: odróżnić wynik od przypadku | mały | funkcja offline na zapisanym `validation_*.json` / `.epochs.npz` |
| 2 | **Samoocena uczestnika (SAM 1–9 valence/arousal)**, np. po ~20% prób albo dla wszystkich obrazów po sesji | Lim, Fiorini, Garg, Brewe: etykiety per osoba są lepsze | średni | nowy krok GUI; kolumna w logu; target z `columns` |
| 3 | **Kontrola confoundu wizualnego:** klasyfikator na cechach obrazu (jasność, kolor, kontrast). Jeśli przewiduje valence tak dobrze jak EEG, wynik EEG jest podejrzany. Dobrać obrazy pozytywne i negatywne o podobnej niskopoziomowej statystyce. | Lim | mały/średni | skrypt offline na `data/training/*` |
| 4 | **Włączyć cechy ERP (LPP)**: okna 100–200, 250–400 i 400–700 ms na Cz, Pz, P3, P4 | Fiorini (LPP 400–500 ms, klaster centralny), Lim | mały | `eeg.features += ["erp"]`, dopasować `erp_windows_ms` i `regions` |
| 5 | **Dodać theta 4–8 Hz** | Castiblanco, Chen, Lim | bardzo mały | `bands_hz["theta"] = [4, 8]` |
| 6 | **Ablacja pasm, zwłaszcza gammy**, i porównanie wyników z i bez niej. Kontrola EMG: moc 50–100 Hz lub test z zaciśnięciem zębów. | Garg, Lim, Chen, Galvão | mały | offline na `.epochs.npz` |
| 7 | **Kalibracja dla celu 2:** krótki blok, w którym uczestnik przez 20–30 s świadomie wywołuje stan (wspomnienie, wyobrażenie) przy obrazach z treningu. Uczyć osobny model „stanu” na przesuwanych oknach 2–4 s. | Chałupnik, Galvão (4 s), Castiblanco (2 s), Brewe (3 s) | średni | nowy typ sesji: okna przesuwne zamiast epok od onsetu |
| 8 | **Dłuższe okno i silniejsze wygładzanie w teście:** dekodować z 2–4 s i uśredniać 3–10 s | Chałupnik, Galvão, Fiorini | mały | `feedback.smoothing` (np. 0,1) plus okno dłuższe od epoki treningowej po wdrożeniu #7 |

### P2: jakość sygnału i modelu

| # | Zmiana | Uzasadnienie (źródło) | Koszt |
|---|---|---|---|
| 9 | **Odrzucanie epok przy ±100–150 µV** (przy suchych elektrodach może być potrzebny wyższy próg) i raport odsetka odrzuceń | Yuvaraj, Fiorini (Autoreject ok. 5%) | mały |
| 10 | **Common average reference (CAR)** jako opcja w configu. Przy 8 kanałach ostrożnie: porównać z referencją sprzętową. | Garg, Yuvaraj, Fiorini | mały |
| 11 | **Selekcja cech i strojenie C** w zagnieżdżonej CV, tylko na danych treningowych (np. `SelectKBest` w pipeline) | Garg (najlepsze były 1–2 cechy), Lim | mały |
| 12 | **Welch lub multitaper** zamiast pojedynczego FFT z oknem Hanninga dla krótkich okien | Chen | mały |
| 13 | **Asymetria alfa C4−C3 i P4−P3** (rodzina `asymmetry`) jako substytut F4−F3, oraz indeks β/α dla arousal | Castiblanco, Galvão, Lim | bardzo mały |
| 14 | **Hjorth (mobility, complexity), Katz/Petrosian FD, wariancja** | Yuvaraj, Garg, Galvão | mały |
| 15 | **`class_weight="balanced"`** w klasyfikatorach. Trening jest zbalansowany po folderach, ale arousal i walidacja nie są. | Yuvaraj | bardzo mały |
| 16 | **Wyrzucenie środkowego tercyla arousal** (obrazy przy medianie) albo trzy klasy z pominięciem środka | Fiorini (polarised vs neutral) | bardzo mały |
| 17 | **Próg mediany arousal liczony tylko na części treningowej** | Chen | bardzo mały |
| 18 | **Regresja zamiast klasyfikacji binarnej** dla płaszczyzny V-A, np. ridge na ciągłych `valence` i `arousal`. Daje ciągłą wartość w tej samej skali co cel. | Garg, Galvão | średni |
| 19 | **Baseline −500…−200 ms** zamiast −200…0 ms, żeby uniknąć aktywności antycypacyjnej. Wymaga dłuższej epoki w lewo. | Fiorini | mały |

### P3: dodatkowe

| # | Zmiana | Uzasadnienie (źródło) | Koszt |
|---|---|---|---|
| 20 | Random Forest / gradient boosting w rejestrze klasyfikatorów (jedna linia `register_classifier`) | Yuvaraj, Garg, Castiblanco | bardzo mały |
| 21 | Cleanlab `find_label_issues`, czyli wykrywanie podejrzanych prób i etykiet | Fiorini | mały |
| 22 | Nagranie 1–2 min spoczynku z oczami **otwartymi** jako referencja indywidualna | Castiblanco (zamknięte oczy zawyżają alfę) | mały |
| 23 | Reguła zatrzymania kalibracji: trenować, aż kroczące balanced accuracy przekroczy próg (realnie ok. 60–65%, nie 85%) | Brewe | mały |
| 24 | Czytelniejszy feedback: jedna duża kropka, kolor zależny od odległości, dźwięk przy celu, sesja próbna przed pomiarem | Brewe (feedback niezrozumiały dla uczestników) | mały |
| 25 | Logistyka: zapasowe elektrody i żel, próba sprzętu przed uczestnikiem | Brewe (awarie czepka kosztowały uczestników) | – |
| 26 | Pomiar timingu fotodiodą lub triggerem | standard ERP; ważne przy cechach LPP | średni |

## 4. Proponowana kolejność działań

1. **Dziś, bez uczestnika:** #5 (theta), #4 (ERP), #13 (asymetria), #15 (class_weight), #9 (próg ±150 µV). To głównie zmiany w configu. Potem symulacja, żeby sprawdzić, czy pipeline działa.
2. **Pierwszy uczestnik (pilot):** trening (80%), walidacja (20%), zapis epok. Offline: #1 (permutacje), #6 (ablacja gammy), #3 (confound wizualny), #11 (selekcja cech). Na tej podstawie decyzja o ostatecznym zestawie cech.
3. **Przed sesją testową:** #2 (samoocena SAM), #7 (kalibracja stanu na oknach 2–4 s), #8 (wygładzanie), #24 (czytelny feedback). Bez #7 wynik celu 2 trzeba raportować jako eksploracyjny.
4. **Raport:** balanced accuracy z przedziałem ufności i p-value z permutacji, baseline klasy większościowej i jawne zastrzeżenie o confoundzie kategorii obrazu.

## 5. Uwagi do zaimplementowanych sesji walidacyjnej i testowej

- **Walidacja** pokazuje każdy z 44 obrazów raz, a modele są zamrożone. Na płaszczyźnie: ◯ to cel (znormalizowana ocena obrazu), ● to odczyt EEG po zakończeniu epoki, w podpisie pole „correct: …”. Tabela pokazuje accuracy, balanced accuracy i baseline większościowy.
- **Test** dekoduje co `feedback.update_s` (0,25 s) z ostatniego okna o długości epoki treningowej, z wygładzaniem wykładniczym (`feedback.smoothing`). Zapisuje cały przebieg (`live_decoding`) oraz średnią i końcową odległość od celu w każdej próbie.
- **Okno dekodowania w teście jest takie samo jak w treningu** (zgodność cech), co jest jednocześnie źródłem problemu z pkt 2.2.4. Po wdrożeniu #7 należy dekodować modelem „stanu” z dłuższego okna.
- **Brak obrazów testowych:** aplikacja proponuje użycie obrazów walidacyjnych. Pozwala to sprawdzić przebieg sesji, ale nie daje wyniku naukowego.
