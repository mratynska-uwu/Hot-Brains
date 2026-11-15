# Gorzki smak porażki i słodko-słony sukces (?) — procedura i uzasadnienie

Alternatywna wersja eksperymentu: klasyfikujemy z EEG, czy osoba badana czuje smak **słodki** czy **słony**.
Wersja ze scenami/emocjami działa bez zmian. Wersję smakową uruchamia się w oknie treningu:
**Features and classifiers → Preset → „Gorzki smak porażki i słodko-słony sukces (?)”**. Okna eksperymentu
zostają zastąpione oknami eksperymentu smakowego, a połączenie z Unicornem jest przekazywane bez ponownego
łączenia przez Bluetooth. Powrót do eksperymentu ze scenami: **Experiment → Configuration → Scene emotion trainer**.

Kod: [`hotbrain/taste.py`](../hotbrain/taste.py) (logika, konfiguracja `TASTE_CONFIG`, ewaluacja),
[`hotbrain/taste_gui.py`](../hotbrain/taste_gui.py) (okna), testy w [`tests/test_taste.py`](../tests/test_taste.py).

## 1. Przebieg jednej próby (domyślne wartości)

| Czas od pojawienia się słowa | Ekran uczestnika | Co robi uczestnik | Co robi analiza |
|---|---|---|---|
| −7…−5 s → 0 s (losowo 5–7 s) | `+` | siedzi spokojnie, usta puste, szczęka rozluźniona | **−3,0…−0,5 s: baseline tej próby** |
| 0–3 s | **SŁODKIE** / **SŁONE** | kładzie bodziec smakowy na język, zamyka usta | marker na osi próbek Unicorna; nic nie jest klasyfikowane |
| 3–20 s | mała kropka `•`, taka sama dla obu smaków | trzyma smak na języku, nie połyka, patrzy na kropkę | — |
| **5–15 s** | `•` | jw. | **klasyfikowany sygnał:** okna 2 s co 1 s (9 okien), każde minus baseline |
| 20–40 s | „Przełknij i przepłucz usta wodą” | połyka, płucze usta wodą (2×) | — |

Jedna próba trwa średnio ≈ 46 s; 20 prób na smak (40 prób) ≈ 31 min. Kolejność smaków jest losowa,
dokładnie zbalansowana i bez więcej niż 2 takich samych smaków z rzędu.

### Dlaczego tak

- **Analiza dopiero od +5 s.** Pierwsze sekundy po wskazówce zawierają ruch ręki, otwieranie ust, ruchy
  języka i szczęki (artefakty mięśniowe i glossokinetyczne o amplitudzie dziesiątek–setek µV) oraz
  wzrokową odpowiedź na pojawienie się słowa. Cukier w kryształkach rozpuszcza się też wolniej niż sól, więc
  początek wrażenia smakowego jest rozmyty o kilka sekund — nie znamy momentu kontaktu bodźca z językiem
  (nie ma gustometru), więc potencjałów wywołanych smakiem (GEP, ~100–600 ms po kontakcie) nie da się tu
  wiarygodnie uśrednić. Klasyfikujemy więc **podtrzymany stan** w czasie trzymania smaku.
- **Słowo znika po 3 s i zastępuje je identyczna kropka.** Słowa „SŁODKIE” i „SŁONE” mają różną długość i
  kształt — to inny bodziec wzrokowy. Gdyby słowo zostawało na ekranie, klasyfikator mógłby nauczyć się
  „czytać” słowo z aktywności potylicznej, a nie rozpoznawać smak. Zniknięcie słowa wywołuje krótką
  odpowiedź wzrokową, dlatego program wymaga ≥ 1 s przerwy między zniknięciem słowa a początkiem analizy.
  (Ustawienie *Cue word visible* = *Tasting period* zostawia słowo na cały czas próby, zgodnie z dosłowną
  wersją procedury — wtedy nie ma przejścia ekranu, ale zostaje różnica bodźca wzrokowego.)
- **Analiza kończy się na +15 s, a próba na +20 s.** Po ~15 s narasta ślina i potrzeba przełknięcia;
  ostatnie sekundy są też najbardziej zaadaptowane (wrażenie smaku słabnie przy stałym bodźcu).
- **Płukanie 20 s i fiksacja ≥ 5 s.** Usuwa resztki smaku (przeniesienie smaku na kolejną próbę), a baseline
  następnej próby zaczyna się co najmniej 2 s po zakończeniu płukania, gdy usta są już nieruchome.
- **Maks. 2 te same smaki z rzędu.** Ogranicza kumulowaną adaptację do jednego smaku i sprawia, że wolny dryf
  sygnału w trakcie sesji nie koreluje z etykietą.

## 2. Przygotowanie bodźców (zalecenia)

Procedura działa z cukrem i solą „na sucho”, ale pomiar będzie czystszy z **roztworami**:

- **Roztwory** w temperaturze pokojowej, w identycznych, nieprzezroczystych pojemnikach/strzykawkach
  oznaczonych kodem: np. sacharoza ~0,6 M (≈ 20 g / 100 ml wody) i NaCl ~0,3 M (≈ 1,75 g / 100 ml).
  Przed sesją warto w krótkim pilotażu dobrać stężenia tak, by **intensywność** obu smaków była subiektywnie
  podobna — inaczej klasyfikator może rozpoznawać „mocniej/słabiej”, a nie „słodko/słono”.
- **Dawka 1–2 ml** pipetą/strzykawką na środek języka (albo łyżeczka). Roztwór działa od razu i nie wymaga
  rozpuszczania językiem — mniej ruchów języka = mniej artefaktów.
- Jeśli kryształki: równe, małe porcje (~0,2–0,5 g) na zwilżonej łyżeczce albo pasku bibuły; wtedy tym
  ważniejsze jest, by nie rozcierać ich językiem.
- **Woda do płukania** w temperaturze pokojowej, kubek w zasięgu ręki; pojemnik na wyplucie, jeśli
  uczestnik woli nie połykać soli.
- Bodźce przygotowane z góry w kolejności zgodnej z listą w panelu operatora (panel pokazuje **Next: …**
  przed każdą próbą i **NOW: …** w momencie wskazówki).

## 3. Instrukcja dla uczestnika

1. Gdy widzisz `+`: siedź spokojnie, usta puste, szczęka luźna, patrz na krzyżyk.
2. Gdy pojawi się słowo: spokojnie połóż smak na języku i zamknij usta (bez zaciskania zębów).
3. Gdy widzisz kropkę: patrz na kropkę, **nie ruszaj językiem, nie połykaj, nie mów**, staraj się nie
   mrugać zbyt często. Skup się na smaku.
4. Gdy pojawi się „Przełknij i przepłucz usta wodą”: przełknij (lub wypluj), przepłucz usta dwa razy.

Przed sesją: 2 próby na sucho (z wodą) dla oswojenia z rytmem.

## 4. Obsługa przez operatora

- **Diagnostyka EEG** przed startem: dobre sygnały przede wszystkim na **Fz, C3, Cz, C4, Pz** (z nich liczone
  są cechy domyślnego presetu). Poproś o zaciśnięcie zębów — zobaczysz, jak wygląda EMG; to samo dzieje się
  przy ruchach języka.
- **OSCAR** włączony, oczekiwanie 15 s przed pierwszą próbą.
- **Klawisz B / przycisk „Mark trial bad”**: rozlany bodziec, połknięcie przed czasem, kaszel, mówienie —
  próba nie jest uczona ani oceniana (można ją oznaczyć do 2 s po zakończeniu okresu smakowania).
- **Escape** zatrzymuje sesję. Próba przerwana przed końcem okna analizy jest odrzucana.
- **Blind trials** (próby ślepe): uczestnik widzi tylko „SMAKUJ”, a operator podaje smak wskazany w panelu.
  To najważniejszy test trafności (patrz §6).

### Zalecany plan sesji

1. Diagnostyka sygnału (2–3 min), 2 próby próbne.
2. **Training**: 20 prób na smak (≈ 31 min), w razie zmęczenia dwa bloki z przerwą (dwa kolejne treningi
   z tym samym ID i zapis obu).
3. **Test** (zamrożone modele, po treningu wybierany automatycznie): 5–10 prób na smak, najlepiej **blind**.

## 5. Pomiar i analiza

- **Elektrody i pasma (preset 1, 18 cech).** Fz, C3, Cz, C4, Pz · theta 4–8, alpha 8–13, niska beta 13–20 Hz ·
  log-moc w paśmie + asymetria C4−C3. Kora smakowa (przednia wyspa / wieczko czołowe) i kora oczodołowo-czołowa
  leżą głęboko; na skalpie ich aktywność najlepiej widać w odprowadzeniach czołowo-centralnych. Słodki i słony
  różnią się też walencją hedoniczną (słodki zwykle przyjemniejszy), co wiąże się z asymetrią
  czołowo-centralną (dążenie/unikanie). Pasma powyżej 20 Hz i gamma są w czasie smakowania zdominowane przez
  EMG szczęki i języka, więc nie ma ich w domyślnym presecie (są dostępne do porównania w presecie 3).
  Elektrody potyliczne są pominięte w presecie 1, by ograniczyć wpływ resztkowych różnic wzrokowych.
- **Korekta względem baseline’u próby** (ERD/ERS): od cech każdego okna odejmowane są cechy z fiksacji
  −3…−0,5 s tej samej próby. Usuwa to wolny dryf kontaktu elektrod (suche/hybrydowe elektrody, 30 min sesji)
  i różnice poziomu między próbami. Gdy baseline próby jest zaartefaktowany, używana jest mediana baseline’ów
  **wcześniejszych** prób (bez zaglądania w przyszłość).
- **Odrzucanie artefaktów okno po oknie:** amplituda > ±150 µV po filtracji 1–48 Hz + notch 50 Hz oraz
  **strażnik EMG**: RMS w paśmie 30–45 Hz > 10 µV na którymkolwiek kanale modelu. Próba wymaga ≥ 3 czystych
  okien.
- **Klasyfikator:** domyślnie **shrinkage LDA** (Ledoit–Wolf) i regresja logistyczna L2 — liniowe, dobrze
  zregularyzowane przy kilkudziesięciu próbach. Uczą się na oknach, ale **decyzja dotyczy całej próby**:
  P(słodkie) = średnia prawdopodobieństw jej okien.
- **Ocena w trakcie (prequential):** każda próba jest najpierw przewidywana przez model, który jej jeszcze nie
  widział, dopiero potem jej okna są dodawane do uczenia. Tabela i wykres pokazują trafność na poziomie prób.
- **Raport po treningu** (`training_reports/taste_*.md/.json`): walidacja krzyżowa **grupowana po próbach**
  (StratifiedGroupKFold — okna jednej próby nigdy nie są jednocześnie w zbiorze uczącym i testowym; zwykła
  walidacja po oknach dawałaby zawyżone wyniki, bo okna jednej próby są silnie skorelowane), trafność
  zbalansowana, ROC AUC i **test permutacyjny**, który tasuje etykiety całych prób.
- **Zapis:** `saved_models/taste_<ID>_<czas>.json` (manifest z kolejnością, czasami, logiem prób, cechami),
  `.joblib` dla każdego modelu i `.epochs.npz` (surowe epoki −3…+15 s, etykiety). Sesja testowa:
  `taste_test_<ID>_<czas>.json`.

## 6. Kontrole trafności i ograniczenia

- **Oczekiwanie vs smak.** W zwykłych próbach uczestnik wie, co zaraz poczuje — model może częściowo
  dekodować oczekiwanie lub znaczenie słowa, a nie samo wrażenie smakowe. **Próby ślepe** (blind) to
  rozstrzygają: jeśli model trenowany na próbach ze słowem działa także w próbach „SMAKUJ”, dekoduje smak.
- **Walencja zamiast jakości smaku.** Jeśli dla danej osoby słony jest wyraźnie mniej przyjemny, model może
  rozpoznawać przyjemność, a nie „słodkość”. Warto zebrać ocenę przyjemności i intensywności (np. 1–9) po
  sesji i podać ją przy wynikach.
- **Ruchy i mięśnie.** Jeśli jeden ze smaków wywołuje więcej ruchów języka/przełykania, różnice mogą pochodzić
  z EMG. Strażnik EMG i pasma ≤ 20 Hz to ograniczają; raport podaje liczbę odrzuconych okien na przyczynę —
  wyraźnie więcej odrzuceń w jednej klasie jest sygnałem ostrzegawczym.
- **Montaż.** Czapka Unicorna nie ma elektrod skroniowych (T7/T8), najbliższych korze smakowej, ani F3/F4 do
  klasycznej asymetrii czołowej. To ogranicza osiągalną trafność.
- **Interpretacja:** poziom przypadku to 50% trafności zbalansowanej; przy 40 próbach wynik ≈ 65% lub wyżej
  z p < 0,05 w teście permutacyjnym jest sensownym sygnałem dla jednej osoby. Wynik bliski 100% przy słowie
  widocznym przez całą próbę należy traktować podejrzliwie (możliwe dekodowanie bodźca wzrokowego).

## 7. Konfiguracja

Wszystkie wartości domyślne są w `TASTE_CONFIG` w [`hotbrain/taste.py`](../hotbrain/taste.py): teksty dla
uczestnika, czasy (`timing`), okna (`windows`), liczba prób i maks. seria (`trials`), korekta baseline’u,
progi artefaktów (`artifacts`), presety cech (`profiles`), domyślne klasyfikatory i ustawienia raportu.
Elektrody i okablowanie są brane z konfiguracji eksperymentu ze scenami (`CONFIG["eeg"]`), bo to ten sam
headset. Większość czasów i ustawień cech można też zmienić w panelu operatora przed sesją.

## Literatura

- Crouzet, S. M., Busch, N. A., & Ohla, K. (2015). Taste quality decoding parallels taste sensations.
  *Current Biology*, 25(7), 890–896.
- Wallroth, R., & Ohla, K. (2018). As soon as you taste it: evidence for sequential and parallel processing of
  gustatory information. *eNeuro*, 5(5).
- Small, D. M. (2010). Taste representation in the human insula. *Brain Structure and Function*, 214, 551–561.
- Blankertz, B., Lemm, S., Treder, M., Haufe, S., & Müller, K.-R. (2011). Single-trial analysis and
  classification of ERP components — a tutorial. *NeuroImage*, 56(2), 814–825.
