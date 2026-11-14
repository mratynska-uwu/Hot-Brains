# Podsumowanie: „Research on the emotion recognition with classifier using EEG interface data”

## 1. Metadane

- **Tytuł:** Research on the emotion recognition with classifier using EEG interface data
- **Autor:** inż. Rafał Chałupnik (promotor wymieniony w podziękowaniach: dr Kędziora)
- **Rok:** 2020
- **Typ / miejsce:** praca dyplomowa magisterska, Politechnika Wrocławska, Wydział Informatyki i Zarządzania (Wrocław). Kierunek/specjalność nieuzupełnione w szablonie strony tytułowej.
- **Objętość:** ok. 52 strony tekstu głównego + załączniki (bodźce, kod źródłowy C#), łącznie ok. 77 stron PDF.

## 2. Streszczenie

Praca sprawdza, czy jednoelektrodowy, konsumencki EEG (NeuroSky MindWave Mobile 2, elektroda nad lewą brwią) pozwala rozróżnić 4 emocje z ćwiartek modelu walencja–pobudzenie (złość, „depresja”, relaks, szczęście). Sześć osób oglądało cztery ok. 5-minutowe teledyski/klipy (jeden na emocję), a z urządzenia pobierano gotowe, producenckie moce 8 pasm EEG z częstością 1 próbki/s. Sygnał wygładzono średnią ruchomą (okno 10, krok 1), po czym AutoML z ML.NET wskazał FastTree One-vs-All jako najlepszy model; ręcznie zbudowany model osiągnął w 10-krotnej walidacji krzyżowej 99,80% trafności. Analiza ablacyjna pasm wskazała, że alfa, beta i gamma niosą znacznie więcej informacji niż delta i theta. Ze względu na sposób walidacji (losowe tasowanie silnie nakładających się próbek z tego samego ciągłego nagrania, jeden bodziec na klasę) wynik należy uznać za skrajnie zawyżony.

## 3. Aspekty implementacyjne

**Bodźce i etykiety**
- 4 klasy = 4 ćwiartki modelu 2D: złość (negatywna walencja / wysokie pobudzenie) – scena z filmu *Wanted* (2008); depresja (neg./niskie) – teledysk Marina „To be human”; relaks (poz./niskie) – montaż krajobrazów z relaksującą muzyką; szczęście (poz./wysokie) – teledysk Pharrell Williams „Happy”.
- **Jeden klip na klasę.** Etykieta = kategoria bodźca, przypisywana z nazwy pliku CSV (kod `GetEmotion(filePath)`).
- Celowo **zrezygnowano z SAM** (self-assessment manikin) – brak jakichkolwiek ocen uczestników; autor zakłada, że bodziec „gwarantuje” emocję.

**Uczestnicy i protokół**
- 6 osób (kobiety i mężczyźni), wiek 15–45 lat.
- Każdy klip ok. 5 min, potem ok. 1 min odpoczynku. Uczestnik nie był informowany, jaką emocję nagrywano. Słuchawki dla redukcji hałasu; komputer rejestrujący odwrócony od uczestnika.
- Kolejność klipów (randomizacja/kontrbalansowanie) – **nie podano**.

**Sprzęt**
- NeuroSky MindWave Mobile 2: **1 elektroda sucha nad lewą brwią (okolica Fp1)**, referencja – klips na uchu; Bluetooth → ThinkGear Connector → TCP localhost (port 13854, JSON) → własny klient C# → CSV.
- **Surowy sygnał nie był używany** (`enableRawOutput: false`). Cechami są producenckie wartości „ASIC_EEG_POWER” (jednostki względne, nieliniowo związane z V²/Hz) dla 8 pasm: delta, theta, low alpha, high alpha, low beta, high beta, low gamma, high gamma. Dodatkowo metryki eSense (Attention, Meditation, 0–100) i blink.
- Rozdzielczość czasowa: **1 próbka/s**.
- Tabela pasm w części teoretycznej: delta < 4 Hz, theta 4–7 Hz, alfa 8–15 Hz, beta 16–31 Hz, gamma > 32 Hz. Dokładne granice podpasm low/high – nie podano.

**Przetwarzanie**
- Brak filtracji częstotliwościowej, brak usuwania artefaktów, brak referencjonowania czy baseline’u po stronie autora (wszystko wewnątrz układu ASIC NeuroSky). Wyjątek: odrzucanie pakietów, gdy urządzenie jest wyłączone albo źle założone (attention = meditation = 0).
- Filtr „uwagi”: odrzucanie próbek z Attention < 50. W wersji końcowej **wyłączony** (zakomentowany w kodzie), bo nie zmieniał wyniku. Bez niego ok. 4900 próbek (z macierzy pomyłek: 4908).
- Wygładzanie: Simple Moving Average na każdym z 8 pasm, **okno 10 próbek (10 s), krok 1**, czyli kolejne próbki dzielą 9/10 danych.
- Próbki traktowane jako niezależne wektory 8 cech, bez kontekstu czasowego.

**Modele**
- ML.NET AutoML (Model Builder, scenariusz multi-class, budżet 300 s): FastTreeOva – MicroAccuracy 0,9233 / MacroAccuracy 0,9264; LightGbmMulti 0,9000 / 0,9001; LightGbmMulti 0,8185 / 0,8199; FastForestOva 0,8027 / 0,8051; FastTreeOva 0,8006 / 0,8035. Ekran ewaluacji: 14 modeli, 186,67 s, „overall accuracy 90%” (zbiór 3321 wierszy w podglądzie, prawdopodobnie wersja z filtrem Attention).
- Model końcowy: One-vs-All z `FastTreeBinaryTrainer` (MART, gradient boosting), `NumberOfTrees = 50`, `FeatureFirstUsePenalty = 0.1`, `EarlyStoppingMetric = L2Norm`. Brak skalowania cech.
- Teoria opisuje też Averaged Perceptron, FastForest, LBFGS LogReg/MaxEnt, LightGBM, Linear SVM, SDCA, Symbolic SGD, ale nie raportuje ich wyników poza tabelą AutoML.

**Walidacja**
- 10-krotna CV: losowe przetasowanie **wszystkich próbek wszystkich uczestników razem**, podział na 10 batchy, średnia z foldów. Model jest więc **wspólny dla grupy**, bez podziału po osobach, sesjach czy blokach czasowych.
- Wygładzone próbki sąsiednie są niemal identyczne i trafiają jednocześnie do treningu i testu, co oznacza **przeciek danych (leakage)**. Dodatkowo klasa = jedno ciągłe nagranie, więc model może rozpoznawać dryf sygnału / „która sesja” zamiast emocji.

**Wyniki**
- Trafność 4-klasowa: **99,80%**. Recall: złość 0,997743, depresja 0,997662, szczęście 0,999211, relaks 0,996109. Precision: 0,997743 / 0,997662 / 0,996853 / 0,999024.
- Ablacja pasm (AutoML; podpasma low/high łączone, więc 5 grup):
  - 4 z 5 pasm: 99,77% (bez theta), 99,77% (bez delta), 99,60% (bez gamma), 99,59% (bez beta), 99,38% (bez alfa).
  - 3 z 5: od 99,58% (alfa+beta+theta; alfa+beta+delta) do 97,83% (gamma+delta+theta).
  - 2 z 5: od 99,01% (alfa+beta) do 81,64% (delta+theta).
  - Pojedyncze pasmo: beta 90,04%, alfa 85,95%, gamma 84,64%, delta 56,48%, theta 48,8%.
  - Współczynnik poprawy (średnia trafność modeli z pasmem / bez pasma): alfa 1,0994, beta 1,1076, gamma 1,0917, delta 1,0231, theta 1,0109.
- Rozpoznawanie na żywo: prosty program pobiera 1 próbkę/s, wygładza ostatnie 10 próbek i podaje klasę (argmax score). Według autora **nie był testowany ani ewaluowany**.

**Oprogramowanie:** C#/.NET, ML.NET + AutoML (Model Builder), NeuroSky ThinkGear Connector, Newtonsoft.Json; pełny kod w załączniku B.

## 4. Co jest istotne dla naszego projektu

**Cel 1 (klasyfikator walencji/pobudzenia):**
- **Ostrzeżenie metodologiczne jest najcenniejszą lekcją z tej pracy.** 99,8% to skutek losowej CV na nakładających się oknach z jednego nagrania na klasę. U nas:
  - dzielimy 80/20 **po próbach/obrazach**, nigdy po oknach z tej samej próby;
  - jeśli kiedykolwiek będziemy wycinać wiele (nakładających się) okien z jednego triala, wszystkie muszą trafić do tego samego foldu;
  - najbezpieczniej robić podział chronologiczny/blokowy (np. ostatnie 20% sesji), bo nasz schemat „test-before-train” już to częściowo zapewnia.
- **Dryf jako confound:** u nich klasa = blok czasowy. U nas walencja = folder obrazu, więc jeśli obrazy „joy” i „sadness” byłyby pokazywane blokami, klasyfikator nauczy się dryfu impedancji suchych elektrod, zmęczenia lub alfy, a nie walencji. **Wymagana pełna randomizacja kolejności** (przeplot klas). Kontrola: test permutacyjny etykiet oraz porównanie z klasyfikatorem uczonym na numerze próby/czasie.
- **Pasma:** wynik wspiera nasz wybór alfa/beta/gamma i pominięcie delta/theta (pojedynczo beta 90,04% > alfa 85,95% > gamma 84,64% ≫ delta 56,48%, theta 48,8%). Uwaga: u nich elektroda czołowa (Fp1), więc „beta/gamma” w dużej mierze może być EMG mięśni twarzy (mimika przy złości i śmiechu). Na naszych elektrodach (Fz, C3, Cz, C4, P3, Pz, P4, Oz) gamma 30–45 Hz też jest podatna na EMG; warto sprawdzić, czy wynik nie zależy głównie od gammy, np. ablacją pasm w stylu ich tabel 4.4–4.7. To tania i przydatna analiza do raportu hackathonowego.
- **Realistyczne oczekiwania:** praca nie daje wiarygodnej referencji dla naszej trafności. Z 6 osób, 1 kanału i bez poprawnej walidacji nie da się wnioskować; dla binarnej walencji/pobudzenia z 8 kanałów i ~170 prób treningowych należy oczekiwać wyników znacznie bliższych 50–70% niż 99%.
- Modele drzewiaste (FastTree/LightGBM) wypadły u nich najlepiej. Przy naszych 32 cechach i małej liczbie prób można dorzucić np. `LightGBM`/`GradientBoosting` jako porównanie, ale z silną regularyzacją (płytkie drzewa); przy tak małym N regresja logistyczna/SVM pozostają rozsądnym domyślnym wyborem.

**Cel 2 (neurofeedback / samoregulacja):**
- Ich tryb na żywo to **wygładzanie średnią ruchomą 10 s z krokiem 1 s** i predykcja co sekundę. To sensowny wzorzec do wyświetlania dekodowanej walencji/pobudzenia: surowe predykcje z pojedynczych okien są zbyt zaszumione dla uczestnika. Proponujemy EMA lub średnią ruchomą 3–10 s wyjść probabilistycznych (`predict_proba`), nie etykiet 0/1.
- Ich podejście („stan” w długich oknach, bez blokowania do bodźca) jest bliższe Celowi 2 niż nasz trening na epokach 700 ms po onsecie obrazu. Samoregulacja to stan utrzymywany przez sekundy, a nie odpowiedź wywołana. Rozważyć:
  - drugi zestaw cech liczony w przesuwnych oknach 2–4 s (ten sam log band power + HFD);
  - zebranie krótkiej sesji kalibracyjnej „przywołaj emocję X przez 20–30 s” pod model dla Celu 2, bo model z epok 700 ms może nie przenosić się na samoindukowane stany (domain shift).
- Ich filtr „Attention ≥ 50” nic nie dał; nie warto szukać odpowiedników.

## 5. Ograniczenia / zastrzeżenia

- **Przeciek danych i pseudoreplikacja:** losowa 10-fold CV na próbkach wygładzonych oknem 10/krok 1, z jednego ciągłego nagrania na klasę. Wynik 99,80% nie świadczy o rozpoznawaniu emocji.
- **Jeden bodziec na klasę:** klasyfikator może rozpoznawać konkretny klip (jasność, ruch, tempo muzyki, mrugnięcia) albo pozycję w sesji/dryf, a nie emocję. Brak kontroli kolejności.
- **Brak oceny subiektywnej (SAM)**, więc nie wiadomo, czy emocje faktycznie wywołano. Wniosek autora, że emocje „są uniwersalne”, nie wynika z danych.
- **Model wspólny dla wszystkich**, bez oceny między- ani wewnątrzosobniczej. Nie wiadomo, czy uczestnicy byli przemieszani w foldach (byli, przez losowe tasowanie).
- **Elektroda Fp1:** silna kontaminacja EOG/EMG. Gotowe moce pasm NeuroSky są czarną skrzynką o niejasnych jednostkach i granicach pasm.
- Ablacja pasm była liczona w AutoML (szczegóły walidacji nie podane) na tych samych, przeciekających danych, więc ranking pasm jest jedynie poglądowy.
- Moduł na żywo nie był ewaluowany.
- **Przenoszalność na nasz układ jest niska:** inne bodźce (5-min wideo vs 1-s obrazy), inna analiza (stan vs odpowiedź na bodziec), inny montaż (1 kanał czołowy vs 8 kanałów centralno-ciemieniowo-potylicznych bez Fp/F3/F4, czyli bez klasycznej asymetrii czołowej alfy), inna częstość (1 Hz band power vs 250 Hz surowy sygnał).
