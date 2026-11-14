# Brewe i in. (2024/2025) – Podsumowanie

## 1. Metadane

- **Tytuł:** A Randomized Trial Utilizing EEG Brain Computer Interface to Improve Facial Emotion Recognition in Autistic Adults
- **Autorzy:** Alexis M. Brewe, Ligia Antezana, Corinne N. Carlton, Denis Gracanin, John A. Richey, Inyoung Kim, Susan W. White
- **Rok:** opublikowano online 28.06.2024; wydanie: Journal of Autism and Developmental Disorders (2025) 55:3217–3230, DOI 10.1007/s10803-024-06436-w
- **Typ:** artykuł oryginalny – pilotażowe randomizowane badanie kontrolowane (RCT), rejestracja ClinicalTrials.gov NCT03376373; finansowanie NIMH R33MH100268-03.

## 2. Streszczenie w 3–5 zdaniach

Autorzy zbudowali „FER Assistant” – interfejs mózg–komputer (BCI) w rzeczywistości mieszanej (HoloLens), który trenuje rozpoznawanie emocji z twarzy (FER) u autystycznych dorosłych za pomocą neurofeedbacku EEG. Najpierw dla każdego uczestnika uczony jest indywidualny klasyfikator (regresja logistyczna, MVPA) na jego własnym EEG z momentu, w którym sam zgłasza rozpoznanie emocji przyciskiem; potem w 10 sesjach awatary pokazują emocje o stopniowo rosnącej intensywności, a dekoder EEG decyduje, czy emocja została „rozpoznana”. 27 mężczyzn z ASD przydzielono losowo do grupy FER Assistant (n=17) lub listy oczekujących (n=10). Akceptowalność była częściowa (retencja 70,59% zamiast założonych ≥80%), a na poziomie grupowym przynależność do grupy interwencyjnej przewidywała wyższy wynik RMET na końcu, choć na poziomie indywidualnym nikt w grupie FER Assistant nie wykazał wiarygodnej zmiany.

## 3. Aspekty implementacyjne

**Uczestnicy i projekt badania**
- 27 autystycznych mężczyzn, wiek 16–29 lat (M=21,12), IQ M=105,78, 85,20% białych; diagnoza potwierdzona ADOS-2, IQ ≥80 (WASI-II).
- Randomizacja prosta (funkcja `sample(1:n)` w R): FER Assistant n=17, lista oczekujących n=10. Ze 40 skierowanych osób 29 przeszło ocenę wstępną, 27 włączono.
- Badanie nie było projektowane jako pełny test skuteczności (mechanizm R21/R33); planowano do n=30.

**Sprzęt**
- EEG: **g.tec g.Nautilus, 32 kanały, bezprzewodowy** (komercyjny czepek). Nie podano: które kanały użyto, częstotliwości próbkowania, referencji, typu elektrod.
- Headset MR: Microsoft HoloLens; opaska Empatica E4 (w tekście też „E3”) – akcelerometr i tętno zbierane, ale **nie raportowane**.
- EEG przesyłane bezprzewodowo do komputera z klasyfikatorem; wynik wysyłany do HoloLens (rys. 2: Bluetooth, router, Wi-Fi).

**Faza kalibracji („training period”)**
- Uczestnik naciska przycisk, gdy subiektywnie rozpozna emocję na twarzy; algorytm zapisuje **3000 ms EEG bezpośrednio przed naciśnięciem**.
- Brak ustalonej liczby prób – trening kończono, gdy model osiągał **≥85% trafności w przewidywaniu naciśnięcia przycisku** danego uczestnika.
- Podejście MVPA „agnostyczne” wobec klasycznych cech EEG – szuka wzorców czasowych przewidujących poprawne rozpoznanie. **Nie podano** konkretnych cech, preprocessingu, filtrów, obsługi artefaktów, schematu walidacji ani sposobu liczenia 85% (ani klasy negatywnej).

**Model**
- Klasyfikator: **regresja logistyczna**; granica decyzyjna = hiperpłaszczyzna z wyuczonych współczynników; wynik jako skalar („scalar decision”) wyświetlany w MR.
- Dwie klasy: „zgodne z rozpoznaniem” vs „niezgodne z rozpoznaniem”.

**Protokół interwencji**
- 10 sesji × 30 min (realnie: średnio 28,14 min, zakres 13–75 min; 10 sesji w średnio 9 tygodni, zakres 6–16).
- Każda sesja: 2 scenariusze społeczne (np. restauracja, przystanek, sklep), w każdym 6 awatarów (3 M, 3 K), 6 emocji podstawowych (radość, smutek, złość, wstręt, zaskoczenie, strach). Tylko bodźce wizualne, bez dźwięku.
- Intensywność ekspresji rośnie od 0% do 100%; ok. 30 s interakcji z każdym awatarem.
- Pętla zamknięta: jeśli cechy EEG = „rozpoznanie”, próba się kończy; jeśli nie – intensywność rośnie dalej aż do rozpoznania wg EEG lub 100%. Przy błędzie tekstowa informacja o złej odpowiedzi.
- Feedback: pasek reprezentujący poziom aktywności neuronalnej, zmieniający kolor na zielony przy wzroście aktywności.

**Miary i statystyka**
- Główny wynik: RMET (36 zdjęć oczu, 4 opcje, α=0,73), na początku i na końcu. SRS-2 (cechy autystyczne), WASI-II (IQ), PSS (satysfakcja, skala 1–10).
- Analizy w IBM SPSS 28; Reliable Change Index (próg 1,96) + regresja hierarchiczna (krok 1: bazowy RMET + IQ; krok 2: grupa).

**Wyniki (liczby dokładnie z pracy)**
- Retencja: 12/17 ukończyło (70,59%); 2 osoby odpadły po **awarii czepka EEG**.
- PSS: pomocność M=6,50 (SD=2,75), przyjemność M=7,50 (SD=2,39).
- RCI: brak wiarygodnej poprawy/pogorszenia w grupie FER Assistant; 2 osoby z listy oczekujących – wiarygodny spadek.
- Regresja: ΔR²=0,15, ΔF(1,17)=7,43, p=0,014; pełny model R²=0,61, F(3,17)=11,41, p<0,001; bazowy RMET β=0,47 (p=0,005), IQ β=0,44 (p=0,008), grupa β=0,39 (p=0,014).
- RMET (tab. 1): FER Assistant 26,63 → 26,64; lista oczekujących 25,40 → 23,60.
- Bazowy RMET vs SRS-2: r(26)=−0,40, p=0,041.
- **Nie raportowano** trafności dekodera online, liczby prób kalibracyjnych ani przebiegu feedbacku w sesjach.

## 4. Co jest istotne dla naszego projektu

**Cel 1 (klasyfikator indywidualny)**
- Potwierdza wybór: **klasyfikator specyficzny dla uczestnika + regresja logistyczna** jako prosty, szybki model online. Nasza LR/SVM na 32 cechach jest w tym samym duchu.
- **Kryterium zakończenia kalibracji** (trenuj, aż trafność ≥ próg, np. 85%) – możemy przenieść na naszą rolling accuracy z 20 prób: kontynuować zbieranie danych treningowych, aż rolling/CV accuracy przekroczy ustalony próg (realistycznie niższy niż 85% przy 8 kanałach i walencji/pobudzeniu), zamiast sztywnych 80/20.
- Etykiety oparte na **subiektywnej odpowiedzi uczestnika** (przycisk), a nie normach bodźca. Dla nas: rozważyć krótką ocenę obrazów przez uczestnika (np. SAM) i porównać z etykietami z ratingów tłumu – szczególnie dla pobudzenia (wąski zakres 1,5–3,9).
- Okno 3000 ms przed reakcją jest dużo dłuższe niż nasze 700 ms – sugeruje, że przy dekodowaniu *stanu* warto testować dłuższe okna.

**Cel 2 (neurofeedback / samoregulacja)**
- Skalarny wynik dekodera (np. `predict_proba`) przekładany na **pasek** – dokładnie ten wzorzec chcemy w GUI. Ale uczestnicy skarżyli się na **niejasny feedback** („co jest poprawne?”); autorzy proponują wyraźne sygnały sukcesu (ptaszek, gwiazdka). W naszym GUI: pokazywać cel (wartość obrazu), aktualny stan oraz jednoznaczny sygnał „trafiono”.
- Pętla adaptacyjna (zwiększanie intensywności aż do sukcesu) – pomysł na Cel 2: utrzymywać trial, aż dekodowany stan osiągnie cel lub upłynie limit czasu.
- **Sesja próbna** (mock session) dla oswojenia ze sprzętem i stopniowe wprowadzanie – warto zrobić krótką rozgrzewkę przed sesją testową.
- **Awaria sprzętu = utrata uczestników** – mieć zapasowe elektrody/żel, ładowarkę Unicorna, procedurę szybkiego sprawdzenia impedancji/jakości sygnału.

## 5. Ograniczenia / zastrzeżenia

- Praca kliniczna, nie metodologiczna: **brak szczegółów przetwarzania EEG** (kanały, fs, filtry, artefakty, cechy, walidacja) – nie da się odtworzyć pipeline’u ani porównać trafności.
- Dekodowane jest „rozpoznanie emocji na twarzy”, **nie walencja/pobudzenie** własnego stanu – inny konstrukt niż nasz.
- 32-kanałowy g.Nautilus vs nasze 8 kanałów Unicorn – kryterium 85% może być nieosiągalne.
- Próba bardzo mała (n=27, 12 ukończyło), tylko mężczyźni z ASD; RCI nie wykazał indywidualnej poprawy; efekt grupowy może wynikać z różnic między grupami (lista oczekujących młodsza, mniej cech autystycznych).
- Brak danych o tym, czy neurofeedback faktycznie zmieniał aktywność EEG w trakcie sesji – nie dowodzi skuteczności samoregulacji.
- Kryterium 85% liczone na danych treningowych może być przeszacowane (brak informacji o walidacji/wycieku).
