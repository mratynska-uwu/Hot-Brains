# Comprehensive Analysis of Feature Extraction Methods for Emotion Recognition from Multichannel EEG Recordings — podsumowanie

## 1. Metadane
- **Tytuł:** Comprehensive Analysis of Feature Extraction Methods for Emotion Recognition from Multichannel EEG Recordings
- **Autorzy:** Rajamanickam Yuvaraj, Prasanth Thagavel, John Thomas, Jack Fogarty, Farhan Ali (NIE/NTU Singapur; Montreal Neurological Institute, McGill)
- **Rok:** 2023 (opublikowano 12.01.2023)
- **Miejsce publikacji:** Sensors 2023, 23, 915 (MDPI). DOI: 10.3390/s23020915
- **Typ:** artykuł badawczy, porównanie zestawów cech i klasyfikatorów na 5 publicznych zbiorach danych (19 stron)

## 2. Streszczenie
Autorzy porównują pięć zestawów cech EEG (statystyczne, falkowe, wymiar fraktalny FD, parametry Hjortha, spektra wyższych rzędów HOS) i ich połączenie w binarnej klasyfikacji high/low valence i arousal. Porównanie obejmuje dwa klasyfikatory (SVM z jądrem RBF i CART) na pięciu publicznych zbiorach: MAHNOB-HCI, DEAP, SEED, AMIGOS i DREAMER. Najlepszy okazał się zestaw **FD (Katz, Petrosian, Higuchi) + CART**: średnio 85,06% dla valence i 84,55% dla arousal. FD dawał też najmniejsze odchylenie standardowe między uczestnikami, a różnica względem pozostałych zestawów była istotna statystycznie. Autorzy rekomendują FD jako tanie, stabilne cechy do systemu online.

## 3. Aspekty implementacyjne
- **Zbiory danych (tab. 1):**

  | Zbiór | Osoby | Próby | Kanały / urządzenie / Fs | Skala (próg) |
  |---|---|---|---|---|
  | MAHNOB-HCI | 27 | 20 klipów, 34,9–117 s | 32 / BioSemi / 256 Hz | 1–9 (4,5) |
  | DEAP | 32 | 40 × 60 s | 32 / 512 Hz | 1–9 (4,5) |
  | SEED | 15 | 10 prób, tylko pozytywne/negatywne (1. sesja) | 62 / ESI NeuroScan / 1000 Hz | – (tylko valence) |
  | AMIGOS | 40 (po wykluczeniach 28) | 16 klipów | 14 / Emotiv EPOC / 128 Hz | 1–9 (4,5) |
  | DREAMER | 23 | 18 klipów, 65–393 s | 14 / Emotiv EPOC / 128 Hz | 1–5 (2,5) |

  Uczestnicy łącznie: średni wiek 27,4 (SD 9,4), 55% mężczyzn.
- **Etykiety:** binaryzacja względem **środka skali** (np. DREAMER: < 2,5 low, ≥ 2,5 high). Z AMIGOS i SEED użyto tylko pierwszej sesji.
- **Liczba prób:** średnio 569,6 na zbiór (SD 423,4): MAHNOB 540, DEAP 1280, SEED 150, AMIGOS 464, DREAMER 414.
- **Preprocessing (Python 3.7.1 + MATLAB R2020b):**
  - notch 50/60 Hz,
  - **górnoprzepustowy 1 Hz, Butterworth 4. rzędu**,
  - downsampling do **128 Hz**,
  - **re-referencja do średniej wspólnej (CAR)**,
  - segmentacja na **2-sekundowe epoki bez nakładania**,
  - automatyczne odrzucanie artefaktów: epoka odrzucana, jeśli **|amplituda| > ±100 µV**.
  - Wynik: średnio 1046 (SD 411) epok/osobę dla valence i 1036 (SD 438) dla arousal.
  - Brak dolnoprzepustowego/pasmowego filtra poza notchem i HP 1 Hz.
- **Cechy (z każdego kanału i każdej epoki):**
  - **Statystyczne (9):** średnia, mediana, SD, skośność, kurtoza, średnia |1. różnicy| δ, średnia |2. różnicy| γ, znormalizowane δ/σ i γ/σ.
  - **Falkowe (24):** CWT z falką Morleta. Przy 128 Hz jest 18 skal, z czego wzięto 12 pierwszych (> 1,25 Hz; środki skal 61,115; 43,59; 31,09; 22,17; 15,81; 11,28; 8,04; 5,74; 4,09; 2,92; 2,08; 1,48 Hz). Dla każdej skali średnia i SD z |współczynników|.
  - **FD (3):** Katz KFD = log(N)/(log(N)+log(d/L)), Petrosian PFD = log(m)/(log(m)+log(m/(m+0,4·N_δ))), Higuchi HFD (krzywa L(k) ∝ k^(−FD)). **Parametr kmax dla HFD nie jest podany.**
  - **Hjorth (2):** mobilność σ_d/σ_x i złożoność (σ_dd/σ_d)/(σ_d/σ_x). Brak „activity”.
  - **HOS (4):** bispektrum: średnia magnituda, H1 (suma log |Bis|), H2 (suma log elementów diagonalnych), H3 (moment 1. rzędu diagonali).
  - Testowano też 10 zestawów (m.in. moc widmowa, entropia, nieliniowe, konektywność, grafowe), ale **raportowano tylko 5 najlepszych**. Moc widmowa nie znalazła się w top-5, a jej wyników nie podano.
- **Klasyfikatory:**
  - Testowano CART, SVM z RBF (GSVM), RF i KNN, raportowano dwa najlepsze (CART, GSVM).
  - Hiperparametry dobierano **optymalizacją bayesowską w każdym wewnętrznym foldzie**. GSVM: box constraint i kernel scale. CART: liczba cykli uczenia, learning rate, minimalny rozmiar liścia. Te parametry sugerują, że „CART” to w praktyce **zespół drzew z boostingiem**.
  - Użyto **random under-sampling boosting** (RUSBoost) na niezbalansowane klasy i standaryzacji predyktorów.
- **Walidacja:** **subject-dependent, 4-fold CV w obrębie danych każdej osoby**: 3 foldy treningowe, 1 testowy, średnia z 4 foldów, potem średnia i SD po osobach. **Nie podano, czy foldy dzielą po próbach/klipach, czy po epokach.** Przy 2-s epokach z tych samych klipów to istotne ryzyko przecieku. Statystyka: dwustronny sparowany t-test, delta Cliffa, korekta Holma–Bonferroniego, p < 0,05.
- **Wyniki (tab. 3–4, średnia ± SD):**
  - Valence: **FD-CART 85,06 ± 0,066%** (DEAP 78,18; DREAMER 87,59; MAHNOB 83,98; AMIGOS 79,07; SEED 96,50). Combined-ALL CART 84,49%, wavelet CART 84,76%, statystyczne CART 84,03%, Hjorth GSVM 80,43%, HOS GSVM 81,11%.
  - Arousal: **FD-CART 84,55 ± 0,045%** (DEAP 79,90; DREAMER 91,60; MAHNOB 85,58; AMIGOS 81,11). Combined-ALL CART 83,76%, wavelet CART 83,94%, statystyczne CART 83,45%.
  - FD-GSVM: valence 82,75%, arousal 82,09%. CART > GSVM we wszystkich zbiorach.
  - Tab. 5, FD vs inne: p < 0,05 we wszystkich porównaniach, ale delta Cliffa vs wavelet/statystyczne/Combined jest **mała (0,02–0,06)**. Duża jest tylko vs Hjorth/HOS (0,26–0,42).
  - Niespójność: w konkluzji podano 86,79% dla valence, a w abstrakcie i tabeli 85,06%.
- **Topografia:** z-scorowane FD na 14 wspólnych kanałach. Wyższa walencja ~ niższa złożoność (KFD, PFD) czołowo, zwłaszcza nad lewą półkulą. Wyższe pobudzenie ~ niższa złożoność lewostronnie (KFD, PFD), przy przeciwnym wzorcu HFD. Autorzy przyznają, że wyniki są niespójne między zbiorami i należy je traktować „tentatywnie”.

## 4. Co jest istotne dla naszego projektu
- **Potwierdza nasz wybór HFD:** wymiar fraktalny to najstabilniejszy zestaw cech w pięciu zbiorach. Tanio rozszerzymy nasze 8 cech HFD o **KFD i PFD** (+16 cech, liczone w mikrosekundach, nadają się do online). Pakiety: `antropy` (katz_fd, petrosian_fd, higuchi_fd) albo własna implementacja.
- **Długość okna:** autorzy liczą FD na epokach 2 s (256 próbek przy 128 Hz). Nasze okno 700 ms przy 250 Hz daje ~175 próbek. HFD z kmax = 10 jest na tej długości jeszcze sensowny (N/kmax ≈ 17), ale estymata będzie szumna. Warto przetestować kmax 5–8. W Celu 2 (ciągłe dekodowanie) można przejść na okna przesuwne 1–2 s.
- **Tanie cechy czasowe:** parametry Hjortha (mobilność, złożoność) i 9 cech statystycznych (w tym znormalizowane różnice) są niemal darmowe. Statystyczne + CART dały 84% valence. Warto dodać je jako kandydatów i porównać w naszym rolling accuracy.
- **Preprocessing do rozważenia:**
  - **CAR** (u nas brak informacji o referencji; przy 8 kanałach CAR jest mniej wiarygodny, ale do przetestowania),
  - **odrzucanie epok > ±100 µV** jako prosty, deterministyczny filtr artefaktów, niezależny od OSCAR, a dla suchych elektrod może wymagać wyższego progu,
  - HP 1 Hz jest zgodny z naszym pasmem 1–48 Hz.
- **Klasyfikator:** CART z boostingiem pokonał SVM RBF. Przy naszych ~170 próbkach treningowych na osobę warto przetestować **gradient boosting / RUSBoost / random forest** obok LR/SVM. Trzeba jednak pamiętać, że drzewa przy małym N łatwo przeuczają, a w online learningu boosting jest droższy (pełny retrening).
- **Niezbalansowanie (143 joy vs 74 sad):** autorzy używali undersamplingu (RUSBoost). U nas: `class_weight='balanced'` w LR/SVM lub undersampling i raportowanie balanced accuracy.
- **Cel 2:** autorzy rekomendują FD jako podstawę „online feature extraction framework”. FD liczy się szybko i nie wymaga transformacji do dziedziny częstotliwości, więc nadaje się do wyświetlania dekodowanej walencji/pobudzenia w czasie rzeczywistym.

## 5. Ograniczenia / zastrzeżenia
- **Prawdopodobny przecięk danych:** CV 4-fold na 2-s epokach w obrębie osoby bez deklaracji, że epoki z tego samego klipu są w jednym foldzie. Sąsiednie epoki z jednego klipu są silnie skorelowane, a etykieta jest wspólna dla całego klipu. Wyniki 80–97% są więc najpewniej mocno zawyżone (SEED 96–97% to sygnał ostrzegawczy). **Nie należy oczekiwać podobnych liczb u nas**, gdzie każda próba to inny obraz, a walidacja jest uczciwa.
- Bodźce to długie klipy wideo (1–6,5 min), a nie 1-sekundowe obrazy. Cechy uśredniane są po stanie trwającym minuty, nie po odpowiedzi wywołanej w 700 ms.
- Zbiory mają 14–62 kanały żelowe (Emotiv: półsuche), a my mamy 8 kanałów Unicorn. Topografie FD są niespójne i oparte na 14 kanałach.
- Brak cech mocy widmowej w raporcie, więc nie wiadomo, jak FD wypada względem naszej log-mocy pasm. Delta Cliffa FD vs falki/statystyki jest marginalna (0,02–0,06): „najlepszy” zestaw jest w praktyce porównywalny z innymi.
- Brak parametru kmax HFD, nieprecyzyjne nazwanie „CART” (faktycznie zespół z boostingiem) i niespójne liczby (85,06% vs 86,79%).
- Tylko analiza subject-dependent. LOSO autorzy zapowiadają jako przyszłą pracę.
