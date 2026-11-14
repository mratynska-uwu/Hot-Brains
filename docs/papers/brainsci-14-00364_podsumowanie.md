# Review of EEG Affective Recognition with a Neuroscience Perspective — podsumowanie

## 1. Metadane
- **Tytuł:** Review of EEG Affective Recognition with a Neuroscience Perspective
- **Autorzy:** Rosary Yuting Lim, Wai-Cheong Lincoln Lew (równy wkład), Kai Keng Ang (autor korespondencyjny); A*STAR Institute for Infocomm Research oraz Nanyang Technological University, Singapur
- **Rok:** 2024 (opublikowano 8.04.2024)
- **Miejsce publikacji:** Brain Sciences 2024, 14, 364 (MDPI). DOI: 10.3390/brainsci14040364
- **Typ:** przegląd systematyczny (PRISMA), 47 stron (z bibliografią)

## 2. Streszczenie
Przegląd łączy perspektywę neuronauki (gdzie i jak powstają emocje: struktury mózgu, ERP, oscylacje) z perspektywą affective computing (zbiory danych, preprocessing, cechy, modele, ewaluacja). Autorzy pokazują, że nie ma dedykowanych regionów mózgu dla poszczególnych emocji (np. ciało migdałowate koduje zarówno walencję dodatnią, jak i ujemną). Pokazują też, że ERP sygnalizują raczej „zmianę/nowość bodźca” niż konkretną emocję, a informacja o emocjach jest rozproszona w oscylacjach wielu pasm. Część obliczeniowa zestawia ~30 metod (głównie DE + modele grafowe/RNN na SEED/DEAP) i omawia przesunięcie domeny (domain shift) między osobami, wybór pasm i elektrod oraz ryzyko przecieku danych przy segmentacji prób. Konkluzja: najlepsze wyniki daje wykorzystanie wszystkich pasm w krótkich oknach (≤5 s), a kolejnym krokiem w dziedzinie ma być modelowanie regulacji emocji.

## 3. Aspekty implementacyjne
Jest to przegląd, więc nie ma jednego protokołu. Poniżej zebrano konkretne fakty i liczby z tekstu.
- **Metodologia przeglądu:** PRISMA, wyszukiwanie od 25.04.2023 (PubMed, Google Scholar). Ok. 200 000 trafień, 400 rekordów, 100 przesianych, 85 ocenionych, **77 włączonych, 66 raportowanych**. Dla części neuronaukowej: 12 artykułów o regionach mózgu, 8 o ERP, 7 o oscylacjach.
- **Zbiory danych (tab. 4):**
  - DEAP: 32 osoby, 40 klipów, 32 kanały, oryginalnie 512 Hz, wersja preprocessed 128 Hz, Biosemi ActiveTwo.
  - MAHNOB-HCI: 27 osób, 32 kanały.
  - SEED: 15 osób, 15 klipów ~4 min, 62 kanały, ESI NeuroScan.
  - DREAMER: 23–25 osób, 14 kanałów, Emotiv EPOC.
  - SEED-IV: 72 klipy, 4 emocje.
  - MPED: 23 osoby, 28 klipów, 7 emocji.
  - AMIGOS: 40 osób, 14 kanałów, 128 Hz.
  - SEED-V: 20 osób, 5 emocji.
- **Schematy walidacji:** subject-dependent (leave-K-trial-out) i subject-independent (LOSO). Autorzy ostrzegają przed **„subject-bias”**: losowe przypisanie segmentów tej samej próby do zbioru treningowego i walidacyjnego zawyża wyniki. W ewaluacji przy niezbalansowanych klasach zalecają F1 zamiast dokładności.
- **Etykiety:** etykieta przypisana do bodźca (np. przez obserwatorów lub średnią ocen) jest szumem względem stanu konkretnej osoby. Autorzy piszą, że „najlepsze etykiety nie powinny być agregowane, lecz unikalne dla każdego badanego”.
- **Artefakty i preprocessing:**
  - EOG < 4 Hz (mrugnięcia), EKG ~1,2 Hz, EMG > 30 Hz, szum sieciowy/środowiskowy > 50 Hz. Filtry pasmowe usuwają też potencjalnie użyteczną wysoką gammę (> 50–80 Hz).
  - ICA i DWT jako alternatywy.
  - Referencja: elektroda odniesienia (płatek ucha, nos, wyrostek sutkowaty) albo średnia wszystkich kanałów (CAR).
- **Cechy:**
  - Dziedzina czasu: statystyki, parametry Hjortha, HOC, ERP. ERP uznano za niepraktyczne online, bo wymagają uśredniania wielu prób.
  - Dziedzina częstotliwości: PSD przez FFT/Welch (okno Hanninga) oraz **Differential Entropy**: DE = ½·log(2πe·σ²_band) = ½·log P_band + ½·log(2πe/N) (wzór 1). DE to w praktyce log-moc pasma plus stała.
  - Czas-częstotliwość: STFT, DWT/CWT, WPT, HHT/EMD, MEMD, VMD, transformata Stockwella, VFCDM.
  - Nieliniowe: wymiar fraktalny i in. Wysoki koszt obliczeniowy i wrażliwość na parametry.
  - Asymetria: DASM (różnica DE par półkul) i RASM (iloraz).
  - Sieciowe: Pearson, PLV, Granger.
- **Pasma i lokalizacja (sekcja 4.4):**
  - Arousal: najbardziej rozróżniająca jest PSD z **niskiej alfy w kanałach potylicznych** (Soleymani).
  - Valence: **beta i gamma w kanałach skroniowych**. Beta/gamma obustronnie skroniowo są silniejsze dla emocji pozytywnych (Zheng i Lu).
  - Wyniki rosną w kierunku wyższych pasm, a **„wszystkie pasma” > pojedyncze pasmo** (rys. 4, SEED, SVM/RGNN/BiDANN/DGCNN).
  - Kanały istotne: skroniowe T7, T8, TP7, TP8; czołowe F7, F5, F3, F1, FZ, F2, F4, F6, F8; przedczołowe AF3, Fp1, FP2, FPZ, AF4.
- **Wnioski neuronaukowe (oscylacje):**
  - Alpha: hamowanie/bramkowanie, lateralizacja zależna od walencji (prawa półkula ~ walencja ujemna, lewa ~ dodatnia).
  - Beta: silniejsza dla walencji ujemnej i dla wysokiego pobudzenia.
  - Theta: silniejsza dla walencji ujemnej i wyższego pobudzenia.
  - Delta: silniejsza synchronizacja dla bodźców emocjonalnych (pobudzenie).
  - Gamma: silniejsza dla walencji ujemnej, ale prawdopodobnie odzwierciedla głównie wczesne przetwarzanie sensoryczne i uwagę.
  - Spadek mocy alfa koreluje z wysokim pobudzeniem, z równoczesnym spadkiem bety. Sama alfa wystarcza do arousal, ale nie do valence.
- **Wnioski o ERP:**
  - Twarze emocjonalne vs neutralne dają wzrost pozytywności w 100–200 ms.
  - LPP jest większy dla bodźców emocjonalnych, szczególnie negatywnych (Bondy 2017, parietalnie, obrazy statyczne).
  - P300/LPP rosną dla negatywnych obrazów nad prawą półkulą (Martini 2012).
  - **Wczesne komponenty korelują z walencją, późne z pobudzeniem.**
  - Specyficzność przestrzenna ERP znika po ~200 ms.
- **Wyniki z literatury (tab. 5, wybór):**
  - Atkinson i Campos 2016 (DEAP, mRMR + SVM RBF, subject-independent): valence 73,14%, arousal 73,06%.
  - Lew 2020 RODAN (DEAP): subject-dependent valence 62,93%, arousal 63,97%; subject-independent 56,78% / 56,60%.
  - Miranda-Correa 2021 (AMIGOS, PSD + PCA + Gaussian NB, subject-independent): valence 56,4%, arousal 57,7%.
  - Modele głębokie na SEED: ponad 90% (subject-dependent).
  - Ackermann: okno 1 s, mRMR, RF/SVM.
  - Thammasan: segmenty 1–4 s lub 5–8 s.
- **Adaptacja domeny:** kalibracja/wyrównanie rozkładów, TCA, GFK, DANN, WGANDA, FLADA, T-DANN, meta-learning (MAML), SRDA (adaptacyjna batch-norm). W dyskusji autorzy podkreślają, że **online adaptive learning** wykorzystujący „niedawną historię” mózgu jest rzetelnym sposobem na niestacjonarność.

## 4. Co jest istotne dla naszego projektu
- **Cel 1, dobór cech:**
  - Nasza log-moc pasm to w praktyce DE (różnica o stałą), więc podejście jest zgodne z literaturą.
  - Przegląd sugeruje **dodanie theta (i ewentualnie delta)**, bo theta/delta niosą informację o pobudzeniu i walencji. „Wszystkie pasma” biją pojedyncze pasma.
  - Rozważyć **cechy asymetrii** dostępne w naszym montażu: C3−C4, P3−P4 (różnica log-mocy, analog DASM), szczególnie alfa dla walencji.
- **Cel 1, okno 700 ms:** w krótkim oknie po bodźcu dominują ERP. Przegląd mówi, że wczesne ERP (100–200 ms) wiążą się z walencją, a późne (LPP, P300, parietalnie) z pobudzeniem. Uzasadnia to dodanie **cech czasowych ERP**: średnia amplituda w oknach 100–200, 250–400 i 400–700 ms na Pz/Cz/P3/P4. Mamy pojedyncze próby, więc trzeba liczyć się z szumem.
- **Ostrzeżenie o etykietach:** mamy etykiety z ocen tłumu przypisane do obrazu, a autorzy wprost wskazują to jako źródło szumu. Warto zebrać **własne oceny SAM uczestnika** (choćby dla podzbioru obrazów) i porównać je z etykietami tłumu.
- **Ostrzeżenie o konfundowaniu walencji:** walencja jest u nas idealnie skorelowana z folderem (joy/sadness). Przegląd przypomina, że sygnał EEG to interpretacja bodźca przez mózg w danym stanie (uwaga, pamięć). Klasyfikator może więc uczyć się różnic wizualnych/kategorialnych zamiast emocji. Warto sprawdzić kontrolnie, np. permutacje w obrębie folderu albo regresję po ocenach w obrębie kategorii.
- **Walidacja bez przecieku:** u nas jedna próba = jeden epok, więc „subject-bias” w sensie segmentów nam nie grozi. Trzeba jednak pilnować, żeby **ten sam obraz** (przy powtórzeniach) nie trafiał jednocześnie do zbioru treningowego i testowego. Raportować F1/balanced accuracy, bo klasy są niezbalansowane (143 vs 74).
- **Cel 2 (neurofeedback / samoregulacja):** autorzy wskazują regulację emocji jako „następny krok” i podkreślają, że brak ustalonego algorytmu dla tego zadania. Najprostszy, neuronaukowo uzasadniony wskaźnik arousal online to **moc alfa (parietalno-potyliczna, Pz/P3/P4/Oz), której spadek oznacza wyższe pobudzenie**. Wspiera to nasz pomysł ciągłego dekodowania. Do walencji potrzeba czegoś więcej niż alfa (beta/gamma, asymetria). Przegląd popiera też **adaptację online** („test-before-train”, aktualizacja na niedawnej historii), co jest zgodne z naszym podejściem.
- **Artefakty:** gamma (30–45 Hz) jest podatna na EMG (> 30 Hz), a przy suchych elektrodach Unicorn to istotne ryzyko. OSCAR lub progowanie amplitudy są wskazane, a znaczenie gammy w modelu trzeba interpretować ostrożnie.

## 5. Ograniczenia / zastrzeżenia
- To przegląd w dużej części teoretyczny (neuronauka, w tym badania na zwierzętach, fMRI i rejestracje wewnątrzczaszkowe). Większość wniosków to korelacje bez gotowych przepisów implementacyjnych.
- Prawie wszystkie cytowane wyniki ML dotyczą 14–62 kanałów i długich bodźców wideo (minuty) z segmentacją 1–8 s. Zalecane kanały to skroniowe, czołowe i przedczołowe, których **w naszym montażu nie mamy** (poza Fz).
- Wyniki z tab. 5 pochodzą z różnych protokołów i nie są wprost porównywalne. Wysokie wyniki subject-dependent mogą być zawyżone przez opisany przez samych autorów „subject-bias”.
- Brak konkretów o oknach poniżej 1 s, o suchych elektrodach i o neurofeedbacku w czasie rzeczywistym.
- Dla obrazów statycznych przegląd cytuje głównie badania twarzy. Nasze bodźce to wygenerowane przez AI sceny, a np. N170 jest specyficzny dla twarzy i nas nie dotyczy.
