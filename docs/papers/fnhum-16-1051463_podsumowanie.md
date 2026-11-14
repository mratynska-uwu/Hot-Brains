# Decoding the neural signatures of valence and arousal from portable EEG headset — podsumowanie

## 1. Metadane

- **Tytuł:** Decoding the neural signatures of valence and arousal from portable EEG headset
- **Autorzy:** Nikhil Garg, Rohit Garg, Apoorv Anand, Veeky Baths (BITS Pilani Goa / Université de Sherbrooke / Université de Lille)
- **Rok / czasopismo:** 2022, *Frontiers in Human Neuroscience* 16:1051463, doi: 10.3389/fnhum.2022.1051463 (sekcja Brain-Computer Interfaces)
- **Typ:** artykuł oryginalny (Original Research); 16 stron (plik PDF ma 16 stron, nie 63)
- **Kod / dane:** github.com/rohitgarg025/Decoding_EEG; zbiór „OASIS EEG” na Zenodo (rekord 7332684)

## 2. Streszczenie

Autorzy nagrali EEG przenośnym headsetem Emotiv Epoc X (14 kanałów, 128 Hz) od 15 osób oglądających 40 statycznych obrazów z bazy OASIS i porównali wyniki z publicznymi zbiorami DEAP i DREAMER (bodźce filmowe). Zbudowali pipeline: CAR, filtr 0,1–40 Hz, usuwanie artefaktów ruchowych (ICA + akcelerometr), AutoReject, 36 cech z biblioteki EEGExtract liczonych w oknach 1 s, selekcja SelectKBest (f_regression) i **regresja** walencji/pobudzenia lasem losowym. Uszeregowali elektrody (najważniejszy obszar czołowy, potem skroniowy) i cechy (najlepsze: entropia/moc w paśmie gamma i beta oraz parametry Hjortha). Na ich własnym zbiorze z obrazami wyniki są bardzo słabe (R² 0,042 dla walencji, 0,236 dla pobudzenia), a w walidacji leave-one-subject-out błąd rośnie.

## 3. Aspekty implementacyjne

**Bodźce i protokół (zbiór OASIS EEG)**
- 40 obrazów z 900 w OASIS, dobranych tak, by pokryć wszystkie cztery ćwiartki płaszczyzny walencja–pobudzenie.
- 15 uczestników (10 M, 5 K), średni wiek 22 lata; zamknięte pomieszczenie, jedynym źródłem światła monitor Samsung 21" 1080p, odległość 80–100 cm.
- Każdy obraz wyświetlany **5 s** (PsychoPy), następnie ocena walencji i pobudzenia na skali **1–10** przez uczestnika; w analizie użyto tylko EEG z okresu oglądania (segmentacja markerami).
- MSE między ocenami uczestników a normami OASIS: 1,34 (walencja) i 1,39 (pobudzenie).
- Etykiety to ciągłe oceny (zadanie regresji, nie klasyfikacji).

**Sprzęt**
- Emotiv Epoc X, 14 elektrod (10–20): AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4; 128 Hz; wbudowany żyroskop/akcelerometr.
- DEAP: 32 osoby, 40 teledysków po 1 min, 128 Hz, użyto tych samych 14 kanałów. DREAMER: 23 osoby, 18 klipów, 128 Hz, 14 elektrod.

**Preprocessing**
- Referencja: średnia ze wszystkich elektrod (CAR) w każdej chwili czasu.
- Filtr: Butterworth 4. rzędu, pasmo 0,1–40 Hz.
- Artefakty ruchowe: ICA na każdym trialu, korelacja Pearsona każdego źródła z każdą osią akcelerometru; źródła z korelacją > średnia + 2 SD dla dowolnej osi filtrowane górnoprzepustowo 3 Hz (Butterworth), potem rzutowane z powrotem macierzą mieszania.
- Odrzucanie złych triali: AutoReject (Jas i in., 2017), wariant globalny – próg peak-to-peak dobierany walidacją K-krotną (norma Frobeniusa między średnią dobrych triali a medianą zbioru walidacyjnego). Na DEAP/DREAMER nie stosowane.
- Brak wzmianki o korekcji linii bazowej.

**Cechy (36, EEGExtract, okno przesuwne 1 s bez nakładania)**
- Pasma: delta 0,5–4, theta 4–8, alfa 8–12, beta 12–30, gamma 30–45 Hz (uwaga: gamma wychodzi poza filtr 40 Hz – niespójność w pracy).
- Złożoność: entropia Shannona; Subband Information Quantity (entropia Shannona sygnału po filtrze Butterwortha 7. rzędu + FIR/IIR) dla 5 pasm; Hjorth mobility = sqrt(var(dx/dt)/var(x)); Hjorth complexity = mobility(dx/dt)/mobility(x); False Nearest Neighbor; DASM = DE(lewa) − DE(prawa) i RASM = DE(lewa)/DE(prawa) dla 7 par półkulowych i 5 pasm, gdzie DE = ½·log(2πeσ²).
- Ciągłość: mediana częstotliwości; moc w pasmach (Butterworth 7. rzędu + periodogram, średnia PSD, scipy.signal); odchylenie standardowe; diffuse slowing; spikes (próg średnia + 3 SD, find_peaks na 7 próbkach); sharp spikes (<70 ms); delta burst after spike; liczba/długość burstów; liczba/długość supresji (próg 10 µV).

**Selekcja cech**
- VarianceThreshold usuwa cechy stałe; zostaje 25 cech wspólnych dla 3 zbiorów (odpadło 11: S.E., F.N.N., D.S., S.K., D.B.A.S., N.B., B.L.M., B.L.S., N.S., S.L.M., S.L.S.).
- Macierz [n_segmentów, n_kanałów × n_cech]; ranking SelectKBest z f_regression (F = ρ²/(1−ρ²)·(n−2)).
- Ranking elektrod i cech: dodawanie po kolei top-N i ocena RMSE.

**Model i walidacja**
- RandomForestRegressor (scikit-learn), 100 drzew, kryterium squared error.
- Metryki: RMSE, R², MAE, explained variance.
- Podział 80:20 na pomieszanych segmentach (z tekstu wynika, że losowo na poziomie segmentów, bez rozdziału po trialach – ryzyko przecieku). Dodatkowo „incremental learning” (dokładanie kolejnych osób) i leave-one-subject-out (LOSO).

**Wyniki (test 20%, optymalny zestaw cech, Tabela 4)**
- DREAMER: walencja RMSE 0,905, R² 0,702 (11 cech); pobudzenie RMSE 0,749, R² 0,683 (9 cech).
- DEAP: walencja RMSE 1,902, R² 0,247 (8 cech); pobudzenie RMSE 1,769, R² 0,273 (9 cech).
- OASIS EEG: walencja RMSE 2,728, R² 0,042, MAE 2,320 (**2 cechy**); pobudzenie RMSE 2,300, R² 0,236, MAE 1,815 (**1 cecha**).
- Optymalna liczba elektrod (Tabela 5): OASIS walencja 7 elektrod (RMSE 2,765, R² 0,020), pobudzenie 14 (RMSE 2,417, R² 0,182); DEAP/DREAMER – 14.
- LOSO (średnia ± SD RMSE): DREAMER 1,356 ± 0,130 / 1,126 ± 0,190; DEAP 2,112 ± 0,416 / 2,025 ± 0,519; OASIS 2,933 ± 0,582 / 2,642 ± 0,845 (walencja / pobudzenie). Błąd LOSO wyższy niż przy 80:20 o 50% (DREAMER), 14% (DEAP), 9% (OASIS).
- Incremental learning: dla DEAP poprawa z liczbą osób; dla DREAMER i OASIS pogorszenie, błąd nasyca się po ~10 (DREAMER) i ~5 (OASIS) osobach – silny bias międzyosobniczy.
- Elektrody: dla OASIS najwyżej F7 (1. dla obu etykiet), T7/T8 (2–3), F8 (4); najniżej O2, P7, P8. Zbiorczo: walencja – w top 25% 33% czołowe, 33% skroniowe, 22% ciemieniowe, 11% potyliczne; pobudzenie – 55% czołowe, 44% skroniowe.
- Cechy: dla OASIS najwyżej S.E.G (SIQ gamma: 1. walencja, 2. pobudzenie), Hjorth mobility (1. pobudzenie, 2. walencja), S.E.B (3/3), Hjorth complexity (4/4), SD (6/6). Moc alfa nisko (25. pobudzenie, 15. walencja), moc gamma 5. dla pobudzenia. Ogólny wniosek autorów: pasma beta i gamma ważniejsze niż pozostałe.
- Tabela 1 (przegląd literatury, głównie DEAP, klasyfikacja): kNN/SVM/GNB z dokładnościami ok. 61–88%, np. SVM + Hjorth/entropia/RASM/DASM 65,72% (10-fold) i 65,92% (LOO).

## 4. Co jest istotne dla naszego projektu

**Cel 1 (klasyfikator walencja–pobudzenie z obrazów)**
- To jedna z niewielu prac z **statycznymi obrazami i tanim headsetem** – i wynik na obrazach jest bliski zeru dla walencji (R² 0,042). Trzeba realistycznie zakładać niewielką przewagę nad losowym zgadywaniem i porównywać się z klasyfikatorem losowym/permutacyjnym.
- Warto tanio dodać do naszych 32 cech: **Hjorth mobility i complexity, odchylenie standardowe, medianę częstotliwości, entropię Shannona per pasmo (SIQ)**. W tej pracy wypadały wyżej niż moc w pasmach, szczególnie alfa. Wszystkie da się policzyć dla okna 700 ms.
- Przy małej liczbie danych (u nich OASIS: 15 osób × 40 obrazów) najlepsze wyniki dawały **1–2 cechy**; więcej cech = przeuczenie. Przy ~170 próbach treningowych na osobę i 32 cechach warto testować SelectKBest (k ≈ 5–10) lub silną regularyzację – ale **selekcję robić wyłącznie na foldzie treningowym**.
- Brak nakładania się elektrod: najważniejsze u nich F7, F8, T7, T8, a my mamy Fz, C3, Cz, C4, P3, Pz, P4, Oz. Z naszego zestawu najbliżej „frontalnych” jest tylko Fz. Asymetrię (DASM/RASM na DE) możemy liczyć na parach C3–C4 i P3–P4.
- Rozdzielanie triali przy podziale 80/20: jeśli dzielimy okna lub segmenty, robić to blokowo po obrazach/trialach (u nich losowy podział segmentów 1 s prawdopodobnie zawyża wyniki).
- Etykiety: oceny uczestników różniły się od norm OASIS (MSE ok. 1,3–1,4 na skali 1–10). Warto po sesji treningowej zebrać choćby krótkie oceny własne uczestnika dla części obrazów – nasze etykiety z crowd-ratingów mogą nie odpowiadać przeżyciom danej osoby.

**Cel 2 (neurofeedback / samoregulacja)**
- Regresja ciągła (RF regressor) naturalnie daje wartość do wyświetlenia na suwaku; można rozważyć `predict_proba` z naszego klasyfikatora jako ciągły sygnał zwrotny.
- Przy RMSE ok. 2,3–2,7 na skali 1–10 (OASIS) sygnał zwrotny będzie bardzo zaszumiony – konieczne wygładzanie (średnia krocząca kilku okien) i komunikowanie niepewności uczestnikowi.
- Modele przenoszą się słabo między osobami (LOSO gorsze), co potwierdza nasz wybór modelu per uczestnik.

## 5. Ograniczenia / zastrzeżenia

- Zadanie regresji z RMSE/R², a nie klasyfikacja binarna – wyników nie da się bezpośrednio porównać z naszą rolling accuracy.
- Bodziec trwał 5 s, a cechy liczono z okien 1 s; nasze okno 700 ms po onsecie to inna skala czasowa (bardziej ERP niż stan emocjonalny) – rankingi cech mogą się nie przenieść.
- Inny montaż (14 kanałów z dominacją czołowo-skroniową) i 128 Hz vs nasze 8 kanałów centralno-ciemieniowych przy 250 Hz.
- Wysoka ranga cech gamma na elektrodach F7/F8/T7/T8 może odzwierciedlać EMG (mięśnie skroniowe, mimika), a nie aktywność mózgu; autorzy tego nie analizują. U nas, z suchymi elektrodami, gamma 30–45 Hz też jest podatna na artefakty mięśniowe.
- Niespójność: filtr 0,1–40 Hz, a pasmo gamma zdefiniowane do 45 Hz.
- CAR na 8 kanałach (jak u nas) jest słabiej uzasadniony niż na 14+ – nie przenosić bezrefleksyjnie.
- Losowy podział segmentów 80:20 bez rozdziału triali/osób → prawdopodobny przeciek; brak testów statystycznych i porównania z poziomem losowym.
- Usuwanie artefaktów ruchowych wymaga akcelerometru zsynchronizowanego z EEG (Unicorn ma akcelerometr/żyroskop, ale trzeba to sprawdzić w g.Pype); ICA na 8 kanałach i krótkich trialach ma ograniczoną skuteczność.
- Mała próba (15 osób), brak pracy nad samoregulacją/neurofeedbackiem; nie testowano wymiaru fraktalnego Higuchiego.
