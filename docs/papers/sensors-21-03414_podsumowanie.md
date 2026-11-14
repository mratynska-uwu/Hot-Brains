# Predicting Exact Valence and Arousal Values from EEG

## 1. Metadane

- **Tytuł:** Predicting Exact Valence and Arousal Values from EEG
- **Autorzy:** Filipe Galvão, Soraia M. Alarcão, Manuel J. Fonseca (LASIGE, Faculdade de Ciências, Universidade de Lisboa)
- **Rok:** 2021 (otrzymano 29.03.2021, przyjęto 11.05.2021, opublikowano 14.05.2021)
- **Miejsce publikacji:** Sensors 2021, 21, 3414 (MDPI), artykuł badawczy open access, doi:10.3390/s21103414

## 2. Streszczenie

Autorzy zamiast klasyfikacji do kilku klas próbują przewidywać **ciągłe wartości** walencji i pobudzenia z EEG. Przetestowali systematycznie 7 regresorów, pasma alfa, beta i gamma, 8 cech (Hjorth, entropia spektralna, energia i entropia falkowa, energia i entropia IMF) oraz asymetrię międzypółkulową. Najlepszy okazał się KNN (K = 1, metryka Manhattan) na cechach z alfa, beta i gamma bez entropii spektralnej, uzupełnionych o różnicową asymetrię alfa. Na DEAP, AMIGOS i DREAMER raportują MAE < 0,06–0,08 i PCC > 0,79. Po zamianie predykcji na klasy uzyskują 89,8% (binarnie, DEAP) i 84,4% (4 kwadranty, DEAP). Protokół walidacji (losowy 10-fold na nakładających się oknach) budzi jednak poważne zastrzeżenia co do wycieku danych.

## 3. Aspekty implementacyjne

**Zbiory danych**
- **DEAP:** 32 osoby, 40 teledysków po 60 s (10 na kwadrant), Biosemi ActiveTwo, 32 kanały.
- **AMIGOS:** 40 osób, 16 krótkich filmów (< 250 s, 4 na kwadrant), Emotiv EPOC, 14 kanałów.
- **DREAMER:** 23 osoby, 18 klipów (65–393 s, 9 emocji dwukrotnie), Emotiv EPOC, 14 kanałów.
- Etykiety to samoocena SAM (walencja, pobudzenie, dominacja). Wszystkie epoki z danego filmu dostają **tę samą** etykietę.
- Metryki MAE i RMSE są wyrażone w zakresie 0–1, czyli etykiety były znormalizowane. Sposób normalizacji nie jest opisany.
- Model budowano na DEAP. AMIGOS i DREAMER służyły do oceny uogólniania, ale z tymi samymi ustawieniami i ponownym treningiem. Nie był to transfer modelu.

**Preprocessing**
- Detrending, notch 50 Hz.
- Artefakty EKG usuwane filtrowaniem adaptacyjnym, artefakty EOG progowaniem falkowym. Szczegóły parametrów nie są podane.
- Pasma alfa (8–13 Hz), beta (13–30 Hz) i gamma (> 30 Hz) wydzielane trzema filtrami pasmowymi FIR. Parametry filtrów nie są podane.
- Częstotliwość próbkowania użyta w analizie nie jest podana. Jest to istotne dla poziomów DWT, patrz niżej.
- **Okno 4 s z 50% nakładaniem.** Wybrane „na podstawie literatury i testów wstępnych”.

**Cechy (liczone dla każdego kanału)**
- **Parametry Hjortha z każdego pasma:**
  - H1 Activity = var(x)
  - H2 Mobility = √(var(x′)/var(x))
  - H3 Complexity = Mobility(x′)/Mobility(x)
- **Entropia spektralna z każdego pasma:** H = −Σ p·log₂p, gdzie p_i = PSD_i/ΣPSD.
- **Energia i entropia falkowa:** DWT z falką db4, 3 poziomy. D1 to gamma, D2 beta, D3 alfa. Energia to suma kwadratów współczynników, entropia liczona analogicznie do spektralnej.
- **Energia i entropia IMF:** EMD. IMF1 odpowiada gammie, IMF2 becie, IMF3 alfie.
- Na pasmo: 8 cech × 32 kanały = 256 cech (DEAP).
- **Asymetria** liczona między kanałami homologicznymi (np. F3–F4, F7–F8):
  - różnicowa: różnica cech,
  - racjonalna: iloraz cech.
  - 8 cech × 14 par = 112 cech.

**Selekcja**
- Etap 1: wybór regresora na wszystkich cechach.
- Etap 2: wybór typu asymetrii.
- Etap 3: forward selection (wrapper). Pary pasmo–cecha sortowane po PCC i dodawane, dopóki wynik się poprawia.
- Etap 4: strojenie hiperparametrów.
- Wszystkie etapy prowadzone na tym samym DEAP i tym samym 10-fold CV.

**Modele (Weka 3.8, domyślne parametry na etapie wyboru)**
- Testowane: Additive Regression, Decision Tree, KNN, Linear Regression, RF, SVR liniowe, SVR RBF.
- Najlepsze RF i KNN (K = 1). SVR, AR i LR wypadły najsłabiej. Przykład dla gammy: SVR RBF PCC 0,213 / 0,176, RF 0,669 / 0,667 (pobudzenie / walencja).
- Asymetria: różnicowa asymetria alfa najlepsza dla walencji (RF PCC 0,862, MAE 0,120). Dla pobudzenia najlepsze były asymetrie gammy (RF) lub racjonalna asymetria bety (KNN).
- Najlepsze pojedyncze cechy: Wavelet Energy i H1 (Activity), czyli cechy mocy. Najgorsze: entropia spektralna i H3. Pasma beta i gamma lepsze od alfy.
- **Finalny KNN:** wszystkie cechy oprócz entropii spektralnej z α, β, γ oraz asymetria różnicowa alfa (bez H3). Wymiar: DEAP 770, AMIGOS i DREAMER 343.
- **Finalny RF:** H1 i WP z β i γ oraz asymetria alfa (H1, WP, WE). Wymiar: 170 / 77.
- **Strojenie:** KNN dla K ∈ {1, 3, 5, 7, 11, 21} z metryką Manhattan. K = 1 najlepsze: PCC 0,794 / 0,795. Przy K = 21 PCC spada do 0,579 / 0,595. RF dla T ∈ {50, 100, 500, 750, 1000}. Wybrano 500 drzew.

**Walidacja**
- Autorzy nazywają ją „subject-independent”. W praktyce zbiór epok wszystkich osób **losowo** dzielono na 10 foldów, więc te same osoby, te same filmy i sąsiednie, nakładające się okna trafiają jednocześnie do treningu i testu.

**Wyniki regresji (Tabela 7, pobudzenie / walencja)**

| Model | Zbiór | PCC | MAE | RMSE |
|---|---|---|---|---|
| KNN | DEAP | 0,794 / 0,795 | 0,062 / 0,066 | 0,163 / 0,172 |
| KNN | AMIGOS | 0,830 / 0,808 | 0,045 / 0,063 | 0,129 / 0,175 |
| KNN | DREAMER | 0,806 / 0,812 | 0,058 / 0,076 | 0,165 / 0,213 |
| RF | DEAP | 0,755 / 0,852 | 0,135 / 0,118 | 0,174 / 0,153 |

**Wyniki klasyfikacji z predykcji ciągłych**
- Binarnie, KNN (pobudzenie / walencja): DEAP 89,84 / 89,83%, AMIGOS 92,46 / 90,69%, DREAMER 93,72 / 92,16%.
- Binarnie, RF: DEAP 80,62 / 85,91%, DREAMER 93,79 / 93,65%.
- 4 kwadranty, KNN: DEAP 84,40%, AMIGOS 87,57%, DREAMER 90,47%.
- 4 kwadranty, RF: 69,99%, 72,32%, 85,88%.
- 4 kwadranty, KNN dla pobudzenia + RF dla walencji: 77,10%, 77,45%, 87,00%.

## 4. Co jest istotne dla naszego projektu

**Cel 2 (neurofeedback):**
- **Regresja ciągłych wartości zamiast klasyfikacji binarnej** pasuje do naszego zadania. Uczestnik ma dopasować swój stan do konkretnych wartości walencji i pobudzenia obrazu, więc ciągły odczyt jest naturalniejszy niż etykieta 0/1. Prosta wersja w naszym pipeline: używać `predict_proba` lub `decision_function` jako ciągłej skali albo wytrenować regresor (np. RF lub Ridge) bezpośrednio na ocenach obrazów (positive_mean − negative_mean, aroused_mean). Następnie przeskalować wynik do zakresu wartości obrazów testowych.
- **Okno 4 s z 50% nakładaniem**, czyli aktualizacja co 2 s, to rozsądny punkt startowy do wyświetlania stanu w czasie rzeczywistym. Uwaga: nasz klasyfikator trenujemy na oknach 700 ms wywołanych obrazem. Zastosowanie go do podtrzymywanego, samodzielnie wywołanego stanu oznacza zmianę rozkładu danych (domain shift). Warto rozważyć dodatkową kalibrację na dłuższych oknach.
- **KNN i RF są szybkie**, co autorzy podkreślają jako zaletę w aplikacjach interaktywnych.

**Cel 1 (cechy):**
- **Najlepsze były cechy mocy:** H1 (wariancja sygnału po filtrze pasmowym, czyli de facto moc pasma), energia falkowa i energia IMF. Potwierdza to nasz wybór log-mocy pasm. Hjorth H1 i H2 oraz energię DWT można tanio dodać jako alternatywę dla FFT z krótkiego okna.
- **Entropia spektralna pogarszała wyniki**, więc lepiej jej nie dodawać. Wymiaru fraktalnego (Higuchi) artykuł nie testował, więc nie ocenia naszej cechy HFD.
- **Różnicowa asymetria alfa wyraźnie poprawiała walencję.** Mamy pary homologiczne C3–C4 i P3–P4, ale **nie mamy F3 i F4**, a klasyczna czołowa asymetria alfa (FAA) jest niedostępna (tylko Fz). Proponuję dodać log α(C4) − log α(C3) i log α(P4) − log α(P3), ewentualnie także dla bety i gammy (4–6 cech).
- **Beta i gamma były najlepsze.** Pokrywa się to z naszymi pasmami. Gamma jest jednak bardzo podatna na EMG (napięcie mięśni twarzy przy emocjach), szczególnie przy suchych elektrodach, a dobre wyniki gammy mogą częściowo odzwierciedlać artefakty mięśniowe.
- **Uwaga techniczna:** przy 250 Hz poziomy DWT D1/D2/D3 odpowiadają ok. 62,5–125 / 31–62,5 / 15,6–31 Hz, a nie gammie, becie i alfie. Przy przenoszeniu cech falkowych trzeba użyć poziomów D3–D5.

**Walidacja (ważne):**
- Ten artykuł pokazuje, **czego nie robić**. Losowy k-fold na nakładających się oknach z jedną etykietą na film, do tego 1-NN, sprawia, że najbliższym sąsiadem jest niemal ten sam fragment sygnału. Wynik K = 1 > K = 21 to typowy objaw wycieku danych.
- U nas: nie mieszać losowo epok tego samego obrazu ani sąsiednich okien między treningiem a testem. Dzielić chronologicznie lub według obrazów. Nasze „test-before-train” jest pod tym względem poprawne.

## 5. Ograniczenia i zastrzeżenia

- **Prawdopodobny silny wyciek danych.** Losowy 10-fold na oknach 4 s z 50% nakładaniem, ta sama etykieta dla wszystkich okien filmu i ci sami uczestnicy w treningu i teście. Do tego selekcja cech i modelu na tym samym CV. Wyniki 84–93% należy traktować jako mocno zawyżone. Nazwa „subject-independent” jest myląca, bo nie był to podział według osób.
- Bodźce to filmy i teledyski trwające 60–393 s z jedną oceną na film. My mamy 1 s obrazy i okna 700 ms, więc dynamika, długość okna i rozdzielczość spektralna są zupełnie inne.
- Sprzęt: 32 kanały (DEAP) lub 14 kanałów Emotiv, z elektrodami czołowymi i asymetrią czołową. My mamy 8 kanałów bez F3/F4/F7/F8.
- Nie podano częstotliwości próbkowania, parametrów filtrów FIR, metody normalizacji etykiet ani szczegółów usuwania artefaktów EKG i EOG.
- Brak porównania z prostym baseline'em (np. predykcja średniej lub losowe etykiety) i brak analizy per uczestnik.
- Etykiety to samoocena osoby badanej (SAM). U nas są to oceny tłumu obrazów, więc dodatkowo pojawia się szum etykiet.
