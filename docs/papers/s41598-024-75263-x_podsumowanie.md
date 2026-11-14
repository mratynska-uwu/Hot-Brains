# EEG-based emotional valence and emotion regulation classification: a data-centric and explainable approach

## 1. Metadane

- **Tytuł:** EEG-based emotional valence and emotion regulation classification: a data-centric and explainable approach
- **Autorzy:** Linda Fiorini, Francesco Bossi (równy wkład), Francesco Di Gruttola (IMT School for Advanced Studies Lucca; Uniwersytet w Pizie; Uniwersytet Boloński)
- **Rok:** 2024 (otrzymano 16.11.2023, przyjęto 03.10.2024, opublikowano 14.10.2024)
- **Miejsce publikacji:** Scientific Reports 14:24046, artykuł badawczy open access, doi:10.1038/s41598-024-75263-x
- **Dane:** dostępne w OSF (https://osf.io/yv468/)

## 2. Streszczenie

Autorzy nagrali 64-kanałowe EEG 30 osób, które oglądały pozytywne i negatywne obrazy OASIS w trzech blokach: zwykłe patrzenie (Look), poznawcza reinterpretacja (Reappraise) i tłumienie ekspresji (Suppress), a po każdym obrazie oceniały jego walencję. Najpierw użyli Random Forest z biblioteką Cleanlab (confident learning), żeby ocenić jakość etykiet i wybrać schemat klas. Wyszło z tego połączenie „neutralne vs spolaryzowane” (walencja) oraz trzy klasy strategii regulacji. Następnie na surowym sygnale (64 kanały × 500 ms) trenowali sieci MINI-VGG, MINI-VGG+GRU, MINI-VGG+BiGRU i Transformer metodą curriculum learning z głosowaniem większościowym trzech modeli. Interpretowali je przez Integrated Gradients. Wyniki są skromne: ważone F1 wynosi ≈0,42–0,43, a strategie regulacji klasyfikowane są niemal na poziomie przypadku. Zbiór dzielono jednak rygorystycznie według osób (cross-subject).

## 3. Aspekty implementacyjne

**Uczestnicy i bodźce**
- 30 płatnych uczestników (17 kobiet, wiek 26 ± 6,12). Wykluczano m.in. osoby z chorobami neurologicznymi lub psychiatrycznymi, a także po nikotynie w ciągu 2 h i alkoholu w ciągu 24 h przed badaniem.
- 60 obrazów z OASIS (baza 900 obrazów, oceny 822 osób, skala Likerta 1–7). Kryteria doboru: brak treści seksualnych i przemocy, pobudzenie > 4, skrajne walencje.
- 30 obrazów pozytywnych (walencja 5,7 ± 1,06, pobudzenie 4,7 ± 1,6) i 30 negatywnych (walencja 2,1 ± 1,1, pobudzenie 4,7 ± 1,8). Pobudzenie było więc wyrównane między klasami.
- W tej próbie 21 obrazów miało oceny istotnie różne od norm OASIS, ale żaden nie zmienił znaku walencji względem neutralnego 4.

**Protokół**
- 3 bloki (Look / Reappraise / Suppress) w kolejności zbalansowanej, po 60 prób. Łącznie 180 prób na osobę.
- Każda próba: krzyż fiksacyjny 1000 ms, obraz 1000 ms, potem ocena walencji 1–7 myszą, bez limitu czasu.
- Nazwa strategii była wyświetlana nad obrazem przez cały blok.
- Pomieszczenie ekranowane elektrycznie i dźwiękowo, monitor 19″ w odległości 1 m. Przed zadaniem nagrywano 5 min spoczynku, ale go nie wykorzystano.

**Sprzęt**
- EGI, 64 kanały w systemie 10–20, referencja online Cz, 1000 Hz, impedancja < 50 kΩ.

**Preprocessing (MNE-Python)**
- Filtr pasmowy 1–48 Hz i notch 50 Hz. Złe kanały wykrywane metodą RANSAC.
- Epoka od −500 do +1000 ms. Linia bazowa −500 do −200 ms (celowo przed onsetem, z pominięciem ostatnich 200 ms). Sygnał przycięty do **0–500 ms po onsecie**.
- Krótkie okno wybrano celowo, „żeby być precyzyjnym dla rozpoznawania emocji w czasie rzeczywistym”. Autorzy powołują się na wczesne korelaty neuronalne emocji.
- Autoreject: próg konsensusu od 0,2 do 0,5 co 0,1 oraz interpolacja 1–8 elektrod. Średnio odrzucano 9,16 ± 9,97 epok na osobę.
- Rereferencja offline do średniej.
- Kształt pojedynczego przykładu: (1, 64, 500).

**Podział danych i augmentacja**
- Podział **według osób**: 22 osoby w treningu, 4 w walidacji, 4 w teście.
- Na zbiorze treningowym PCA do 100 komponentów (87% wariancji), potem SMOTE (k = 5) w przestrzeni PCA i odwrotna PCA. Zgodność rekonstrukcji: korelacja odległościowa 0,99.

**Confident learning (Random Forest + Cleanlab)**
- Osobne modele RF dla walencji i dla strategii regulacji. Etykiety walencji to **własne oceny uczestników**, nie normy z bazy.
- Wyniki RF na zbiorze walidacyjnym:
  - 7 klas: F1 = 0,15 (poziom przypadku 0,14).
  - 3 klasy (1–2 / 3–5 / 6–7): accuracy 0,47. Klasa neutralna ma F1 0,64, pozostałe 0,20–0,23.
  - 2 klasy, neutralne vs spolaryzowane: accuracy 0,55, F1 0,39 / 0,64.
  - Strategia regulacji (3 klasy): accuracy 0,33.
- Jakość etykiety każdego przykładu uśredniano po walencji i strategii, a zbiór dzielono na percentylach 33 i 66 na części łatwą, średnią i trudną.

**Curriculum learning i ensemble**
- Trzy modele o tej samej architekturze: pierwszy trenowany na części łatwej, drugi na łatwej i średniej, trzeci na całości. Decyzję podejmuje głosowanie większościowe (2 z 3).
- Zadanie wieloetykietowe z 5 etykietami (Neutral, Polarised, Look, Reappraise, Suppress). Wyjście sigmoid, próg 0,5.

**Architektury**
- **MINI-VGG 1D:** dwa bloki po 2 warstwy konwolucyjne z ReLU (32 filtry, k = 3, potem 64 filtry, k = 3), max-pooling 2, batch normalization, dropout 0,15, warstwa Dense z sigmoidą.
- **+GRU / +BiGRU:** trzy warstwy GRU po 128 jednostek (tanh) po drugim bloku konwolucyjnym, Monte Carlo Dropout 0,15.
- **Transformer:** 2 enkodery (head size 64, 2 głowy, dropout 0,4, konwolucje z 64 filtrami, k = 1), global average pooling, Dense 128 z ReLU, dropout 0,4, wyjście 5 × sigmoid.
- Trening: Adam z lr = 1e-6, maks. 20 000 epok, early stopping z cierpliwością 20 (Keras).

**Wyniki na zbiorze testowym (ważone F1)**

| Model | Ważone F1 | Neutral | Polarised | Look / Reappraise / Suppress |
|---|---|---|---|---|
| MINI-VGG | 0,42 | 0,55 | 0,51 | 0,36 / 0,27 / 0,29 |
| MINI-VGG+GRU | 0,43 | 0,57 | 0,41 | 0,31 / 0,39 / 0,39 |
| MINI-VGG+BiGRU | 0,43 | 0,43 | 0,50 | 0,44 / 0,38 / 0,36 |
| Transformer | 0,28 | 0,53 | 0,56 | 0,02 każda |

**Wyjaśnialność (Integrated Gradients)**
- Biblioteka Alibi, baseline zerowy, 25 kroków.
- Atrybucje zgrupowano w 10 przedziałów czasowych po 50 ms i 3 klastry kanałów (przedni 23, centralny 23, tylny 18), a następnie porównano liniowymi modelami mieszanymi (LMM) z poprawką Tukeya.
- Klaster **centralny** był najważniejszy we wszystkich modelach.
- CNN: maksimum w przedziale 150–200 ms.
- GRU: praktycznie tylko ostatnie 100 ms (400–500 ms). BiGRU: pierwsze 50 ms i ostatnie 50 ms.
- Transformer: stabilny wzorzec w czasie, z przewagą klastra centralnego.
- Autorzy wiążą to z LPP (400–500 ms, okolice centro-parietalne).
- **Behawioralnie:** interakcja strategia × walencja, χ² = 1106,7, df = 40, p < 0,001. Reappraisal i suppression podnoszą oceny obrazów negatywnych i obniżają oceny pozytywnych względem Look.
- Replikacja na DEAP jest tylko w materiałach dodatkowych. Liczby nie są podane w artykule.
- **Narzędzia:** MNE-Python, autoreject, Cleanlab, Keras, Alibi, R (lmer, emmeans).

## 4. Co jest istotne dla naszego projektu

**Cel 1 (klasyfikator walencji i pobudzenia):**
- **Preprocessing jest niemal identyczny z naszym:** 1–48 Hz i notch 50 Hz. Różnią się baseline (−500…−200 ms zamiast −200…0 ms) i okno analizy (0–500 ms). Warto przetestować baseline kończący się przed −200 ms, żeby nie łapać aktywności antycypacyjnej. Wymaga to wydłużenia epoki w lewo.
- **Klaster centralny i okno ok. 400–500 ms (LPP) były najbardziej informatywne.** Mamy dobre pokrycie: C3, Cz, C4, Pz, P3, P4. Proponuję dodać do 32 cech spektralnych cechy czasowe (ERP), np. średnią amplitudę 300–700 ms na Cz, Pz, P3 i P4. Nasza epoka sięga do +700 ms, więc da się to zrobić. FFT z okna 700 ms ma rozdzielczość ok. 1,4 Hz i słabo uchwyci składowe wywołane.
- **Odrzucanie artefaktów jest istotne.** Autoreject odrzucał średnio ok. 9 z 180 epok. U nas warto choćby wprowadzić próg amplitudy albo użyć OSCAR, zwłaszcza przy suchych elektrodach.
- **Szum etykiet:** nawet przy skrajnie spolaryzowanych obrazach (OASIS: 2,1 vs 5,7) uczestnicy często oceniali je jako 3–5. Nasze etykiety pochodzą z ocen tłumu, a nie od uczestnika, więc szum jest prawdopodobnie jeszcze większy.
  - Warto zebrać własne oceny uczestnika choćby dla części obrazów.
  - Przy pobudzeniu (1,5–3,9, próg na medianie) można wyrzucić środkowy tercyl obrazów. Odpowiada to podejściu „polarised vs neutral” z artykułu.
  - Cleanlab (`find_label_issues`) można tanio uruchomić na naszych cechach, żeby oznaczyć podejrzane próby.
- **Wycieki danych:** autorzy podkreślają, że losowe mieszanie epok zawyża wyniki, i cytują Kamrud i in. z błędem niedoszacowanym o 35–3900%. Nasz model jest per-osoba, ale analogiczny problem dotyczy powtórzeń tego samego obrazu oraz dryfu w czasie. Nasze „test-before-train” jest tu poprawne. Przy podziale 80/20 lepiej dzielić chronologicznie lub według obrazów niż losowo według epok.

**Cel 2 (neurofeedback / samoregulacja):**
- Reappraisal i suppression **zmieniają subiektywną walencję**: obrazy negatywne oceniane wyżej, pozytywne niżej. Uczestnik może więc realnie regulować swój stan.
- Samej strategii regulacji nie da się jednak wiarygodnie zdekodować z EEG: accuracy 0,33 w RF, F1 ok. 0,3–0,44 w sieciach. Nie ma sensu wyświetlać „jaką strategię stosujesz”. Lepiej pokazywać tylko walencję i pobudzenie.
- Okno 500 ms wybrano z myślą o czasie rzeczywistym. Przy samoregulacji (stan podtrzymywany, a nie odpowiedź na bodziec) i tak trzeba będzie wygładzać predykcje z kilku okien.

## 5. Ograniczenia i zastrzeżenia

- Model jest **cross-subject**, a nasz per-osoba, więc liczby nie są bezpośrednio porównywalne. Wyniki są też słabe: walencja niewiele powyżej przypadku, strategie na poziomie przypadku.
- Zadanie to „neutral vs polarised”, a nie pozytywne vs negatywne. Nie mówi to wprost, jak rozdzielać walencję dodatnią i ujemną.
- Głębokie sieci na 64 kanałach × 500 próbek (1000 Hz) przy ~180 próbach na osobę. U nas jest 8 kanałów, 250 Hz i dane z jednej osoby, więc sieci tej wielkości raczej nie mają zastosowania. Bardziej przenośne są wnioski o czasie (LPP) i lokalizacji (kanały centralne).
- IG analizowano na poziomie sensorów. Autorzy sami nazywają interpretację anatomiczną spekulatywną.
- Bodźce miały wysokie pobudzenie (> 4) przy wyrównanym pobudzeniu między klasami. Artykuł **nie klasyfikuje pobudzenia**, więc nie pomaga w naszej osi pobudzenia.
- Nietypowe ustawienia: lr = 1e-6 i 20 000 epok. Szczegóły DEAP są tylko w suplemencie. Liczby SMOTE i curriculum nie są porównane z wariantem bez nich, więc nie da się ocenić zysku z każdego elementu.
