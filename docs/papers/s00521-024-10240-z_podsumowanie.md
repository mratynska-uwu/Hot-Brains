# Castiblanco Jimenez i in. (2024) – Podsumowanie

## 1. Metadane

- **Tytuł:** Effective affective EEG-based indicators in emotion-evoking VR environments: an evidence from machine learning
- **Autorzy:** Ivonne Angelica Castiblanco Jimenez, Elena Carlotta Olivetti, Enrico Vezzetti, Sandro Moos, Alessia Celeghin, Federica Marcolin (Politecnico di Torino, Università degli Studi di Torino)
- **Rok:** 2024 (przyjęto 12.07.2024, online 19.09.2024)
- **Miejsce/typ:** Neural Computing and Applications 36:22245–22263, DOI 10.1007/s00521-024-10240-z; artykuł oryginalny, open access (CC BY 4.0).

## 2. Streszczenie w 3–5 zdaniach

30 uczestników eksplorowało 5 środowisk VR (wyświetlanych na monitorze 27″) wywołujących złość, wstręt, strach, radość i smutek, a ich EEG rejestrowano 14-kanałowym Emotiv EPOC X. Z mocy pasm liczono dwa „wskaźniki afektywne”: walencję jako asymetrię alfa (F4 − F3) i pobudzenie jako β/α na F3 i F4. Szeregi czasowe tych wskaźników były wejściem dla XGBoost i Random Forest, które klasyfikowały 3 poziomy (niski/neutralny/wysoki) walencji i pobudzenia według samooceny SAM (skala 9-punktowa). Autorzy podają trafność 93% (pobudzenie) i 97% (walencja) dla XGBoost oraz 93%/91% dla RF, i wnioskują o spójności EEG z samoopisem. Opisowo: theta była najsilniejszym pasmem we wszystkich VE, a płat czołowy najbardziej aktywny.

## 3. Aspekty implementacyjne

**Uczestnicy i bodźce**
- 30 osób, wiek 19–36, głównie Włosi, równy rozkład płci; wykluczenia: choroby neurologiczne/psychiatryczne, leki wpływające na mózg; normalny lub skorygowany wzrok.
- 10 zwalidowanych VE (Dozio i in. 2022) – po 2 na emocję (wnętrze/plener): radość, wstręt, strach, smutek, złość. Każdy uczestnik widział **5 VE (po jednym na emocję), w losowej kolejności**.
- Nawigacja myszą/klawiaturą przed monitorem 27″, dźwięk z głośnika zewnętrznego (UE Boom) – **nie** headset VR.

**Protokół i czasy**
- Baseline: **15 s z zamkniętymi oczami**.
- Każde VE: **60 s**; potem **90 s przerwy** (30 s na SAM 9-punktowy dla walencji i pobudzenia + 60 s odpoczynku).
- Projekt wewnątrzosobniczy; analiza oparta na **różnicy między baseline a aktywnością w trakcie zadania**.

**Sprzęt**
- **Emotiv EPOC X**, 14 elektrod solnych: AF3, AF4, F3, F4, F7, F8, FC5, FC6, P7, P8, T7, T8, O1, O2; referencje CMS/DRL na P3/P4.
- **fs = 128 Hz**.

**Preprocessing i cechy**
- Filtr górnoprzepustowy przed FFT (częstotliwość odcięcia **nie podana**); brak opisu filtru sieciowego, ICA ani referencjonowania. Do analizy brano tylko sygnały „wolne od artefaktów” (metoda **nie opisana**); 3 osoby wykluczono z powodu szumu/braku danych → 27 zbiorów.
- FFT z oknem Hanninga **2 s (256 próbek)**, przesuwanym o **16 próbek** (125 ms); kwadrat modułu FFT uśredniany w pasmach, moc w µV²/Hz.
- Pasma: **θ 4–8 Hz, α 8–12 Hz, β 12–25 Hz, γ 25–45 Hz**.
- Wskaźniki (tab. 1; wybrane wzory 2 i 5):
  - **Walencja = αF4 − αF3** (tekst: różnica logarytmów naturalnych mocy α prawej i lewej półkuli). Wyższa wartość → względnie większa aktywacja lewej strony czołowej → walencja pozytywna.
  - **Pobudzenie = (βF3 + βF4) / (αF3 + αF4)**.
  - Alternatywy z literatury: walencja αF4/βF4 − αF3/βF3; walencja jako różnica β/α w AF3,F3 vs AF4,F4; pobudzenie β/α na AF3, AF4, F3, F4.
- Wektor cech: dla każdego uczestnika i VE szereg czasowy wskaźnika z całej nawigacji → **134 próbki, każda o długości 537 elementów**.

**Etykiety i modele**
- Etykiety z SAM: 1–3 = „niski”, 4–6 = „neutralny”, 7–9 = „wysoki” (osobno dla walencji i pobudzenia).
- **XGBoost** (pakiet xgb 1.7.4), funkcja celu z regularyzacją Ω = γT + ½λΣw²; strojenie hiperparametrów ~40 min, trening <5 min (zakresów i wybranych wartości **nie podano**).
- **Random Forest** (scikit-learn 1.1.3), trening <2 min; hiperparametry **nie podane**.
- Walidacja: **pojedynczy podział, zbiór testowy = 0,3 całego zbioru**; brak CV; nie podano, czy podział był rozdzielny po uczestnikach – próbki tych samych osób mogły trafić do treningu i testu (model zbiorczy, nie osobniczy).

**Wyniki (tab. 3)**
- XGBoost pobudzenie: F1 niski/neutralny/wysoki = 0,77/0,94/0,97, accuracy **0,93**; walencja: 0,97/1,00/0,93, accuracy **0,97**.
- RF pobudzenie: 0,77/0,94/0,97, accuracy **0,93**; walencja: 0,91/0,95/0,86, accuracy **0,91**.
- Macierze pomyłek: test pobudzenia ok. 40 próbek (8/16/16), walencji ok. 35 (16/11/8) – klasy niezrównoważone (mało „niskiego” pobudzenia i „wysokiej” walencji).
- Opisowo (rys. 9, mediany 27 osób): θ najwyższe we wszystkich VE; Indoor Fear najwyższe γ i β; Indoor Anger najniższe γ i β; płat czołowy najaktywniejszy w θ, α, β; γ w okolicach czołowych/skroniowych.
- Przykład pojedynczej osoby (rys. 8): w Outdoor Fear ok. 2 s po starcie gwałtowny wzrost pobudzenia i spadek walencji.

## 4. Co jest istotne dla naszego projektu

**Cel 1**
- **Brak F3/F4 w Unicornie**: wzory na walencję/pobudzenie wymagają elektrod czołowych bocznych. Mamy tylko Fz (linia środkowa). Przybliżenia: asymetria α **C4 − C3** i **P4 − P3** (ln mocy), oraz β/α uśrednione po C3/C4/Fz jako indeks pobudzenia. Warto dodać je jako 2–4 cechy „wiedzowe” obok obecnych 32 i sprawdzić ich wkład.
- **Dodać pasmo theta (4–8 Hz)** – autorzy uznali je za kluczowe (najsilniejsze, związane z regulacją emocji). Nasz pipeline pomija θ, mimo że filtr 1–48 Hz je przepuszcza. Fz jest dobrą lokalizacją dla θ czołowej linii środkowej.
- Granice pasm różnią się od naszych (β 12–25, γ 25–45 vs nasze β 13–30, γ 30–45) – nie jest to krytyczne, ale γ przy suchych elektrodach i 8 kanałach jest podatne na EMG; warto sprawdzić ważność cech γ.
- Okno 2 s daje rozdzielczość 0,5 Hz; nasze 700 ms (~1,4 Hz) jest akceptowalne dla α/β, ale zgrubne dla θ – rozważyć Welch z zerowym dopełnianiem lub dłuższe okna.
- Etykiety z **własnej oceny uczestnika (SAM)** zamiast norm – rozważyć to zwłaszcza dla pobudzenia.

**Cel 2 (neurofeedback)**
- Paradygmat ciągłego stanu (60 s) + **okno przesuwne 2 s co 125 ms** daje płynny sygnał walencji/pobudzenia w czasie (rys. 8) – to wzorzec odpowiedni dla wyświetlania na żywo. Nasz klasyfikator epokowy (700 ms po onsecie) jest dopasowany do potencjałów wywołanych; do samoregulacji lepiej użyć cech z okna przesuwnego 1–2 s i wygładzania.
- Proste wzory (asymetria α, β/α) są interpretowalne dla uczestnika („zwiększ lewą aktywność czołową”) i mogą służyć jako fallback/druga ścieżka feedbacku obok klasyfikatora.
- **Normalizacja względem baseline** (różnica zadanie − spoczynek) – warto nagrać krótki baseline na początku sesji testowej. Uwaga: baseline z zamkniętymi oczami ma zawyżoną α – do porównań z oglądaniem obrazów lepszy baseline z otwartymi oczami.

## 5. Ograniczenia / zastrzeżenia

- **Wysokie trafności (91–97%) są mało wiarygodne**: 134 próbki × 537 cech, pojedynczy podział 70/30, brak CV, brak informacji o separacji uczestników, strojenie hiperparametrów bez opisanego zbioru walidacyjnego – duże ryzyko przeuczenia/wycieku. Nie należy tych liczb traktować jako realnego punktu odniesienia dla nas.
- Klasy niezrównoważone, a etykieta jest jedna na całe 60 s VE – to klasyfikacja „sesji”, nie pojedynczych prób jak u nas.
- Inny sprzęt (14 kanałów Emotiv, 128 Hz, elektrody czołowe) – kluczowe elektrody F3/F4 niedostępne w Unicornie.
- Opis preprocessingu skąpy: brak częstotliwości filtru, metody usuwania artefaktów, referencji.
- Środowiska VR 60 s wywołują silniejsze i dłuższe emocje niż nasze obrazy 1 s – efekty u nas będą słabsze.
- Wnioski o θ/β/γ są opisowe (mediany), bez testów statystycznych między warunkami.
- Model zbiorczy, nie osobniczy – nie testuje naszego scenariusza (klasyfikator per uczestnik).
