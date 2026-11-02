# Pokazówka wyników i Dashboard Databricks AI/BI (`EuroSAT.lvdash.json`)

Na podstawie tabel z warstw **Bronze** oraz **Gold** w Unity Catalog przygotowano interaktywny dashboard analityczny Databricks AI/BI (jego definicja znajduje się w pliku [`Photos/EuroSAT.lvdash.json`](../Photos/EuroSAT.lvdash.json)). Poniżej znajduje się przegląd wyników uzyskanych po przetworzeniu **20 paczek strumieniowych (9,5 tys. zdjęć treningowych)**.

---

## 1. Główne wskaźniki KPI oraz krzywe uczenia (Error Rate i Test Log-Loss)

![Główny widok dashboardu - KPI, Error Rate i Log-Loss](../Photos/Zrzut%20ekranu%202026-11-02%20203238.png)

- **Wskaźniki podsumowujące:**
  - **Executed batches:** `20` przetworzonych paczek danych.
  - **Count of training pictures:** `9,5 tys.` zdjęć wykorzystanych w procesie douczania.
  - **Minimum error rate:** `0,05` (błąd klasyfikacji spadł do poziomu **5%**).
  - **Maximum accuracy:** `0,95` (**95%** skuteczności na zbalansowanym zbiorze testowym).
- **Trend uczenia przyrostowego:** Wykresy `Error rate for each model` oraz `Test log loss for each model` wyraźnie pokazują systematyczny spadek błędu klasyfikacji (z ok. `11–15%` w pierwszej paczce do `5–7%` w kolejnych paczkach) oraz stabilizację funkcji straty Log-Loss wraz z napływem nowych danych.

---

## 2. Ewolucja miary F1-Score oraz skuteczność klasyfikacji per klasa

![Średni wynik F1 po paczkach oraz metryki dla każdej klasy](../Photos/Zrzut%20ekranu%202026-11-02%20202244.png)

- **Average F1 score for each model:** Wraz z kolejnymi paczkami wszystkie trzy modele poprawiają wynik `F1-macro` z poziomu `0,85–0,89` (paczka `#1`) do okolic **`0,93–0,95`** (od paczki `#13` wzwyż). Najwyższą wartość szczytową osiąga sieć wielowarstwowa `PyTorch_Deep_MLP`.
- **Results of classification for each class:** Zestawienie średnich wartości `Accuracy`, `F1`, `Precision` i `Recall` pokazuje bardzo wysoką skuteczność dla klas o wyrazistej strukturze (`Forest` – 0,98, `Residential` – 0,98, `Industrial` – 0,97, `AnnualCrop` – 0,96) oraz nieco większą trudność dla klas wizualnie zbliżonych (`HerbaceousVegetation` – 0,79, `Highway` – 0,84).

---

## 3. Czas treningu oraz pewność predykcji (Confidence Analysis)

![Czas treningu oraz średnia pewność predykcji poprawnych i błędnych](../Photos/Zrzut%20ekranu%202026-11-02%20203258.png)

- **Average training time for each model:**
  - `Sklearn_SGD_LogReg` trenuje się najszybciej (ułamki sekund na paczkę dzięki operacji `partial_fit`).
  - `PyTorch_Linear_Head` potrzebuje średnio poniżej `1 s` na paczkę (12 epok).
  - `PyTorch_Deep_MLP` wymaga ok. `3,2 s` na paczkę (15 epok z warstwami `BatchNorm1d` i `Dropout`), oferując w zamian najwyższą dokładność i najlepszą kalibrację pewności.
- **Average confidence when wrong and correct:**
  - Wszystkie modele wykazują zdrową separację pewności między predykcjami trafnymi a błędnymi (np. dla `PyTorch_Deep_MLP` średnia pewność przy poprawnej odpowiedzi wynosi **`0,97`**, natomiast przy pomyłce spada do **`0,74`**).

---

## 4. Porównanie dokładności modeli w rozbiciu na klasy

![Porównanie dokładności trzech modeli dla każdej z 10 klas](../Photos/Zrzut%20ekranu%202026-11-02%20202351.png)

Widok `Per-Class Classification Performance & Error Analysis` wyposażony jest w interaktywne filtry (`Batch number`, `Model name`, `Class name`) i pozwala bezpośrednio zestawić dokładność cząstkową (`class_accuracy`) każdego z trzech modeli dla każdej z 10 klas pokrycia terenu.

---

## 5. Macierz pomyłek (Confusion Matrix)

![Macierz pomyłek dla modelu PyTorch_Deep_MLP](../Photos/Zrzut%20ekranu%202026-11-02%20202335.png)

Zagregowana macierz pomyłek wyliczona na podstawie tabeli `gold_model_predictions` pozwala zidentyfikować naturalne podobieństwa spektralne między klasami zdjęć satelitarnych (np. sporadyczne mylenie `PermanentCrop` z `HerbaceousVegetation` lub `River` z `Highway`).

---

## 6. Statystyki fizyczne zdjęć satelitarnych w warstwie Bronze

![Profile kolorów RGB oraz średnia jasność klas w warstwie Bronze](../Photos/Zrzut%20ekranu%202026-11-02%20203341.png)

Sekcja oparta na tabeli `bronze_batch_stats` prezentuje charakterystykę fizyczną surowych obrazów Sentinel-2:
- **RGB color breakdown in each class:** Rozkład średnich wartości kanałów Red, Green i Blue dla poszczególnych typów terenu.
- **Average brightness for each class:** Wyraźne różnice jasności między ciemnymi obszarami leśnymi i wodnymi (`Forest` – `60,1`, `SeaLake` – `70,37`) a jasnymi terenami zurbanizowanymi i rolniczymi (`Industrial` – `127,3`, `AnnualCrop` – `123,12`).

