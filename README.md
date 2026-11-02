# Databricks Vision MLOps – Przyrostowa Klasyfikacja Zdjęć Satelitarnych EuroSAT

Kompletny potok **Continual / Incremental Vision MLOps** zrealizowany w postaci modularnych notebooków na platformie **Databricks** (z wykorzystaniem **Unity Catalog**, architektury **Medallion (Bronze / Silver / Gold)** w **Delta Lake**, śledzenia eksperymentów w **MLflow**, harmonogramowania zadań **Databricks Workflows (Serverless Jobs)** oraz wizualizacji w **Databricks AI/BI Dashboards**).

---

## Spis treści i szczegółowa dokumentacja (`docs/`)

Szczegółowy opis poszczególnych części systemu został podzielony na dedykowane pliki Markdown w katalogu [`docs/`](docs/):

1. **[Opis zbioru danych EuroSAT RGB i cel projektu](docs/01_dataset_and_problem.md)** – źródło danych z Hugging Face, charakterystyka 10 klas pokrycia terenu Sentinel-2 oraz strategia strumieniowego pobierania paczek i budowy zbalansowanego zbioru Holdout.
2. **[Architektura potoku MLOps, Notebooki i Databricks Workflow Job](docs/02_pipeline_and_job.md)** – szczegółowy opis 5 notebooków (`00`–`04`), przepływ sterowania, kontrola jakości (Data Quality) oraz konfiguracja zadania Databricks Job.
3. **[Ekstrakcja cech, specyfikacja klasyfikatorów i uczenie przyrostowe](docs/03_models_and_incremental_learning.md)** – działanie zamrożonego ekstraktora `MobileNetV3-Small` (`576` cech), mechanizm bufora pamięci `Experience Replay` oraz pełna tabela hiperparametrów 3 klasyfikatorów (`PyTorch` i `Scikit-Learn`).
4. **[Baza danych Unity Catalog (Medallion) i wyliczane wartości](docs/04_database_schema_and_metrics.md)** – schemat relacji i pełny słownik wszystkich 8 tabel Delta wraz ze sposobem obliczania statystyk RGB/jasności oraz metryk ML.
5. **[Pokazówka wyników i Dashboard Databricks AI/BI](docs/05_dashboard_and_results_gallery.md)** – galeria zrzutów ekranu z uruchomionego systemu po przetworzeniu 20 paczek (`9,5 tys.` zdjęć), krzywe uczenia, macierz pomyłek oraz analiza profili kolorystycznych.

---

## 1. Zbiór danych (Dataset)

- **Nazwa zbioru:** EuroSAT RGB (`giswqs/EuroSAT_RGB`)
- **Link do zbioru na Hugging Face:** [https://huggingface.co/datasets/giswqs/EuroSAT_RGB](https://huggingface.co/datasets/giswqs/EuroSAT_RGB)
- **Charakterystyka:** Zdjęcia satelitarne RGB z misji **Sentinel-2** podzielone na **10 klas pokrycia terenu**:
  `AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`.
- **Sposób przetwarzania:**
  - **Zbiór testowy Holdout (`500` zdjęć):** Pobierany jednorazowo ze splitu `test` w sposób idealnie zbalansowany (**po 50 zdjęć na każdą z 10 klas**) i zapisywany na stałe w tabeli `silver_test_holdout`.
  - **Strumieniowe paczki treningowe (`500` zdjęć na paczkę):** Pobierane cyklicznie ze splitu `train` w trybie `streaming=True` za pomocą `.skip(current_offset).take(500)`.

> Szczegóły: [docs/01_dataset_and_problem.md](docs/01_dataset_and_problem.md)

---

## 2. Co jest robione i jak działa potok na platformie Databricks

Projekt został zaimplementowany w **5 notebookach** w katalogu [`notebooks/`](notebooks/) z centralną konfiguracją w pliku [`config.yaml`](config.yaml):

| Notebook | Warstwa / Rola | Co jest robione w danej sekcji |
| :--- | :--- | :--- |
| **[`00_setup_and_holdout_test.ipynb`](notebooks/00_setup_and_holdout_test.ipynb)** | **Inicjalizacja (jednorazowa)** | Tworzy Volume `/Volumes/workspace/default/raw_data` oraz tabelę stanu `pipeline_state` (`offset = 0`). Pobiera po `50` zdjęć testowych dla każdej z `10` klas, przepuszcza je przez `MobileNetV3-Small` i zapisuje `500` wektorów cech w tabeli `silver_test_holdout`. |
| **[`01_ingest_bronze_batch.ipynb`](notebooks/01_ingest_bronze_batch.ipynb)** | **Warstwa Bronze** | Odczytuje `last_offset` z `pipeline_state`, pobiera strumieniowo `500` kolejnych zdjęć treningowych do katalogu tymczasowego w Volume, oblicza dla każdego zdjęcia wymiary, średnie wartości kanałów `R, G, B` oraz jasność (`brightness`) i zapisuje je w tabeli `bronze_batch_stats`. |
| **[`02_silver_extract_features.ipynb`](notebooks/02_silver_extract_features.ipynb)** | **Warstwa Silver** | Przeprowadza **kontrolę jakości (Data Quality)** – odrzuca pliki uszkodzone, zbyt ciemne (`brightness < 5.0`) lub prześwietlone (`brightness > 250.0`). Dla poprawnych zdjęć ekstrahuje **wektory 576 cech** przy użyciu zamrożonej sieci `MobileNetV3-Small`, zapisuje je do `silver_current_batch` oraz aktualizuje bufor pamięci `silver_replay_buffer` (do `100` historycznych wektorów przeciw zjawisku *catastrophic forgetting*). |
| **[`03_train_and_compare_mlflow.ipynb`](notebooks/03_train_and_compare_mlflow.ipynb)** | **Warstwa Gold + MLflow** | Łączy wektory bieżącej paczki z `silver_replay_buffer`, wczytuje checkpointy wag z poprzedniego przebiegu i doucza przyrostowo **3 modele klasyfikacyjne**. Loguje parametry, czasy, metryki i pliki wag w **MLflow**, a szczegółowe wyniki zapisuje w 3 tabelach warstwy Gold. |
| **[`04_cleanup_batch.ipynb`](notebooks/04_cleanup_batch.ipynb)** | **Czyszczenie i Stan** | Usuwa fizyczne pliki `.jpg` bieżącej paczki z Volume, czyści tabelę tymczasową `silver_current_batch` (`TRUNCATE TABLE`) i dopisuje nowy rekord w `pipeline_state` (`last_offset + 500`), przygotowując potok do pobrania kolejnej paczki. |

### Automatyzacja jako Databricks Workflow Job

Cztery notebooki operacyjne (`01` → `02` → `03` → `04`) działają jako zautomatyzowany potok zadań (**Databricks Job**) uruchamiany na zasobach **Serverless**:

![Databricks Workflow Job DAG](Photos/Zrzut%20ekranu%202026-11-02%20204508.png)

> Szczegóły: [docs/02_pipeline_and_job.md](docs/02_pipeline_and_job.md)

---

## 3. Specyfikacja ekstraktora cech i porównywanych klasyfikatorów

### Ekstraktor cech obrazu (Backbone)
- **Sieć:** `MobileNetV3-Small` (`MobileNet_V3_Small_Weights.DEFAULT`), z odciętą warstwą klasyfikującą (`backbone.classifier = nn.Identity()`) i zamrożonymi wagami (`requires_grad = False`).
- **Wejście:** Obraz RGB przeskalowany do `224x224` i znormalizowany (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- **Wyjście:** Wektor `576` cech zmiennoprzecinkowych (`embedding_dim = 576`).

### Porównywane klasyfikatory i ich hiperparametry

| Model (`model_name`) | Rodzina | Architektura i warstwy | Optymalizator i funkcja straty | Epoki na paczkę | Współczynnik uczenia (`learning_rate`) | Mechanizm przyrostowy (Incremental) |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| **`PyTorch_Linear_Head`** | `PyTorch` | `nn.Linear(576, 10)` | `AdamW` (`weight_decay=1e-4`), `CrossEntropyLoss`, `batch_size=64` | `12` | Paczka `#1`: `0.002`<br/>Paczka `>= #2`: `0.001` (`0.5 * base_lr`) | Zapis/odczyt wag `state_dict` (`.pt`) + Replay Buffer |
| **`PyTorch_Deep_MLP`** | `PyTorch` | `Linear(576, 256)` → `BatchNorm1d(256)` → `ReLU` → `Dropout(0.3)` → `Linear(256, 10)` | `AdamW` (`weight_decay=1e-4`), `CrossEntropyLoss`, `batch_size=64` | `15` | Paczka `#1`: `0.0015`<br/>Paczka `>= #2`: `0.00075` (`0.5 * base_lr`) | Zapis/odczyt wag `state_dict` (`.pt`) + Replay Buffer |
| **`Sklearn_SGD_LogReg`** | `Scikit-Learn` | `SGDClassifier(loss="log_loss", random_state=42)` | Stochastyczny spadek gradientowy (`learning_rate="adaptive"`) | `5` | `eta0 = 0.01` (adaptacyjny) | Wywołania `partial_fit()` + serializacja `joblib` (`.pkl`) + Replay Buffer |

> Szczegóły: [docs/03_models_and_incremental_learning.md](docs/03_models_and_incremental_learning.md)

---

## 4. Baza danych Unity Catalog (`workspace.default`) i wyliczane wartości

W projekcie utrzymywanych jest **8 tabel Delta Lake** podzielonych zgodnie z architekturą Medallion:

| Warstwa | Nazwa tabeli Delta | Jakie wartości są liczone i przechowywane |
| :--- | :--- | :--- |
| **Sterowanie** | `pipeline_state` | Nazwa zbioru (`dataset_name`), numer ostatniej paczki (`last_batch_id`), przesunięcie w strumieniu (`last_offset`) oraz czas aktualizacji (`updated_at`). |
| **Bronze** | `bronze_batch_stats` | Metadane pliku (`batch_id`, `global_idx`, `file_path`, `label_id`, `class_name`), wymiary obrazu (`width`, `height`), średnie wartości kanałów kolorystycznych **`mean_r`, `mean_g`, `mean_b`** oraz ogólna jasność **`brightness`**. |
| **Silver** | `silver_test_holdout` | Stały zbiór testowy `500` obrazów (po `50` na klasę) z wyliczonymi wektorami **`features` (`ARRAY<DOUBLE>` o długości `576`)**. |
| **Silver** | `silver_current_batch` | Zwalidowane wektory `576` cech (`features`) zdjęć z bieżącej paczki treningowej (po odrzuceniu obrazów uszkodzonych lub o skrajnej jasności). |
| **Silver** | `silver_replay_buffer` | Bufor pamięci przechowujący do `100` losowo wybranych wektorów cech z poprzednich paczek (po 30 próbek losowanych z każdej paczki). |
| **Gold** | `gold_model_runs` | Ogólne wyniki każdego modelu po każdej paczce: czasy (**`train_time_sec`, `inference_time_sec`, `total_time_sec`**), błędy (**`train_loss`, `test_log_loss`, `test_error_rate`**), metryki jakości (**`test_accuracy`, `test_f1_macro`, `test_precision_macro`, `test_recall_macro`**), średnia pewność predykcji trafnych i błędnych (**`avg_conf_when_correct`, `avg_conf_when_wrong`**) oraz `mlflow_run_id`. |
| **Gold** | `gold_model_class_metrics` | Skuteczność każdego modelu w rozbiciu na każdą z 10 klas: **`class_accuracy`, `class_error_rate`, `class_log_loss`, `class_precision`, `class_recall`, `class_f1`**. |
| **Gold** | `gold_model_predictions` | Pojedyncze predykcje każdego modelu na wszystkich `500` zdjęciach testowych: klasa prawdziwa i przewidziana, flaga trafienia (`is_correct`, `classification_error`), strata jednostkowa (**`sample_log_loss`**) oraz pewność modelu (**`confidence`**). |

> Pełny schemat kolumn i wzorów: [docs/04_database_schema_and_metrics.md](docs/04_database_schema_and_metrics.md)

---

## 5. Pokazówka zdjęć – Dashboard Databricks AI/BI i wyniki eksperymentów

W katalogu [`Photos/`](Photos/) znajduje się wyeksportowany szablon dashboardu Databricks Lakeview ([`Photos/EuroSAT.lvdash.json`](Photos/EuroSAT.lvdash.json)) oraz zrzuty ekranu dokumentujące działanie potoku po przetworzeniu **20 paczek (9,5 tys. zdjęć)**:

### Podsumowanie uczenia przyrostowego (KPI, spadek Error Rate i Log-Loss)
![Główny widok dashboardu - KPI, Error Rate i Log-Loss](Photos/Zrzut%20ekranu%202026-11-02%20203238.png)

### Wzrost F1-Score w kolejnych paczkach oraz metryki dla poszczególnych klas
![Średni wynik F1 po paczkach oraz metryki dla każdej klasy](Photos/Zrzut%20ekranu%202026-11-02%20202244.png)

### Porównanie czasów treningu oraz pewności predykcji (Confidence)
![Czas treningu oraz średnia pewność predykcji poprawnych i błędnych](Photos/Zrzut%20ekranu%202026-11-02%20203258.png)

### Dokładność modeli w podziale na 10 klas pokrycia terenu
![Porównanie dokładności trzech modeli dla każdej z 10 klas](Photos/Zrzut%20ekranu%202026-11-02%20202351.png)

### Macierz pomyłek (Confusion Matrix)
![Macierz pomyłek dla modelu PyTorch_Deep_MLP](Photos/Zrzut%20ekranu%202026-11-02%20202335.png)

### Warstwa Bronze – profile RGB i średnia jasność klas satelitarnych
![Profile kolorów RGB oraz średnia jasność klas w warstwie Bronze](Photos/Zrzut%20ekranu%202026-11-02%20203341.png)

> Szczegółowy opis wszystkich wykresów: [docs/05_dashboard_and_results_gallery.md](docs/05_dashboard_and_results_gallery.md)

---

## 6. Jak uruchomić projekt na Databricks

1. Zaimportuj repozytorium do **Databricks Workspace (Repos / Git Folders)**.
2. Zweryfikuj parametry docelowego katalogu i schematu w pliku [`config.yaml`](config.yaml) (domyślnie `catalog_name: "workspace"`, `schema_name: "default"`, `volume_name: "raw_data"`).
3. Uruchom jednorazowo notebook inicjalizacyjny **[`notebooks/00_setup_and_holdout_test.ipynb`](notebooks/00_setup_and_holdout_test.ipynb)**, który utworzy Volume, tabelę `pipeline_state` oraz zbalansowany zbiór testowy `silver_test_holdout`.
4. Skonfiguruj w **Databricks Workflows** zadanie (Job) składające się z 4 kolejnych kroków:
   `01_ingest_bronze_batch` → `02_silver_extract_features` → `03_train_and_compare_mlflow` → `04_cleanup_batch`.
5. Zaimportuj plik [`Photos/EuroSAT.lvdash.json`](Photos/EuroSAT.lvdash.json) w sekcji **Dashboards (AI/BI)**, aby monitorować postęp uczenia modeli i statystyki obrazów na żywo.
