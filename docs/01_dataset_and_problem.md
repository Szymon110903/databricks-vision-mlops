# Zbiór danych EuroSAT RGB i cel projektu

## 1. Źródło i link do zbioru danych

Projekt wykorzystuje zbiór zdjęć satelitarnych **EuroSAT RGB** dostępny na platformie Hugging Face:

- **Link do zbioru danych:** [https://huggingface.co/datasets/giswqs/EuroSAT_RGB](https://huggingface.co/datasets/giswqs/EuroSAT_RGB)
- **Identyfikator w konfiguracji (`config.yaml`):** `giswqs/EuroSAT_RGB`
- **Pochodzenie obrazów:** Europejska Agencja Kosmiczna – misja satelitarna **Sentinel-2** (kanały widzialne RGB).

---

## 2. Charakterystyka danych i klas

Zbiór składa się z wycinków zdjęć satelitarnych o wymiarach `64x64` pikseli (skalowanych w potoku do `224x224` pikseli na wejście sieci `MobileNetV3-Small`), przedstawiających **10 klas pokrycia i użytkowania terenu (Land Use and Land Cover – LULC)**:

| ID klasy (`label_id`) | Nazwa klasy (`class_name`) | Opis pokrycia terenu |
| :---: | :--- | :--- |
| `0` | **AnnualCrop** | Uprawy jednoroczne (pola uprawne sezonowe) |
| `1` | **Forest** | Obszary leśne |
| `2` | **HerbaceousVegetation** | Roślinność zielna, łąki naturalne i wrzosowiska |
| `3` | **Highway** | Autostrady i główne drogi szybkiego ruchu |
| `4` | **Industrial** | Tereny przemysłowe, hale produkcyjne i magazynowe |
| `5` | **Pasture** | Pastwiska i tereny wypasu |
| `6` | **PermanentCrop** | Uprawy trwałe (sady, winnice, plantacje wieloletnie) |
| `7` | **Residential** | Zabudowa mieszkaniowa (jednorodzinna i wielorodzinna) |
| `8` | **River** | Rzeki i cieki wodne |
| `9` | **SeaLake** | Morza, jeziora i duże zbiorniki wodne |

---

## 3. Strategia pobierania i podziału danych na Databricks

Zamiast jednorazowego pobierania całego zbioru danych do pamięci klastra, projekt symuluje **produkcyjny strumień napływających danych obrazowych (Incremental / Continual Learning)**:

1. **Stały, zbalansowany zbiór testowy (Holdout Test Set – `500` obrazów):**
   - Tworzony jednorazowo w notebooku [`00_setup_and_holdout_test.ipynb`](../notebooks/00_setup_and_holdout_test.ipynb) ze splitu `test`.
   - Zawiera dokładnie **po 50 obrazów dla każdej z 10 klas** (`500 // 10 = 50`).
   - Zapisany na stałe w tabeli `silver_test_holdout`, dzięki czemu każdy model po każdej kolejnej paczce treningowej jest oceniany na identycznym, w pełni zbalansowanym zbiorze referencyjnym.
2. **Strumieniowe paczki treningowe (`batch_size = 500` obrazów):**
   - Pobierane ze splitu `train` w trybie `streaming=True` za pomocą `.skip(current_offset).take(500)` w notebooku [`01_ingest_bronze_batch.ipynb`](../notebooks/01_ingest_bronze_batch.ipynb).
   - Po przetworzeniu paczki surowe pliki `.jpg` są usuwane z Volume, a licznik `last_offset` w tabeli `pipeline_state` przesuwa się o kolejne `500` zdjęć.

