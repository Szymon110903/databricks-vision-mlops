# Ekstrakcja cech, specyfikacja klasyfikatorów i uczenie przyrostowe

## 1. Ekstraktor cech (Backbone): `MobileNetV3-Small`

Zamiast trenować od zera ciężką sieć konwolucyjną na każdej paczce zdjęć, zastosowano podejście **Transfer Learning / Feature Extraction**:

- **Model bazowy:** `torchvision.models.mobilenet_v3_small` z pre-trenowanymi wagami `MobileNet_V3_Small_Weights.DEFAULT` (ImageNet-1K).
- **Modyfikacja architektury:** oryginalna głowica klasyfikująca została zastąpiona warstwą tożsamościową (`backbone.classifier = nn.Identity()`), a wszystkie wagi zostały zamrożone (`requires_grad = False`, tryb `eval()`).
- **Potok przekształceń obrazu (`preprocess`):**
  - przeskalowanie obrazu do wymiaru `224 x 224` (`transforms.Resize((224, 224))`),
  - konwersja do tensora (`transforms.ToTensor()`),
  - normalizacja kanałów RGB według statystyk ImageNet: `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`.
- **Wyjście ekstraktora:** wektor zmiennoprzecinkowy o długości **`576` cech** (`embedding_dim = 576`) dla każdego zdjęcia.

---

## 2. Mechanizm przeciw zapominaniu: Experience Replay (`silver_replay_buffer`)

W uczeniu przyrostowym (Incremental Learning) modele douczane wyłącznie na nowej paczce danych są narażone na zjawisko **katastrofalnego zapominania (*catastrophic forgetting*)**. Aby temu zapobiec:

1. Z każdej przetworzonej paczki w warstwie Silver losowane jest do **`30` wektorów cech** (`random.sample`).
2. Są one dołączane do tabeli `silver_replay_buffer`, która przechowuje maksymalnie **`100` historycznych wektorów** (`replay_buffer_size: 100`). Jeśli po dodaniu nowych próbek bufor przekracza `100` rekordów, wykonywane jest losowe próbkowanie bez zwracania do rozmiaru `100`.
3. Począwszy od paczki `#2`, zbiór treningowy przekazywany do klasyfikatorów składa się z:
   - bieżącej paczki z `silver_current_batch` (`500` nowych wektorów),
   - bufora historycznego z `silver_replay_buffer` (`do 100` wektorów z poprzednich paczek),
   - co daje łącznie **`530–600` próbek treningowych** na pojedynczy cykl nauki.

---

## 3. Specyfikacja porównywanych klasyfikatorów

W notebooku [`03_train_and_compare_mlflow.ipynb`](../notebooks/03_train_and_compare_mlflow.ipynb) trenowane, douczane i porównywane są **trzy modele** z dwóch rodzin bibliotek (`PyTorch` oraz `Scikit-Learn`):

| Parametr / Cecha | `PyTorch_Linear_Head` | `PyTorch_Deep_MLP` | `Sklearn_SGD_LogReg` |
| :--- | :--- | :--- | :--- |
| **Rodzina (`model_family`)** | `PyTorch` | `PyTorch` | `Scikit-Learn` |
| **Architektura (`architecture`)** | `Linear(576->10)` | `MLP(576->256->10, BN, Dropout=0.3)` | `SGDClassifier(loss=log_loss, partial_fit)` |
| **Szczegóły warstw / estymatora** | Pojedyncza warstwa gęsta `nn.Linear(576, 10)` | `Linear(576, 256)` → `BatchNorm1d(256)` → `ReLU()` → `Dropout(0.3)` → `Linear(256, 10)` | Liniowy klasyfikator regresji logistycznej uczony stochastycznym spadkiem wzdłuż gradientu (`loss="log_loss"`) |
| **Optymalizator / Algorytm** | `AdamW` (`weight_decay = 1e-4`) | `AdamW` (`weight_decay = 1e-4`) | `SGD` (`learning_rate="adaptive"`, `random_state=42`) |
| **Funkcja straty** | `CrossEntropyLoss` | `CrossEntropyLoss` | `log_loss` (entropia krzyżowa) |
| **Liczba epok na paczkę (`epochs`)** | `12` | `15` | `5` (kolejnych wywołań `partial_fit`) |
| **Rozmiar mini-batcha** | `64` (`DataLoader(..., shuffle=True)`) | `64` (`DataLoader(..., shuffle=True)`) | Pełna paczka przekazywana do `partial_fit` |
| **Bazowy współczynnik uczenia (`base_lr`)** | `0.002` | `0.0015` | `0.01` (`eta0`) |
| **Adaptacja LR w kolejnych paczkach** | Paczka `#1`: `0.002`<br/>Paczka `>= #2`: `0.001` (`0.5 * base_lr`) | Paczka `#1`: `0.0015`<br/>Paczka `>= #2`: `0.00075` (`0.5 * base_lr`) | Tryb `adaptive` wewnątrz `SGDClassifier` (start od `eta0 = 0.01`) |
| **Format zapisu wag (Checkpoint)** | `/Volumes/.../model_checkpoints/PyTorch_Linear_Head.pt` | `/Volumes/.../model_checkpoints/PyTorch_Deep_MLP.pt` | `/Volumes/.../model_checkpoints/Sklearn_SGD_LogReg.pkl` |

---

## 4. Uzasadnienie doboru hiperparametrów dla poszczególnych modeli

Różnice w liczbie epok (`12` vs `15` vs `5`) oraz współczynnikach uczenia (`0.002` vs `0.0015` vs `0.01`) wynikają bezpośrednio z odmiennej złożoności architektur oraz sposobu optymalizacji w każdym z trzech modeli:

1. **`PyTorch_Linear_Head` (`epochs = 12`, `base_lr = 0.002`):**
   - Posiada wyłącznie jedną warstwę liniową (`576 × 10 + 10 = 5 770` parametrów) bez nieliniowości i bez `Dropoutu`.
   - Ze względu na wypukłą powierzchnię funkcji straty (identyczną z wieloklasową regresją logistyczną) model zbiega szybciej i bezpiecznie toleruje nieco wyższy krok uczenia (`base_lr = 0.002` z optymalizatorem `AdamW`) oraz umiarkowaną liczbę **12 epok** na paczkę, bez ryzyka niestabilności gradientów.

2. **`PyTorch_Deep_MLP` (`epochs = 15`, `base_lr = 0.0015`, `Dropout = 0.3`, `BatchNorm1d`):**
   - Zawiera dodatkową warstwę ukrytą o wymiarze `256` (`576 × 256 + 256 × 10 ≈ 150 tys.` parametrów – ponad **25× więcej wag** niż `LinearHead`).
   - Obecność losowego wygaszania neuronów **`Dropout(0.3)`** (30% neuronów wyłączanych w każdym kroku treningowym) celowo utrudnia dopasowanie do pojedynczej paczki, aby zapobiec przeuczeniu (*overfitting*) na małej próbce 500–600 zdjęć. Z tego powodu sieć potrzebuje **większej liczby epok (`15`)**, aby w pełni wyuczyć reprezentacje w warstwie ukrytej.
   - Zastosowano nieco **niższy współczynnik uczenia (`base_lr = 0.0015`)**, ponieważ w głębszej architekturze z nieliniowością `ReLU` i normalizacją `BatchNorm1d` zbyt duży krok początkowy powodowałby gwałtowne oscylacje wag i niszczenie cech wyuczonych w poprzednich paczkach.

3. **`Sklearn_SGD_LogReg` (`epochs = 5`, `base_lr / eta0 = 0.01`, `learning_rate = "adaptive"`):**
   - W klasycznym `SGDClassifier` (bez momentów pędu drugiego rzędu, jakie ma `AdamW`) aktualizacja wag odbywa się czystym stochastycznym spadkiem gradientowym po każdej pojedynczej próbce wewnątrz `partial_fit`.
   - Z tego względu punkt startowy kroku uczenia dla zwykłego SGD musi być wyższy (`eta0 = 0.01`) niż dla `AdamW` (`0.0015–0.002`), a tryb `learning_rate="adaptive"` automatycznie zmniejsza krok, gdy błąd przestaje maleć.
   - Ponieważ `partial_fit` wykonuje aktualizację wag po każdym pojedynczym wektorze w paczce (500–600 aktualizacji na jedno wywołanie), już **5 przejść (`epochs = 5`)** wystarcza do zbieżności na bieżącej paczce, a większa liczba epok prowadziłaby do nadmiernego dopasowania do ostatniej paczki kosztem wcześniejszych.

4. **Redukcja `learning_rate` o połowę (`0.5 * base_lr`) od paczki `#2` wzwyż (modele PyTorch):**
   - W pierwszej paczce (`#1`) wagi sieci są losowe, więc potrzebny jest pełny krok `base_lr`, aby szybko znaleźć właściwy obszar przestrzeni wag.
   - W kolejnych paczkach (`>= #2`) model posiada już ukształtowaną wiedzę z poprzednich tysięcy zdjęć. Zmniejszenie kroku uczenia o 50% (`0.001` dla `LinearHead` i `0.00075` dla `DeepMLP`) działa jak **łagodny fine-tuning** – pozwala przyswoić nowe wzorce z bieżącej paczki bez agresywnego wymazywania dotychczas wyuczonych granic decyzyjnych.

5. **Regularyzacja `weight_decay = 1e-4` w `AdamW`:**
   - Ogranicza nadmierny wzrost wartości wag przy wielokrotnym douczaniu na kolejnych 20+ paczkach, co przekłada się na lepszą kalibrację prawdopodobieństw (niższy `test_log_loss` widoczny na wykresach).

---

## 5. Mechanizm douczania przyrostowego (Incremental Checkpoint Loading)

- **Dla modeli PyTorch (`LinearHead` oraz `DeepMLPHead`):**
  - Przy paczce `#1` wagi są inicjalizowane losowo, a trening przebiega z bazowym współczynnikiem uczenia `base_lr`.
  - Po zakończeniu treningu słownik wag (`model.state_dict()`) zapisywany jest do pliku `.pt` w Volume oraz logowany jako artefakt w MLflow.
  - Przy paczkach `#2, #3, ...` skrypt wykrywa istniejący plik `.pt`, wczytuje wagi (`model.load_state_dict(torch.load(ckpt_path))`), ustawia flagę `incremental_loaded = True` i redukuje współczynnik uczenia o połowę (`lr = base_lr * 0.5`), aby stabilnie dostroić wagi bez gwałtownego nadpisywania wcześniejszej wiedzy.
- **Dla modelu Scikit-Learn (`SGDClassifier`):**
  - Wykorzystywana jest natywna metoda przyrostowa `clf.partial_fit(X_train, y_train, classes=np.arange(10))`.
  - Obiekt klasyfikatora wraz ze stanem optymalizatora jest serializowany przez `joblib.dump` do pliku `.pkl` i wczytywany przed treningiem kolejnej paczki.

