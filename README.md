[README.md](https://github.com/user-attachments/files/26930900/README.md)
# EuroMillions-Predictor
EuroMillions LSTM Hybrid Predictor
# EuroMillions LSTM Hybrid Predictor

Experimental project for generating suggested EuroMillions combinations using LSTM neural networks and a recent-frequency ensemble.

## Overview

This project loads historical EuroMillions draw results from a CSV file and builds a multi-step modeling pipeline:

- separates main numbers and Lucky Stars
- trains independent LSTM models
- evaluates multiple time-window lengths
- blends neural network output with a recent-frequency baseline
- compares configurations using historical hit-based metrics
- generates a final suggested combination and a top 5 list of combinations

The goal is to explore time-series modeling and hybrid ensemble techniques on lottery-history data. It is **not** intended to claim reliable predictive power on a fundamentally random process.

---

## Features

- Uses real historical EuroMillions results from CSV
- Treats separately:
  - 5 main numbers
  - 2 Lucky Stars
- Trains separate LSTM models for each target group
- Tests multiple temporal windows (`window_length`)
- Combines:
  - LSTM output
  - recent-frequency scoring
- Evaluates with more meaningful metrics than only MSE/MAE:
  - average main-number hits
  - average star hits
  - % of draws with at least 1 correct number
  - % of draws with at least 1 correct star
  - % of draws with 2 or more correct numbers
  - % of draws with 2 correct stars
- Outputs:
  - raw rounded network prediction
  - final blended combination
  - top 5 suggested combinations
  - optional CSV exports

---

## Project Structure

```bash
.
├── EuroMillions_numbers.csv
├── euromillones.ipynb
├── euromillones_corregido.ipynb
├── euromillones_v2_separado.ipynb
├── euromillones_v3_optimizacion.ipynb
├── euromillones_v4_top5_export.ipynb
└── README.md
```

### Notebook Versions

- **euromillones.ipynb**: original version
- **euromillones_corregido.ipynb**: corrected baseline version
- **euromillones_v2_separado.ipynb**: separate modeling for numbers and stars
- **euromillones_v3_optimizacion.ipynb**: optimization of windows and ensemble weights
- **euromillones_v4_top5_export.ipynb**: top 5 combinations, raw-vs-final comparison, optional CSV export

---

## Dataset

The CSV file should contain these columns:

- `Date`
- `N1`, `N2`, `N3`, `N4`, `N5`
- `E1`, `E2`

Example:

```csv
Date;N1;N2;N3;N4;N5;E1;E2
2024-01-02;8;13;24;29;42;5;10
```

The notebooks use:

- `N1, N2, N3, N4, N5` for the main numbers
- `E1, E2` for the Lucky Stars

---

## Requirements

Install the main dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

If you are using Jupyter:

```bash
pip install notebook
```

On some Windows environments, TensorFlow may need a CPU-only install:

```bash
pip install tensorflow-cpu
```

---

## How to Run

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Make sure the CSV is in the project folder
The file should be named:

```bash
EuroMillions_numbers.csv
```

### 3. Launch Jupyter
```bash
jupyter notebook
```

### 4. Open the most complete notebook
Recommended notebook:

```bash
euromillones_v4_top5_export.ipynb
```

---

## V4 Notebook Workflow

### 1. Data loading
Reads the CSV file and separates:
- main numbers
- Lucky Stars

### 2. Preprocessing
- scales features with `StandardScaler`
- builds rolling temporal sequences for LSTM input

### 3. Training
Trains two networks:
- one for main numbers
- one for stars

### 4. Optimization
Tests multiple combinations of:
- temporal window lengths
- blending weights between network output and recent-frequency baseline

### 5. Evaluation
Ranks all configurations using historical hit-based metrics on the test split.

### 6. Final selection
Chooses the best configuration and produces:
- raw rounded prediction
- final blended combination
- top 5 suggested combinations

### 7. Export
Allows optional CSV export of the ranking and summaries.

---

## Example Output

### Raw network prediction
```text
[9, 18, 26, 34, 43] + stars [4, 8]
```

### Final blended combination
```text
[8, 13, 24, 29, 42] + stars [5, 10]
```

### Top 5 suggested combinations
```text
[8, 13, 24, 29, 42] + stars [5, 10]
[8, 13, 24, 29, 42] + stars [5, 10]
[8, 13, 24, 42, 44] + stars [5, 10]
[8, 13, 17, 24, 42] + stars [5, 10]
[8, 13, 24, 42, 44] + stars [5, 6]
```

---

## Metrics Used

Besides regression metrics such as:
- MSE
- MAE

the project also evaluates hit-based metrics:

- average number hits
- average star hits
- % of draws with >=1 correct number
- % of draws with >=1 correct star
- % of draws with >=2 correct numbers
- % of draws with 2 correct stars

These metrics are more interpretable for this type of task.

---

## Hybrid Model Idea

The project does not rely only on the direct network output.

The final combination is built by blending:

1. **LSTM prediction**
   - attempts to model temporal structure in the historical sequence

2. **Recent frequency**
   - gives additional weight to numbers and stars that appeared more often in the latest draws

The blend is controlled by:
- `alpha_num`
- `alpha_star`

Example:
- `alpha = 0.3` means 30% network influence and 70% recent-frequency influence

---

## Limitations

This project is **experimental and educational**.

EuroMillions is a fundamentally random process, so:
- reliable predictive power should not be expected
- results should be interpreted as a modeling exercise
- outperforming a simple baseline on historical slices does not guarantee future predictive value

The main value of the project is in:
- structuring an ML pipeline
- comparing models with baselines
- evaluating multiple configurations
- documenting results clearly

---

## Possible Future Improvements

- rolling-window validation
- classification-based modeling instead of direct regression
- score generation for all numbers from 1 to 50
- score generation for all stars from 1 to 12
- automatic hyperparameter search
- result dashboard
- automatic report export
- comparison with other models:
  - Random Forest
  - XGBoost
  - GRU
  - lightweight Transformers

---

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook

---

## Author

Built as a practical machine-learning and time-series experiment using historical EuroMillions data.

---

## License

You can add any license you prefer, for example:

```text
MIT License
```
