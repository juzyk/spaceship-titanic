# Spaceship Titanic — Binary Classification

Solution for the Kaggle competition [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic).
The goal is to predict if a passenger was **transported** to another dimension (`Transported` = True / False).

## What is inside

| File | Description |
|---|---|
| `Spaceship_Titanic.ipynb` | Main notebook: EDA, feature engineering, models, submission |
| `requirements.txt` | Python libraries |
| `data/` | Put `train.csv` and `test.csv` here |

## Steps

1. **EDA** — target balance, transported rate by category (`HomePlanet`, `CryoSleep`, `Destination`, `VIP`), distributions of numeric features, correlation.
2. **Feature engineering**
   - `Cabin` → `Deck`, `CabinNum`, `Side`
   - `GroupSize` from `PassengerId`
   - `TotalSpend` (sum of all spending) and `NoSpending` flag
3. **Models** — Logistic Regression, SVM, KNN, Decision Tree, Random Forest, AdaBoost, XGBoost, LightGBM.
   Hyperparameters tuned with `RandomizedSearchCV`.
4. **Submission** — best model (XGBoost) trained on full train data, predictions saved to `submission.csv`.

## Results (validation set)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **XGBoost** | **0.813** | 0.812 | 0.817 | **0.815** |
| LightGBM | 0.811 | 0.817 | 0.806 | 0.811 |
| AdaBoost | 0.799 | 0.784 | 0.831 | 0.807 |
| Random Forest | 0.797 | 0.797 | 0.801 | 0.799 |
| SVM | 0.796 | 0.807 | 0.783 | 0.795 |
| Logistic Regression | 0.792 | 0.788 | 0.803 | 0.795 |
| Decision Tree | 0.792 | 0.784 | 0.809 | 0.797 |

## How to run

```bash
pip install -r requirements.txt
```

1. Download `train.csv` and `test.csv` from the [competition page](https://www.kaggle.com/competitions/spaceship-titanic/data).
2. Put them in the `data/` folder.
3. Open `Spaceship_Titanic.ipynb` and run all cells.
