# House Price Prediction

A Python project that trains and compares machine learning models to predict
house prices from property features (square footage, bedrooms, location
factors, etc.).

## What it does

1. **Loads data** — generates a realistic synthetic housing dataset (3,000
   homes) so the script runs anywhere with no internet access or API key
   required. You can swap in your own CSV instead (see [Using your own
   data](#using-your-own-data)).
2. **Explores the data** — prints shape, summary statistics, and missing
   values, and saves a correlation heatmap.
3. **Preprocesses** — splits into train/test sets (80/20) and scales
   features with `StandardScaler`.
4. **Trains four models**:
   - Linear Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
5. **Evaluates** each model with RMSE, MAE, and R².
6. **Picks the best model** (highest R²), plots its feature importance
   (for tree-based models) and an actual-vs-predicted scatter plot, then
   saves it to disk with `joblib`.
7. **Demonstrates a prediction** on a sample house.

## Project structure

```
.
├── house_price_prediction.py   # main script
├── README.md                   # this file
├── requirements.txt            # dependencies
```

After running, these files are also generated:

```
├── correlation_heatmap.png
├── feature_importance.png
├── actual_vs_predicted.png
└── best_house_price_model.joblib
```

## Requirements

- Python 3.9+
- Packages listed in `requirements.txt`:
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - joblib

Install them with:

```bash
pip install -r requirements.txt
```

## Usage

Run the script directly:

```bash
python house_price_prediction.py
```

You'll see console output like:

```
=== Dataset shape ===
(3000, 12)

...

--- Linear Regression ---
RMSE: 25797.68
MAE : 20624.02
R^2 : 0.9402

--- Random Forest ---
RMSE: 31747.44
MAE : 24828.55
R^2 : 0.9095

=== Best model: Linear Regression (R^2 = 0.9402) ===

Example prediction for a sample house: $421,322.70
Actual value for that sample: $423,461.00
```

Three chart images and a saved model file are written to the working
directory.

## Dataset features

| Feature | Description |
|---|---|
| `SqFtLiving` | Living area in square feet |
| `Bedrooms` | Number of bedrooms |
| `Bathrooms` | Number of bathrooms |
| `Floors` | Number of floors |
| `HouseAge` | Age of the house in years |
| `LotSize` | Lot size in square feet |
| `Garage` | Number of garage spaces |
| `Condition` | Overall condition rating (1–5) |
| `SchoolRating` | Nearby school rating (1–10) |
| `DistanceToCity` | Distance to city center (miles) |
| `CrimeRate` | Local crime index |
| `PRICE` | Target: sale price in USD |

## Using your own data

If you have a real housing dataset (e.g., a Kaggle CSV), open
`house_price_prediction.py` and set:

```python
CSV_PATH = "path/to/your_data.csv"
```

Make sure your CSV has a column named `PRICE` as the prediction target;
every other numeric column is treated as a feature. Non-numeric columns
(e.g., neighborhood names) should be encoded first, for example with
`pd.get_dummies()`.

## How it works — key steps in the code

- `generate_synthetic_housing_data()` builds features with realistic
  relationships to price plus random noise, so models have genuine signal
  to learn from.
- `explore_data()` runs basic exploratory data analysis (EDA).
- `preprocess()` splits the data and scales features so that models like
  Linear Regression aren't biased by feature scale.
- `train_models()` fits four different regressors and returns their
  metrics side by side, so you can compare approaches easily.
- `plot_feature_importance()` and `plot_actual_vs_predicted()` visualize
  how the best model behaves.
- The best model (by R²) and the fitted scaler are bundled together and
  saved with `joblib.dump()` so they can be reloaded later without
  retraining:

```python
import joblib
bundle = joblib.load("best_house_price_model.joblib")
model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]

# predict on new data (as a DataFrame with the same feature_names/order)
new_scaled = scaler.transform(new_data[feature_names])
prediction = model.predict(new_scaled)
```

## Notes

- Metrics will vary slightly depending on the random seed and how much
  noise is in the synthetic data — this is expected and mirrors real-world
  model comparison.
- Swap in a real dataset for more meaningful results in a production
  setting; the synthetic generator is meant as a runnable, self-contained
  demonstration of the full workflow.
