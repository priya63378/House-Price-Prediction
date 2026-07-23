"""
House Price Prediction
-----------------------
Trains and compares regression models to predict house prices.

By default this generates a realistic synthetic housing dataset (no
internet access required). If you have your own CSV of house data,
just point CSV_PATH at it (see the load_data() function) and make
sure it has a "PRICE" column as the target.

Run:
    python house_price_prediction.py

Outputs:
    - Console metrics for each model
    - feature_importance.png
    - actual_vs_predicted.png
    - correlation_heatmap.png
    - best_house_price_model.joblib  (saved best model + scaler)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42
N_SAMPLES = 3000

# Set this to a CSV file path to use your own dataset instead of the
# synthetic generator below. The CSV must contain a "PRICE" column.
CSV_PATH = None


def generate_synthetic_housing_data(n_samples=N_SAMPLES, random_state=RANDOM_STATE):
    """
    Create a realistic synthetic housing dataset with meaningful
    relationships between features and price, plus some noise.
    """
    rng = np.random.default_rng(random_state)

    sqft_living = rng.normal(1800, 700, n_samples).clip(300, 8000)
    bedrooms = rng.integers(1, 7, n_samples)
    bathrooms = (rng.integers(1, 5, n_samples) + rng.choice([0, 0.5], n_samples))
    floors = rng.choice([1, 1.5, 2, 2.5, 3], n_samples, p=[0.35, 0.1, 0.35, 0.1, 0.1])
    house_age = rng.integers(0, 100, n_samples)
    lot_size = rng.normal(6000, 2500, n_samples).clip(500, 30000)
    garage = rng.integers(0, 4, n_samples)
    condition = rng.integers(1, 6, n_samples)  # 1 (poor) - 5 (excellent)
    school_rating = rng.integers(1, 11, n_samples)  # 1-10
    distance_to_city = rng.exponential(8, n_samples).clip(0.2, 60)  # miles
    crime_rate = rng.exponential(3, n_samples).clip(0.1, 25)  # index

    # Base price formula with realistic weights, then add noise
    price = (
        50000
        + sqft_living * 140
        + bedrooms * 6000
        + bathrooms * 9000
        + floors * 4000
        + garage * 5000
        + condition * 8000
        + school_rating * 7000
        - house_age * 550
        - distance_to_city * 1800
        - crime_rate * 2200
        + lot_size * 3.5
    )
    noise = rng.normal(0, 25000, n_samples)
    price = (price + noise).clip(40000, None)

    df = pd.DataFrame(
        {
            "SqFtLiving": sqft_living.round(0),
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Floors": floors,
            "HouseAge": house_age,
            "LotSize": lot_size.round(0),
            "Garage": garage,
            "Condition": condition,
            "SchoolRating": school_rating,
            "DistanceToCity": distance_to_city.round(2),
            "CrimeRate": crime_rate.round(2),
            "PRICE": price.round(0),
        }
    )
    return df


def load_data():
    """Load the housing dataset into a DataFrame (CSV if provided, else synthetic)."""
    if CSV_PATH:
        df = pd.read_csv(CSV_PATH)
        if "PRICE" not in df.columns:
            raise ValueError("CSV must contain a 'PRICE' column as the target.")
        return df
    return generate_synthetic_housing_data()


def explore_data(df: pd.DataFrame):
    """Print basic EDA info and save a correlation heatmap."""
    print("\n=== Dataset shape ===")
    print(df.shape)

    print("\n=== First 5 rows ===")
    print(df.head())

    print("\n=== Summary statistics ===")
    print(df.describe().T)

    print("\n=== Missing values ===")
    print(df.isnull().sum())

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.close()
    print("\nSaved correlation_heatmap.png")


def preprocess(df: pd.DataFrame):
    """Split features/target, train/test split, and scale features."""
    X = df.drop(columns=["PRICE"])
    y = df["PRICE"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def evaluate(name, model, X_test, y_test):
    """Compute and print regression metrics for a fitted model."""
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n--- {name} ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R^2 : {r2:.4f}")

    return {"name": name, "rmse": rmse, "mae": mae, "r2": r2, "preds": preds}


def train_models(X_train_scaled, X_test_scaled, y_train, y_test):
    """Train several regressors and return their evaluation results."""
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=8),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    results = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        result = evaluate(name, model, X_test_scaled, y_test)
        results.append(result)
        fitted[name] = model

    return results, fitted


def plot_feature_importance(model, feature_names):
    """Plot feature importance for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)

    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh", color="steelblue")
    plt.title("Feature Importance (Best Model)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()
    print("Saved feature_importance.png")


def plot_actual_vs_predicted(y_test, preds, model_name):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, preds, alpha=0.3, edgecolor="k", linewidth=0.2)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Actual vs Predicted ({model_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=150)
    plt.close()
    print("Saved actual_vs_predicted.png")


def main():
    print("Loading data...")
    df = load_data()

    explore_data(df)

    (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
    ) = preprocess(df)

    print("\nTraining models...")
    results, fitted = train_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Pick best model by R^2
    best = max(results, key=lambda r: r["r2"])
    best_model = fitted[best["name"]]

    print(f"\n=== Best model: {best['name']} (R^2 = {best['r2']:.4f}) ===")

    plot_feature_importance(best_model, X_train.columns)
    plot_actual_vs_predicted(y_test, best["preds"], best["name"])

    joblib.dump(
        {"model": best_model, "scaler": scaler, "feature_names": list(X_train.columns)},
        "best_house_price_model.joblib",
    )
    print("\nSaved best_house_price_model.joblib")

    # Example: predict on a single new sample (first row of test set)
    sample = X_test.iloc[[0]]
    sample_scaled = scaler.transform(sample)
    prediction = best_model.predict(sample_scaled)[0]
    print(f"\nExample prediction for a sample house: ${prediction:,.2f}")
    print(f"Actual value for that sample: ${y_test.iloc[0]:,.2f}")


if __name__ == "__main__":
    main()
