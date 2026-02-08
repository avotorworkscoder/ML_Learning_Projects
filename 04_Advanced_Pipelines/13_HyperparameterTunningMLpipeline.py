# ==========================================================
# MODULAR CUSTOMER CHURN PIPELINE
# Professional-grade single codebase
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

RANDOM_STATE = 42
TEST_SIZE = 0.3

np.random.seed(RANDOM_STATE)

# ==========================================================
# 2. DATA GENERATION
# ==========================================================

def generate_dataset(n=1000):
    """Simulate telecom churn dataset."""

    usage = np.random.normal(50, 15, n)
    support_calls = np.random.randint(0, 10, n)
    tenure = np.random.randint(1, 60, n)

    logit = (0.5 * support_calls) - (0.08 * usage) - (0.04 * tenure) + 2.0
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (np.random.random(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "usage": usage,
        "support_calls": support_calls,
        "tenure": tenure,
        "churn": churn
    })

    return df


# ==========================================================
# 3. FEATURE ENGINEERING
# ==========================================================

def engineer_features(df):
    """Add engineered behavioral features."""
    df = df.copy()

    df["usage_per_call"] = df["usage"] / (df["support_calls"] + 1)
    df["loyalty_index"] = df["tenure"] / (df["support_calls"] + 1)

    return df


# ==========================================================
# 4. HANDLE IMBALANCE
# ==========================================================

def balance_dataset(df):
    """Oversample minority class."""

    majority = df[df.churn == 0]
    minority = df[df.churn == 1]

    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=RANDOM_STATE
    )

    balanced_df = pd.concat([majority, minority_upsampled])
    return balanced_df


# ==========================================================
# 5. DATA PREPARATION
# ==========================================================

def prepare_data():
    """Generate, balance, engineer features, split data."""

    df = generate_dataset()
    df = balance_dataset(df)
    df = engineer_features(df)

    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test


# ==========================================================
# 6. HYPERPARAMETER TUNING
# ==========================================================

def tune_logistic_regression(X_train, y_train):
    """GridSearch tuning for Logistic Regression."""

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    param_grid = {
        "model__C": [0.01, 0.1, 1, 10],
        "model__solver": ["lbfgs"]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=StratifiedKFold(n_splits=5),
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_score_


def tune_random_forest(X_train, y_train):
    """RandomizedSearch tuning for Random Forest."""

    rf = RandomForestClassifier(random_state=RANDOM_STATE)

    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    random_search = RandomizedSearchCV(
        rf,
        param_dist,
        n_iter=10,
        cv=StratifiedKFold(n_splits=5),
        scoring="f1",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    random_search.fit(X_train, y_train)

    return random_search.best_estimator_, random_search.best_score_


# ==========================================================
# 7. MODEL SELECTION
# ==========================================================

def select_best_model(X_train, y_train):

    log_model, log_score = tune_logistic_regression(X_train, y_train)
    rf_model, rf_score = tune_random_forest(X_train, y_train)

    print("Logistic F1:", log_score)
    print("RandomForest F1:", rf_score)

    if rf_score > log_score:
        print("Selected Model: Random Forest")
        return rf_model
    else:
        print("Selected Model: Logistic Regression")
        return log_model


# ==========================================================
# 8. MODEL EVALUATION
# ==========================================================

def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    fpr, tpr, _ = roc_curve(y_test, probs)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1])
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()

    print("ROC AUC:", roc_auc_score(y_test, probs))

    return probs


# ==========================================================
# 9. MONITORING AND EXPORT
# ==========================================================

def monitor_predictions(probabilities, input_features):

    avg_risk = np.mean(probabilities)
    high_risk_mask = probabilities > 0.7

    print("\nAverage Risk:", avg_risk)
    print("High Risk Customers:", np.sum(high_risk_mask))

    retention_list = input_features.copy()
    retention_list["churn_probability"] = probabilities
    retention_list = retention_list[high_risk_mask]

    retention_list.to_csv("high_risk_customers.csv", index=False)
    print("High-risk customers exported.")


# ==========================================================
# 10. MAIN PIPELINE
# ==========================================================

def main():

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data()

    # Select best model
    best_model = select_best_model(X_train, y_train)

    # Evaluate
    probs = evaluate_model(best_model, X_test, y_test)

    # Monitor
    monitor_predictions(probs, X_test)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
