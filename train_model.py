import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import joblib


# ---------------------------------------------------
# 1. CREATE SYNTHETIC FINANCIAL DATA
# ---------------------------------------------------

np.random.seed(42)

n_samples = 2000

data = pd.DataFrame({
    "age": np.random.randint(21, 61, n_samples),

    "annual_income": np.random.randint(
        200000, 1500000, n_samples
    ),

    "credit_score": np.random.randint(
        300, 851, n_samples
    ),

    "loan_amount": np.random.randint(
        50000, 1000000, n_samples
    ),

    "loan_term": np.random.choice(
        [12, 24, 36, 48, 60],
        n_samples
    ),

    "existing_debt": np.random.randint(
        0, 500000, n_samples
    ),

    "employment_type": np.random.choice(
        [0, 1, 2],
        n_samples,
        p=[0.25, 0.55, 0.20]
    ),

    "previous_defaults": np.random.randint(
        0, 4, n_samples
    ),

    "dependents": np.random.randint(
        0, 5, n_samples
    )
})


# ---------------------------------------------------
# 2. CREATE A CREDIT RISK SCORE
# ---------------------------------------------------

score = (
    0.35 * (data["credit_score"] - 300) / 550
    + 0.20 * (data["annual_income"] / 1500000)
    - 0.20 * (data["loan_amount"] / 1000000)
    - 0.10 * (data["existing_debt"] / 500000)
    + 0.05 * data["employment_type"]
    - 0.15 * data["previous_defaults"]
    - 0.03 * data["dependents"]
)


# Add some randomness so the model does not learn a perfect rule
noise = np.random.normal(0, 0.12, n_samples)

score = score + noise


# ---------------------------------------------------
# 3. CREATE TARGET
# ---------------------------------------------------

# 1 = Loan Approved
# 0 = Loan Not Approved

data["loan_approved"] = (score > 0.18).astype(int)


# ---------------------------------------------------
# 4. SAVE DATASET
# ---------------------------------------------------

data.to_csv("credit_data.csv", index=False)

print("Dataset created successfully!")
print(data.head())


# ---------------------------------------------------
# 5. SEPARATE FEATURES AND TARGET
# ---------------------------------------------------

X = data.drop("loan_approved", axis=1)

y = data["loan_approved"]


# ---------------------------------------------------
# 6. TRAIN / TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------
# 7. CREATE RANDOM FOREST MODEL
# ---------------------------------------------------

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=42
)


# ---------------------------------------------------
# 8. TRAIN MODEL
# ---------------------------------------------------

model.fit(X_train, y_train)


# ---------------------------------------------------
# 9. MAKE PREDICTIONS
# ---------------------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------------------
# 10. EVALUATE MODEL
# ---------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n-----------------------------")
print("MODEL PERFORMANCE")
print("-----------------------------")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ---------------------------------------------------
# 11. DISPLAY FEATURE IMPORTANCE
# ---------------------------------------------------

features = X.columns

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)


# ---------------------------------------------------
# 12. SAVE MODEL
# ---------------------------------------------------

joblib.dump(
    model,
    "credit_risk_model.pkl"
)

print("\nModel saved as credit_risk_model.pkl")