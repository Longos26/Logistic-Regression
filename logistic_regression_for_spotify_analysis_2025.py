import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# --- Load Data ---
df = pd.read_csv('/content/spotify_churn_dataset.csv')
display(df.head())

# --- Identify Features and Target ---
exclude_cols = {'user_id', 'is_churned'}
numerical_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
categorical_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if c not in exclude_cols]

print("Numerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)

# --- Prepare Data ---
X = df.drop(columns=[c for c in exclude_cols if c in df.columns])
y = df['is_churned']

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Preprocessing ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='drop'
)

# --- Build Pipeline ---
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# --- Train Model ---
model.fit(X_train, y_train)

# --- Transform Data for Analysis ---
def to_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix

X_train_processed = to_dense(model.named_steps['preprocessor'].transform(X_train))
X_test_processed = to_dense(model.named_steps['preprocessor'].transform(X_test))

print(f"Processed Training Data Shape: {X_train_processed.shape}")
print(f"Processed Testing Data Shape: {X_test_processed.shape}")

# --- Evaluate Model ---
y_pred = model.predict(X_test)

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1-Score": f1_score(y_test, y_pred, zero_division=0)
}

print("\n--- Evaluation Metrics ---")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# --- Confusion Matrix ---
plt.figure(figsize=(7, 5))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True, fmt='d', cmap='Blues',
    xticklabels=['Not Churned (0)', 'Churned (1)'],
    yticklabels=['Not Churned (0)', 'Churned (1)']
)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# --- Feature Importance ---
try:
    feature_names = list(model.named_steps['preprocessor'].get_feature_names_out())
except Exception:
    num_names = numerical_cols
    cat_encoder = model.named_steps['preprocessor'].named_transformers_.get('cat')
    cat_names = list(cat_encoder.get_feature_names_out(categorical_cols)) if hasattr(cat_encoder, 'get_feature_names_out') else []
    feature_names = num_names + cat_names

coef = model.named_steps['classifier'].coef_[0]
intercept = model.named_steps['classifier'].intercept_[0]

if len(feature_names) != len(coef):
    print("⚠️ Feature name count mismatch. Using generic feature names.")
    feature_names = [f"f_{i}" for i in range(len(coef))]

coeff_df = pd.DataFrame({
    'Feature': feature_names + ['Intercept'],
    'Coefficient': list(coef) + [intercept]
}).assign(AbsCoeff=lambda d: d['Coefficient'].abs()) \
 .sort_values('AbsCoeff', ascending=False) \
 .drop(columns='AbsCoeff')

print("\n--- Logistic Regression Coefficients ---")
display(coeff_df)

# --- Visualization (Age vs Listening Time) ---
processed_test_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
if {'num__age', 'num__listening_time'}.issubset(processed_test_df.columns):
    y_proba = model.predict_proba(X_test)[:, 1]
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(
        processed_test_df['num__age'], processed_test_df['num__listening_time'],
        c=y_proba, cmap='viridis', alpha=0.6
    )
    plt.colorbar(sc, label='Predicted Probability of Churn')
    plt.xlabel('Age (scaled)')
    plt.ylabel('Listening Time (scaled)')
    plt.title('Predicted Churn Probability by Age & Listening Time')
    plt.show()
else:
    print("\nNo scatter plot: 'age' or 'listening_time' not found in features.")
