import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv('data/processed_data.csv')
X = df.drop(columns=["TPV", "TSV", "ID", "T_g", "T_mrt"])

numeric_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()


with open("feature_selection_results/selected_features.txt", "r") as f:
    selected_features = [line.strip() for line in f.readlines()]

selected_orig_columns = set()
for feat in selected_features:
    if '__' not in feat:
        continue
    _, col = feat.split('__', 1)
    if col in numeric_features or col in categorical_features:
        selected_orig_columns.add(col)


numeric_selected = [col for col in numeric_features if col in selected_orig_columns]
categorical_selected = [col for col in categorical_features if col in selected_orig_columns]


final_preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", MinMaxScaler())
    ]), numeric_selected),
    ("cat", OneHotEncoder(drop="first"), categorical_selected)
])


X_subset = X[numeric_selected + categorical_selected]
final_preprocessor.fit(X_subset)

# Save
joblib.dump(final_preprocessor, "feature_selection_results/preprocessor_selected.pkl")
