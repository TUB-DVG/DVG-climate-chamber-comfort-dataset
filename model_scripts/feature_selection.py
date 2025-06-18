import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv('../data/processed_data.csv')
X = df.drop(columns=["TPV", "TSV", "ID"])
y = df["TPV"]

# Drop correlated features
X = X.drop(columns=["T_g", "T_mrt"])

numeric_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", MinMaxScaler())
    ]), numeric_features),
    ("cat", OneHotEncoder(drop="first"), categorical_features)
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_proc, y_train)

feature_names = preprocessor.get_feature_names_out()
importances = pd.Series(rf.feature_importances_, index=feature_names)
selected_features = importances[importances > 0.02].index.tolist()

X_train_final = pd.DataFrame(X_train_proc, columns=feature_names)[selected_features]
X_test_final = pd.DataFrame(X_test_proc, columns=feature_names)[selected_features]

X_train_final.to_csv("../feature_selection_results/X_train_selected.csv", index=False)
X_test_final.to_csv("../feature_selection_results/X_test_selected.csv", index=False)
y_train.to_csv("../feature_selection_results/y_train.csv", index=False)
y_test.to_csv("../feature_selection_results/y_test.csv", index=False)

with open("../feature_selection_results/selected_features.txt", "w") as f:
    for feat in selected_features:
        f.write(f"{feat}\n")

joblib.dump(preprocessor, "../feature_selection_results/preprocessor.pkl")