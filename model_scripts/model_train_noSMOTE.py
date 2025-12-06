import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import KNNImputer
import pickle
import os
import json
import argparse
from pathlib import Path
from config import PARAM_GRID_ET_RF, PARAM_GRID_SVC, RANDOM_STATE, TEST_SIZE, N_SPLITS, FEATURE_SETS
from utils import wrangle_data, save_model, save_train_results, save_pred_results


def train_models_no_smote(data_path, output_dir, project_root):
    df = wrangle_data(data_path)
    project_root = Path(project_root)

    results = {}
    results_y = {}

    os.makedirs(output_dir, exist_ok=True)

    for feature_set_name, feature_cols in FEATURE_SETS.items():
        results[feature_set_name] = {}
        results_y[feature_set_name] = {}

        if feature_set_name == 'F_selected':
            fs_path = project_root / "feature_selection_results"

            x_train = pd.read_csv(fs_path / "X_train_selected.csv")
            x_test = pd.read_csv(fs_path / "X_test_selected.csv")
            y_train = pd.read_csv(fs_path / "y_train.csv").values.ravel()
            y_test = pd.read_csv(fs_path / "y_test.csv").values.ravel()

            for model_name, base_model, param_grid in [
                ('ET', ExtraTreesClassifier(random_state=RANDOM_STATE), PARAM_GRID_ET_RF),
                ('RF', RandomForestClassifier(random_state=RANDOM_STATE), PARAM_GRID_ET_RF),
                ('SVC', SVC(probability=True, random_state=RANDOM_STATE), PARAM_GRID_SVC)
            ]:
                pipeline = Pipeline([
                    ('clf', base_model)
                ])

                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=param_grid,
                    n_iter=30,
                    cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    verbose=1,
                    error_score='raise'
                )
                search.fit(x_train, y_train)

                best_model = search.best_estimator_
                y_pred = best_model.predict(x_test)
                y_proba = best_model.predict_proba(x_test)

                save_model(best_model, os.path.join(output_dir, f"{model_name}_best_model-{feature_set_name}_noSMOTE.pkl"))

                results[feature_set_name][model_name] = {
                    'model_name': model_name,
                    'feature_set': feature_set_name,
                    'best_params': search.best_params_,
                    'cv_accuracy': search.best_score_,
                    'test_accuracy': accuracy_score(y_test, y_pred),
                    'test_precision': precision_score(y_test, y_pred, average='macro'),
                    'test_recall': recall_score(y_test, y_pred, average='macro'),
                    'test_f1': f1_score(y_test, y_pred, average='macro'),
                    'cohen_kappa': cohen_kappa_score(y_test, y_pred)
                }

                results_y[feature_set_name][model_name] = {
                    'y_test': y_test.tolist(),
                    'y_pred': y_pred.tolist(),
                    'y_proba': y_proba.tolist()
                }

        else:
            x = df[feature_cols]
            y = df['TPV']
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
            )

            cat_attr = x.select_dtypes(include=['object']).columns.tolist()
            num_attr = x.select_dtypes(exclude=['object']).columns.tolist()

            num_pipeline = Pipeline([
                ('imputer', KNNImputer(n_neighbors=5)),
                ('scaler', MinMaxScaler())
            ])
            cat_pipeline = Pipeline([
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])
            preprocessors = ColumnTransformer([
                ("num", num_pipeline, num_attr),
                ("cat", cat_pipeline, cat_attr)
            ])

            for model_name, base_clf, param_grid in [
                ('ET', ExtraTreesClassifier(random_state=RANDOM_STATE, class_weight='balanced'), PARAM_GRID_ET_RF),
                ('RF', RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'), PARAM_GRID_ET_RF),
                ('SVC', SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced'), PARAM_GRID_SVC),
            ]:
                pipeline = Pipeline([
                    ('preprocess', preprocessors),
                    ('clf', base_clf)
                ])
                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=param_grid,
                    n_iter=30,
                    cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    verbose=1,
                    error_score='raise'
                )
                search.fit(x_train, y_train)

                best_model = search.best_estimator_
                y_pred = best_model.predict(x_test)
                y_proba = best_model.predict_proba(x_test)

                save_model(best_model, os.path.join(output_dir, f"{model_name}_best_model-{feature_set_name}_noSMOTE.pkl"))

                results[feature_set_name][model_name] = {
                    'model_name': model_name,
                    'feature_set': feature_set_name,
                    'best_params': search.best_params_,
                    'cv_accuracy': search.best_score_,
                    'test_accuracy': accuracy_score(y_test, y_pred),
                    'test_precision': precision_score(y_test, y_pred, average='macro'),
                    'test_recall': recall_score(y_test, y_pred, average='macro'),
                    'test_f1': f1_score(y_test, y_pred, average='macro'),
                    'cohen_kappa': cohen_kappa_score(y_test, y_pred)
                }

                results_y[feature_set_name][model_name] = {
                    'y_test': y_test.tolist(),
                    'y_pred': y_pred.tolist(),
                    'y_proba': y_proba.tolist()
                }

    save_train_results(results, output_dir)
    save_pred_results(results_y, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ET, RF and SVC model training for TPV prediction without SMOTE")
    parser.add_argument("--data_path", type=str, default=None, help="Path to processed_data.csv")
    parser.add_argument("--output_dir", type=str, default="results_noSMOTE", help="saved results directory")
    args = parser.parse_args()

    script_file = Path(__file__).resolve()
    project_root = script_file.parent.parent

    # Data path
    if args.data_path:
        data_path = Path(args.data_path).resolve()
    else:
        data_path = project_root / "data" / "processed_data.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"\nData file not found!\n   Expected: {data_path}\n"
                                f"   Make sure 'data/processed_data.csv' exists in the project root.")


    output_dir = project_root / args.output_dir
    output_dir.mkdir(exist_ok=True)

    train_models_no_smote(data_path=str(data_path), output_dir=str(output_dir), project_root=str(project_root))

#python model_scripts/model_train_noSMOTE.py