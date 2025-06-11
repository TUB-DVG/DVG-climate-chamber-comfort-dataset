import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import KNNImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC, SMOTE
import pickle
import os
import json
import argparse
from config import PARAM_GRID_ET_RF, PARAM_GRID_SVC, RANDOM_STATE, TEST_SIZE, N_SPLITS, FEATURE_SETS


def wrangle_data(data_path):
    return pd.read_csv(data_path)


def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def save_train_results(results, output_dir):
    results_path = os.path.join(output_dir, 'model_results.json')
    with open(results_path, 'w') as file:
        json.dump(results, file, indent=4)


def save_pred_results(results_y, output_dir):
    results_path = os.path.join(output_dir, 'pred_results.json')
    with open(results_path, 'w') as file:
        json.dump(results_y, file, indent=4)


def train_models(data_path, output_dir):
    df = wrangle_data(data_path)

    results = {}
    results_y = {}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for feature_set_name, feature_cols in FEATURE_SETS.items():
        results[feature_set_name] = {}
        results_y[feature_set_name] = {}

        if feature_set_name == 'F_selected':
            X_train = pd.read_csv('../feature_selection_results/X_train_selected.csv')
            X_test = pd.read_csv('../feature_selection_results/X_test_selected.csv')
            y_train = pd.read_csv('../feature_selection_results/y_train.csv').values.ravel()
            y_test = pd.read_csv('../feature_selection_results/y_test.csv').values.ravel()

            for model_name, base_model, param_grid in [
                ('ET', ExtraTreesClassifier(random_state=RANDOM_STATE), PARAM_GRID_ET_RF),
                ('RF', RandomForestClassifier(random_state=RANDOM_STATE), PARAM_GRID_ET_RF),
                ('SVC', SVC(probability=True, random_state=RANDOM_STATE), PARAM_GRID_SVC)
            ]:
                pipeline = Pipeline([
                    ('clf', base_model)
                ])
                smote = SMOTE(random_state=RANDOM_STATE)
                X_train_smt, y_train_smt = smote.fit_resample(X_train, y_train)

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
                search.fit(X_train_smt, y_train_smt)

                best_model = search.best_estimator_
                y_pred = best_model.predict(X_test)
                y_proba = best_model.predict_proba(X_test)

                save_model(best_model, os.path.join(output_dir, f"{model_name}_best_model-{feature_set_name}.pkl"))

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
            X = df[feature_cols]
            y = df['TPV']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
            )

            cat_attr = X.select_dtypes(include=['object']).columns.tolist()
            num_attr = X.select_dtypes(exclude=['object']).columns.tolist()

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

            cat_indices = [X.columns.get_loc(col) for col in cat_attr]
            smote = SMOTENC(categorical_features=cat_indices, random_state=RANDOM_STATE) if cat_attr else SMOTE(random_state=RANDOM_STATE)

            for model_name, model_pipeline, param_grid in [
                ('ET', ImbPipeline([('preprocessors', preprocessors), ('smote', smote), ('clf', ExtraTreesClassifier(random_state=RANDOM_STATE))]), PARAM_GRID_ET_RF),
                ('RF', ImbPipeline([('preprocessors', preprocessors), ('smote', smote), ('clf', RandomForestClassifier(random_state=RANDOM_STATE))]), PARAM_GRID_ET_RF),
                ('SVC', ImbPipeline([('preprocessors', preprocessors), ('smote', smote), ('clf', SVC(probability=True, random_state=RANDOM_STATE))]), PARAM_GRID_SVC)
            ]:
                search = RandomizedSearchCV(
                    estimator=model_pipeline,
                    param_distributions=param_grid,
                    n_iter=30,
                    cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    verbose=1,
                    error_score='raise'
                )
                search.fit(X_train, y_train)

                best_model = search.best_estimator_
                y_pred = best_model.predict(X_test)
                y_proba = best_model.predict_proba(X_test)

                save_model(best_model, os.path.join(output_dir, f"{model_name}_best_model-{feature_set_name}.pkl"))

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
    parser = argparse.ArgumentParser(description="ET, RF and SVC model training for TPV prediction")
    parser.add_argument("--data_path", type=str, required=True, help="processed dataset path")
    parser.add_argument("--output_dir", type=str, default="output", help="saved results directory")
    args = parser.parse_args()

    train_models(data_path=args.data_path, output_dir=args.output_dir)


#python model_train.py --data_path ..\data\processed_data.csv --output_dir ..\results
