import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, make_scorer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import KNNImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC, SMOTE
import os
import argparse
from pathlib import Path
from config import PARAM_GRID_ET_RF, PARAM_GRID_SVC, RANDOM_STATE, TEST_SIZE, N_SPLITS, N_REPEATS, FEATURE_SETS
from utils import wrangle_data, save_model, save_train_results, save_pred_results


def evaluate_repeated_cv(model, x, y, n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE):

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )

    scoring = {
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro',
        'cohen_kappa': make_scorer(cohen_kappa_score)
    }

    scores = cross_validate(
        model,
        x,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    summary = {}
    for key, values in scores.items():
        if key.startswith('test_'):
            metric = key.replace('test_', '')
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std())
    return summary


def train_models_repeated_cv(data_path: str, output_dir: str, project_root: str):
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

            # Reconstruct full x, y
            x_full = pd.concat([x_train, x_test], ignore_index=True)
            y_full = np.concatenate([y_train, y_test])

            for model_name, base_model, param_grid in [
                ('ET', ExtraTreesClassifier(random_state=RANDOM_STATE), PARAM_GRID_ET_RF),
                ('RF', RandomForestClassifier(random_state=RANDOM_STATE), PARAM_GRID_ET_RF),
                ('SVC', SVC(probability=True, random_state=RANDOM_STATE), PARAM_GRID_SVC)
            ]:
                pipeline = Pipeline([
                    ('clf', base_model)
                ])

                smote = SMOTE(random_state=RANDOM_STATE)
                x_train_smt, y_train_smt = smote.fit_resample(x_train, y_train)

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
                search.fit(x_train_smt, y_train_smt)

                best_model = search.best_estimator_
                y_pred = best_model.predict(x_test)
                y_proba = best_model.predict_proba(x_test)

                cv_summary = evaluate_repeated_cv(
                    best_model,
                    x_full,
                    y_full,
                    n_splits=N_SPLITS,
                    n_repeats=N_REPEATS,
                    random_state=RANDOM_STATE
                )

                save_model(best_model, os.path.join(output_dir, f"{model_name}_best_model-{feature_set_name}_repeatedCV.pkl"))

                results[feature_set_name][model_name] = {
                    'model_name': model_name,
                    'feature_set': feature_set_name,
                    'best_params': search.best_params_,
                    'cv_accuracy': float(search.best_score_),
                    'test_accuracy': float(accuracy_score(y_test, y_pred)),
                    'test_precision': float(precision_score(y_test, y_pred, average='macro')),
                    'test_recall': float(recall_score(y_test, y_pred, average='macro')),
                    'test_f1': float(f1_score(y_test, y_pred, average='macro')),
                    'cohen_kappa': float(cohen_kappa_score(y_test, y_pred)),

                    # Repeated-Stratified-KFold metrics
                    'rep_cv_accuracy_mean': cv_summary.get('accuracy_mean'),
                    'rep_cv_accuracy_std': cv_summary.get('accuracy_std'),
                    'rep_cv_precision_mean': cv_summary.get('precision_macro_mean'),
                    'rep_cv_precision_std': cv_summary.get('precision_macro_std'),
                    'rep_cv_recall_mean': cv_summary.get('recall_macro_mean'),
                    'rep_cv_recall_std': cv_summary.get('recall_macro_std'),
                    'rep_cv_f1_mean': cv_summary.get('f1_macro_mean'),
                    'rep_cv_f1_std': cv_summary.get('f1_macro_std'),
                    'rep_cv_cohen_kappa_mean': cv_summary.get('cohen_kappa_mean'),
                    'rep_cv_cohen_kappa_std': cv_summary.get('cohen_kappa_std'),
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

            if cat_attr:
                cat_indices = [x.columns.get_loc(col) for col in cat_attr]
                smote = SMOTENC(categorical_features=cat_indices, random_state=RANDOM_STATE
                )
            else:
                smote = SMOTE(random_state=RANDOM_STATE)
            for model_name, model_pipeline, param_grid in [
                ('ET', ImbPipeline([('preprocessors', preprocessors), ('smote', smote),
                                    ('clf', ExtraTreesClassifier(random_state=RANDOM_STATE))]), PARAM_GRID_ET_RF),
                ('RF', ImbPipeline([('preprocessors', preprocessors), ('smote', smote),
                                    ('clf', RandomForestClassifier(random_state=RANDOM_STATE))]), PARAM_GRID_ET_RF),
                ('SVC', ImbPipeline([('preprocessors', preprocessors), ('smote', smote),
                                     ('clf', SVC(probability=True, random_state=RANDOM_STATE))]), PARAM_GRID_SVC)
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
                search.fit(x_train, y_train)

                best_model = search.best_estimator_
                y_pred = best_model.predict(x_test)
                y_proba = best_model.predict_proba(x_test)

                # Repeated CV
                cv_summary = evaluate_repeated_cv(
                    best_model,
                    x,
                    y,
                    n_splits=N_SPLITS,
                    n_repeats=N_REPEATS,
                    random_state=RANDOM_STATE
                )

                save_model(best_model, os.path.join(output_dir, f"{model_name}_best_model-{feature_set_name}_repeatedCV.pkl"))

                results[feature_set_name][model_name] = {
                    'model_name': model_name,
                    'feature_set': feature_set_name,
                    'best_params': search.best_params_,
                    'cv_accuracy': float(search.best_score_),
                    'test_accuracy': float(accuracy_score(y_test, y_pred)),
                    'test_precision': float(precision_score(y_test, y_pred, average='macro')),
                    'test_recall': float(recall_score(y_test, y_pred, average='macro')),
                    'test_f1': float(f1_score(y_test, y_pred, average='macro')),
                    'cohen_kappa': float(cohen_kappa_score(y_test, y_pred)),

                    # Repeated-Stratified-KFold metrics
                    'rep_cv_accuracy_mean': cv_summary.get('accuracy_mean'),
                    'rep_cv_accuracy_std': cv_summary.get('accuracy_std'),
                    'rep_cv_precision_mean': cv_summary.get('precision_macro_mean'),
                    'rep_cv_precision_std': cv_summary.get('precision_macro_std'),
                    'rep_cv_recall_mean': cv_summary.get('recall_macro_mean'),
                    'rep_cv_recall_std': cv_summary.get('recall_macro_std'),
                    'rep_cv_f1_mean': cv_summary.get('f1_macro_mean'),
                    'rep_cv_f1_std': cv_summary.get('f1_macro_std'),
                    'rep_cv_cohen_kappa_mean': cv_summary.get('cohen_kappa_mean'),
                    'rep_cv_cohen_kappa_std': cv_summary.get('cohen_kappa_std'),
                }

                results_y[feature_set_name][model_name] = {
                    'y_test': y_test.tolist(),
                    'y_pred': y_pred.tolist(),
                    'y_proba': y_proba.tolist()
                }

    save_train_results(results, output_dir)
    save_pred_results(results_y, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ET, RF and SVC model training for TPV prediction with repeatedCV")
    parser.add_argument("--data_path", type=str, default=None, help="Path to processed_data.csv")
    parser.add_argument("--output_dir", type=str, default="results_repeatedCV", help="saved results directory")
    args = parser.parse_args()

    script_file = Path(__file__).resolve()
    project_root = script_file.parent.parent

    # Data path
    if args.data_path:
        data_path = Path(args.data_path).resolve()
    else:
        data_path = project_root / "data" / "processed_data.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"\nData file not found!\n   Expected: {data_path}\n"
            f"   Make sure 'data/processed_data.csv' exists in the project root."
        )

    output_dir = project_root / args.output_dir
    output_dir.mkdir(exist_ok=True)

    train_models_repeated_cv(data_path=str(data_path), output_dir=str(output_dir), project_root=str(project_root))


#python model_scripts/model_train_repeatedkfold.py