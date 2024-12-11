import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.impute import KNNImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC, SMOTE
import pickle
import os
import json
import argparse
from config import PARAM_GRID_ET, PARAM_GRID_SVC, RANDOM_STATE, TEST_SIZE, N_SPLITS, FEATURE_SETS

#Import data
def wrangle_data(data_path):
    return pd.read_csv(data_path)

#Save SVC and ET models
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

#save training results
def save_train_results(results, output_dir):
    results_path = os.path.join(output_dir, 'model_results.json')
    with open(results_path, 'w') as file:
        json.dump(results, file, indent=4)

#save target results(y_test and y_pred)
def save_pred_results(results_y, output_dir):
    results_path = os.path.join(output_dir, 'pred_results.json')
    with open(results_path, 'w') as file:
        json.dump(results_y, file, indent=4)

# Function for model training
def train_models(data_path, output_dir):
    df = wrangle_data(data_path)
    
    y = df['TPV']

    results = {}
    results_y = {}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

        
    for feature_set_name, feature_cols in FEATURE_SETS.items():
        if feature_set_name not in results:
           results[feature_set_name] = {}

        if feature_set_name not in results_y:
            results_y[feature_set_name] = {}

        X = df[feature_cols]
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=TEST_SIZE, 
                                                            stratify=y, 
                                                            random_state=RANDOM_STATE)

        # Pipeline for feature preprocess
        cat_attr = X.select_dtypes(include=['object']).columns.tolist()
        #cat_attr = ['sex']
        num_attr = list(X.drop(columns=cat_attr))

        num_pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', MinMaxScaler()),
        ]) 
        cat_pipeline = Pipeline([
            ('encoder', OneHotEncoder()),
        ])
        preprocessors = ColumnTransformer(transformers=[
        ("num", num_pipeline, num_attr),
        ("cat", cat_pipeline, cat_attr)])

        cat_indices = [X.columns.get_loc(col) for col in cat_attr]
        smote_type = SMOTENC(random_state=RANDOM_STATE, 
                             categorical_features=cat_indices
                             ) if cat_attr else SMOTE(random_state=RANDOM_STATE)
        
        #Pipeline for ET and SVC
        for model_name, pipeline, param_grid in [
            ('ET', ImbPipeline([('preprocessors', preprocessors),
                                ('smote', smote_type), 
                                ('clf', ExtraTreesClassifier(random_state=RANDOM_STATE))
            ]), PARAM_GRID_ET),
            
            ('SVC', ImbPipeline([('preprocessors', preprocessors),
                              ('smote', smote_type),
                              ('clf', SVC(probability=True, random_state=RANDOM_STATE))
            ]), PARAM_GRID_SVC)
        
        ]:
                     
            #Stratified Kfold
            #5-fold cross-validation
            stratified_kfold = StratifiedKFold(n_splits=N_SPLITS, 
                                       shuffle=True, 
                                       random_state=RANDOM_STATE)
    
            grid_search = GridSearchCV(pipeline,
                                       param_grid=param_grid, 
                                       cv=stratified_kfold, 
                                       n_jobs = -1)

            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            cv_accuracy = grid_search.best_score_
                        
            #Saving models as pickle
            #save_model(best_model, os.path.join(output_dir, f"{model_name}_best_model-{feature_set_name}.pkl"))

            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)
            
            #Evaluation metrics
            results[feature_set_name][model_name] = {
                'model_name': model_name, 
                'feature_set': feature_set_name, 
                'best_params': grid_search.best_params_,
                'cv_accuracy': cv_accuracy,
                'test_accuracy': accuracy_score(y_test, y_pred),
                'test_precision': precision_score(y_test, y_pred, average='macro'),
                'test_recall': recall_score(y_test, y_pred, average='macro'),
                'test_f1': f1_score(y_test, y_pred, average='macro'),
                'cohen_kappa': cohen_kappa_score(y_test, y_pred)

            }

            #save y_test and y_pred
            results_y[feature_set_name][model_name] = {
                'y_test': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'y_proba': y_proba.tolist(),
            }

    save_train_results(results, output_dir)
    save_pred_results(results_y, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ET and SVC model training for TPV prediction.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input processed dataset(csv file).")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save results.")
    args = parser.parse_args()

    train_models(data_path=args.data_path, output_dir=args.output_dir)

#python model_train.py --data_path ..\data\processed_data.csv --output_dir ..\results