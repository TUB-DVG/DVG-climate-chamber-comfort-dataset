import pandas as pd
import pickle
import os
import json


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

