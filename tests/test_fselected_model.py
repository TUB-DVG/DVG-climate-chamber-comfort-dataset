import pickle
import joblib
import pandas as pd

feature_set_name = "F_selected"
#Define model name (ET, SVC, RF)
model_name = "ET"

# Load model and preprocessor
model_path = f"results/{model_name}_best_model-F_selected.pkl"
model = pickle.load(open(model_path, "rb"))

preprocessor = joblib.load("feature_selection_results/preprocessor_selected.pkl")

#Adjust input accordingly
cols = ['age', 'clo', 'CO_2', 'RH_in', 'T_in', 'T_out', 'RH_out', 'w_st', 'a_st', 'HR', 'V_a', 'BMI']
input_data = [[30, 0.88, 500, 52.4, 23.55, 8.63, 87.4, 28.43, 28.58, 86.94, 0.04, 22.83]]

df = pd.DataFrame(input_data, columns=cols)
df = df[preprocessor.feature_names_in_]

# Transform and predict
df_proc_array = preprocessor.transform(df)
feature_names = preprocessor.get_feature_names_out()
df_proc = pd.DataFrame(df_proc_array, columns=feature_names)
pred = model.predict(df_proc)

# Run prediction (0:no change; -1:cooler; 1:warmer)
print(f"{model_name} - {feature_set_name} Prediction:", pred)





