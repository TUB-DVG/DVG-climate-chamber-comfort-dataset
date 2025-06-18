import pickle
import pandas as pd

# Choose model: "F_accessible" or "F_fanger"
feature_set_name = "F_accessible"
#Define model name (ET, SVC, RF)
model_name = "ET"

model_path = f"results/{model_name}_best_model-{feature_set_name}.pkl"
model = pickle.load(open(model_path, "rb"))

if feature_set_name == "F_accessible":
    cols = ['met', 'age', 'sex', 'RH_in', 'T_in', 'T_out', 'RH_out', 'w_st', 'HR', 'BMI']
    
elif feature_set_name == "F_fanger":
    cols = ['T_in', 'T_mrt', 'V_a', 'RH_in', 'clo', 'met']
    

#adjust input data accordingly
input_data = [[1.2, 30, 'Female', 39.77, 18.63, 1.336, 94.70, 28.582, 86.94, 22.83]] 
df = pd.DataFrame(input_data, columns=cols)

# Run prediction (0:no change; -1:cooler; 1:warmer)
pred = model.predict(df)
print(f"{model_name} - {feature_set_name} Prediction:", pred)
