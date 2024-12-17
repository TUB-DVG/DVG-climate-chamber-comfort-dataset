# **DVG Climate Chamber Comfort dataset**

## **Project Overview**  
This project focuses on analyzing climate chamber data, preprocessing it, and building machine learning models to predict thermal preference based on the data collected during a thermal comfort study at the climate chamber of RWTH Aachen University, Germany. The repository includes notebooks for Exploratory Data Analysis (EDA), scripts for model training, result analysis, and pre-trained models.

## **Repository Structure**  

```
DVG-CLIMATE-CHAMBER-COMFORT-DATASET/
│
├── data/                                        # Directory containing datasets and information about the experiment
│   ├── ...                                 
│   └── README.md                                   # Readme file for the dataset
│
├── EDA_notebooks/                               # Notebooks for data preprocessing and exploratory data analysis
│   ├── preprocessing.ipynb                         # Notebook for data cleaning and preprocessing
│   └── rawdata_analysis.ipynb                      # Exploratory analysis of raw data
│
├── EDA_plots/                                   # Directory for storing generated EDA plots and visualizations
│
├── model_scripts/                               # Scripts for model training and evaluation
│   ├── config.py                                   # Configuration file for Gridsearch cv parameters and feature sets
│   ├── model_train.py                              # Script for training machine learning models
│   └── plot_results.ipynb                          # Notebook for generating and analyzing result plots
│
├── results/                                     # Saved models and output results
│   ├── ET_best_model-F_accessible.pkl              # ExtraTrees model trained on f_accessible feature sets
│   ├── ET_best_model-F_all.pkl                     # ExtraTrees model trained on f_all feature sets
│   ├── ET_best_model-F_fanger.pkl                  # ExtraTrees model trained on f_fanger feature sets
│   ├── model_results.csv                           # Model performance results in CSV format
│   ├── model_results.json                          # Model performance results in JSON format
│   ├── pred_results.json                           # Model predictions in JSON format
│   ├── SVC_best_model-F_accessible.pkl             # SVC model trained on f_accessible feature sets
│   ├── SVC_best_model-F_all.pkl                    # SVC model trained on f_all feature sets
│   └── SVC_best_model-F_fanger.pkl                 # SVC model trained on f_fanger feature sets
│
├── results_plots/                               # Plots generated from model predictions and evaluations
│
├── requirements.txt                             # Python dependencies needed to run the project
│
├── LICENSE.mos                                  # MIT License file 
│
└── README.md                                    # Project documentation

```
## **How to use**

### **Clone the Repository**
```bash
git clone https://github.com/TUB-DVG/DVG-climate-chamber-comfort-dataset.git
cd DVG-climate-chamber-comfort-dataset
```

### **Dependencies**  
Ensure you have Python 3.8+ installed. 
The project uses the following Python libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## **Data Preprocessing and Analysis**

- **Raw Data Analysis**: Use `rawdata_analysis.ipynb` for initial EDA.
- **Preprocessing**: Run the `preprocessing.ipynb` notebook to clean and preprocess the raw data.  

## **Model Training**

The script `model_train.py` trains machine learning models (ExtraTrees and SVC). The results are saved in the `results/` folder as `.pkl` files for pre-trained models and `plot_results.ipynb` to visualize model performance.

-data_path: Path to the processed data file

-output_dir: Directory where model outputs and results will be saved

Run the script as follows:
```bash
python model_scripts/model_train.py --data_path ./data/processed_data.csv --output_dir ./results

```

## **Usage**
- **Feature Sets**:
The project uses three predefined feature sets for training the models. When using the pre-trained models or running predictions, input features must match one of these formats:

    -`F_fanger: ['rt', 'mrt', 'v', 'rh', 'clo', 'met']`
  
    -`F_all: ['met', 'age', 'sex', 'clo', 'CO2', 'rh', 'rt','out_t', 'out_rh', 'w_st', 'a_st', 'hr', 'v', 'bmi']`
  
    -`F_accessible: ['met', 'age', 'sex', 'rh', 'rt', 'out_t', 'out_rh', 'w_st', 'hr', 'bmi']`

The pre-trained models can be imported as follows:

```python
import pickle
import pandas as pd

# Load a pre-trained model
model_path = "results/ET_best_model-F_accessible.pkl"  
with open(model_path, "rb") as file:
    et_model = pickle.load(file)

# Example input for f_accessible
input_data = [[1.2, 30, 'Female', 39.77, 18.63, 1.336, 94.70, 28.582, 86.94, 22.83]]

# Define column names
cols = ['met', 'age', 'sex', 'rh', 'rt', 'out_t', 'out_rh', 'w_st', 'hr', 'bmi']

# Convert to DataFrame
df = pd.DataFrame(input_data, columns=cols)

# Run prediction (0:No change; -1:Cooler; 1:Warmer)
prediction = et_model.predict(df)
print("Prediction:", prediction)
```

## **License**

This project is licensed under the [MIT License](LICENSE.mos). 


## **Contact**

info@dvg.tu-berlin.de

j.odeyemi@tu-berlin.de
