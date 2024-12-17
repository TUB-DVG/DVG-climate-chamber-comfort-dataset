# **DVG Climate Chamber Comfort**

## **Project Overview**  
This project focuses on analyzing climate chamber data, preprocessing it, and building machine learning models to predict thermal preference based on the data collected during a thermal comfort study at the climate chamber of RWTH Aachen University, Germany. The repository includes notebooks for Exploratory Data Analysis (EDA), scripts for model training, result analysis, and pre-trained models for quick predictions.

---

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

---

## **How to use**

### **1. Clone the Repository**
```bash
git clone https://github.com/username/DVG-CLIMATE-CHAMBER-COMFORT-DATASET.git
cd DVG-CLIMATE-CHAMBER-COMFORT-DATASET
```

### **2. Set Up the Environment**  
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```
---

## **Dependencies**

The project uses the following Python libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

---
---

## **Data Preprocessing and Analysis**

- **Preprocessing**: Run the `preprocessing.ipynb` notebook to clean and preprocess the raw climate chamber data.  
- **Raw Data Analysis**: Use `rawdata_analysis.ipynb` for initial EDA and insights into the dataset.

---

## **Model Training**

The script `model_train.py` trains machine learning models (ExtraTrees and SVC). The results are saved in the `results/` folder as `.pkl` files for pre-trained models.
--data_path: Path to the processed data file
--output_dir: Directory where model outputs and results will be saved

Run the script as follows:
```bash
python model_scripts/model_train.py --data_path ./data/processed_data.csv --output_dir ./results

```

---

## **Results and Predictions**

- **Pre-trained Models**:  
   Pre-trained models are available in the `results/` folder as a .pkl file. These models include:  

- **Visualizations**:  
   Use `plot_results.ipynb` to visualize model performance and generate comparison plots.

---

## **Usage**
- **Feature Sets**:
The project uses three predefined feature sets for training the models. When using the pre-trained models or running predictions, input features must match one of these formats:
    'F_fanger': ['rt', 'mrt', 'v', 'rh', 'clo', 'met'],
    'F_all': ['met', 'age', 'sex', 'clo', 'CO2', 'rh', 'rt','out_t', 'out_rh', 'w_st', 'a_st', 'hr', 'v', 'bmi'],
    'F_accessible': ['met', 'age', 'sex', 'rh', 'rt', 'out_t', 'out_rh', 'w_st', 'hr', 'bmi']

You can load the pre-trained models as follows:

```python
import pickle
import pandas as pd

# Load a pre-trained model
model_path = "results/ET_best_model-F_accessible.pkl"  
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Example input: replace with actual values for the selected feature set
input_data = [[1.2, 30, 'Female', 39.77, 18.63, 1.336, 94.70, 28.582, 86.94, 22.83]]

# Define column names
cols = ['met', 'age', 'sex', 'rh', 'rt', 'out_t', 'out_rh', 'w_st', 'hr', 'bmi']

# Convert to DataFrame
df = pd.DataFrame(input_data, columns=cols)

# Run prediction (0:No change; -1:Cooler; 1:Warmer)
prediction = model.predict(df)
print("Prediction:", prediction)
```

---

## **License**

This project is licensed under the [MIT License](LICENSE.mos). 

---

## **Contributions**

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to the branch and submit a Pull Request.

---

## **Contact**

info@dvg.tu-berlin.de

---
