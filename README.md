# **DVG Climate Chamber Comfort dataset**

## **Project Overview**  
This project focuses on analyzing climate chamber data, preprocessing it, and building machine learning models to predict thermal preference based on the data collected during a thermal comfort study at the climate chamber of RWTH Aachen University, Germany. The repository includes notebooks for Exploratory Data Analysis (EDA), scripts for model training, result analysis, and testing pre-trained models.

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
├── feature_selection_results/                   # Directory for storing results from the feature selection for F_selected feature sets
│
├── model_scripts/                               # Scripts for model training and evaluation
│   ├── config.py                                   # Configuration file for Gridsearch cv parameters and feature sets
│   ├── correlation.ipynb                           # Correlation analysis
│   ├── f_selected_preprocessor.py                  # Script to save the preprocesser used in the training of f_selected feature set
│   ├── feature_selection.py                        # Script for feature selection process for the F_selected feature set
│   ├── model_train.py                              # Script for training machine learning models
│   └── plot_results.ipynb                          # Notebook for generating and analyzing result plots
│
├── results/                                     # Saved models and output results
│
├── results_plots/                               # Plots generated from model predictions and evaluations
│
├── tests/                                       # Test Prediction Using Pre-trained Models
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

The script `model_train.py` trains machine learning models (Random forest, ExtraTrees and SVC). The results are saved in the `results/` folder as `.pkl` files for pre-trained models and `plot_results.ipynb` to visualize model performance.

-data_path: Path to the processed data file

-output_dir: Directory where model outputs and results will be saved

Run the following script in the root folder.
```bash
python model_scripts/model_train.py --data_path data/processed_data.csv --output_dir results

```

## **Usage**
### **Feature Sets**:
The project uses three predefined feature sets for training the models. When using the pre-trained models or running predictions, input features must match one of these formats:
- **F_fanger**
      `['T_in', 'T_mrt', 'V_a', 'RH_in', 'clo', 'met']`
  
- **F_selected**
      `['age', 'clo', 'CO_2', 'RH_in', 'T_in', 'T_out', 'RH_out', 'w_st', 'a_st', 'HR', 'V_a', 'BMI']`

- **F_accessible**
      `['met', 'age', 'sex', 'RH_in', 'T_in', 'T_out', 'RH_out', 'w_st', 'HR', 'BMI']`

### **Model Testing**
All model test scripts are located in the `tests/` folder. You can run all tests in the root folder.
- `test_fselected_model.py` Tests prediction using the `F_selected` feature set.
- `test_fanger_or_accessible.py` Tests prediction using either`F_accessible` or `F_fanger` feature set.

## **License**

This project is licensed under the [MIT License](LICENSE.mos). 

## **Acknowledgments**
Special thanks to RWTH Aachen University for providing access to the climate chamber facilities and supporting data collection efforts. Also to the Einstein Center Digital Future (ECDF), Berlin for funding this project.

## **Contact**

info@dvg.tu-berlin.de
julianah.odeyemi@eonerc.rwth-aachen.de
