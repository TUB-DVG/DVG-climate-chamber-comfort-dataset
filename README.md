# **DVG Climate Chamber Analysis and Model Training**

## **Project Overview**  
This project focuses on analyzing climate chamber data, preprocessing it, and building machine learning models to predict thermal preference based on the data collected during a thermal comfort study at the climate chamber of RWTH Aachen University, Germany. The repository includes notebooks for Exploratory Data Analysis (EDA), scripts for model training, result analysis, and pre-trained models for quick predictions.

---

## **Repository Structure**  

```
DVG-CLIMATE-CHAMBER-COMPOSITION/
│
├── EDA_notebooks/            # Notebooks for exploratory data analysis
│   ├── preprocessing.ipynb   # Data preprocessing notebook
│   └── rawdata_analysis.ipynb # Initial raw data analysis
│
├── EDA_plots/                # Folder for visualizations and EDA-related plots
│
├── model_scripts/            # Scripts for training and evaluating models
│   ├── config.py             # Configuration for model parameters
│   ├── model_train.py        # Model training script
│   └── plot_results.ipynb    # Notebook to visualize model results
│
├── results/                  # Saved models and result files
│   ├── ET_best_model-F_accessible.pkl    # Pre-trained ExtraTrees model (accessible)
│   ├── ET_best_model-F_all.pkl           # ExtraTrees model (all features)
│   ├── SVC_best_model-F_accessible.pkl   # SVC model (accessible features)
│   ├── model_results.csv                 # Results in CSV format
│   ├── model_results.json                # Results in JSON format
│   ├── pred_results.json                 # Prediction results
│   └── ...                               # Additional models and results
│
├── results_plots/            # Plots generated from model predictions and evaluations
│
├── data/                     # Placeholder for datasets
├── data_dictionary.csv       # Metadata or description of the dataset columns
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── SubjectiveQs.pdf          # Supporting subjective questions
```

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/username/DVG-CLIMATE-CHAMBER-COMPOSITION.git
cd DVG-CLIMATE-CHAMBER-COMPOSITION
```

### **2. Set Up the Environment**  
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## **Data Preprocessing and Analysis**

- **Preprocessing**: Run the `preprocessing.ipynb` notebook to clean and preprocess the raw climate chamber data.  
- **Raw Data Analysis**: Use `rawdata_analysis.ipynb` for initial EDA and insights into the dataset.

---

## **Model Training**

The script `model_train.py` trains machine learning models, such as ExtraTrees and SVC. The results are saved in the `results/` folder as `.pkl` files for pre-trained models.

Run the script as follows:
```bash
python model_scripts/model_train.py
```

---

## **Results and Predictions**

- **Pre-trained Models**:  
   Pre-trained models are available in the `results/` folder. These models include:  
   - `ET_best_model-F_accessible.pkl`  
   - `SVC_best_model-F_accessible.pkl`  

- **Visualizations**:  
   Use `plot_results.ipynb` to visualize model performance and generate comparison plots.

---

## **Usage**

You can load the pre-trained models for inference as follows:

```python
import pickle

# Load a pre-trained model
with open("results/ET_best_model-F_accessible.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Example prediction
example_input = [[1.2, 3.4, 5.6, 7.8]]  # Replace with actual feature values
prediction = model.predict(example_input)
print("Prediction:", prediction)
```

---

## **Results and Outputs**

1. **Model Results**:
   - Metrics like accuracy, precision, and recall are saved in `model_results.json`.
   - CSV format results are available in `model_results.csv`.

2. **Plots**:
   - Visualizations for predictions and performance can be found in the `results_plots/` folder.

---

## **Dependencies**

The project uses the following Python libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Install all dependencies via:
```bash
pip install -r requirements.txt
```

---

## **Dataset**

- A detailed description of the dataset's features can be found in `data_dictionary.csv`.  
- Add or download the dataset into the `data/` folder.

---

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for details.

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

For questions or feedback, feel free to reach out via GitHub Issues or email.

---
