#Grid Search CV
#model hyperparameters for Randomized search

PARAM_GRID_ET_RF = {
    'clf__bootstrap': [True],
    'clf__max_depth': [30, 60, 90, 120],
    'clf__max_features': [ 'sqrt','log2', None],
    'clf__min_samples_leaf': [1, 2, 4],
    'clf__min_samples_split': [8, 10, 12],
    'clf__n_estimators': [100, 200, 500],
    'clf__criterion': ['gini', 'entropy', 'log_loss']
}

PARAM_GRID_SVC = {
    'clf__C': [0.1, 1, 10, 100],
    'clf__kernel': ['linear', 'rbf', 'poly'],
    'clf__gamma': ['scale', 'auto'],
    'clf__degree': [1, 2, 3]
}

RANDOM_STATE = 42

#Test train size
TEST_SIZE = 0.3
N_SPLITS = 5

#Define the 3 Feature sets
#ID,TSV,TPV,met,age,sex,w_st,a_st,HR,T_g,RH_in,T_in,CO_2,V_a,T_out,RH_out,clo,BMI,T_mrt
FEATURE_SETS = {
    #f_fanger (Fanger 6 variables)
    'F_fanger': ['T_in', 'T_mrt', 'V_a', 'RH_in', 'clo', 'met'],
    #F_selected (features after data driven feature selection)
    'F_selected': ['age', 'clo', 'CO_2', 'RH_in', 'T_in', 
              'T_out', 'RH_out', 'w_st', 'a_st', 'HR', 'V_a', 'BMI'],
    #F_accessible 
    'F_accessible': ['met', 'age', 'sex', 'RH_in', 'T_in', 'T_out', 
                     'RH_out', 'w_st', 'HR', 'BMI']
                }
