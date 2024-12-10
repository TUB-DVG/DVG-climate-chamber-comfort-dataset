#Grid Search CV
#model hyperparameters for grid search
PARAM_GRID_ET = {
    'clf__bootstrap': [True],
    'clf__max_depth': [30, 60, 90, 120],
    'clf__max_features': ['sqrt', 'log2', None],
    'clf__min_samples_leaf': [1, 2, 4],
    'clf__min_samples_split': [8, 10, 12],
    'clf__n_estimators': [100, 200, 500]
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
FEATURE_SETS = {
    #f_fanger (Fanger 6 variables)
    'F_fanger': ['rt', 'mrt', 'v', 'rh', 'clo', 'met'],
    #F_all (all available features without mrt and gt)
    'F_all': ['met', 'age', 'sex', 'clo', 'CO2', 'rh', 'rt', 
              'out_t', 'out_rh', 'w_st', 'a_st', 'hr', 'v', 'bmi'],
    #F_wearable-accessible 
    'F_accessible': ['met', 'age', 'sex', 'rh', 'rt', 'out_t', 
                     'out_rh', 'w_st', 'hr', 'bmi']
                }
