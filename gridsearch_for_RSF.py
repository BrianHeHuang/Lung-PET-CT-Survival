import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, ParameterGrid
from pysurvival.models.survival_forest import RandomSurvivalForestModel
from pysurvival.utils.metrics import concordance_index,_concordance_index, _brier_score
from pysurvival.utils.display import integrated_brier_score
#from pysurvival.utils.sklearn_adapter import sklearn_adapter
from pysurvival import utils
from config import config
import os

from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import RandomSurvivalForest

#models = ["e620607d-d43f-457e-984f-b923a3b6641f", "da37531f-98fc-4da2-97ae-7e947c0a67e9"]
models = ["e620607d-d43f-457e-984f-b923a3b6641f"]
seed = 42
ensemble_risks = []
ensemble_survivals = []
for model in models:
    training_set = pd.read_csv(os.path.join(config.EXTRACTED_FEATURES, "{}-train_features.csv".format(str(model))))
    validation_set = pd.read_csv(os.path.join(config.EXTRACTED_FEATURES, "{}-validation_features.csv".format(str(model))))
    test_set = pd.read_csv(os.path.join(config.EXTRACTED_FEATURES, "{}-test_features.csv".format(str(model))))

    #dropping a sample with vital_status = 2
    training_set = training_set[training_set["vital_status"].astype(int) != 2]
    validation_set = validation_set[validation_set["vital_status"].astype(int) != 2]
    test_set = test_set[test_set["vital_status"].astype(int) != 2]

    clinical_features = ["volume", "age", "gender", "race", "missing_race", "suv"]
    extracted_features = ["feature{}".format(str(i)) for i in range(1,17)]
    survival_label = "os"
    #vital_status = 1: Alive, censored; vital_status = 0: deceased, uncensored
    censored_label = "vital_status"

    #combining training and validation from feature extraction into new training set
    full_training_set = pd.concat([training_set, validation_set])

    train_labels = full_training_set[survival_label]
    if censored_label == "vital_status":
        train_censor = pd.Series([1 if x == 0 else 0 for x in full_training_set[censored_label]])
    else:
        train_censor = full_training_set[censored_label]
    train_image_feat = full_training_set[extracted_features]
    train_clinical_feat = full_training_set[clinical_features]

    test_labels = test_set[survival_label]
    if censored_label == "vital_status":
        test_censor = pd.Series([1 if x == 0 else 0 for x in test_set[censored_label]])
    else:
        test_censor = test_set[censored_label]
    test_image_feat = test_set[extracted_features]
    test_clinical_feat = test_set[clinical_features]

    params = {'n_estimators':[30, 10, 200],'max_depth':[5, 10, 20], 'min_samples_node':[5, 10, 20],'max_features':['sqrt', 'log2', 'all'], "sample_size_pct":[0.4, 0.64, 0.9]}
    param_grid = ParameterGrid(params)
    #construct the image feature model

    #gridsearch
    best_params = {}
    best_c_index = 0

    for dict in param_grid:
        c_index = []
        seeds = [42, 75, 125, 200, 350]
        rsf = RandomSurvivalForestModel(num_trees=dict['n_estimators'])
        for grid_seed in seeds:
            X_train, X_test, label_train, label_test = train_test_split(train_image_feat, train_labels, test_size=0.3, random_state=grid_seed)
            X_train, X_test, censor_train, censor_test = train_test_split(train_image_feat, train_censor, test_size=0.3, random_state=grid_seed)
            rsf.fit(X_train, label_train, censor_train, max_features=dict["max_features"], max_depth=dict["max_depth"],
                    min_node_size=dict["min_samples_node"], sample_size_pct=dict["sample_size_pct"], seed=seed)
            c_index.append(concordance_index(rsf, X_test, label_test, censor_test))
        summed_index = sum(c_index)/len(c_index)
        if summed_index>best_c_index:
            best_c_index = summed_index
            best_params = dict
            # print(best_params)
            # print(best_c_index)

    print("Best image params are " + str(best_params))
    print("Best image c-index is " + str(best_c_index))

    best_clinparams = {}
    best_clin_c_index = 0

    for dict in param_grid:
        c_index = []
        seeds = [42, 75, 125, 200, 350]
        rsf = RandomSurvivalForestModel(num_trees=dict['n_estimators'])
        for grid_seed in seeds:
            X_train, X_test, label_train, label_test = train_test_split(train_clinical_feat, train_labels, test_size=0.3,
                                                                        random_state=grid_seed)
            X_train, X_test, censor_train, censor_test = train_test_split(train_clinical_feat, train_censor, test_size=0.3,
                                                                          random_state=grid_seed)
            rsf.fit(X_train, label_train, censor_train, max_features=dict["max_features"], max_depth=dict["max_depth"],
                    min_node_size=dict["min_samples_node"], sample_size_pct=dict["sample_size_pct"], seed=seed)
            c_index.append(concordance_index(rsf, X_test, label_test, censor_test))
        summed_index = sum(c_index) / len(c_index)
        if summed_index > best_clin_c_index:
            best_clin_c_index = summed_index
            best_clinparams = dict

    print("Best clin params are " + str(best_clinparams))
    print("Best clin c-index is " + str(best_clin_c_index))