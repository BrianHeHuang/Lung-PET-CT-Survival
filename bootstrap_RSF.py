import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, ParameterGrid
from pysurvival.models.survival_forest import RandomSurvivalForestModel
from pysurvival.utils.metrics import concordance_index,_concordance_index, _brier_score, brier_score
from pysurvival.utils.display import integrated_brier_score, create_risk_groups
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test

from db import db, Result, Expand_Result, CalculatedResult
import efficientnet.tfkeras
from tensorflow.keras.models import load_model
from pysurvival import utils
from config import config
import os

from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import RandomSurvivalForest

def plotless_integrated_brier_score(model, X, T, E, t_max=None, use_mean_point=True,
    figure_size=(20, 6.5)):
    """ The Integrated Brier Score (IBS) provides an overall calculation of
        the model performance at all available times.
    """

    # Computing the brier scores
    times, brier_scores = brier_score(model, X, T, E, t_max, use_mean_point)

    # Getting the proper value of t_max
    if t_max is None:
        t_max = max(times)
    else:
        t_max = min(t_max, max(times))

    # Computing the IBS
    ibs_value = np.trapz(brier_scores, times)/t_max
    return ibs_value

def ensemble_concordance_index(risk_sum, T, E, include_ties=True,
                      additional_results=False, **kwargs):
    risk = risk_sum
    risk, T, E = utils.check_data(risk, T, E)

    # Ordering risk, T and E in descending order according to T
    order = np.argsort(-T)
    risk = risk[order]
    T = T[order]
    E = E[order]

    # Calculating th c-index
    results = _concordance_index(risk, T, E, include_ties)

    if not additional_results:
        return results[0]
    else:
        return results

def ensemble_brier_score(survival_times, model, T, E, t_max=None, use_mean_point=True, **kwargs):

    T, E = utils.check_data(T, E)
    Survival = survival_times

    # Extracting the time buckets
    times = model.times
    time_buckets = model.time_buckets

    # Ordering Survival, T and E in descending order according to T
    order = np.argsort(-T)
    Survival = Survival[order, :]
    T = T[order]
    E = E[order]

    if t_max is None or t_max <= 0.:
        t_max = max(T)

    # Calculating the brier scores at each t <= t_max
    results = _brier_score(Survival, T, E, t_max, times, time_buckets,
                           use_mean_point)
    times = results[0]
    brier_scores = results[1]

    return (times, brier_scores)

def ensemble_integrated_brier_score(survival_times, model, T, E, t_max=None, use_mean_point=True, figure_size = (20,6.5)):
    """ The Integrated Brier Score (IBS) provides an overall calculation of
        the model performance at all available times.
    """

    # Computing the brier scores
    times, brier_scores = ensemble_brier_score(survival_times, model, T, E, t_max, use_mean_point)

    # Getting the proper value of t_max
    if t_max is None:
        t_max = max(times)
    else:
        t_max = min(t_max, max(times))

    # Computing the IBS
    ibs_value = np.trapz(brier_scores, times) / t_max

    title = 'Prediction error curve with IBS(t = {:.1f}) = {:.2f}'
    title = title.format(t_max, ibs_value)
    fig, ax = plt.subplots(figsize=figure_size)
    ax.plot( times, brier_scores, color = 'blue', lw = 3)
    ax.set_xlim(-0.01, max(times))
    ax.axhline(y=0.25, ls = '--', color = 'red')
    ax.text(0.90*max(times), 0.235, '0.25 limit', fontsize=20, color='brown',
        fontweight='bold')
    plt.title(title, fontsize=20)
    plt.show()
    return ibs_value

def generate_rsf(mask_form = 'automated', mode = 'ct'):
    training_clin = pd.read_csv('***')
    validation_clin = pd.read_csv('***')
    test_clin = pd.read_csv('***')

    if mask_form == 'automated_radiomics':
        if mode == "ct":
            training_set = pd.read_csv('***')
            validation_set = pd.read_csv('***')
            test_set = pd.read_csv('***')
        else:
            training_set = pd.read_csv('***')
            validation_set = pd.read_csv('***')
            test_set = pd.read_csv('***')
    elif mask_form == 'manual_radiomics':
        if mode == 'ct':
            training_set = pd.read_csv('***')
            validation_set = pd.read_csv('***')
            test_set = pd.read_csv('***')
        else:
            training_set = pd.read_csv('***')
            validation_set = pd.read_csv('***')
            test_set = pd.read_csv('***')
    elif mask_form == 'automated_DL':
        if mode == "ct":
            training_set =  pd.read_csv('***')
            validation_set = pd.read_csv('***')
            test_set = pd.read_csv('***')
        else:
            training_set =  pd.read_csv('***')
            validation_set = pd.read_csv('***')
            test_set = pd.read_csv('***')
    elif mask_form=='manual_DL':
        if mode == "ct":
            training_set =  pd.read_csv('***')
            validation_set = pd.read_csv('***')
            test_set = pd.read_csv('***')
        else:
            training_set =  pd.read_csv('***')
            validation_set = pd.read_csv('***')
            test_set = pd.read_csv('***')
    elif mask_form =="clinical":
        training_set = pd.read_csv('***')
        validation_set = pd.read_csv('***')
        test_set = pd.read_csv('***')
    else:
        print("invalid mask_form")

    if mask_form == "clinical":
        pass
    else:
        training_set = pd.concat([training_set, training_clin], axis=1)
        validation_set = pd.concat([validation_set, validation_clin], axis = 1)
        test_set = pd.concat([test_set, test_clin], axis=1)

    index_train = pd.read_csv('***')
    index_vali = pd.read_csv('***')
    index_test = pd.read_csv('***')

    index_train.set_index('accession', inplace=True)
    training_set.set_index('accession', inplace=True)
    training_set = training_set.reindex(index_train.index)
    index_train.reset_index(inplace=True)
    training_set.reset_index(inplace=True)

    index_vali.set_index('accession', inplace=True)
    validation_set.set_index('accession', inplace=True)
    validation_set = validation_set.reindex(index_vali.index)
    index_vali.reset_index(inplace=True)
    validation_set.reset_index(inplace=True)

    index_test.set_index('accession', inplace=True)
    test_set.set_index('accession', inplace=True)
    test_set = test_set.reindex(index_test.index)
    index_test.reset_index(inplace=True)
    test_set.reset_index(inplace=True)

    training_set = training_set[training_set["vital_status"].astype(int) != 2]
    validation_set = validation_set[validation_set["vital_status"].astype(int) != 2]
    test_set = test_set[test_set["vital_status"].astype(int) != 2]

    #non harmonized features
    if mask_form=='automated_radiomics':
        if mode == 'ct':
            clinical_features = ["age", "ct_automated_volume", "gender", "race", "missing_race", "suv"]
            extracted_features = [col for col in training_set.columns if col.startswith('original')]
        else:
            clinical_features = ["age", "pet_automated_volume", "gender", "race", "missing_race", "suv"]
            extracted_features = [col for col in training_set.columns if col.startswith('original')]
    elif mask_form =='manual_radiomics':
        if mode == 'ct':
            clinical_features = ["age",'ct_manual_volume', "gender", "race", "missing_race", "suv"]
            extracted_features = [col for col in training_set.columns if col.startswith('original')]
        else:
            clinical_features = ["age", 'pet_manual_volume', "gender", "race", "missing_race", "suv"]
            extracted_features = [col for col in training_set.columns if col.startswith('original')]
    elif mask_form == 'automated_DL':
        if mode == 'ct':
            clinical_features = ["age", "ct_automated_volume", "gender", "race", "missing_race", "suv"]
            extracted_features = ["feature{}".format(str(i)) for i in range(1,17)]
        else:
            clinical_features = ["age","pet_automated_volume", "gender", "race", "missing_race", "suv"]
            extracted_features = ["feature{}".format(str(i)) for i in range(1,17)]
    elif mask_form == "manual_DL":
        if mode == 'ct':
            clinical_features = ["age", "ct_manual_volume", "gender", "race", "missing_race", "suv"]
            extracted_features = ["feature{}".format(str(i)) for i in range(1,17)]
        else:
            clinical_features = ["age","pet_manual_volume", "gender", "race", "missing_race", "suv"]
            extracted_features = ["feature{}".format(str(i)) for i in range(1,17)]
    elif mask_form == "clinical":
        if mode == "automated_clin":
            clinical_features = ["age", "ct_automated_volume", "pet_automated_volume", "gender", "race", "missing_race", "suv"]
            extracted_features = ["age", "ct_automated_volume","pet_automated_volume", "gender", "race", "missing_race", "suv"]
        elif mode == "manual_clin":
            clinical_features = ["age", "ct_manual_volume", "pet_manual_volume", "gender", "race", "missing_race","suv"]
            extracted_features = ["age", "ct_manual_volume", "pet_manual_volume", "gender", "race","missing_race", "suv"]
        else:
            print("invalid clin mode")
    else:
        print("invalid mask_form")

    #####harmonized features
    # if mask_form=='automated_radiomics':
    #     extracted_features = [str(i) for i in range(107)]
    # elif mask_form =='manual_radiomics':
    #     extracted_features = [str(i) for i in range(107)]
    # elif mask_form == 'automated_DL':
    #     extracted_features = [str(i) for i in range(16)]
    # elif mask_form == "manual_DL":
    #     extracted_features = [str(i) for i in range(16)]
    # elif mask_form == "clinical":
    #     if mode == "automated_clin":
    #         clinical_features = ["age", "ct_automated_volume", "pet_automated_volume", "gender", "race", "missing_race", "suv"]
    #         extracted_features = ["age", "ct_automated_volume","pet_automated_volume", "gender", "race", "missing_race", "suv"]
    #     elif mode == "manual_clin":
    #         clinical_features = ["age", "ct_manual_volume", "pet_manual_volume", "gender", "race", "missing_race","suv"]
    #         extracted_features = ["age", "ct_manual_volume", "pet_manual_volume", "gender", "race","missing_race", "suv"]
    #     else:
    #         print("invalid clin mode")
    # else:
    #     print("invalid mask_form")

    survival_label = "os"

    #vital_status = 1: Alive, censored; vital_status = 0: deceased, uncensored
    censored_label = "vital_status"

    #combining training and validation from feature extraction into new training set
    full_training_set = pd.concat([training_set, validation_set])
    full_training_set = full_training_set[full_training_set["accession"] != "Lung-Brown-558"]

    train_labels = full_training_set[survival_label]
    # if censored_label == "vital_status_x":
    if censored_label == "vital_status":
        train_censor = pd.Series([1 if x == 0 else 0 for x in full_training_set[censored_label]])
    else:
        train_censor = full_training_set[censored_label]
    train_image_feat = full_training_set[extracted_features]

    test_labels = test_set[survival_label]
    # if censored_label == "vital_status_x":
    if censored_label == "vital_status":
        test_censor = pd.Series([1 if x == 0 else 0 for x in test_set[censored_label]])
    else:
        test_censor = test_set[censored_label]
        test_image_feat = test_set[extracted_features]
        test_clinical_feat = test_set[clinical_features]

    seed = 42
    n_estimators = 30
    max_depth = 20
    # min_samples_split = 8
    min_samples_node = 10
    max_features = "log2"
    sample_size_pct = 0.4
    n_jobs = 6
    bootstrap = True

    rsf = RandomSurvivalForestModel(num_trees=n_estimators)

    if mask_form == "clinical":
        train_clinical_feat = full_training_set[clinical_features]
        rsf.fit(train_clinical_feat, train_labels, train_censor, max_features=max_features, max_depth=max_depth,
                     min_node_size=min_samples_node, sample_size_pct=sample_size_pct, seed=seed)
    else:
        rsf.fit(train_image_feat, train_labels, train_censor, max_features = max_features, max_depth = max_depth,
        min_node_size = min_samples_node, sample_size_pct = sample_size_pct, seed = seed)
    return rsf

def gen_bootstrap(ct_rsf, pet_rsf, clin_rsf, iterations = 100, mask_form = 'automated'):

    ct_cind = []
    ct_ibs = []

    pet_cind = []
    pet_ibs = []

    clin_cind = []
    clin_ibs = []

    ctpet_cind = []
    ctpet_ibs = []

    full_cind = []
    full_ibs = []

    if mask_form == 'automated_radiomics':
        ct_test_set = pd.read_csv('***')
        pet_test_set = pd.read_csv('***')
        clin_test_set = pd.read_csv('***')
    elif mask_form == 'manual_radiomics':
        ct_test_set = pd.read_csv('***')
        pet_test_set = pd.read_csv('***')
        clin_test_set = pd.read_csv('***')
    elif mask_form == 'automated_DL':
        ct_test_set = pd.read_csv('***')
        pet_test_set = pd.read_csv('***')
        clin_test_set = pd.read_csv('***')
        # manual DL
    elif mask_form == 'manual_DL':
        ct_test_set = pd.read_csv('***')
        pet_test_set = pd.read_csv('***')
        clin_test_set = pd.read_csv('***')
    else:
        print("invalid mask_form")

    for i in range(iterations):
        indices = ct_test_set.shape[0]
        random_indices = list(np.random.choice(indices,size = round(2/3*indices),replace = False))
        random_ct_rows = ct_test_set.iloc[random_indices]
        random_pet_rows = pet_test_set.iloc[random_indices]
        random_clin_rows = clin_test_set.iloc[random_indices]

        if mask_form == "automated_radiomics" or mask_form == 'automated_DL':
            clinical_features = ["age", "ct_automated_volume", "pet_automated_volume", "gender", "race", "missing_race", "suv"]
        elif mask_form == 'manual_DL' or mask_form == 'manual_radiomics':
            clinical_features = ["age", "ct_manual_volume", "pet_manual_volume", "gender", "race", "missing_race", "suv"]
        else:
            print("invalid mask_form")

            #harmonized features:#######
        if mask_form == 'automated_radiomics' or mask_form == 'manual_radiomics':
            ct_extracted_features = [col for col in training_set.columns if col.startswith('original')]
            pet_extracted_features = [col for col in training_set.columns if col.startswith('original')]
        else:
            ct_extracted_features = ["feature{}".format(str(i)) for i in range(1,17)]
            pet_extracted_features = ["feature{}".format(str(i)) for i in range(1,17)]
        survival_label = "os"

        # vital_status = 1: Alive, censored; vital_status = 0: deceased, uncensored
        censored_label = "vital_status"
        ct_test_labels = random_clin_rows[survival_label]
        pet_test_labels = random_clin_rows[survival_label]

        ct_test_censor = pd.Series([1 if x == 0 else 0 for x in random_clin_rows[censored_label]])
        pet_test_censor = pd.Series([1 if x == 0 else 0 for x in random_clin_rows[censored_label]])

        ct_test_image_feat = random_ct_rows[ct_extracted_features]
        pet_test_image_feat = random_pet_rows[pet_extracted_features]

        test_clinical_feat = random_clin_rows[clinical_features]

        ct_c_index = concordance_index(ct_rsf, ct_test_image_feat, ct_test_labels, ct_test_censor)
        ct_cind.append(ct_c_index)
        print('CT_C-index: {:.3f}'.format(ct_c_index))
        ct_ibs_score = plotless_integrated_brier_score(ct_rsf, ct_test_image_feat, ct_test_labels, ct_test_censor,
                    figure_size=(20, 6.5))
        ct_ibs.append(ct_ibs_score)

        pet_c_index = concordance_index(pet_rsf, pet_test_image_feat, pet_test_labels, pet_test_censor)
        pet_cind.append(pet_c_index)
        #print('PET_C-index: {:.3f}'.format(pet_c_index))
        pet_ibs_score = plotless_integrated_brier_score(pet_rsf, pet_test_image_feat, pet_test_labels, pet_test_censor,
                    figure_size=(20, 6.5))
        pet_ibs.append(pet_ibs_score)

        clin_c_index = concordance_index(clin_rsf, test_clinical_feat, ct_test_labels, ct_test_censor)
        clin_cind.append(clin_c_index)
        #print('Clin_C-index: {:.3f}'.format(clin_c_index))
        clin_ibs_score = plotless_integrated_brier_score(clin_rsf, test_clinical_feat, ct_test_labels, ct_test_censor,
                                                        figure_size=(20, 6.5))
        clin_ibs.append(pet_ibs_score)

        ct_risk = ct_rsf.predict_risk(ct_test_image_feat)
        pet_risk = pet_rsf.predict_risk(pet_test_image_feat)
        clin_risk = clin_rsf.predict_risk(test_clinical_feat)

        ct_survival = ct_rsf.predict_survival(ct_test_image_feat)
        pet_survival = pet_rsf.predict_survival(pet_test_image_feat)
        clin_survival = clin_rsf.predict_survival(test_clinical_feat)

        ctpet_risk = (ct_risk + pet_risk)/2
        ctpetclin_risk = (ct_risk + pet_risk + clin_risk)/3

        ctpet_surv = (ct_survival + pet_survival)/2
        ctpetclin_surv = (ct_survival + pet_survival + clin_survival)/3

        ctpet_c_index = ensemble_concordance_index(ctpet_risk, ct_test_labels, ct_test_censor)
        #print('CT + PET Ensemble C-index: {:.3f}'.format(ctpet_c_index))
        ctpet_cind.append(ctpet_c_index)

        ctpet_ibs_score = ensemble_integrated_brier_score(ctpet_surv, ct_rsf, ct_test_labels, ct_test_censor,
                                              figure_size=(20, 6.5))
        #print('CT + PET Ensemble IBS: {:.3f}'.format(ctpet_ibs_score))
        ctpet_ibs.append(ctpet_ibs_score)

        ctpetclin_c_index = ensemble_concordance_index(ctpetclin_risk, ct_test_labels, ct_test_censor)
        #print('CT + PET + Clin Ensemble C-index: {:.3f}'.format(ctpetclin_c_index))
        full_cind.append(ctpetclin_c_index)

        ctpetclin_ibs_score = ensemble_integrated_brier_score(ctpetclin_surv, ct_rsf, ct_test_labels, ct_test_censor,
                                              figure_size=(20, 6.5))
        #print('CT + PET + Clin Ensemble IBS: {:.3f}'.format(ctpetclin_ibs_score))
        full_ibs.append(ctpetclin_ibs_score)
    return ct_cind, ct_ibs, pet_cind, pet_ibs, clin_cind, clin_ibs, ctpet_cind, ctpet_ibs, full_cind, full_ibs

def generate_RSF_results(mask_form = 'automated_radiomics', iterations=100):
    ct_rsf = generate_rsf(mask_form, "ct")
    pet_rsf = generate_rsf(mask_form, 'pet')
    if mask_form == 'automated_radiomics' or mask_form =='automated_DL':
        clin_rsf = generate_rsf("clinical", "automated_clin")
    elif mask_form == 'manual_radiomics' or mask_form == 'manual_DL':
        clin_rsf = generate_rsf("clinical", "manual_clin")

    ct_cind, ct_ibs, pet_cind, pet_ibs, clin_cind, clin_ibs, ctpet_cind, ctpet_ibs, full_cind, full_ibs = gen_bootstrap(ct_rsf, pet_rsf, clin_rsf, iterations, mask_form)

    dataframe_index = ["ct", "pet", "clin", "ct_pet", "ct_pet_clin"]
    c_ind_rows = [ct_cind, pet_cind, clin_cind, ctpet_cind, full_cind]
    ibs_rows = [ct_ibs, pet_ibs, clin_ibs, ctpet_ibs, full_ibs]
    all_c_ind = pd.DataFrame(data = c_ind_rows, index = dataframe_index)
    all_ibs = pd.DataFrame(data=ibs_rows, index = dataframe_index)

    # harm_fldr = "***"
    bootstrap_fldr = "***"
    if mask_form == 'automated_radiomics':
        all_c_ind.to_csv(os.path.join(bootstrap_fldr,"***"))
        all_ibs.to_csv(os.path.join(bootstrap_fldr, "***"))
    elif mask_form == 'manual_radiomics':
        all_c_ind.to_csv(os.path.join(bootstrap_fldr, "***"))
        all_ibs.to_csv(os.path.join(bootstrap_fldr, "***"))
    elif mask_form == 'manual_DL':
        all_c_ind.to_csv(os.path.join(bootstrap_fldr, "***"))
        all_ibs.to_csv(os.path.join(bootstrap_fldr, "***"))
    elif mask_form == 'automated_DL':
        all_c_ind.to_csv(os.path.join(bootstrap_fldr, "***"))
        all_ibs.to_csv(os.path.join(bootstrap_fldr, "***"))
    else:
        print("invalid mask_form")

generate_RSF_results('automated_radiomics', iterations=100)
generate_RSF_results('manual_radiomics', iterations=100)
generate_RSF_results('automated_DL', iterations=100)
generate_RSF_results('manual_DL', iterations=100)

print("Complete")