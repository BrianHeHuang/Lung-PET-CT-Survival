import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, ParameterGrid
from pysurvival.models.survival_forest import RandomSurvivalForestModel
from pysurvival.utils.metrics import concordance_index,_concordance_index, _brier_score
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

models = ["e620607d-d43f-457e-984f-b923a3b6641f", "da37531f-98fc-4da2-97ae-7e947c0a67e9"]
ensemble_risks = []
ensemble_survivals = []
train_risks = []
model_directory = config.MODEL_DIR

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

    seed = 42
    n_estimators = 30
    max_depth = 20
    min_samples_node = 10
    max_features = "log2"
    sample_size_pct = 0.4
    n_jobs = 6
    bootstrap = True
    #gridsearch parameters

    rsf = RandomSurvivalForestModel(num_trees=n_estimators)
    rsf.fit(train_image_feat, train_labels, train_censor, max_features = max_features, max_depth = max_depth,
        min_node_size = min_samples_node, sample_size_pct = sample_size_pct, seed = seed)

    c_index = concordance_index(rsf, test_image_feat, test_labels, test_censor)
    print('C-index: {:.3f}'.format(c_index))

    ibs = integrated_brier_score(rsf, test_image_feat, test_labels, test_censor,
                figure_size=(20, 6.5))
    print('IBS: {:.3f}'.format(ibs))

    #constuct clin feat only model
    rsf_clin = RandomSurvivalForestModel(num_trees=n_estimators)
    #rsf_clin.fit(train_clinical_feat, train_labels, train_censor, seed= seed)
    rsf_clin.fit(train_clinical_feat, train_labels, train_censor, max_features = max_features, max_depth = max_depth,
        min_node_size = min_samples_node, sample_size_pct = sample_size_pct, seed=seed)

    c_index = concordance_index(rsf_clin, test_clinical_feat, test_labels, test_censor)
    print('Clin_Feat C-index: {:.3f}'.format(c_index))
    ibs = integrated_brier_score(rsf, test_clinical_feat, test_labels, test_censor,
                figure_size=(20, 6.5))
    print('Clin_Feat IBS: {:.3f}'.format(ibs))

    ct_risk = rsf.predict_risk(test_image_feat)
    clin_risk = rsf_clin.predict_risk(test_clinical_feat)
    ensemble_risks.append(ct_risk)
    ensemble_risks.append(clin_risk)

    ct_survival = rsf.predict_survival(test_image_feat)
    clin_survival = rsf_clin.predict_survival(test_clinical_feat)
    ensemble_survivals.append(ct_survival)
    ensemble_survivals.append(clin_survival)

ensemble_risks = ensemble_risks[:-1]
ensemble_survivals = ensemble_survivals[:-1]
ensemble_risk = sum(ensemble_risks)/len(ensemble_risks)
ensemble_survival = (sum(ensemble_survivals))/3

c_index = ensemble_concordance_index(ensemble_risk, test_labels, test_censor)
print('Ensemble C-index: {:.3f}'.format(c_index))

ibs = ensemble_integrated_brier_score(ensemble_survival, rsf, test_labels, test_censor,
            figure_size=(20, 6.5))
print('Ensemble IBS: {:.3f}'.format(ibs))

#kaplan meier curve creation
kmf = KaplanMeierFitter()
full_array = pd.concat([test_labels, test_censor], axis=1)
full_array.rename(columns = {'os' : 'os', 0 : 'censor'}, inplace = True)
KM_curve_array = pd.concat([full_array, pd.Series(ensemble_risk)], axis=1)
low_risk = KM_curve_array[KM_curve_array[0] <= mean_value_train]
high_risk = KM_curve_array[KM_curve_array[0] > mean_value_train]

#construct sub-survival curves
ax = plt.subplot(111)

kmf_control = KaplanMeierFitter()
ax = kmf_control.fit(low_risk["os"], event_observed=low_risk["censor"], label="Low RSF Risk").plot_survival_function(ax=ax)

kmf_exp = KaplanMeierFitter()
ax = kmf_exp.fit(high_risk["os"], event_observed=high_risk["censor"],label = "High RSF Risk").plot_survival_function(ax=ax)

add_at_risk_counts(kmf_exp, kmf_control, ax=ax)
ax.set_xlabel('Time (Days)')
ax.set_ylabel('Percent Survival')
ax.grid()
plt.tight_layout()

plt.title("Survival by High and Low RSF Risk")

#log rank test
logrank = logrank_test(high_risk["os"], low_risk["os"], event_observed_A=high_risk["censor"], event_observed_B=low_risk["censor"])
logrank.print_summary()
print(logrank.p_value)

print("Complete")
