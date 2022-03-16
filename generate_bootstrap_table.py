import pandas as pd
from config import config
import functools
import numpy as np
import statistics as stat
import json
import sklearn.metrics as metrics
import os
import scipy

def conjunction(*conditions):
    return functools.reduce(np.logical_and, conditions)

def orconjunction(*conditions):
    return functools.reduce(np.logical_or, conditions)

folder_path = "***"

auto_rad_cind_f = "***"
manual_rad_cind_f = "***"

auto_DL_cind_f = "***"
manual_DL_cind_f = "***"

auto_rad_ibs_f = "***"
manual_rad_ibs_f = "***"

auto_DL_ibs_f = "***"
manual_DL_ibs_f = "***"

auto_rad_cind = pd.read_csv(os.path.join(folder_path, auto_rad_cind_f))
auto_rad_ibs = pd.read_csv(os.path.join(folder_path, auto_rad_ibs_f))

auto_DL_cind = pd.read_csv(os.path.join(folder_path, auto_DL_cind_f))
auto_DL_ibs = pd.read_csv(os.path.join(folder_path, auto_DL_ibs_f))

manual_rad_cind = pd.read_csv(os.path.join(folder_path, manual_rad_cind_f))
manual_rad_ibs = pd.read_csv(os.path.join(folder_path, manual_rad_ibs_f))

manual_DL_cind = pd.read_csv(os.path.join(folder_path, manual_DL_cind_f))
manual_DL_ibs = pd.read_csv(os.path.join(folder_path, manual_DL_ibs_f))

# folder_path_harm = "***"
#
# auto_rad_cind_f_harm = "***"
# manual_rad_cind_f_harm = "***"
# auto_DL_cind_f_harm = "***"
# manual_DL_cind_f_harm = "***"
#
# auto_rad_ibs_f_harm = "***"
# manual_rad_ibs_f_harm = "***"
# auto_DL_ibs_f_harm ="***"
# manual_DL_ibs_f_harm = "***"
#
# auto_rad_cind_harm = pd.read_csv(os.path.join(folder_path_harm, auto_rad_cind_f_harm))
# auto_rad_ibs_harm = pd.read_csv(os.path.join(folder_path_harm, auto_rad_ibs_f_harm))
# auto_DL_cind_harm = pd.read_csv(os.path.join(folder_path_harm, auto_DL_cind_f_harm))
# auto_DL_ibs_harm = pd.read_csv(os.path.join(folder_path_harm, auto_DL_ibs_f_harm))
# manual_rad_cind_harm = pd.read_csv(os.path.join(folder_path_harm, manual_rad_cind_f_harm))
# manual_rad_ibs_harm = pd.read_csv(os.path.join(folder_path_harm, manual_rad_ibs_f_harm))
# manual_DL_cind_harm = pd.read_csv(os.path.join(folder_path_harm, manual_DL_cind_f_harm))
# manual_DL_ibs_harm = pd.read_csv(os.path.join(folder_path_harm, manual_DL_ibs_f_harm))

def process_df(model):
    df = pd.read_csv(model).transpose()
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    return df

models = ["auto_rad", "manual_rad", "auto_DL", "manual_DL"]
results = ["ct", "pet", "clin", "ct_pet", "ct_pet_clin"]

result_dict = {
  "ct": 0,
  "pet": 1, "clin": 2,
  "ct_pet": 3,
  "ct_pet_clin": 4
}

results_table = pd.DataFrame(index = models, columns=results)

for i in models:
    if i == "auto_rad":
        df = auto_rad_cind
    elif i== "manual_rad":
        df = manual_rad_cind
    elif i == "auto_DL":
        df = auto_DL_cind
    elif i=="manual_DL":
        df = manual_DL_cind
    elif i == "cnn_auto":
        df = cnn_auto
    elif i == "cnn_manual":
        df = cnn_manual
    else:
        print("error")
    df = df.iloc[:, 1:]
    for j in results:
        index = result_dict[j]
        lst = [float(i) for i in df.iloc[index]]
        mean = round(stat.mean(lst), 3)
        interval = round(1.96*scipy.stats.sem(lst),3)
        lower_ci = round(np.percentile(lst, 2.5),3)
        upper_ci = round(np.percentile(lst, 97.5),3)
        string = str(mean) + " (" + str(lower_ci) +", " +str(upper_ci) + ")"
        results_table.at[i, j] = string

results_table.to_csv(os.path.join(folder_path, "***"))