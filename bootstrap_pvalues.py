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
folder_path_harm = "***"

auto_rad_cind_f = "***"
manual_rad_cind_f = "***"
auto_DL_cind_f = "***"
manual_DL_cind_f = "***"

auto_rad_ibs_f = "***"
manual_rad_ibs_f = "***"
auto_DL_ibs_f = "***"
manual_DL_ibs_f = "***"

auto_rad_cind_f_harm = "***"
manual_rad_cind_f_harm = "***"
auto_DL_cind_f_harm = "***"
manual_DL_cind_f_harm = "***"

auto_rad_ibs_f_harm = "***"
manual_rad_ibs_f_harm = "***"
auto_DL_ibs_f_harm = "***"
manual_DL_ibs_f_harm = "***"

auto_rad_cind = pd.read_csv(os.path.join(folder_path, auto_rad_cind_f))
auto_rad_ibs = pd.read_csv(os.path.join(folder_path, auto_rad_ibs_f))
auto_DL_cind = pd.read_csv(os.path.join(folder_path, auto_DL_cind_f))
auto_DL_ibs = pd.read_csv(os.path.join(folder_path, auto_DL_ibs_f))
manual_rad_cind = pd.read_csv(os.path.join(folder_path, manual_rad_cind_f))
manual_rad_ibs = pd.read_csv(os.path.join(folder_path, manual_rad_ibs_f))
manual_DL_cind = pd.read_csv(os.path.join(folder_path, manual_DL_cind_f))
manual_DL_ibs = pd.read_csv(os.path.join(folder_path, manual_DL_ibs_f))

auto_rad_cind_harm = pd.read_csv(os.path.join(folder_path_harm, auto_rad_cind_f_harm))
auto_rad_ibs_harm = pd.read_csv(os.path.join(folder_path_harm, auto_rad_ibs_f_harm))
auto_DL_cind_harm = pd.read_csv(os.path.join(folder_path_harm, auto_DL_cind_f_harm))
auto_DL_ibs_harm = pd.read_csv(os.path.join(folder_path_harm, auto_DL_ibs_f_harm))
manual_rad_cind_harm = pd.read_csv(os.path.join(folder_path_harm, manual_rad_cind_f_harm))
manual_rad_ibs_harm = pd.read_csv(os.path.join(folder_path_harm, manual_rad_ibs_f_harm))
manual_DL_cind_harm = pd.read_csv(os.path.join(folder_path_harm, manual_DL_cind_f_harm))
manual_DL_ibs_harm = pd.read_csv(os.path.join(folder_path_harm, manual_DL_ibs_f_harm))


def process_df(model):
    df = pd.read_csv(model).transpose()
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    return df

models = ["auto_rad", "manual_rad", "auto_DL", "manual_DL"]
results = ["ct", "pet", "clin", "ct_pet","ct_pet_clin"]

result_dict = {
  "ct": 0,
  "pet": 1, "clin": 2,
  "ct_pet": 3,
  "ct_pet_clin": 4
}

results_table = pd.DataFrame(index = models, columns=results)

def pval(df1,df2, modality1, modality2):
    index1 = result_dict[modality1]
    index2 = result_dict[modality2]
    df1 = df1.iloc[:, 1:]
    df2 = df2.iloc[:, 1:]
    lst1 = [float(i) for i in df1.iloc[index1]]
    lst2 = [float(i) for i in df2.iloc[index2]]
    diff = []
    zip_object = zip(lst1, lst2)
    for list1_i, list2_i in zip_object:
        diff.append(list1_i - list2_i)
    overz = len([i for i in diff if i>0])/len(diff)
    underz = len([i for i in diff if i<0])/len(diff)
    small = min(overz, underz)
    pval = 2*small
    return pval

ctpet = pval(manual_DL_cind, manual_DL_cind, "ct", "pet")
ctclin = pval(manual_DL_cind, manual_DL_cind, "ct", "clin")
clinpet = pval(manual_DL_cind, manual_DL_cind, "clin", "pet")

ct_ctpet = pval(manual_DL_cind, manual_DL_cind, "ct", "ct_pet")
pet_ctpet = pval(manual_DL_cind, manual_DL_cind, "pet", "ct_pet")
clin_ctpet = pval(manual_DL_cind, manual_DL_cind, "clin", "ct_pet")

ct_ctpetcllin = pval(manual_DL_cind, manual_DL_cind, "ct", "ct_pet_clin")
pet_ctpetclin = pval(manual_DL_cind, manual_DL_cind, "pet", "ct_pet_clin")
clin_ctpetclin = pval(manual_DL_cind, manual_DL_cind, "clin", "ct_pet_clin")

ctpet_ensemble = pval(manual_DL_cind, manual_DL_cind, "ct_pet", "ct_pet_clin")

ctpet = pval(auto_DL_cind, auto_DL_cind, "ct", "pet")
ctclin = pval(auto_DL_cind, auto_DL_cind, "ct", "clin")
clinpet = pval(auto_DL_cind, auto_DL_cind, "clin", "pet")

ct_ctpet = pval(auto_DL_cind, auto_DL_cind, "ct", "ct_pet")
pet_ctpet = pval(auto_DL_cind, auto_DL_cind, "pet", "ct_pet")
clin_ctpet = pval(auto_DL_cind, auto_DL_cind, "clin", "ct_pet")

ct_ctpetcllin = pval(auto_DL_cind, auto_DL_cind, "ct", "ct_pet_clin")
pet_ctpetclin = pval(auto_DL_cind, auto_DL_cind, "pet", "ct_pet_clin")
clin_ctpetclin = pval(auto_DL_cind, auto_DL_cind, "clin", "ct_pet_clin")

ctpet_ensemble = pval(auto_DL_cind, auto_DL_cind, "ct_pet", "ct_pet_clin")

# ctpet = pval(auto_rad_cind, auto_rad_cind, "ct", "pet")
# ctclin = pval(auto_rad_cind, auto_rad_cind, "ct", "clin")
# clinpet = pval(auto_rad_cind, auto_rad_cind, "clin", "pet")
#
# ct_ctpet = pval(auto_rad_cind, auto_rad_cind, "ct", "ct_pet")
# pet_ctpet = pval(auto_rad_cind, auto_rad_cind, "pet", "ct_pet")
# clin_ctpet = pval(auto_rad_cind, auto_rad_cind, "clin", "ct_pet")
#
# ct_ctpetcllin = pval(auto_rad_cind, auto_rad_cind, "ct", "ct_pet_clin")
# pet_ctpetclin = pval(auto_rad_cind, auto_rad_cind, "pet", "ct_pet_clin")
# clin_ctpetclin = pval(auto_rad_cind, auto_rad_cind, "clin", "ct_pet_clin")
#
# ctpet_ensemble = pval(auto_rad_cind, auto_rad_cind, "ct_pet", "ct_pet_clin")

# ctct = pval(manual_DL_cind, manual_rad_cind, "ct", "ct")
# petpet = pval(manual_DL_cind, manual_rad_cind, "pet", "pet")
# clinclin = pval(manual_DL_cind, manual_rad_cind, "clin", "clin")
# ctpet = pval(manual_DL_cind, manual_rad_cind, "ct_pet", "ct_pet")
# ctpetclin = pval(manual_DL_cind, manual_rad_cind, "ct_pet_clin", "ct_pet_clin")

# ctct = pval(manual_DL_cind, auto_DL_cind, "ct", "ct")
# petpet = pval(manual_DL_cind, auto_DL_cind, "pet", "pet")
# clinclin = pval(manual_DL_cind, auto_DL_cind, "clin", "clin")
# ctpet = pval(manual_DL_cind, auto_DL_cind, "ct_pet", "ct_pet")
# ctpetclin = pval(manual_DL_cind, auto_DL_cind, "ct_pet_clin", "ct_pet_clin")

# ctct = pval(manual_rad_cind, auto_rad_cind, "ct", "ct")
# petpet = pval(manual_rad_cind, auto_rad_cind, "pet", "pet")
# clinclin = pval(manual_rad_cind, auto_rad_cind, "clin", "clin")
# ctpet = pval(manual_rad_cind, auto_rad_cind, "ct_pet", "ct_pet")
# ctpetclin = pval(manual_rad_cind, auto_rad_cind, "ct_pet_clin", "ct_pet_clin")
print("test complete")
