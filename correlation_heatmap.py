import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

file_folder = "***"
file_folder_rad = "***"

ct_manual_dl = os.path.join(file_folder, "***")
ct_automasks_dl = os.path.join(file_folder, "***")

ct_manual_radiomics = os.path.join(file_folder_rad, "***")
ct_automasks_radiomics = os.path.join(file_folder_rad, "***")

pet_manual_dl = os.path.join(file_folder, "***")
pet_manual_radiomics = os.path.join(file_folder, "***")

pet_automasks_dl = os.path.join(file_folder, "***")
pet_automasks_radiomics = os.path.join(file_folder, "***")

feature1 = pd.read_csv(pet_automasks_dl)
feature2 = pd.read_csv(pet_automasks_radiomics)

full_feat = pd.concat([feature1, feature2], axis=1).corr()

extracted_features_rad = [col for col in feature2.columns if col.startswith('original')]
extracted_features_dl = ["feature{}".format(str(i)) for i in range(1, 17)]

new_dl_name = ["DL Feat. {}".format(str(i)) for i in range(1, 17)]
new_rad_name = ["Rad. Feat. {}".format(str(i)) for i in range(1, len(extracted_features_rad)+1)]

#CT manual
# sorted_feat = ***

#PET manual
# sorted_feat = ***

#CT automatic
# sorted_feat = ***

#PET manual

#for example
sorted_feat = ['feature11', 'feature6', 'feature16', 'feature5', 'feature15', 'feature4', 'feature3', 'feature13', 'feature7', 'feature12', 'feature14', 'feature9', 'feature1', 'feature2', 'feature8', 'feature10']

numbers = [i.split('e')[2] for i in sorted_feat]
updated_names = ["DL Feat. {}".format(str(i)) for i in numbers]
full_feat=full_feat.loc[extracted_features_dl, extracted_features_rad]
full_feat = full_feat.loc[sorted_feat, :].abs()
full_feat = full_feat.set_axis(updated_names, axis = 0)
dl_name_update = []
full_feat = full_feat.set_axis(new_rad_name, axis=1)

full_feat['nan'] = 0
full_feat['Sum'] = full_feat[list(full_feat.columns)].sum(axis=1)
full_feat['nan'] = 0

max = full_feat['Sum'].max()
min = full_feat['Sum'].min()
full_feat['Sum'] = (full_feat['Sum'] - min)/(max - min)

f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
plt.title('PET CNN (Automatic Masks)')
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(full_feat, cmap="Blues" , vmax=1,
            square=True, xticklabels = "auto", linewidths=.7, cbar_kws={"shrink": .4})
plt.show()
print("test")