#script to process radiomics features csvs before feeding into random survival forest
import pandas as pd
from config import config
import numpy as np
import os
from sklearn.impute import SimpleImputer

#load in clinical csvs
train = pd.read_csv(os.path.join(config.DATASET_RECORDS, "***"))
validation = pd.read_csv(os.path.join(config.DATASET_RECORDS, "***"))
test = pd.read_csv(os.path.join(config.DATASET_RECORDS, "***"))

#load in radiomics features csvs
train_rad_pred = pd.read_csv(os.path.join(config.DATASET_RECORDS, "***"))
validation_rad_pred = pd.read_csv(os.path.join(config.DATASET_RECORDS,  "***"))
test_rad_pred = pd.read_csv(os.path.join(config.DATASET_RECORDS,  "***"))
train_rad = pd.read_csv(os.path.join(config.DATASET_RECORDS,  "***"))
validation_rad = pd.read_csv(os.path.join(config.DATASET_RECORDS,  "***"))
test_rad = pd.read_csv(os.path.join(config.DATASET_RECORDS, "***"))

PETtrain_rad_pred = pd.read_csv(os.path.join(config.DATASET_RECORDS, "***"))
PETvalidation_rad_pred = pd.read_csv(os.path.join(config.DATASET_RECORDS,  "***"))
PETtest_rad_pred = pd.read_csv(os.path.join(config.DATASET_RECORDS, "***"))
PETtrain_rad = pd.read_csv(os.path.join(config.DATASET_RECORDS,  "***"))
PETvalidation_rad = pd.read_csv(os.path.join(config.DATASET_RECORDS,  "***"))
PETtest_rad = pd.read_csv(os.path.join(config.DATASET_RECORDS, "***"))

#drop all columns that are not radiomics features to be used in RSF model
train_rad.drop(train_rad.columns[range(25)], axis=1, inplace=True)
train_rad_pred.drop(train_rad_pred.columns[range(25)], axis=1, inplace=True)
validation_rad.drop(validation_rad.columns[range(25)], axis=1, inplace=True)
validation_rad_pred.drop(validation_rad_pred.columns[range(25)], axis=1, inplace=True)
test_rad.drop(test_rad.columns[range(25)], axis=1, inplace=True)
test_rad_pred.drop(test_rad_pred.columns[range(25)], axis=1, inplace=True)

PETtrain_rad.drop(PETtrain_rad.columns[range(25)], axis=1, inplace=True)
PETtrain_rad_pred.drop(PETtrain_rad_pred.columns[range(25)], axis=1, inplace=True)
PETvalidation_rad.drop(PETvalidation_rad.columns[range(25)], axis=1, inplace=True)
PETvalidation_rad_pred.drop(PETvalidation_rad_pred.columns[range(25)], axis=1, inplace=True)
PETtest_rad.drop(PETtest_rad.columns[range(25)], axis=1, inplace=True)
PETtest_rad_pred.drop(PETtest_rad_pred.columns[range(25)], axis=1, inplace=True)


#impute missing data for radiomics features extracted from automatically segmented masks
for column in train_rad_pred:
    train_rad_pred[column]= train_rad_pred[column].fillna(train_rad_pred[column].mean())

for column in PETtrain_rad_pred:
    PETtrain_rad_pred[column]= PETtrain_rad_pred[column].fillna(PETtrain_rad_pred[column].mean())

for column in PETtrain_rad:
    PETtrain_rad[column]= PETtrain_rad[column].fillna(PETtrain_rad[column].mean())

for column in validation_rad_pred:
    validation_rad_pred[column]= validation_rad_pred[column].fillna(validation_rad_pred[column].mean())

for column in PETvalidation_rad_pred:
    PETvalidation_rad_pred[column] = PETvalidation_rad_pred[column].fillna(PETvalidation_rad_pred[column].mean())

for column in PETvalidation_rad:
    PETvalidation_rad[column] = PETvalidation_rad[column].fillna(PETvalidation_rad[column].mean())
    # print(validation_rad_pred[column])

for column in test_rad_pred:
    test_rad_pred[column]= test_rad_pred[column].fillna(test_rad_pred[column].mean())

for column in PETtest_rad_pred:
    PETtest_rad_pred[column] = PETtest_rad_pred[column].fillna(PETtest_rad_pred[column].mean())
    # print(test_rad_pred[column])

#concatenate clinical csvs with radiomics features to reduce clutter
train_feat = pd.concat([train, train_rad], axis=1)
train_feat_pred = pd.concat([train, train_rad_pred], axis=1)
validation_feat = pd.concat([validation, validation_rad], axis=1)
validation_feat_pred = pd.concat([validation, validation_rad_pred], axis=1)
test_feat = pd.concat([test, test_rad], axis=1)
test_feat_pred = pd.concat([test, test_rad_pred], axis=1)

PETtrain_feat = pd.concat([train, PETtrain_rad], axis=1)
PETtrain_feat_pred = pd.concat([train, PETtrain_rad_pred], axis=1)
PETvalidation_feat = pd.concat([validation, PETvalidation_rad], axis=1)
PETvalidation_feat_pred = pd.concat([validation, PETvalidation_rad_pred], axis=1)
PETtest_feat = pd.concat([test, PETtest_rad], axis=1)
PETtest_feat_pred = pd.concat([test, PETtest_rad_pred], axis=1)

#save new csvs
train_feat.to_csv( "***", index=False)
train_feat_pred.to_csv( "***", index=False)
validation_feat.to_csv( "***", index=False)
validation_feat_pred.to_csv("***", index=False)
test_feat.to_csv( "***", index=False)
test_feat_pred.to_csv("***", index=False)

PETtrain_feat.to_csv("***", index=False)
PETtrain_feat_pred.to_csv("***", index=False)
PETvalidation_feat.to_csv("***", index=False)
PETvalidation_feat_pred.to_csv("***", index=False)
PETtest_feat.to_csv("***", index=False)
PETtest_feat_pred.to_csv("***", index=False)