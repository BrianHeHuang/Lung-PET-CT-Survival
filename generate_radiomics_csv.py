# script used to generate formatted csv files specific for train/test/validation splits that can be run with pyradiomics batch
# to generate radiomics features

import os
import glob
import csv
import pandas as pd
from config import config

if config.MODE == "CT":
    preprocess_directory = config.PREPROCESSED_DIR_CT
    # print(preprocess_directory)

    # radiomics features for train set
    train_csv_path = os.path.join(config.DATASET_RECORDS, 'original_train_split.csv')
    df = pd.read_csv(train_csv_path)
    # print(df.head())

    patient_list = df["accession"].tolist()
    # print(patient_list)

    image_loc = '-CT-imagingVolume.nrrd'
    image_list = [x + image_loc for x in patient_list]
    print(image_list)

    mask_loc = '-CT-segMask_pred.nrrd'
    # mask_loc = '-CT-segMask_GTV.nrrd'
    mask_list = [x + mask_loc for x in patient_list]
    print(mask_list)

    dict = {'Patient': patient_list, 'Image': image_list, 'Mask': mask_list}
    df_rad = pd.DataFrame(dict)
    df_rad['Image'] = preprocess_directory + '/' + df_rad['Image']
    df_rad['Mask'] = preprocess_directory + '/' + df_rad['Mask']
    df_rad.to_csv('radiomicsbatch_train_pred.csv', index=False)
    # df_rad.to_csv('radiomicsbatchCT_train.csv', index=False)

    #radiomics features for test set
    test_csv_path = os.path.join(config.DATASET_RECORDS, 'original_test_split.csv')
    df = pd.read_csv(test_csv_path)
    # print(df.head())

    patient_list = df["accession"].tolist()
    # print(patient_list)

    image_loc = '-CT-imagingVolume.nrrd'
    image_list = [x + image_loc for x in patient_list]
    print(image_list)

    mask_loc = '-CT-segMask_pred.nrrd'
    # mask_loc = '-CT-segMask_GTV.nrrd'
    mask_list = [x + mask_loc for x in patient_list]
    print(mask_list)

    dict = {'Patient': patient_list, 'Image': image_list, 'Mask': mask_list}
    df_rad = pd.DataFrame(dict)
    df_rad['Image'] = preprocess_directory + '/' + df_rad['Image']
    df_rad['Mask'] = preprocess_directory + '/' + df_rad['Mask']
    df_rad.to_csv('radiomicsbatch_test_pred.csv', index=False)
    # df_rad.to_csv('radiomicsbatchCT_test.csv', index=False)



    #radiomics features for validation set
    validation_csv_path = os.path.join(config.DATASET_RECORDS, 'original_validation_split.csv')
    df = pd.read_csv(validation_csv_path)
    # print(df.head())

    patient_list = df["accession"].tolist()
    # print(patient_list)

    image_loc = '-CT-imagingVolume.nrrd'
    image_list = [x + image_loc for x in patient_list]
    print(image_list)

    mask_loc = '-CT-segMask_pred.nrrd'
    # mask_loc = '-CT-segMask_GTV.nrrd'
    mask_list = [x + mask_loc for x in patient_list]
    print(mask_list)

    dict = {'Patient': patient_list, 'Image': image_list, 'Mask': mask_list}
    df_rad = pd.DataFrame(dict)
    df_rad['Image'] = preprocess_directory + '/' + df_rad['Image']
    df_rad['Mask'] = preprocess_directory + '/' + df_rad['Mask']
    df_rad.to_csv('radiomicsbatch_validation_pred.csv', index=False)
    # df_rad.to_csv('radiomicsbatchCT_validation.csv', index=False)

if config.MODE == "PET":
    preprocess_directory = config.PREPROCESSED_DIR_PET
    # print(preprocess_directory)

    # radiomics features for train set
    train_csv_path = os.path.join(config.DATASET_RECORDS, 'original_train_split.csv')
    df = pd.read_csv(train_csv_path)
    # print(df.head())

    patient_list = df["accession"].tolist()
    # print(patient_list)

    image_loc = '-PET-imagingVolume.nrrd'
    image_list = [x + image_loc for x in patient_list]
    print(image_list)

    mask_loc = '-PET-segMask_pred.nrrd'
    # mask_loc = '-PET-segMask_GTV.nrrd'
    mask_list = [x + mask_loc for x in patient_list]
    print(mask_list)

    dict = {'Patient': patient_list, 'Image': image_list, 'Mask': mask_list}
    df_rad = pd.DataFrame(dict)
    df_rad['Image'] = preprocess_directory + '/' + df_rad['Image']
    df_rad['Mask'] = preprocess_directory + '/' + df_rad['Mask']
    df_rad.to_csv('radiomicsbatchPET_train_pred.csv', index=False)
    # df_rad.to_csv('radiomicsbatchPET_train.csv', index=False)

    # radiomics features for test set
    test_csv_path = os.path.join(config.DATASET_RECORDS, 'original_test_split.csv')
    df = pd.read_csv(test_csv_path)
    # print(df.head())

    patient_list = df["accession"].tolist()
    # print(patient_list)

    image_loc = '-PET-imagingVolume.nrrd'
    image_list = [x + image_loc for x in patient_list]
    print(image_list)

    mask_loc = '-PET-segMask_pred.nrrd'
    # mask_loc = '-PET-segMask_GTV.nrrd'
    mask_list = [x + mask_loc for x in patient_list]
    print(mask_list)

    dict = {'Patient': patient_list, 'Image': image_list, 'Mask': mask_list}
    df_rad = pd.DataFrame(dict)
    df_rad['Image'] = preprocess_directory + '/' + df_rad['Image']
    df_rad['Mask'] = preprocess_directory + '/' + df_rad['Mask']
    df_rad.to_csv('radiomicsbatchPET_test_pred.csv', index=False)
    # df_rad.to_csv('radiomicsbatchPET_test.csv', index=False)

    # radiomics features for validation set
    validation_csv_path = os.path.join(config.DATASET_RECORDS, 'original_validation_split.csv')
    df = pd.read_csv(validation_csv_path)
    # print(df.head())

    patient_list = df["accession"].tolist()
    # print(patient_list)

    image_loc = '-PET-imagingVolume.nrrd'
    image_list = [x + image_loc for x in patient_list]
    print(image_list)

    mask_loc = '-PET-segMask_pred.nrrd'
    # mask_loc = '-PET-segMask_GTV.nrrd'
    mask_list = [x + mask_loc for x in patient_list]
    print(mask_list)

    dict = {'Patient': patient_list, 'Image': image_list, 'Mask': mask_list}
    df_rad = pd.DataFrame(dict)
    df_rad['Image'] = preprocess_directory + '/' + df_rad['Image']
    df_rad['Mask'] = preprocess_directory + '/' + df_rad['Mask']
    df_rad.to_csv('radiomicsbatchPET_validation_pred.csv', index=False)
    # df_rad.to_csv('radiomicsbatchPET_validation.csv', index=False)

