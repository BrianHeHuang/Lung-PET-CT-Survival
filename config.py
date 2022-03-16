import os
import logging

class Config(object):
    IMAGE_SIZE = 224

    TRIALS = 2
    BATCH_SIZE = 32

    EPOCHS = 150
    PATIENCE = 50
    SAMPLES_VALIDATION = 300
    VALIDATION_SPLIT = 0.12
    TEST_SPLIT = 0.06

    DEVELOPMENT = True
    DEBUG = True
    PRINT_SQL = False
    SECRET = "example secret key"
    LOG_LEVEL = logging.DEBUG
    UUID = "2a4fdf1d-f485-42ab-8ccb-c5ad81d2d54d"

    #root for raw NRRD files
    RAW_NRRD_ROOT = ""
    DIAGNOSIS_METHOD = "none"
    #CT or PET
    MODE = "PET"
    RAW_FEATURES = [
        "./full_new_outcome.csv",
        "./full_new_institution.csv",
        "./full_new_clinical.csv",
        "./full_new_sort.csv"
        ]

    DATA = ""
    # PREPROCESSED_DIR = os.path.join(DATA, "prenoN4_CT")
    PREPROCESSED_DIR = os.path.join(DATA, "prenoN4_PET")

    PREPROCESSED_DIR_PET = os.path.join(DATA, "prenoN4_PET")
    PREPROCESSED_DIR_CT = os.path.join(DATA, "prenoN4_CT")
    PREPROCESSED_DIR_CT_PRED = os.path.join(DATA, "prenoN4_CT_pred")
    PREPROCESSED_DIR_PET_PRED = os.path.join(DATA, "prenoN4_PET_pred")

    TRAIN_DIR = os.path.join(DATA, "train")
    TEST_DIR = os.path.join(DATA, "test")
    VALIDATION_DIR = os.path.join(DATA, "validation")

    # FEATURES_DIR = "./featuresCT"
    FEATURES_DIR = "./featuresPET"
    NRRD_FEATURES = os.path.join(FEATURES_DIR, "nrrd-features.pkl")
    FEATURES = os.path.join(FEATURES_DIR, "training-features.pkl")
    PREPROCESS = os.path.join(FEATURES_DIR, "preprocess.pkl")

    INPUT_FORM = "all"

    OUTPUT = ""
    DB_URL = "sqlite:///{}/results.db".format(OUTPUT)
    MODEL_DIR = os.path.join(OUTPUT, "models")
    STDOUT_DIR = os.path.join(OUTPUT, "stdout")
    STDERR_DIR = os.path.join(OUTPUT, "stderr")
    DATASET_RECORDS = os.path.join(OUTPUT, "datasets")
    EXTRACTED_FEATURES = os.path.join(OUTPUT,"extracted_features")

    MAIN_TEST_HOLDOUT = 0.2
    NUMBER_OF_FOLDS = 5
    SPLIT_TRAINING_INTO_VALIDATION = 0.2

config = Config()
