from data import generate_from_features, relist, Dataset, load_image, INPUT_FORM_PARAMETERS, SHAPES_OUTPUT, T1, SEGMENTATION, IMAGE, input_data_form
import evaluate
from sklearn.metrics import accuracy_score, average_precision_score, cohen_kappa_score, hamming_loss, roc_auc_score, recall_score, confusion_matrix, precision_recall_curve, auc
import efficientnet.keras as efn
import os
import numpy as np
from glob import glob
from db import db, Result, Expand_Result, CalculatedResult
import ast
import pandas as pd
from config import config
from keras.models import load_model
from keras.models import Model
import math
from tqdm import tqdm
import traceback
from uuid import uuid4, UUID
from evaluate_ensemble import dim_two_pet, dim_one_pet, dim_zero_pet, dim_two_ct, dim_one_ct, dim_zero_ct

model_directory = config.MODEL_DIR

#CT model
model_file = os.path.join(model_directory, "***.h5")
model_id = "***"

#PET model
# model_file = os.path.join(model_directory, "***.h5")
# model_id = "***"

dataset_directory = config.DATASET_RECORDS
query = db.session.query(Expand_Result).filter(Expand_Result.uuid == model_id).all()
result = query[0]
split_id = result.split_uuid
label = result.label
label = ast.literal_eval(result.label)

training_set = pd.read_csv(os.path.join(config.DATASET_RECORDS, "{}-{}-train.csv".format(str(split_id),str(label))))
validation_set = pd.read_csv(os.path.join(config.DATASET_RECORDS, "{}-{}-validation.csv".format(str(split_id),str(label))))
test_set = pd.read_csv(os.path.join(config.DATASET_RECORDS, "{}-{}-test.csv".format(str(split_id),str(label))))

clinical_features = [
    "volume",
    "Age",
    "Gender",
    "Race",
    "missing_race",
    "SUV",
    "outcome",
    "Unique_ID",
    "vital_status",
    "pfs",
    "os",
]

def outcome_feature(row):
    label = row["outcome"]
    features = [row[f] for f in clinical_features]
    return label, features

def adenocarcinoma_feature(row):
    label = row["adenocarcinoma"]
    features = [row[f] for f in clinical_features]
    return label, features

def progression_feature(row):
    label = row["progression"]
    features = [row[f] for f in clinical_features]
    return label, features

LABEL_FORMS = {
    "outcome": outcome_feature,
    "adenocarcinoma": adenocarcinoma_feature,
    "progression": progression_feature,
}

def get_label_features(row, label="outcome"):
    """returns label, features, sample name"""
    return (*LABEL_FORMS[label](row), row["patient"])

def extract_generate_from_features(df, input_form=config.INPUT_FORM, label_form="outcome", verbose=False, source=config.PREPROCESSED_DIR, dimension = 2):
    parameters = INPUT_FORM_PARAMETERS[input_form]

    for row in tqdm(df.iterrows(), total=len(df)):
        patient_row = row[1]
        t1_image_file = os.path.join(source, "{}-{}-{}".format(patient_row["patient"], T1, IMAGE))
        print("#################~~~~~~~~~~~~~~~~~~~~~~~~for test")
        print(t1_image_file)
        t1_seg_file = os.path.join(source, "{}-{}-{}".format(patient_row["patient"], T1, SEGMENTATION))
        print(t1_seg_file)
        try:
            t1_masked = None
            #t2_masked = None
            if parameters["t1"] or parameters["features"]: # load in case of features so that files that error out aren't included in analysis
                if verbose:
                    print(SHAPES_OUTPUT.format("t1"))
                t1_masked = load_image(t1_image_file, t1_seg_file, verbose=verbose, dimension = dimension)
            #if parameters["t2"]:
            #    if verbose:
            #        print(SHAPES_OUTPUT.format("t2"))
            #    t2_masked = load_image(t2_image_file, t2_seg_file, verbose=verbose)
            labels, features, name = get_label_features(patient_row, label=label_form)
            images, features, labels = input_data_form(t1_masked, features, labels, input_form="all")
            yield images, features, labels, name

        except Exception as e:
            print()
            print("#" * 80)
            print("Exception occurred for: {}\n{}".format(patient_row, e))
            print(traceback.format_exc())
            continue


train_images, train_features, train_labels, train_names = relist(extract_generate_from_features(training_set, input_form=result.input_form, label_form=str(label)))
validation_images, validation_features, validation_labels, validation_names = relist(extract_generate_from_features(validation_set, input_form=result.input_form, label_form=str(label)))
test_images, test_features, test_labels, test_names = relist(extract_generate_from_features(test_set, input_form=result.input_form, label_form=str(label)))
train_features = relist(train_features)
validation_features = relist(validation_features)
test_features = relist(test_features)

test_images, test_features, test_labels, test_names = relist(extract_generate_from_features(test_set, input_form="all", label_form="outcome"))
test_features = relist(test_features)

def generate_combined_array(names, labels, features):
    name_column = np.array([names]).T
    label_column = np.array([labels]).T
    feature_column = np.transpose(np.array([features]), axes=[2, 1, 0])
    feature_column = feature_column.reshape((feature_column.shape[0], feature_column.shape[1]))
    combined = np.concatenate((name_column, label_column, feature_column), axis=1)
    return combined

train_combined = generate_combined_array(names = train_names, labels = train_labels, features = train_features)
validation_combined = generate_combined_array(names = validation_names, labels = validation_labels, features = validation_features)
test_combined = generate_combined_array(names = test_names, labels = test_labels, features = test_features)


train_generator = Dataset(
    train_images,
    train_features,
    train_labels,
    train_names,
    augment=False,
    shuffle=False,
    input_form=result.input_form,
    seed=split_id,
)
validation_generator = Dataset(
    validation_images,
    validation_features,
    validation_labels,
    validation_names,
    augment=False,
    shuffle=False,
    input_form=result.input_form,
    seed=split_id,
)
test_generator = Dataset(
    test_images,
    test_features,
    test_labels,
    test_names,
    augment=False,
    shuffle=False,
    input_form=result.input_form,
    seed=split_id,
)

def extract_get_results(model, data):
    results = model.predict_generator(data, steps=math.ceil(len(data)/config.BATCH_SIZE))
    data.reset()
    return results

def test_model(model, dataset):
    results = extract_get_results(model, dataset)
    #probabilities = list(evaluate.transform_binary_probabilities(results))
    # labels = dataset.classes
    # features = dataset.features
    # names = dataset.names
    return results

ct_models = [dim_two_ct, dim_one_ct, dim_zero_ct]

def generate_results(model_name, dimension):
    model_file = os.path.join(model_directory, model_name+"-enet.h5")
    model_id = model_name
    test_images, test_features, test_labels, test_names = relist(extract_generate_from_features(test_set, input_form="all", label_form="outcome", dimension = dimension))
    test_features = relist(test_features)
    test_generator = Dataset(
        test_images,
        test_features,
        test_labels,
        test_names,
        augment=False,
        shuffle=False,
        input_form="all",
        seed=UUID(config.UUID),
    )
    dataset_directory = config.DATASET_RECORDS
    query = db.session.query(Expand_Result).filter(Expand_Result.uuid == model_id).all()
    result = query[0]
    split_id = result.split_uuid
    label = result.label
    model = load_model(model_file)
    ind_layer = 2
    model_cut = Model(inputs=model.inputs, output=model.layers[-ind_layer].output)
    results = extract_get_results(model_cut, test_generator)
    return results

model = load_model(model_file)
ind_layer = 2
model_cut = Model(inputs=model.inputs, output=model.layers[-ind_layer].output)

extracted_test_features = test_model(model_cut, test_generator)
exported_test_data = np.concatenate((test_combined,extracted_test_features), axis = 1)
pd.DataFrame(exported_test_data).to_csv(os.path.join(config.EXTRACTED_FEATURES, "{}-test_features.csv".format(str(model_id))))
print("finished test")

extracted_validation_features = test_model(model_cut, validation_generator)
exported_validation_data = np.concatenate((validation_combined,extracted_validation_features), axis = 1)
pd.DataFrame(exported_validation_data).to_csv(os.path.join(config.EXTRACTED_FEATURES, "{}-validation_features.csv".format(str(model_id))))
print("finished validation")

extracted_train_features = test_model(model_cut, train_generator)
exported_train_data = np.concatenate((train_combined, extracted_train_features), axis = 1)
pd.DataFrame(exported_train_data).to_csv(os.path.join(config.EXTRACTED_FEATURES, "{}-train_features.csv".format(str(model_id))))
print("finished train")
