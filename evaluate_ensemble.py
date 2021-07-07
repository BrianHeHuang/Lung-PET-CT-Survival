from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import operator
import math
from db import db, Result, Expand_Result, CalculatedResult, CxResult
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import json
import sklearn.metrics as metrics
import ast
import evaluate
from sklearn.metrics import auc, precision_recall_curve, roc_curve, confusion_matrix, roc_auc_score, accuracy_score

def transform_binary_predictions(results):
    binary_predictions = results.copy()
    for i in range(len(results)):
        prediction = results[i]
        if prediction >= 0.5:
            binary_predictions[i] = 1
        else:
            binary_predictions[i] = 0
    return binary_predictions

def ensemble_calculate_confusion_matrix_stats(labels, results):
    binary_results = transform_binary_predictions(results)
    confusion_mat = confusion_matrix(labels, binary_results)
    TN, FP, FN, TP = confusion_mat.ravel()

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    if (TP+FP) == 0:
        PPV = -1
    else:
        PPV = TP/(TP+FP)
    # Negative predictive value
    if (TN+FN) == 0:
        NPV = -1
    else:
        NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    Acc = (TN + TP)/(TN + TP + FN + FP)
    return {
        "Acc": Acc,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TPR": TPR,
        "TNR": TNR,
        "PPV": PPV,
        "NPV": NPV,
        "FPR": FPR,
        "FNR": FNR,
        "FDR": FDR,
        "AM": (TPR+TNR)/2,
        "GM": np.sqrt(TPR*TNR),
    }

model_list = ["e620607d-d43f-457e-984f-b923a3b6641f", "da37531f-98fc-4da2-97ae-7e947c0a67e9"]



predictions_list = []
test_labels = []
ensemble_predictions = []
model_accuracies = []
model_aocs = []
model_sens = []
model_spec = []
model_npvs = []
model_ppvs = []
model_hyperparameters = []

for i in model_list:
    query = db.session.query(Expand_Result).filter(Expand_Result.uuid == i).all()
    result = query[0]
    predictions_list.append(ast.literal_eval(result.test_probabilities))
    test_labels.append(ast.literal_eval(result.test_labels))
    model_accuracies.append(result.test_accuracy)
    model_aocs.append(result.auc)
    model_sens.append(result.sensitivity)
    model_spec.append(result.specificity)
    model_npvs.append(result.npv)
    model_ppvs.append(result.ppv)
    model_hyperparameters.append(result.hyperparameters)

def complex_avg_predictions(pet_weight, ct_weight):
    pet_model_list = [dim_two_pet, dim_one_pet, dim_zero_pet]
    ct_model_list = [dim_two_ct, dim_one_ct, dim_zero_ct]
    predictions_list = []
    test_labels = []
    model_accuracies = []
    model_aocs = []
    for i in pet_model_list:
        query = db.session.query(Expand_Result).filter(Expand_Result.uuid == i).all()
        result = query[0]
        predictions_list.append(ast.literal_eval(result.test_probabilities))
        test_labels.append(ast.literal_eval(result.test_labels))
        model_accuracies.append(result.test_accuracy)
        model_aocs.append(result.auc)
    PET_predictions = [sum(x)*pet_weight/len(predictions_list) for x in zip(*predictions_list)]

    predictions_list = []
    test_labels = []
    model_accuracies = []
    model_aocs = []
    for i in ct_model_list:
        query = db.session.query(Expand_Result).filter(Expand_Result.uuid == i).all()
        result = query[0]
        predictions_list.append(ast.literal_eval(result.test_probabilities))
        test_labels.append(ast.literal_eval(result.test_labels))
        model_accuracies.append(result.test_accuracy)
        model_aocs.append(result.auc)
    CT_predictions = [sum(x)*ct_weight/len(predictions_list) for x in zip(*predictions_list)]
    average_predictions = [sum(x) for x in zip(PET_predictions, CT_predictions)]
    binary_predict = transform_binary_predictions(average_predictions)
    y_labels = test_labels[0]
    ensemble_acc = accuracy_score(y_labels, binary_predict)
    ensemble_auc = roc_auc_score(y_labels, average_predictions)
    confusion_mat = ensemble_calculate_confusion_matrix_stats(y_labels, average_predictions)
    ensemble_sens = confusion_mat["TPR"]
    ensemble_spec = confusion_mat["TNR"]
    ensemble_npv = confusion_mat["NPV"]
    ensemble_ppv = confusion_mat["PPV"]
    print("PET and CT Ensemble had accuracy of " + str(ensemble_acc) + " , AUC of " + str(ensemble_auc)
          + " , sensitivity of " + str(ensemble_sens) + " , specifiicty of " + str(ensemble_spec) + " , PPV of " + str(ensemble_ppv)  + " , NPV of " + str(ensemble_npv) + "with PET,CT weight of " + str(pet_weight) + "," + str(ct_weight))

average_predictions = [sum(x)/len(predictions_list) for x in zip(*predictions_list)]
binary_predict = transform_binary_predictions(average_predictions)
y_labels = test_labels[0]
ensemble_acc = accuracy_score(y_labels, binary_predict)
ensemble_auc = roc_auc_score(y_labels, average_predictions)
confusion_matrix = ensemble_calculate_confusion_matrix_stats(y_labels, average_predictions)
ensemble_sens = confusion_matrix["TPR"]
ensemble_spec = confusion_matrix["TNR"]
ensemble_npv = confusion_matrix["NPV"]
ensemble_ppv = confusion_matrix["PPV"]

print(str(model_hyperparameters))

print('Individual model accuracies were ' + str(model_accuracies) + " and AUCs were " + str(model_aocs) + ". Ensemble had accuracy of " + str(ensemble_acc) + " and AUC of " + str(ensemble_auc))
print('Individual model sensitivities were ' + str(model_sens) + " and specificities were " + str(model_spec) + ". Ensemble had sensitivity of " + str(ensemble_sens) + " and specificity of " + str(ensemble_spec))
print('Individual model PPVs were ' + str(model_ppvs) + " and NPVs were " + str(model_npvs) + ". Ensemble had ppv of " + str(ensemble_ppv) + " and npv of " + str(ensemble_npv))