from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import operator
import math
from db import db, Result, Expand_Result, CalculatedResult
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import json
import sklearn.metrics as metrics
import ast
import evaluate
from sklearn.metrics import auc, precision_recall_curve, roc_curve, confusion_matrix, roc_auc_score, accuracy_score
from statsmodels.stats.contingency_tables import mcnemar

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

#model uuids here
#model_1 =
#model_2 =

def gen_acc(model1, model2, metric, ensemble = False):
    query1 = db.session.query(Expand_Result).filter(Expand_Result.uuid == model1).all()
    result1 = query1[0]
    predict1 = transform_binary_predictions(ast.literal_eval(result1.test_probabilities))
    test_labels = ast.literal_eval(result1.test_labels)

    query2 = db.session.query(Expand_Result).filter(Expand_Result.uuid == model2).all()
    result2 = query2[0]
    predict2 = transform_binary_predictions(ast.literal_eval(result2.test_probabilities))

    if ensemble == True:
        average_predictions = [sum(x)/2 for x in zip(ast.literal_eval(result1.test_probabilities), ast.literal_eval(result2.test_probabilities))]
        ensemble_binary_predict = transform_binary_predictions(average_predictions)
        predict2 = ensemble_binary_predict

    both_cor = 0
    query1_cor = 0
    query2_cor = 0
    both_incor = 0

    for i in range(len(predict1)):
        if metric == "acc" or (metric == "sens" and test_labels[i] == 1) or (metric == "spec" and test_labels[i] == 0):
            if predict1[i] == predict2[i] == test_labels[i]:
                both_cor +=1
            elif predict1[i] == test_labels[i]:
                query1_cor +=1
            elif predict2[i] == test_labels[i]:
                query2_cor +=1
            elif predict1[i] == predict2[i]:
                both_incor +=1
            else:
                print("error, recheck")
        else:
            continue
    return([[both_cor, query1_cor], [query2_cor, both_incor]])

def gen_ens_acc(model1, model2, model3, model4, metric):
    query1 = db.session.query(Expand_Result).filter(Expand_Result.uuid == model1).all()
    result1 = query1[0]
    test_labels = ast.literal_eval(result1.test_labels)

    query2 = db.session.query(Expand_Result).filter(Expand_Result.uuid == model2).all()
    result2 = query2[0]

    query3 = db.session.query(Expand_Result).filter(Expand_Result.uuid == model3).all()
    result3 = query3[0]

    query4 = db.session.query(Expand_Result).filter(Expand_Result.uuid == model4).all()
    result4 = query4[0]

    average_predictions1 = [sum(x)/2 for x in zip(ast.literal_eval(result1.test_probabilities), ast.literal_eval(result2.test_probabilities))]
    average_predictions2 = [sum(x)/2 for x in zip(ast.literal_eval(result3.test_probabilities), ast.literal_eval(result4.test_probabilities))]

    predict1 = transform_binary_predictions(average_predictions1)
    predict2 = transform_binary_predictions(average_predictions2)

    both_cor = 0
    query1_cor = 0
    query2_cor = 0
    both_incor = 0

    for i in range(len(predict1)):
        if metric == "acc" or (metric == "sens" and test_labels[i] == 1) or (metric == "spec" and test_labels[i] == 0):
            if predict1[i] == predict2[i] == test_labels[i]:
                both_cor +=1
            elif predict1[i] == test_labels[i]:
                query1_cor +=1
            elif predict2[i] == test_labels[i]:
                query2_cor +=1
            elif predict1[i] == predict2[i]:
                both_incor +=1
            else:
                print("error, recheck")
        else:
            continue
    return([[both_cor, query1_cor], [query2_cor, both_incor]])

# acc = gen_acc(model1, model2, "acc", ensemble = False)
# result = mcnemar(acc, exact=True)
# print('acc_statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# #
# sens = gen_acc(model1, model2, "sens", ensemble = False)
# result = mcnemar(sens, exact=True)
# print('sens_statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
#
# spec = gen_acc(model1, model2, "spec", ensemble = False)
# result = mcnemar(spec, exact=True)
# print('spec_statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))

# model1 =
# model2 =
# #
# model3 =
# model4 =
#
# acc = gen_ens_acc(model1,model2,model3,model4, "acc")
# result = mcnemar(acc, exact=True)
# print('ens_acc_statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# sens = gen_ens_acc(model1,model2,model3,model4, "sens")
# result = mcnemar(sens, exact=True)
# print('sens_acc_statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# spec = gen_ens_acc(model1,model2,model3,model4, "spec")
# result = mcnemar(spec, exact=True)
# print('spec_ens_acc_statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))

print("test")