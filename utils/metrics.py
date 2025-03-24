"""
Evaluation metric of downstream task migration
Management evaluation metric, K-fold cross-validation, results saving
"""

import os
import numpy as np
import json

from sklearn.metrics import precision_recall_curve, auc, accuracy_score, average_precision_score
from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_auc_score, f1_score, recall_score


#   main function call interface
def evaluate(refs, preds, task="classification", dataset='', multi_label=True):
    if task == "classification":
        if dataset == 'Angiographic' and multi_label==True:
            metrics = multi_label_metrics(refs, preds)
        else:
            metrics = classification_metrics(refs, preds, dataset)

        if dataset=='AMD':
            print('Metrics: aca=%2.5f - kappa=%2.3f - AMD_kappa=%2.3f - macro f1=%2.3f - AMD_F1=%2.3f - auc=%2.3f -AMD_auc=%2.3f'
                  % (metrics["aca"],metrics["kappa"],metrics["AMD_kappa"],metrics["f1_avg"],metrics["AMD_f1"],metrics["auc_avg"],metrics["AMD_auc"]))
        elif dataset == 'Angiographic' and multi_label==True:
            print('Metrics: acc=%2.5f - auc=%2.5f - aupr=%2.5f - kappa=%2.3f - f1=%2.3f' % (metrics["acc"], metrics["auc"], metrics["AUPR"], metrics["kappa"], metrics["f1"]))
        elif dataset == 'Angiographic' and multi_label==False:
            print('Metrics: acc=%2.5f - auc=%2.5f - aupr=%2.5f - kappa=%2.3f - f1=%2.3f' % (metrics["aca"], metrics["auc_avg"], metrics["aupr_avg"], metrics["kappa"], metrics["f1_avg"]))
        else:
            print('Metrics: aca=%2.5f - TAOP_aca=%2.5f - kappa=%2.3f - macro f1=%2.3f - auc=%2.3f -aupr=%2.3f'
                  % (metrics["aca"], metrics["TAOP_acc"],metrics["kappa"],metrics["f1_avg"],metrics["auc_avg"],metrics["aupr_avg"]))
    else:
        metrics = {}
    return metrics


# auc
def au_prc(true_mask, pred_mask):
    precision, recall, threshold = precision_recall_curve(true_mask, pred_mask)
    au_prc = auc(recall, precision)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    f1[np.isnan(f1)] = 0
    th = threshold[np.argmax(f1)]

    return au_prc, th


#   specificity
def specificity(refs, preds):
    cm = confusion_matrix(refs, preds)
    specificity = cm[0,0]/(cm[0,0]+cm[0,1])
    return specificity


#   Classification evaluation metric
def classification_metrics(refs, preds, dataset):
    k = np.round(cohen_kappa_score(refs, np.argmax(preds, -1), weights="quadratic"), 3)                             # Kappa quadatic

    cm = confusion_matrix(refs, np.argmax(preds, -1))                                                                       #   confusion matrix
    cm_norm = (cm / np.expand_dims(np.sum(cm, -1), 1))

    acc_class = list(np.round(np.diag(cm_norm), 3))
    aca = np.round(np.mean(np.diag(cm_norm)), 3)

    recall_class = [np.round(recall_score(refs == i, np.argmax(preds, -1) == i), 3) for i in np.unique(refs)]       #   recall per class

    specificity_class = [np.round(specificity(refs == i, np.argmax(preds, -1) == i), 3) for i in np.unique(refs)]   #  specificity

    auc_class = [np.round(roc_auc_score(refs == i, preds[:, i]), 3) for i in np.unique(refs)]                       # auc auc

    f1_class = [np.round(f1_score(refs == i, np.argmax(preds, -1) == i), 3) for i in np.unique(refs)]               # f1  f1

    aupr_class = [np.round(average_precision_score(refs == i, preds[:, i]), 3) for i in np.unique(refs)]

    if dataset=='AMD':
        # AMD dataset
        AMD_auc = roc_auc_score(refs, preds[:,1])
        precision, recall, thresholds = precision_recall_curve(refs, preds[:,1])
        kappa_list = []
        f1_list = []
        for threshold in thresholds:
            y_scores = preds[:,1]
            y_scores = np.array(y_scores >= threshold, dtype=float)
            kappa = cohen_kappa_score(refs, y_scores)
            kappa_list.append(kappa)
            f1 = f1_score(refs, y_scores)
            f1_list.append(f1)
        kappa_f1 = np.array(kappa_list) + np.array(f1_list)
        AMD_kappa = kappa_list[np.argmax(kappa_f1)]
        AMD_f1 = f1_list[np.argmax(kappa_f1)]

    # TAOP dataset
    class_acc = accuracy_score(refs, np.argmax(preds, -1))

    if dataset=='AMD':
        metrics = {"aca": aca , "TAOP_acc":class_acc,"acc_class": acc_class,
                   "kappa": k, "AMD_kappa":AMD_kappa,
                   "auc_class": auc_class, "auc_avg": np.mean(auc_class), "AMD_auc":AMD_auc,
                   "f1_class": f1_class, "f1_avg": np.mean(f1_class), "AMD_f1":AMD_f1,
                   "sensitivity_class": recall_class, "sensitivity_avg": np.mean(recall_class),
                   "specificity_class": specificity_class, "specificity_avg": np.mean(specificity_class),
                   "cm": cm, "cm_norm": cm_norm}
    else:
        metrics = {"aca": aca ,"acc_class": acc_class,
                   "kappa": k,"TAOP_acc":class_acc,
                   "auc_class": auc_class, "auc_avg": np.mean(auc_class),
                   "f1_class": f1_class, "f1_avg": np.mean(f1_class),
                   "sensitivity_class": recall_class, "sensitivity_avg": np.mean(recall_class),
                   "specificity_class": specificity_class, "specificity_avg": np.mean(specificity_class),
                   "cm": cm, "cm_norm": cm_norm,
                   "aupr_class":aupr_class, "aupr_avg":np.mean(aupr_class)}
    return metrics


# K  K-fold cross-verify average
def average_folds_results(list_folds_results, task):
    '''
    list_folds_results：K
    '''
    metrics_name = list(list_folds_results[0].keys())               #   Name of all evaluation metric

    out = {}
    for iMetric in metrics_name:
        values = np.concatenate([np.expand_dims(np.array(iFold[iMetric]), -1) for iFold in list_folds_results], -1)
        out[(iMetric + "_avg")] = np.round(np.mean(values, -1), 3).tolist()
        out[(iMetric + "_std")] = np.round(np.std(values, -1), 3).tolist()

    if task == "classification":
        # print('Metrics: aca=%2.3f(%2.3f) - TAOP_aca=%2.3f(%2.3f) - kappa=%2.3f(%2.3f) - macro f1=%2.3f(%2.3f) -'
        #       'AMD_kappa=%2.3f(%2.3f) - AMD_auc=%2.3f(%2.3f) - AMD_f1=%2.3f(%2.3f)' % (
        #     out["aca_avg"], out["aca_std"], out["TAOP_acc_avg"], out["TAOP_acc_std"],
        #     out["kappa_avg"], out["kappa_std"], out["f1_avg_avg"], out["f1_avg_std"],
        #     out["AMD_kappa_avg"], out["AMD_kappa_std"], out["AMD_auc_avg"], out["AMD_auc_std"],
        #     out["AMD_f1_avg"], out["AMD_f1_std"]))
        print('Metrics: aca=%2.3f(%2.3f)  - kappa=%2.3f(%2.3f) - macro f1=%2.3f(%2.3f) - auc=%2.3f(%2.3f) - aupr=%2.3f(%2.3f) '
               % (out["aca_avg"], out["aca_std"], out["kappa_avg"], out["kappa_std"], out["f1_avg_avg"], out["f1_avg_std"],
                  out["auc_avg_avg"], out["auc_avg_std"], out["f1_avg_avg"], out["f1_avg_std"]))
    # elif task=='Angiographic':
    #     print('Metrics: acc=%2.3f(%2.3f) - auc=%2.3f(%2.3f) - aupr=%2.3f(%2.3f) - kappa=%2.3f(%2.3f) - f1=%2.3f(%2.3f) -'
    #           % (out["acc_avg"], out["acc_std"], out["auc_avg"], out["auc_std"], out["AUPR_avg"], out["AUPR_std"],
    #              out["kappa_avg"], out["kappa_std"]))
    elif task=='Angiographic':
        print('Metrics: acc=%2.3f(%2.3f) - auc=%2.3f(%2.3f) - aupr=%2.3f(%2.3f) - kappa=%2.3f(%2.3f) -'
                % (out["acc_avg"], out["acc_std"], out["auc_avg"], out["auc_std"], out["AUPR_avg"], out["AUPR_std"],
                    out["kappa_avg"], out["kappa_std"]))

    return out


#    Save experiment results and adapter weights
def save_results(metrics, out_path, id_experiment=None, id_metrics=None, save_model=False, weights=None):
    '''
    metrics：K  Results of K-fold cross-validation
    id_experiment：id
    '''
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    if id_experiment is None:
        id_experiment = "experiment" + str(np.random.rand())
    else:
        id_experiment = id_experiment
    if not os.path.isdir(out_path + id_experiment):
        os.mkdir(out_path + id_experiment)

    # json  Save the results in json format
    with open(out_path + id_experiment + '/metrics_' + id_metrics + '.json', 'w') as fp:
        json.dump(metrics, fp)

    #   Save adapter weight
    if save_model:
        import torch
        for i in range(len(weights)):
            torch.save(weights[i], out_path + id_experiment + '/weights_' + str(i) + '.pth')


# 
def multi_label_metrics(gt_list, pred_list):
    # pred confidence, 
    avg_acc_list = []
    avg_auc_list = []
    avg_kappa_list = []
    avg_f1_list = []
    aupr_per_label = []

    for i in range(gt_list.shape[1]):
        #  
        # acc
        pred = (pred_list>0.5).astype(int)
        accuracy = accuracy_score(gt_list[:, i], pred[:, i])

        # AUC
        try:
            auc = roc_auc_score(gt_list[:, i], pred_list[:, i])
        except ValueError:
            pass

        # AUPR
        precision, recall, thresholds = precision_recall_curve(gt_list[:, i], pred_list[:, i])
        aupr = average_precision_score(gt_list[:, i], pred_list[:, i])

        # kappa f1
        kappa_list = []
        f1_list = []
        sample_indices = np.linspace(0, len(thresholds) - 1, num=100, dtype=int)  # 100
        sampled_thresholds = thresholds[sample_indices]
        for threshold in sampled_thresholds:
            y_scores = pred_list[:, i]
            y_scores = np.array(y_scores >= threshold, dtype=float)
            kappa = cohen_kappa_score(gt_list[:, i], y_scores)
            kappa_list.append(kappa)
            f1 = f1_score(gt_list[:, i], y_scores)
            f1_list.append(f1)
        kappa_f1 = np.array(kappa_list) + np.array(f1_list)

        avg_acc_list.append(accuracy)
        avg_auc_list.append(auc)
        avg_kappa_list.append(kappa_list[np.argmax(kappa_f1)])
        avg_f1_list.append(f1_list[np.argmax(kappa_f1)])
        aupr_per_label.append(aupr)

    return {"acc":np.mean(avg_acc_list), "auc":np.mean(avg_auc_list), "AUPR":np.mean(aupr_per_label), "kappa":np.mean(avg_kappa_list), "f1":np.mean(avg_f1_list)}