import numpy as np
from .confusion import confusion_matrix

#accuracy = {TP + TN}/ {TP + TN + FP + FN}
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


#precision = {TP}/{TP + FP}
def precision_score(y_true, y_pred):
    cm, _ = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fp = np.sum(cm,axis=0) - tp
    precision = tp / (tp + fp )
    return np.mean(precision)
#recall = {TP}/{TP + FN}
def recall_score(y_true, y_pred):
    cm, _ = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fn = np.sum(cm, axis=1) - tp
    recall = tp / (tp + fn)
    return np.mean(recall)

#f1 = {precision * recall}/{precision + recall}
def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * (p * r) / (p + r)



# this is used when classes are imbalanced
# ici on dois calculer precision et recall par classe puis F1 par classe.
def weighted_f1_score(y_true, y_pred):
    cm, _ = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    support = np.sum(cm, axis=1)
    return np.sum(f1 * support) / np.sum(support)