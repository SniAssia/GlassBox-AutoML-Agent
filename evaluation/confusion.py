import numpy as np

# y_true: lst of true labels
# y_pre : lst of predicted labels 
def confusion_matrix(y_true, y_pred):
    # here we get all classes (values that can be taken by y_true and y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n = len(classes)
    matrix = np.zeros((n, n), dtype=int)
    # in the confusion matrix we assign the rows to real values 
    for i, true_label in enumerate(classes):
        # columns to predicted labels 
        for j, pred_label in enumerate(classes):
            # here we comput FN ,TP,TN,FP
            matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    return matrix, classes