import numpy as np

def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n = len(classes)
    matrix = np.zeros((n, n), dtype=int)
    
    # Map class labels to indices for faster lookup
    label_to_index = {label: i for i, label in enumerate(classes)}
    
    for t, p in zip(y_true, y_pred):
        matrix[label_to_index[t], label_to_index[p]] += 1
        
    return matrix.tolist(), classes.tolist() # Return as lists for JSON readiness