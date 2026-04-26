from .metrics_classification import *
from .metrics_regression import *
from .roc_auc import roc_auc_score
from .confusion import confusion_matrix
import numpy as np

class Evaluator:
    def classification_report(self, y_true, y_pred, y_scores=None):
        report = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "weighted_f1": weighted_f1_score(y_true, y_pred)
        }
        cm, classes = confusion_matrix(y_true, y_pred)
        report["confusion_matrix"] = cm.tolist()
        report["classes"] = classes.tolist()
        if y_scores is not None:
            report["roc_auc"] = roc_auc_score(y_true, y_scores)
        return report
    def regression_report(self, y_true, y_pred):
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)

        }
    def display_classification_report(self, y_true, y_pred, y_scores=None):
        report = self.classification_report(y_true, y_pred, y_scores)
        print("Classification Report : ")
        for key, value in report.items():
            if key not in ["confusion_matrix", "classes"]:
                print(key," : ",value)
    