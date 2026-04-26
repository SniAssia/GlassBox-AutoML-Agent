from .metrics_classification import *
from .metrics_regression import *
from .roc_auc import roc_auc_score
from .confusion import confusion_matrix
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # plotting is optional for the AutoFit pipeline
    plt = None

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
    def display_confusion_matrix(self, y_true, y_pred):
        if plt is None:
            raise RuntimeError("matplotlib is required for display_confusion_matrix")
        cm, classes = confusion_matrix(y_true, y_pred)
        _, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        # afficher les valeurs
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j],
                        ha="center",
                        va="center")
        plt.title("Confusion Matrix")
        plt.show()
    def display_classification_report(self, y_true, y_pred, y_scores=None):
        report = self.classification_report(y_true, y_pred, y_scores)
        print("Classification Report : ")
        for key, value in report.items():
            if key not in ["confusion_matrix", "classes"]:
                print(key," : ",value)
    #interpretation graphique :
    def plot_roc_curve(y_true, y_scores):
        if plt is None:
            raise RuntimeError("matplotlib is required for plot_roc_curve")
        order = np.argsort(-y_scores)
        y_true = y_true[order]
        y_scores = y_scores[order]
        tps = np.cumsum(y_true)
        fps = np.cumsum(1 - y_true)
        tpr = tps / tps[-1]
        fpr = fps / fps[-1]
        auc = np.trapz(tpr, fpr)
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
        plt.plot([0,1], [0,1], linestyle="--", label="Random model")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()
