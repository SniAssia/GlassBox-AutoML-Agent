import numpy as np


#comparer des modèles indépendamment du seuil
#évaluer la qualité des scores probabilistes
# éviter les biais liés à un seuil arbitraire
#mieux fonctionner avec datasets déséquilibrés

# Receiver Operating Characteristic
#roc compute la probabilité que le modèle donne un score plus élevé à un positif qu'à un négatif
#ça veut dire :
#on prend un positif et un négatif au hasard
#et on regarde :
#score positif > score negatif ?
#si oui → correct

def roc_auc_score(y_true, y_scores):
    desc = np.argsort(-y_scores)
    y_true = y_true[desc]
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    auc = np.trapz(tpr, fpr)
    return auc
