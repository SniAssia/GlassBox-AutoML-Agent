---
name: glassbox_automl
description: Outil d'AutoML transparent (White-Box) pour l'audit, le nettoyage et l'entraînement de modèles (Random Forest, Naive Bayes) via NumPy.
endpoint: https://upgraded-capybara-x5pg9gx9r7472pqgx-8000.app.github.dev/run-automl
method: POST
arguments:
  csv_base64:
    type: string
    description: Le fichier CSV converti en chaîne base64.
  target_column:
    type: string
    description: Le nom de la colonne cible pour la prédiction.
  task_type:
    type: string
    description: Le type de tâche (classification ou regression).
    default: "classification"
---

# GlassBox AutoML Agent
Ce module permet d'automatiser le pipeline de Data Science tout en restant auditable :
1. **The Inspector** : Audit statistique (moyenne, médiane, valeurs manquantes).
2. **The Cleaner** : Imputation et Standard Scaling via NumPy.
3. **The Models** : Entraînement de modèles "from scratch".
4. **Explainability** : Retourne l'importance des variables (Gini Impurity).