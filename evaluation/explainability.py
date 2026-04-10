import numpy as np


def _top_feature_pairs(feature_names, values, limit=5):
    values = np.asarray(values, dtype=float)
    order = np.argsort(np.abs(values))[::-1][:limit]
    return [
        {
            "feature": feature_names[idx],
            "value": float(values[idx]),
            "abs_value": float(abs(values[idx])),
        }
        for idx in order
    ]


def _tree_feature_counts(node, counts):
    if node is None or node.prediction is not None:
        return
    counts[node.feature] = counts.get(node.feature, 0) + 1
    _tree_feature_counts(node.left, counts)
    _tree_feature_counts(node.right, counts)


def _explain_tree(model, feature_names):
    counts = {}
    _tree_feature_counts(model.root, counts)
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    top = [
        {"feature": feature_names[idx], "split_count": int(count)}
        for idx, count in sorted_counts[:5]
    ]
    return {
        "summary": "Decision paths are explained by the features used most often in splits.",
        "top_split_features": top,
    }


def _explain_forest(model, feature_names):
    counts = {}
    for tree in model.trees:
        _tree_feature_counts(tree.root, counts)
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    top = [
        {"feature": feature_names[idx], "split_count": int(count)}
        for idx, count in sorted_counts[:5]
    ]
    return {
        "summary": "Feature influence is approximated by how often features are used across tree splits.",
        "top_split_features": top,
        "tree_count": int(len(model.trees)),
    }


def _explain_linear(model, feature_names):
    weights = np.asarray(model.w_[1:], dtype=float)
    return {
        "summary": "Feature influence is derived from learned coefficient magnitude.",
        "top_coefficients": _top_feature_pairs(feature_names, weights),
    }


def _explain_logistic(model, feature_names):
    weights = np.asarray(model.weights_[:, 1:], dtype=float)
    mean_abs = np.mean(np.abs(weights), axis=0)
    return {
        "summary": "Feature influence is estimated from the average absolute OvR coefficient magnitude.",
        "top_coefficients": _top_feature_pairs(feature_names, mean_abs),
        "classes": model.classes_.tolist(),
    }


def _explain_knn(model):
    return {
        "summary": "Predictions are based on nearest training points under the configured distance metric.",
        "k": int(model.k),
        "distance_metric": model.distance_metric,
        "task": model.task,
    }


def _explain_naive_bayes(model, feature_names):
    class_summaries = []
    for cls, params in zip(model.classes, model.parameters):
        top = _top_feature_pairs(feature_names, params["mean"])
        class_summaries.append(
            {
                "class": cls.item() if isinstance(cls, np.generic) else cls,
                "prior": float(params["prior"]),
                "top_mean_features": top,
            }
        )
    return {
        "summary": "Predictions are based on per-class Gaussian feature distributions and learned priors.",
        "var_smoothing": float(model.var_smoothing),
        "classes": class_summaries,
    }


def build_explainability_report(model_name, model, feature_names):
    feature_names = list(feature_names)

    if model_name == "linear_regression":
        return _explain_linear(model, feature_names)
    if model_name == "logistic_regression":
        return _explain_logistic(model, feature_names)
    if model_name == "decision_tree":
        return _explain_tree(model, feature_names)
    if model_name == "random_forest":
        return _explain_forest(model, feature_names)
    if model_name == "knn":
        return _explain_knn(model)
    if model_name == "naive_bayes":
        return _explain_naive_bayes(model, feature_names)

    return {"summary": f"No explainer is configured for model '{model_name}'."}
