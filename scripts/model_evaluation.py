import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, fbeta_score, recall_score, precision_score
from sklearn.metrics import make_scorer


#model_evaluation.py
def eval_preds(model, X, y_true, y_pred, task):
    if task == 'binary':
        cm = confusion_matrix(y_true, y_pred)
        proba = model.predict_proba(X)[:,1]
        acc = accuracy_score(y_true, y_pred)  # Calculate accuracy
        auc = roc_auc_score(y_true, proba)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        f2 = fbeta_score(y_true, y_pred, pos_label=1, beta=2)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
    elif task == 'multi_class':
        cm = confusion_matrix(y_true, y_pred)
        proba = model.predict_proba(X)
        acc = accuracy_score(y_true, y_pred)  # Calculate accuracy
        auc = roc_auc_score(y_true, proba, multi_class='ovr', average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        f2 = fbeta_score(y_true, y_pred, beta=2, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
    metrics = pd.Series(data={'ACC':acc, 'AUC':auc, 'F1':f1, 'F2':f2, 'Recall': recall, 'Precision': precision})
    metrics = round(metrics,3)
    return cm, metrics  # Return confusion matrix and metrics


def predict_and_evaluate(fitted_models, X, y_true, clf_str, task):
    cm_dict = {key: np.nan for key in clf_str}
    metrics = pd.DataFrame(columns=['ACC', 'AUC', 'F1', 'F2', 'Recall', 'Precision'])
    y_pred = pd.DataFrame(columns=clf_str)
    for fit_model, model_name in zip(fitted_models, clf_str):
        y_pred[model_name] = fit_model.predict(X)
        if task == 'binary':
            cm, scores = eval_preds(fit_model, X, y_true, y_pred[model_name], task)
        elif task == 'multi_class':
            cm, scores = eval_preds(fit_model, X, y_true, y_pred[model_name], task)
        cm_dict[model_name] = cm
        metrics.loc[model_name] = scores
    return y_pred, cm_dict, metrics


