#!/usr/bin/env python3
"""
Task 5c — Random Forest (6 vetorizações)
- RandomForestClassifier com class_weight='balanced'
- GridSearchCV com stratified 5-fold no treino
  (n_estimators=[100, 200], max_depth=[20, 50, None])
- Avaliação no dev set
- predict_proba para PR-AUC e ROC-AUC
"""

import numpy as np
import os
import sys
import json
import pickle
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score, accuracy_score,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task5_utils import generate_consolidated_reports

BASE_DIR = "/home/paulo/CascadeProjects/Applied_ML"
VEC_DIR = os.path.join(BASE_DIR, "NEW_training/vectorization")
RESULTS_DIR = os.path.join(BASE_DIR, "NEW_training/training/results")

VECTORIZATIONS = [
    ("tfidf", "sparse"),
    ("fasttext", "dense"),
    ("bert_base", "dense"),
    ("bert_large", "dense"),
    ("albertina_base", "dense"),
    ("albertina_large", "dense"),
]

PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [20, 50, None],
}


def load_data(vec_key, vec_type):
    vec_path = os.path.join(VEC_DIR, vec_key)
    if vec_type == "sparse":
        X_train = sparse.load_npz(os.path.join(vec_path, "train_sparse.npz"))
        X_dev = sparse.load_npz(os.path.join(vec_path, "dev_sparse.npz"))
    else:
        X_train = np.load(os.path.join(vec_path, "train_embeddings.npy"))
        X_dev = np.load(os.path.join(vec_path, "dev_embeddings.npy"))
    y_train = np.load(os.path.join(vec_path, "labels_train.npy"))
    y_dev = np.load(os.path.join(vec_path, "labels_dev.npy"))
    return X_train, X_dev, y_train, y_dev


def evaluate_model(y_true, y_pred, y_scores=None):
    metrics = {
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
    }
    if y_scores is not None:
        metrics['pr_auc'] = average_precision_score(y_true, y_scores)
        metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
    return metrics


def save_results(out_dir, vec_key, best_params,
                 cv_f1, dev_metrics, y_dev, y_pred, model):
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "best_params.json"), 'w') as f:
        json.dump(best_params, f, indent=2)

    with open(os.path.join(out_dir, "model.pkl"), 'wb') as f:
        pickle.dump(model, f)

    report = classification_report(y_dev, y_pred, zero_division=0)
    with open(os.path.join(out_dir, "classification_report.txt"), 'w') as f:
        f.write(f"RandomForest + {vec_key}\n")
        f.write(f"Best params: {best_params}\n")
        f.write(f"class_weight: balanced (fixo)\n")
        f.write(f"CV F1 (mean): {cv_f1:.4f}\n")
        f.write(f"\nDev metrics:\n")
        for k, v in dev_metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write(f"\n{report}\n")

    cm = confusion_matrix(y_dev, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f"RandomForest + {vec_key}")
    fig.colorbar(im)
    classes = ['Negative', 'Positive']
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=classes, yticklabels=classes)
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, text):
        for f in self.files:
            f.write(text)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def run_rf():
    rf_dir = os.path.join(RESULTS_DIR, "random_forest")
    os.makedirs(rf_dir, exist_ok=True)
    log_path = os.path.join(rf_dir, "random_forest_execution_log.txt")
    log_file = open(log_path, 'w')
    tee = Tee(sys.stdout, log_file)
    sys.stdout = tee

    print("=" * 70)
    print("TASK 5c — Random Forest (6 vetorizações, class_weight=balanced)")
    print("=" * 70)

    all_results = []
    n_vecs = len(VECTORIZATIONS)

    for i, (vec_key, vec_type) in enumerate(VECTORIZATIONS):
        print(f"\n{'=' * 60}")
        print(f"[{i+1}/{n_vecs}] RandomForest + {vec_key} ({vec_type})")
        print(f"{'=' * 60}")

        X_train, X_dev, y_train, y_dev = load_data(vec_key, vec_type)
        print(f"  Train: {X_train.shape} | Dev: {X_dev.shape}")
        print(f"  Train pos={int((y_train==1).sum())} neg={int((y_train==0).sum())}")

        clf = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        n_candidates = len(PARAM_GRID['n_estimators']) * len(PARAM_GRID['max_depth'])
        n_fits = n_candidates * 5
        print(f"  GridSearchCV: {n_candidates} candidates × 5 folds = {n_fits} fits")

        grid = GridSearchCV(
            clf, PARAM_GRID, cv=cv, scoring='f1',
            n_jobs=-1, verbose=2
        )
        grid.fit(X_train, y_train)

        print(f"  Best params: {grid.best_params_}")
        print(f"  CV F1 (mean): {grid.best_score_:.4f}")

        best_model = grid.best_estimator_
        print(f"  Predicting on dev...")
        y_pred = best_model.predict(X_dev)
        y_scores = best_model.predict_proba(X_dev)[:, 1]

        dev_metrics = evaluate_model(y_dev, y_pred, y_scores)
        print(f"  Dev metrics:")
        for k, v in dev_metrics.items():
            print(f"    {k}: {v:.4f}")

        out_dir = os.path.join(RESULTS_DIR, vec_key, "random_forest")
        save_results(out_dir, vec_key, grid.best_params_,
                     grid.best_score_, dev_metrics, y_dev, y_pred, best_model)

        all_results.append({
            'vectorization': vec_key,
            'classifier': 'random_forest',
            'best_params': grid.best_params_,
            'cv_f1': grid.best_score_,
            **dev_metrics,
        })

        print(f"  [{i+1}/{n_vecs}] CONCLUÍDO — F1={dev_metrics['f1']:.4f}")

    classifier_desc = (
        "Random Forest é um ensemble de árvores de decisão.\n"
        "Reduz overfitting combinando múltiplas árvores com amostragem aleatória.\n"
        "class_weight='balanced': ajusta pesos inversamente proporcionais à frequência das classes.\n"
        "predict_proba fornece probabilidades para PR-AUC e ROC-AUC."
    )
    config_str = (
        "RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)\n"
        "Grid: n_estimators=[100, 200] x max_depth=[20, 50, None]\n"
        "CV: StratifiedKFold(5, shuffle=True, random_state=42)\n"
        "Scoring: f1"
    )
    generate_consolidated_reports(
        all_results, RESULTS_DIR, "Random Forest",
        classifier_desc, config_str,
        "random_forest_results.json"
    )

    print(f"\n{'=' * 70}")
    print("TASK 5c CONCLUÍDA — Random Forest")
    print(f"{'=' * 70}")

    sys.stdout = sys.__stdout__
    log_file.close()
    print(f"Log: {log_path}")


if __name__ == "__main__":
    run_rf()
