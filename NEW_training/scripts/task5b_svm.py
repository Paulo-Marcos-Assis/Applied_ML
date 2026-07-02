#!/usr/bin/env python3
"""
Task 5b — SVM LinearSVC (6 vetorizações)
- LinearSVC com class_weight='balanced'
- GridSearchCV com stratified 5-fold no treino (C=[0.1, 1, 10])
- Avaliação no dev set
- decision_function para PR-AUC e ROC-AUC (sem probability)
"""

import numpy as np
import os
import sys
import json
import pickle
from scipy import sparse
from sklearn.svm import LinearSVC
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
    'C': [0.1, 1, 10],
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
        f.write(f"LinearSVC + {vec_key}\n")
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
    ax.set_title(f"LinearSVC + {vec_key}")
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


def run_svm():
    svm_dir = os.path.join(RESULTS_DIR, "svm")
    os.makedirs(svm_dir, exist_ok=True)
    log_path = os.path.join(svm_dir, "svm_execution_log.txt")
    log_file = open(log_path, 'w')
    tee = Tee(sys.stdout, log_file)
    sys.stdout = tee

    print("=" * 70)
    print("TASK 5b — SVM LinearSVC (6 vetorizações, class_weight=balanced)")
    print("=" * 70)

    all_results = []
    n_vecs = len(VECTORIZATIONS)

    for i, (vec_key, vec_type) in enumerate(VECTORIZATIONS):
        print(f"\n{'=' * 60}")
        print(f"[{i+1}/{n_vecs}] LinearSVC + {vec_key} ({vec_type})")
        print(f"{'=' * 60}")

        X_train, X_dev, y_train, y_dev = load_data(vec_key, vec_type)
        print(f"  Train: {X_train.shape} | Dev: {X_dev.shape}")
        print(f"  Train pos={int((y_train==1).sum())} neg={int((y_train==0).sum())}")

        clf = LinearSVC(
            class_weight='balanced',
            dual='auto',
            max_iter=10000,
            random_state=42,
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        n_candidates = len(PARAM_GRID['C'])
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
        y_scores = best_model.decision_function(X_dev)

        dev_metrics = evaluate_model(y_dev, y_pred, y_scores)
        print(f"  Dev metrics:")
        for k, v in dev_metrics.items():
            print(f"    {k}: {v:.4f}")

        out_dir = os.path.join(RESULTS_DIR, vec_key, "svm")
        save_results(out_dir, vec_key, grid.best_params_,
                     grid.best_score_, dev_metrics, y_dev, y_pred, best_model)

        all_results.append({
            'vectorization': vec_key,
            'classifier': 'linear_svc',
            'best_params': grid.best_params_,
            'cv_f1': grid.best_score_,
            **dev_metrics,
        })

        print(f"  [{i+1}/{n_vecs}] CONCLUÍDO — F1={dev_metrics['f1']:.4f}")

    classifier_desc = (
        "SVM (Support Vector Machine) encontra o hiperplano ótimo que melhor separa as classes.\n"
        "LinearSVC: kernel linear, mais eficiente para alta dimensionalidade.\n"
        "class_weight='balanced': ajusta pesos inversamente proporcionais à frequência das classes.\n"
        "decision_function fornece scores para PR-AUC e ROC-AUC."
    )
    config_str = (
        "LinearSVC(class_weight='balanced', dual='auto', max_iter=10000, random_state=42)\n"
        "Grid: C=[0.1, 1, 10]\n"
        "CV: StratifiedKFold(5, shuffle=True, random_state=42)\n"
        "Scoring: f1"
    )
    generate_consolidated_reports(
        all_results, RESULTS_DIR, "SVM",
        classifier_desc, config_str,
        "svm_results.json"
    )

    print(f"\n{'=' * 70}")
    print("TASK 5b CONCLUÍDA — SVM LinearSVC")
    print(f"{'=' * 70}")

    sys.stdout = sys.__stdout__
    log_file.close()
    print(f"Log: {log_path}")


if __name__ == "__main__":
    run_svm()
