#!/usr/bin/env python3
"""
Baseline Model: Naive Bayes with TF-IDF Vectorization
Applied ML Project - Fraud Detection in News

Author: Cascade AI Assistant
Date: 19 May 2026
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report, 
                           confusion_matrix)
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

def load_tfidf_data(tfidf_dir):
    """Carregar matriz TF-IDF e labels"""
    print(f"Carregando dados TF-IDF de: {tfidf_dir}")
    
    # Encontrar arquivos mais recentes
    files = os.listdir(tfidf_dir)
    matrix_files = [f for f in files if f.endswith('.npz')]
    label_files = [f for f in files if f.endswith('_labels.npy')]
    
    if not matrix_files or not label_files:
        raise FileNotFoundError("Arquivos TF-IDF nao encontrados")
    
    # Usar o arquivo mais recente
    matrix_file = sorted(matrix_files)[-1]
    label_file = sorted(label_files)[-1]
    
    print(f"Matriz: {matrix_file}")
    print(f"Labels: {label_file}")
    
    # Carregar matriz esparsa
    matrix_path = os.path.join(tfidf_dir, matrix_file)
    X = sp.load_npz(matrix_path)
    
    # Carregar labels
    label_path = os.path.join(tfidf_dir, label_file)
    y = np.load(label_path)
    
    print(f"Matriz carregada: {X.shape}")
    print(f"Labels carregados: {y.shape}")
    print(f"Distribuicao: {np.bincount(y)}")
    
    return X, y

def create_output_dir(base_path):
    """Criar diretório de saída se não existir"""
    os.makedirs(base_path, exist_ok=True)
    print(f"Diretorio criado/verificado: {base_path}")

def train_naive_bayes_baseline(X_train, y_train, X_dev, y_dev, output_dir):
    """Treinar baseline Naive Bayes com validação cruzada"""
    print("\n" + "="*70)
    print("TREINANDO BASELINE NAIVE BAYES COM TF-IDF")
    print("="*70)
    
    print(f"Conjunto de treino: {X_train.shape}")
    print(f"Conjunto de desenvolvimento: {X_dev.shape}")
    
    # Configuração da validação cruzada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Modelo baseline
    model = MultinomialNB(alpha=1.0)  # Laplace smoothing padrão
    
    # Métricas para avaliação
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    print(f"\nModelo: {model}")
    print(f"Validação cruzada: 5-fold stratified")
    print(f"Métricas: {scoring_metrics}")
    
    # Validação cruzada APENAS no conjunto de treino
    print("\nExecutando validação cruzada no conjunto de treino...")
    cv_results = {}
    for metric in scoring_metrics:
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric)
        cv_results[metric] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        print(f"{metric.upper()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # Treinar modelo final no conjunto de TREINO
    print("\nTreinando modelo final no conjunto de treino...")
    model.fit(X_train, y_train)
    
    # Previsões no conjunto de DESENVOLVIMENTO
    print("Avaliando no conjunto de desenvolvimento...")
    y_pred = model.predict(X_dev)
    y_prob = model.predict_proba(X_dev)[:, 1]
    
    # Métricas no conjunto de desenvolvimento
    dev_metrics = {
        'accuracy': accuracy_score(y_dev, y_pred),
        'precision': precision_score(y_dev, y_pred),
        'recall': recall_score(y_dev, y_pred),
        'f1': f1_score(y_dev, y_pred),
        'roc_auc': roc_auc_score(y_dev, y_prob)
    }
    
    print("\nMétricas no conjunto de DESENVOLVIMENTO:")
    for metric, value in dev_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    return model, cv_results, dev_metrics, y_pred, y_prob

def generate_classification_report(y_true, y_pred, output_dir):
    """Gerar relatório de classificação detalhado"""
    print("\n" + "="*50)
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("="*50)
    
    report = classification_report(y_true, y_pred, target_names=['Negativo', 'Positivo'])
    print(report)
    
    # Salvar relatório
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RELATÓRIO DE CLASSIFICAÇÃO - BASELINE NAIVE BAYES + TF-IDF\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    
    print(f"Relatório salvo em: {report_path}")
    
    return report

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plotar e salvar matriz de confusão"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negativo', 'Positivo'],
                yticklabels=['Negativo', 'Positivo'])
    plt.title('Matriz de Confusão - Baseline Naive Bayes')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()
    
    # Salvar figura
    plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Matriz de confusão salva em: {plot_path}")
    
    return cm

def save_model_and_results(model, cv_results, final_metrics, output_dir):
    """Salvar modelo e resultados"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salvar modelo
    model_path = os.path.join(output_dir, f'naive_bayes_baseline_{timestamp}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Salvar resultados
    results = {
        'model_type': 'MultinomialNB',
        'vectorization': 'TF-IDF',
        'timestamp': timestamp,
        'cv_results': cv_results,
        'final_metrics': final_metrics,
        'model_params': model.get_params()
    }
    
    results_path = os.path.join(output_dir, f'baseline_results_{timestamp}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Modelo salvo em: {model_path}")
    print(f"Resultados salvos em: {results_path}")
    
    return model_path, results_path

def print_baseline_summary(cv_results, dev_metrics):
    """Imprimir resumo do baseline"""
    print("\n" + "="*70)
    print("RESUMO DO BASELINE - NAIVE BAYES + TF-IDF")
    print("="*70)
    
    print("\nResultados da Validação Cruzada (5-fold no treino):")
    for metric, results in cv_results.items():
        print(f"  {metric.upper()}: {results['mean']:.4f} (+/- {results['std'] * 2:.4f})")
    
    print("\nMétricas no Conjunto de DESENVOLVIMENTO:")
    for metric, value in dev_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    # Estabelecer baseline de performance
    baseline_f1_cv = cv_results['f1']['mean']
    baseline_f1_dev = dev_metrics['f1']
    
    print(f"\nBASELINE ESTABELECIDO:")
    print(f"  F1-Score (CV no treino): {baseline_f1_cv:.4f}")
    print(f"  F1-Score (Desenvolvimento): {baseline_f1_dev:.4f}")
    print(f"  Accuracy (Desenvolvimento): {dev_metrics['accuracy']:.4f}")
    print(f"  Precision (Desenvolvimento): {dev_metrics['precision']:.4f}")
    print(f"  Recall (Desenvolvimento): {dev_metrics['recall']:.4f}")
    
    return baseline_f1_dev

def main():
    """Função principal"""
    # Configurações
    TFIDF_DIR = "/home/paulo/CascadeProjects/Applied_ML/vectorization/tf_idf"
    OUTPUT_DIR = "/home/paulo/CascadeProjects/Applied_ML/training/results/baseline_naive_bayes"
    
    print("="*70)
    print("BASELINE NAIVE BAYES + TF-IDF PARA DETECÇÃO DE FRAUDES")
    print("="*70)
    print(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    try:
        # Criar diretório de saída
        create_output_dir(OUTPUT_DIR)
        
        # Carregar dados TF-IDF
        X, y = load_tfidf_data(TFIDF_DIR)
        
        # Split treino/desenvolvimento (80/20)
        print("\nDividindo dados em treino (80%) e desenvolvimento (20%)...")
        X_train, X_dev, y_train, y_dev = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Treino: {X_train.shape[0]} exemplos")
        print(f"Desenvolvimento: {X_dev.shape[0]} exemplos")
        print("\nNOTA: Conjunto de TESTE separado ainda nao foi vetorizado.")
        print("Localizacao: /dataset/cleaned_data_no_bias/FOR_TRAINING/Pre_processed_for_Sparse/TEST_PREPROCESSED.csv")
        
        # Treinar baseline
        model, cv_results, dev_metrics, y_pred, y_prob = train_naive_bayes_baseline(
            X_train, y_train, X_dev, y_dev, OUTPUT_DIR
        )
        
        # Gerar relatórios (usando conjunto de desenvolvimento)
        report = generate_classification_report(y_dev, y_pred, OUTPUT_DIR)
        cm = plot_confusion_matrix(y_dev, y_pred, OUTPUT_DIR)
        
        # Salvar modelo e resultados
        model_path, results_path = save_model_and_results(model, cv_results, dev_metrics, OUTPUT_DIR)
        
        # Imprimir resumo
        baseline_f1 = print_baseline_summary(cv_results, dev_metrics)
        
        print("\n" + "="*70)
        print("BASELINE CONCLUÍDO COM SUCESSO!")
        print("="*70)
        print(f"F1-Score baseline: {baseline_f1:.4f}")
        print("Próximos passos:")
        print("1. Comparar com outros modelos (SVM, Random Forest, Logistic Regression)")
        print("2. Experimentar diferentes técnicas de vetorização")
        print("3. Aplicar feature engineering e selection")
        print("4. Otimizar hiperparâmetros (após model selection)")
        
    except Exception as e:
        print(f"ERRO: {str(e)}")
        raise

if __name__ == "__main__":
    main()
