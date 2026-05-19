#!/usr/bin/env python3
"""
Script para avaliar o melhor modelo (TF-IDF + SVM) no conjunto de TESTE

Modelo: TF-IDF + SVM
F1-Score no desenvolvimento: 0.9783
Precision no desenvolvimento: 1.0000 (perfeita!)

Author: Cascade AI Assistant
Date: 19 May 2026
"""

import pickle
import numpy as np
import scipy.sparse as sp
import os
from datetime import datetime
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, classification_report,
                           confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

def load_test_data(test_dir):
    """Carregar dados de teste vetorizados"""
    files = os.listdir(test_dir)
    matrix_files = [f for f in files if f.endswith('.npz')]
    label_files = [f for f in files if f.endswith('_labels.npy')]
    
    if not matrix_files or not label_files:
        raise FileNotFoundError(f"Arquivos de teste nao encontrados em {test_dir}")
    
    matrix_file = sorted(matrix_files)[-1]
    label_file = sorted(label_files)[-1]
    
    print(f"Carregando dados de teste...")
    print(f"Matriz: {matrix_file}")
    print(f"Labels: {label_file}")
    
    X_test = sp.load_npz(os.path.join(test_dir, matrix_file))
    y_test = np.load(os.path.join(test_dir, label_file))
    
    print(f"Matriz carregada: {X_test.shape}")
    print(f"Labels carregados: {y_test.shape}")
    print(f"Distribuicao: {np.bincount(y_test)}")
    
    return X_test, y_test

def load_best_model(model_path):
    """Carregar modelo treinado"""
    print(f"\nCarregando modelo de: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Modelo carregado: {type(model).__name__}")
    return model

def evaluate_model(model, X_test, y_test):
    """Avaliar modelo no conjunto de teste"""
    print("\n" + "="*70)
    print("AVALIANDO MODELO NO CONJUNTO DE TESTE")
    print("="*70)
    
    # Fazer predicoes
    print("\nFazendo predicoes...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calcular metricas
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    print("\nMETRICAS NO CONJUNTO DE TESTE:")
    print(f"  ACCURACY:  {metrics['accuracy']:.4f}")
    print(f"  PRECISION: {metrics['precision']:.4f}")
    print(f"  RECALL:    {metrics['recall']:.4f}")
    print(f"  F1-SCORE:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics, y_pred, y_prob

def generate_confusion_matrix(y_test, y_pred, output_dir):
    """Gerar e salvar matriz de confusao"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix - TEST SET - TF-IDF + SVM', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, 'test_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nMatriz de confusao salva: {cm_path}")
    return cm_path

def generate_classification_report_txt(y_test, y_pred, output_dir):
    """Gerar e salvar classification report"""
    report = classification_report(y_test, y_pred,
                                   target_names=['Negative', 'Positive'],
                                   digits=4)
    
    report_path = os.path.join(output_dir, 'test_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("CLASSIFICATION REPORT - TEST SET - TF-IDF + SVM\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    
    print(f"Classification report salvo: {report_path}")
    return report_path

def save_test_results(metrics, output_dir):
    """Salvar resultados do teste"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {
        'timestamp': timestamp,
        'model': 'TF-IDF + SVM',
        'test_metrics': metrics,
        'description': 'Avaliacao final no conjunto de teste'
    }
    
    results_path = os.path.join(output_dir, f'test_results_{timestamp}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Resultados salvos: {results_path}")
    return results_path

def generate_final_report(metrics, dev_metrics, output_dir):
    """Gerar relatorio final comparando desenvolvimento vs teste"""
    report_path = os.path.join(output_dir, 'FINAL_TEST_REPORT.md')
    
    with open(report_path, 'w') as f:
        f.write("# Relatorio Final - Avaliacao no Conjunto de Teste\n\n")
        f.write("## Modelo Avaliado\n\n")
        f.write("**TF-IDF + SVM (Linear Kernel)**\n\n")
        f.write("- Vetorizacao: TF-IDF (n-grams 1-2, max 10k features)\n")
        f.write("- Classificador: SVM (kernel linear, probability=True)\n")
        f.write("- Melhor modelo entre 18 combinacoes testadas\n\n")
        
        f.write("## Resultados no Conjunto de Desenvolvimento\n\n")
        f.write("| Metrica | Valor |\n")
        f.write("|---------|-------|\n")
        f.write(f"| Accuracy | {dev_metrics['accuracy']:.4f} |\n")
        f.write(f"| Precision | {dev_metrics['precision']:.4f} |\n")
        f.write(f"| Recall | {dev_metrics['recall']:.4f} |\n")
        f.write(f"| F1-Score | {dev_metrics['f1']:.4f} |\n")
        f.write(f"| ROC-AUC | {dev_metrics['roc_auc']:.4f} |\n\n")
        
        f.write("## Resultados no Conjunto de TESTE (Final)\n\n")
        f.write("| Metrica | Valor |\n")
        f.write("|---------|-------|\n")
        f.write(f"| Accuracy | {metrics['accuracy']:.4f} |\n")
        f.write(f"| Precision | {metrics['precision']:.4f} |\n")
        f.write(f"| Recall | {metrics['recall']:.4f} |\n")
        f.write(f"| F1-Score | {metrics['f1']:.4f} |\n")
        f.write(f"| ROC-AUC | {metrics['roc_auc']:.4f} |\n\n")
        
        f.write("## Comparacao Desenvolvimento vs Teste\n\n")
        f.write("| Metrica | Desenvolvimento | Teste | Diferenca |\n")
        f.write("|---------|-----------------|-------|----------|\n")
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            diff = metrics[metric] - dev_metrics[metric]
            diff_pct = (diff / dev_metrics[metric]) * 100
            f.write(f"| {metric.upper()} | {dev_metrics[metric]:.4f} | {metrics[metric]:.4f} | {diff:+.4f} ({diff_pct:+.2f}%) |\n")
        
        f.write("\n## Analise\n\n")
        
        # Analise automatica
        f1_diff = metrics['f1'] - dev_metrics['f1']
        if abs(f1_diff) < 0.01:
            f.write("**Consistencia Excelente:** Metricas de teste muito proximas ao desenvolvimento (diferenca < 1%).\n")
            f.write("O modelo generaliza muito bem para dados nunca vistos.\n\n")
        elif f1_diff > 0:
            f.write("**Performance Superior no Teste:** Modelo teve performance melhor no teste que no desenvolvimento.\n")
            f.write("Isso e incomum mas pode indicar que o conjunto de teste e ligeiramente mais facil.\n\n")
        else:
            f.write("**Leve Queda no Teste:** Performance no teste e inferior ao desenvolvimento.\n")
            if abs(f1_diff) < 0.05:
                f.write("Queda pequena (< 5%), ainda dentro do esperado para generalizacao.\n\n")
            else:
                f.write("Queda significativa, pode indicar overfitting ou diferenca na distribuicao dos dados.\n\n")
        
        f.write("## Conclusao\n\n")
        f.write(f"O modelo TF-IDF + SVM alcancou **F1-Score de {metrics['f1']:.4f}** no conjunto de teste final.\n\n")
        
        if metrics['f1'] >= 0.95:
            f.write("**Performance EXCELENTE** - Modelo pronto para producao.\n")
        elif metrics['f1'] >= 0.90:
            f.write("**Performance MUITO BOA** - Modelo confiavel para uso.\n")
        elif metrics['f1'] >= 0.85:
            f.write("**Performance BOA** - Modelo aceitavel, pode ser melhorado.\n")
        else:
            f.write("**Performance MODERADA** - Considerar melhorias ou modelos alternativos.\n")
    
    print(f"\nRelatorio final salvo: {report_path}")
    return report_path

def main():
    """Funcao principal"""
    BASE_DIR = "/home/paulo/CascadeProjects/Applied_ML"
    TEST_DATA_DIR = os.path.join(BASE_DIR, "vectorization/tf_idf/test")
    MODEL_PATH = os.path.join(BASE_DIR, "training/results/svm_vectorizations/svm_tf_idf_20260519_174152.pkl")
    OUTPUT_DIR = os.path.join(BASE_DIR, "training/results/final_test_evaluation")
    
    print("="*70)
    print("AVALIACAO FINAL - MELHOR MODELO NO CONJUNTO DE TESTE")
    print("="*70)
    print(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Modelo: TF-IDF + SVM")
    print(f"F1-Score no desenvolvimento: 0.9783")
    print()
    
    # Criar diretorio de saida
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Diretorio de saida: {OUTPUT_DIR}")
    
    try:
        # Carregar dados de teste
        X_test, y_test = load_test_data(TEST_DATA_DIR)
        
        # Carregar modelo
        model = load_best_model(MODEL_PATH)
        
        # Avaliar modelo
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)
        
        # Gerar relatorios
        print("\n" + "="*70)
        print("GERANDO RELATORIOS")
        print("="*70)
        
        cm_path = generate_confusion_matrix(y_test, y_pred, OUTPUT_DIR)
        report_path = generate_classification_report_txt(y_test, y_pred, OUTPUT_DIR)
        results_path = save_test_results(metrics, OUTPUT_DIR)
        
        # Metricas do desenvolvimento (para comparacao)
        dev_metrics = {
            'accuracy': 0.9787,
            'precision': 1.0000,
            'recall': 0.9574,
            'f1': 0.9783,
            'roc_auc': 0.9923
        }
        
        final_report_path = generate_final_report(metrics, dev_metrics, OUTPUT_DIR)
        
        print("\n" + "="*70)
        print("AVALIACAO CONCLUIDA COM SUCESSO!")
        print("="*70)
        print(f"\nF1-Score no TESTE: {metrics['f1']:.4f}")
        print(f"F1-Score no DEV:   {dev_metrics['f1']:.4f}")
        print(f"Diferenca:         {metrics['f1'] - dev_metrics['f1']:+.4f}")
        
        print("\nArquivos gerados:")
        print(f"  - {cm_path}")
        print(f"  - {report_path}")
        print(f"  - {results_path}")
        print(f"  - {final_report_path}")
        
    except Exception as e:
        print(f"\nERRO durante avaliacao: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
