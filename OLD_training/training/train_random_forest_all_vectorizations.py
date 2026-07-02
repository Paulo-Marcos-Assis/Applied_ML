#!/usr/bin/env python3
"""
Script para treinar Random Forest com diferentes vetorizacoes
Compara impacto de: TF-IDF, FastText, BERT-Base, BERT-Large, Albertina-Base, Albertina-Large

Author: Cascade AI Assistant
Date: 19 May 2026
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report,
                           confusion_matrix)
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

def create_output_dir(base_path):
    """Criar diretorio de saida se nao existir"""
    os.makedirs(base_path, exist_ok=True)
    print(f"Diretorio criado/verificado: {base_path}")

def load_sparse_data(vec_dir, vec_name):
    """Carregar dados esparsos (TF-IDF)"""
    print(f"\nCarregando {vec_name} (esparso) de: {vec_dir}")
    
    files = os.listdir(vec_dir)
    matrix_files = [f for f in files if f.endswith('.npz')]
    label_files = [f for f in files if f.endswith('_labels.npy')]
    
    if not matrix_files or not label_files:
        raise FileNotFoundError(f"Arquivos de {vec_name} nao encontrados")
    
    matrix_file = sorted(matrix_files)[-1]
    label_file = sorted(label_files)[-1]
    
    print(f"Matriz: {matrix_file}")
    print(f"Labels: {label_file}")
    
    X = sp.load_npz(os.path.join(vec_dir, matrix_file))
    y = np.load(os.path.join(vec_dir, label_file))
    
    print(f"Matriz carregada: {X.shape}")
    print(f"Labels carregados: {y.shape}")
    print(f"Distribuicao: {np.bincount(y)}")
    
    return X, y

def load_dense_data(vec_dir, vec_name):
    """Carregar dados densos (embeddings)"""
    print(f"\nCarregando {vec_name} (denso) de: {vec_dir}")
    
    files = os.listdir(vec_dir)
    matrix_files = [f for f in files if f.endswith('.npy') and 'matrix' in f.lower()]
    label_files = [f for f in files if f.endswith('.npy') and 'label' in f.lower()]
    
    if not matrix_files or not label_files:
        raise FileNotFoundError(f"Arquivos de {vec_name} nao encontrados")
    
    matrix_file = sorted(matrix_files)[-1]
    label_file = sorted(label_files)[-1]
    
    print(f"Matriz: {matrix_file}")
    print(f"Labels: {label_file}")
    
    X = np.load(os.path.join(vec_dir, matrix_file))
    y = np.load(os.path.join(vec_dir, label_file))
    
    print(f"Matriz carregada: {X.shape}")
    print(f"Labels carregados: {y.shape}")
    print(f"Distribuicao: {np.bincount(y)}")
    
    return X, y

def train_and_evaluate_random_forest(X_train, y_train, X_dev, y_dev, vec_name):
    """Treinar e avaliar Random Forest"""
    print(f"\n{'='*70}")
    print(f"TREINANDO RANDOM FOREST COM {vec_name}")
    print(f"{'='*70}")
    
    # Random Forest com 100 estimadores
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Configuracao: RandomForestClassifier(n_estimators=100)")
    
    # Validacao cruzada no treino
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    print("\nValidacao cruzada no conjunto de treino:")
    cv_results = {}
    for metric in scoring_metrics:
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric)
        cv_results[metric] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        print(f"{metric.upper()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # Treinar modelo final
    print("\nTreinando modelo final no conjunto de treino...")
    model.fit(X_train, y_train)
    
    # Avaliar no desenvolvimento
    print("Avaliando no conjunto de desenvolvimento...")
    y_pred = model.predict(X_dev)
    y_prob = model.predict_proba(X_dev)[:, 1]
    
    dev_metrics = {
        'accuracy': accuracy_score(y_dev, y_pred),
        'precision': precision_score(y_dev, y_pred),
        'recall': recall_score(y_dev, y_pred),
        'f1': f1_score(y_dev, y_pred),
        'roc_auc': roc_auc_score(y_dev, y_prob)
    }
    
    print("\nMetricas no conjunto de DESENVOLVIMENTO:")
    for metric, value in dev_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    return model, cv_results, dev_metrics

def compare_vectorizations(results, output_dir):
    """Comparar performance entre vetorizacoes"""
    print("\n" + "="*70)
    print("COMPARACAO ENTRE VETORIZACOES (RANDOM FOREST)")
    print("="*70)
    
    comparison_data = []
    for vec_name, vec_results in results.items():
        dev_metrics = vec_results['dev_metrics']
        comparison_data.append({
            'Vectorization': vec_name,
            'Accuracy': dev_metrics['accuracy'],
            'Precision': dev_metrics['precision'],
            'Recall': dev_metrics['recall'],
            'F1-Score': dev_metrics['f1'],
            'ROC-AUC': dev_metrics['roc_auc']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    print("\nTabela de Comparacao (Conjunto de Desenvolvimento):")
    print(df_comparison.to_string(index=False, float_format='%.4f'))
    
    # Identificar melhor vetorizacao
    best_vec = df_comparison.loc[df_comparison['F1-Score'].idxmax(), 'Vectorization']
    best_f1 = df_comparison['F1-Score'].max()
    print(f"\nMelhor Vetorizacao: {best_vec} (F1-Score: {best_f1:.4f})")
    
    # Salvar tabela
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(output_dir, f'random_forest_vectorizations_comparison_{timestamp}.csv')
    df_comparison.to_csv(csv_path, index=False)
    print(f"\nTabela salva em: {csv_path}")
    
    return df_comparison, best_vec

def plot_comparison(df_comparison, output_dir):
    """Plotar comparacao entre vetorizacoes"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Random Forest: Vectorization Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(df_comparison['Vectorization'], df_comparison[metric], 
                     color='forestgreen', alpha=0.7)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xlabel('Vectorization', fontsize=12)
        ax.set_title(f'{metric}', fontsize=13, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'random_forest_vectorizations_comparison_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Grafico salvo em: {plot_path}")
    return plot_path

def generate_individual_reports(vec_name, y_dev, y_pred, output_dir):
    """Gerar matriz de confusao e classification report individuais"""
    # Criar diretorio para esta vetorizacao
    vec_dir = os.path.join(output_dir, vec_name.lower().replace(' ', '_').replace('-', '_'))
    os.makedirs(vec_dir, exist_ok=True)
    
    # Matriz de confusao
    cm = confusion_matrix(y_dev, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - Random Forest - {vec_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(vec_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Classification report
    report = classification_report(y_dev, y_pred,
                                   target_names=['Negative', 'Positive'],
                                   digits=4)
    report_path = os.path.join(vec_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"CLASSIFICATION REPORT - RANDOM FOREST - {vec_name.upper()}\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    
    print(f"  Matriz de confusao: {cm_path}")
    print(f"  Classification report: {report_path}")

def save_results(results, output_dir):
    """Salvar todos os resultados"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Salvar modelos e resultados
    for vec_name, vec_results in results.items():
        model_path = os.path.join(output_dir, f'random_forest_{vec_name.lower().replace(" ", "_").replace("-", "_")}_{timestamp}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(vec_results['model'], f)
        print(f"Modelo {vec_name} salvo em: {model_path}")
    
    # Salvar metadados
    metadata = {
        'timestamp': timestamp,
        'classifier': 'Random Forest',
        'results_summary': {
            name: {
                'cv_results': results[name]['cv_results'],
                'dev_metrics': results[name]['dev_metrics']
            } for name in results.keys()
        }
    }
    
    metadata_path = os.path.join(output_dir, f'random_forest_all_vectorizations_metadata_{timestamp}.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\nMetadados salvos em: {metadata_path}")
    return metadata_path

def main():
    """Funcao principal"""
    BASE_DIR = "/home/paulo/CascadeProjects/Applied_ML"
    OUTPUT_DIR = os.path.join(BASE_DIR, "training/results/random_forest_vectorizations")
    
    # Configuracao das vetorizacoes (TODAS as 6)
    vectorizations = {
        'TF-IDF': {
            'dir': os.path.join(BASE_DIR, 'vectorization/tf_idf'),
            'is_sparse': True
        },
        'FastText': {
            'dir': os.path.join(BASE_DIR, 'vectorization/fasttext/outputs'),
            'is_sparse': False
        },
        'BERT-Base': {
            'dir': os.path.join(BASE_DIR, 'vectorization/bert_base'),
            'is_sparse': False
        },
        'BERT-Large': {
            'dir': os.path.join(BASE_DIR, 'vectorization/bert_large'),
            'is_sparse': False
        },
        'Albertina-Base': {
            'dir': os.path.join(BASE_DIR, 'vectorization/albertina_base'),
            'is_sparse': False
        },
        'Albertina-Large': {
            'dir': os.path.join(BASE_DIR, 'vectorization/albertina_large'),
            'is_sparse': False
        }
    }
    
    print("="*70)
    print("RANDOM FOREST COM MULTIPLAS VETORIZACOES")
    print("="*70)
    print(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Vetorizacoes: {', '.join(vectorizations.keys())}")
    
    try:
        create_output_dir(OUTPUT_DIR)
        
        results = {}
        
        for vec_name, vec_config in vectorizations.items():
            print(f"\n{'#'*70}")
            print(f"PROCESSANDO: {vec_name}")
            print(f"{'#'*70}")
            
            # Carregar dados
            if vec_config['is_sparse']:
                X, y = load_sparse_data(vec_config['dir'], vec_name)
            else:
                X, y = load_dense_data(vec_config['dir'], vec_name)
            
            # Split treino/desenvolvimento
            print(f"\nDividindo dados em treino (80%) e desenvolvimento (20%)...")
            X_train, X_dev, y_train, y_dev = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            print(f"Treino: {X_train.shape[0]} exemplos")
            print(f"Desenvolvimento: {X_dev.shape[0]} exemplos")
            
            # Treinar e avaliar
            model, cv_results, dev_metrics = train_and_evaluate_random_forest(
                X_train, y_train, X_dev, y_dev, vec_name
            )
            
            # Fazer predicoes para relatorios individuais
            y_pred = model.predict(X_dev)
            
            # Gerar relatorios individuais
            print(f"\nGerando relatorios individuais para {vec_name}...")
            generate_individual_reports(vec_name, y_dev, y_pred, OUTPUT_DIR)
            
            results[vec_name] = {
                'model': model,
                'cv_results': cv_results,
                'dev_metrics': dev_metrics
            }
        
        # Comparar resultados
        df_comparison, best_vec = compare_vectorizations(results, OUTPUT_DIR)
        
        # Plotar comparacao
        plot_path = plot_comparison(df_comparison, OUTPUT_DIR)
        
        # Salvar resultados
        metadata_path = save_results(results, OUTPUT_DIR)
        
        print("\n" + "="*70)
        print("TREINAMENTO CONCLUIDO COM SUCESSO!")
        print("="*70)
        print(f"Melhor vetorizacao: {best_vec}")
        print(f"Total de vetorizacoes testadas: {len(vectorizations)}")
        
    except Exception as e:
        print(f"\nERRO durante execucao: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
