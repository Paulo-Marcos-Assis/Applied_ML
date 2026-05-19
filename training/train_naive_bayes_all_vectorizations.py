#!/usr/bin/env python3
"""
Script para treinar Naive Bayes com diferentes vetorizacoes
Compara impacto de: FastText, BERT-Base, BERT-Large, Albertina-Base, Albertina-Large
(TF-IDF ja coberto pelo baseline)

Author: Cascade AI Assistant
Date: 19 May 2026
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

def create_output_dir(base_path):
    """Criar diretorio de saida se nao existir"""
    os.makedirs(base_path, exist_ok=True)
    print(f"Diretorio criado/verificado: {base_path}")

def load_vectorization_data(vec_dir, vec_name):
    """Carregar matriz de vetorizacao e labels"""
    print(f"\nCarregando {vec_name} de: {vec_dir}")
    
    files = os.listdir(vec_dir)
    
    # Procurar arquivos de matriz
    matrix_files = [f for f in files if f.endswith('.npy') and 'matrix' in f.lower()]
    label_files = [f for f in files if f.endswith('.npy') and 'label' in f.lower()]
    
    if not matrix_files or not label_files:
        raise FileNotFoundError(f"Arquivos de {vec_name} nao encontrados em {vec_dir}")
    
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

def train_and_evaluate_naive_bayes(X_train, y_train, X_dev, y_dev, vec_name, is_dense=True):
    """Treinar e avaliar Naive Bayes"""
    print(f"\n{'='*70}")
    print(f"TREINANDO NAIVE BAYES COM {vec_name}")
    print(f"{'='*70}")
    
    # Escolher tipo de Naive Bayes baseado no tipo de dados
    if is_dense:
        model = GaussianNB()
        print("Usando GaussianNB (dados densos - embeddings)")
    else:
        model = MultinomialNB(alpha=1.0)
        print("Usando MultinomialNB (dados esparsos)")
    
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
    print("COMPARACAO ENTRE VETORIZACOES (NAIVE BAYES)")
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
    csv_path = os.path.join(output_dir, f'naive_bayes_vectorizations_comparison_{timestamp}.csv')
    df_comparison.to_csv(csv_path, index=False)
    print(f"\nTabela salva em: {csv_path}")
    
    return df_comparison, best_vec

def plot_comparison(df_comparison, output_dir):
    """Plotar comparacao entre vetorizacoes"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Naive Bayes: Vectorization Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(df_comparison['Vectorization'], df_comparison[metric], 
                     color='steelblue', alpha=0.7)
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
    plot_path = os.path.join(output_dir, f'naive_bayes_vectorizations_comparison_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Grafico salvo em: {plot_path}")
    return plot_path

def save_results(results, output_dir):
    """Salvar todos os resultados"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Salvar modelos e resultados
    for vec_name, vec_results in results.items():
        model_path = os.path.join(output_dir, f'naive_bayes_{vec_name.lower().replace(" ", "_")}_{timestamp}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(vec_results['model'], f)
        print(f"Modelo {vec_name} salvo em: {model_path}")
    
    # Salvar metadados
    metadata = {
        'timestamp': timestamp,
        'classifier': 'Naive Bayes',
        'results_summary': {
            name: {
                'cv_results': results[name]['cv_results'],
                'dev_metrics': results[name]['dev_metrics']
            } for name in results.keys()
        }
    }
    
    metadata_path = os.path.join(output_dir, f'naive_bayes_all_vectorizations_metadata_{timestamp}.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\nMetadados salvos em: {metadata_path}")
    return metadata_path

def main():
    """Funcao principal"""
    BASE_DIR = "/home/paulo/CascadeProjects/Applied_ML"
    OUTPUT_DIR = os.path.join(BASE_DIR, "training/results/naive_bayes_vectorizations")
    
    # Configuracao das vetorizacoes (excluindo TF-IDF que ja foi rodado no baseline)
    vectorizations = {
        'FastText': {
            'dir': os.path.join(BASE_DIR, 'vectorization/fasttext/outputs'),
            'is_dense': True
        },
        'BERT-Base': {
            'dir': os.path.join(BASE_DIR, 'vectorization/bert_base'),
            'is_dense': True
        },
        'BERT-Large': {
            'dir': os.path.join(BASE_DIR, 'vectorization/bert_large'),
            'is_dense': True
        },
        'Albertina-Base': {
            'dir': os.path.join(BASE_DIR, 'vectorization/albertina_base'),
            'is_dense': True
        },
        'Albertina-Large': {
            'dir': os.path.join(BASE_DIR, 'vectorization/albertina_large'),
            'is_dense': True
        }
    }
    
    print("="*70)
    print("NAIVE BAYES COM MULTIPLAS VETORIZACOES")
    print("="*70)
    print(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Vetorizacoes: {', '.join(vectorizations.keys())}")
    print("\nNOTA: TF-IDF + Naive Bayes ja foi executado no baseline")
    
    try:
        create_output_dir(OUTPUT_DIR)
        
        results = {}
        
        for vec_name, vec_config in vectorizations.items():
            print(f"\n{'#'*70}")
            print(f"PROCESSANDO: {vec_name}")
            print(f"{'#'*70}")
            
            # Carregar dados
            X, y = load_vectorization_data(vec_config['dir'], vec_name)
            
            # Split treino/desenvolvimento
            print(f"\nDividindo dados em treino (80%) e desenvolvimento (20%)...")
            X_train, X_dev, y_train, y_dev = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            print(f"Treino: {X_train.shape[0]} exemplos")
            print(f"Desenvolvimento: {X_dev.shape[0]} exemplos")
            
            # Treinar e avaliar
            model, cv_results, dev_metrics = train_and_evaluate_naive_bayes(
                X_train, y_train, X_dev, y_dev, vec_name, vec_config['is_dense']
            )
            
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
