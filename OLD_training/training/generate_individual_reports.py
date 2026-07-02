#!/usr/bin/env python3
"""
Script para gerar matrizes de confusao e classification reports individuais
a partir dos metadados e modelos ja salvos (sem reexecutar treinamento)

Author: Cascade AI Assistant
Date: 19 May 2026
"""

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import scipy.sparse as sp

def create_confusion_matrix(y_true, y_pred, output_path, title):
    """Gerar e salvar matriz de confusao"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Matriz de confusao salva: {output_path}")

def create_classification_report_txt(y_true, y_pred, output_path, title):
    """Gerar e salvar classification report em TXT"""
    report = classification_report(y_true, y_pred, 
                                   target_names=['Negative', 'Positive'],
                                   digits=4)
    
    with open(output_path, 'w') as f:
        f.write(f"{title}\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    
    print(f"Classification report salvo: {output_path}")

def load_vectorization_data(vec_dir, vec_name, is_sparse):
    """Carregar dados vetorizados"""
    files = os.listdir(vec_dir)
    
    if is_sparse:
        matrix_files = [f for f in files if f.endswith('.npz')]
        label_files = [f for f in files if f.endswith('_labels.npy')]
    else:
        matrix_files = [f for f in files if f.endswith('.npy') and 'matrix' in f.lower()]
        label_files = [f for f in files if f.endswith('.npy') and 'label' in f.lower()]
    
    if not matrix_files or not label_files:
        raise FileNotFoundError(f"Arquivos de {vec_name} nao encontrados")
    
    matrix_file = sorted(matrix_files)[-1]
    label_file = sorted(label_files)[-1]
    
    if is_sparse:
        X = sp.load_npz(os.path.join(vec_dir, matrix_file))
    else:
        X = np.load(os.path.join(vec_dir, matrix_file))
    
    y = np.load(os.path.join(vec_dir, label_file))
    
    return X, y

def process_experiment(experiment_name, metadata_path, vectorizations, base_output_dir):
    """Processar um experimento completo"""
    print(f"\n{'='*70}")
    print(f"PROCESSANDO: {experiment_name}")
    print(f"{'='*70}")
    
    # Carregar metadados
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    for vec_name, vec_config in vectorizations.items():
        print(f"\n{'-'*50}")
        print(f"Vetorizacao: {vec_name}")
        print(f"{'-'*50}")
        
        # Criar diretorio para esta vetorizacao
        vec_output_dir = os.path.join(base_output_dir, vec_name.lower().replace(' ', '_').replace('-', '_'))
        os.makedirs(vec_output_dir, exist_ok=True)
        
        # Carregar dados
        X, y = load_vectorization_data(vec_config['dir'], vec_name, vec_config.get('is_sparse', False))
        
        # Fazer o mesmo split (random_state=42 garante reproducibilidade)
        X_train, X_dev, y_train, y_dev = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Carregar modelo (tentar com underscore e com hifen)
        vec_name_underscore = vec_name.lower().replace(' ', '_').replace('-', '_')
        vec_name_hyphen = vec_name.lower().replace(' ', '-')
        
        model_files = [f for f in os.listdir(base_output_dir) 
                      if (vec_name_underscore in f.lower() or vec_name_hyphen in f.lower())
                      and f.endswith('.pkl')
                      and 'metadata' not in f.lower()]
        
        if not model_files:
            print(f"AVISO: Modelo nao encontrado para {vec_name}")
            continue
        
        model_path = os.path.join(base_output_dir, model_files[0])
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Fazer predicoes no desenvolvimento
        y_pred = model.predict(X_dev)
        
        # Gerar matriz de confusao
        cm_path = os.path.join(vec_output_dir, 'confusion_matrix.png')
        title = f'Confusion Matrix - {experiment_name} - {vec_name}'
        create_confusion_matrix(y_dev, y_pred, cm_path, title)
        
        # Gerar classification report
        report_path = os.path.join(vec_output_dir, 'classification_report.txt')
        title = f'CLASSIFICATION REPORT - {experiment_name.upper()} - {vec_name.upper()}'
        create_classification_report_txt(y_dev, y_pred, report_path, title)

def main():
    """Funcao principal"""
    BASE_DIR = "/home/paulo/CascadeProjects/Applied_ML"
    
    # Configuracao para Naive Bayes
    nb_vectorizations = {
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
    
    # Configuracao para SVM e Random Forest
    all_vectorizations = {
        'TF-IDF': {
            'dir': os.path.join(BASE_DIR, 'vectorization/tf_idf'),
            'is_sparse': True
        },
        **nb_vectorizations
    }
    
    # Processar Naive Bayes
    nb_metadata = os.path.join(BASE_DIR, 'training/results/naive_bayes_vectorizations/naive_bayes_all_vectorizations_metadata_20260519_173213.pkl')
    nb_output = os.path.join(BASE_DIR, 'training/results/naive_bayes_vectorizations')
    
    if os.path.exists(nb_metadata):
        process_experiment('Naive Bayes', nb_metadata, nb_vectorizations, nb_output)
    else:
        print(f"AVISO: Metadados Naive Bayes nao encontrados")
    
    # Processar SVM
    svm_metadata = os.path.join(BASE_DIR, 'training/results/svm_vectorizations/svm_all_vectorizations_metadata_20260519_174152.pkl')
    svm_output = os.path.join(BASE_DIR, 'training/results/svm_vectorizations')
    
    if os.path.exists(svm_metadata):
        process_experiment('SVM', svm_metadata, all_vectorizations, svm_output)
    else:
        print(f"AVISO: Metadados SVM nao encontrados")
    
    print("\n" + "="*70)
    print("GERACAO DE RELATORIOS INDIVIDUAIS CONCLUIDA!")
    print("="*70)

if __name__ == "__main__":
    main()
