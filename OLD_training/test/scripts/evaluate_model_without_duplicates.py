"""
Avaliação REAL do modelo no conjunto de teste SEM duplicados
Remove exemplos duplicados e executa o modelo novamente para métricas exatas
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_without_duplicates():
    """Avalia modelo removendo duplicados e executando predições reais"""
    
    print("="*100)
    print("AVALIAÇÃO REAL DO MODELO SEM DUPLICADOS")
    print("="*100)
    print()
    
    # Caminhos
    base_path = Path("/home/paulo/CascadeProjects/Applied_ML")
    data_path = base_path / "dataset/cleaned_data_no_bias/FOR_TRAINING"
    
    # 1. Carregar dados
    print("1. Carregando dados...")
    train_df = pd.read_csv(data_path / "TRAIN.csv")
    test_df = pd.read_csv(data_path / "TEST.csv")
    
    print(f"   Treino: {len(train_df)} exemplos")
    print(f"   Teste original: {len(test_df)} exemplos")
    print()
    
    # 2. Identificar e remover duplicados
    print("2. Identificando duplicados...")
    train_texts = set(train_df['text'].dropna())
    test_df['is_duplicate'] = test_df['text'].apply(
        lambda x: x in train_texts if pd.notna(x) else False
    )
    
    duplicates_count = test_df['is_duplicate'].sum()
    print(f"   Duplicados encontrados: {duplicates_count} ({duplicates_count/len(test_df)*100:.2f}%)")
    
    # Criar teste limpo
    test_clean_df = test_df[~test_df['is_duplicate']].copy()
    test_duplicates_df = test_df[test_df['is_duplicate']].copy()
    
    print(f"   Teste limpo: {len(test_clean_df)} exemplos")
    print(f"   Duplicados removidos: {len(test_duplicates_df)} exemplos")
    print()
    
    # Mostrar distribuição
    print("   Distribuição de classes no teste limpo:")
    print(f"   {test_clean_df['label'].value_counts().to_dict()}")
    print()
    
    # 3. Salvar teste limpo
    test_clean_path = base_path / "test/dataset/TEST_CLEAN_NO_DUPLICATES.csv"
    test_clean_df[['title', 'text', 'label']].to_csv(test_clean_path, index=False)
    print(f"3. Teste limpo salvo em: {test_clean_path}")
    print()
    
    # 4. Vetorizar teste limpo com TF-IDF
    print("4. Vetorizando teste limpo com TF-IDF...")
    
    # Carregar vetorizador treinado
    vectorizer_path = base_path / "vectorization/tf_idf/tfidf_ngram12_mindf2_maxdf09_maxfeat10k_20260518_220950_vectorizer.pkl"
    
    if not vectorizer_path.exists():
        print(f"   Vetorizador não encontrado em: {vectorizer_path}")
        print("   Buscando em locais alternativos...")
        
        # Tentar outros locais
        possible_paths = [
            base_path / "test/model/tfidf_ngram12_mindf2_maxdf09_maxfeat10k_20260518_220950_vectorizer.pkl",
            base_path / "training/results/baseline_naive_bayes/tfidf_vectorizer.pkl",
            base_path / "training/results/svm_vectorizations/tfidf/tfidf_vectorizer.pkl",
        ]
        
        for path in possible_paths:
            if path.exists():
                vectorizer_path = path
                print(f"   Encontrado em: {vectorizer_path}")
                break
    
    if vectorizer_path.exists():
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Vetorizar
        X_test_clean = vectorizer.transform(test_clean_df['text'])
        y_test_clean = test_clean_df['label'].values
        
        print(f"   ✓ Vetorização concluída: {X_test_clean.shape}")
        print()
        
        # 5. Carregar modelo treinado
        print("5. Carregando modelo TF-IDF + SVM...")
        
        model_path = base_path / "test/model/svm_tf_idf_20260519_174152.pkl"
        
        if not model_path.exists():
            # Buscar em outros locais
            possible_model_paths = [
                base_path / "training/results/svm_vectorizations/svm_tf_idf_20260519_174152.pkl",
            ]
            for path in possible_model_paths:
                if path.exists():
                    model_path = path
                    print(f"   Modelo encontrado em: {model_path}")
                    break
        
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            print(f"   ✓ Modelo carregado")
            print()
            
            # 6. Fazer predições
            print("6. Executando predições no teste limpo...")
            y_pred_clean = model.predict(X_test_clean)
            y_pred_proba_clean = model.predict_proba(X_test_clean)[:, 1]
            
            print(f"   ✓ Predições concluídas")
            print()
            
            # 7. Calcular métricas REAIS
            print("="*100)
            print("MÉTRICAS REAIS NO TESTE LIMPO (SEM DUPLICADOS)")
            print("="*100)
            print()
            
            accuracy_clean = accuracy_score(y_test_clean, y_pred_clean)
            precision_clean = precision_score(y_test_clean, y_pred_clean)
            recall_clean = recall_score(y_test_clean, y_pred_clean)
            f1_clean = f1_score(y_test_clean, y_pred_clean)
            roc_auc_clean = roc_auc_score(y_test_clean, y_pred_proba_clean)
            
            print(f"Accuracy:  {accuracy_clean:.4f}")
            print(f"Precision: {precision_clean:.4f}")
            print(f"Recall:    {recall_clean:.4f}")
            print(f"F1-Score:  {f1_clean:.4f}")
            print(f"ROC-AUC:   {roc_auc_clean:.4f}")
            print()
            
            # 8. Comparação com métricas originais
            print("="*100)
            print("COMPARAÇÃO: TESTE ORIGINAL vs TESTE LIMPO")
            print("="*100)
            print()
            
            original_metrics = {
                'Accuracy': 0.9717,
                'Precision': 0.9826,
                'Recall': 0.9602,
                'F1-Score': 0.9713,
                'ROC-AUC': 0.9922
            }
            
            clean_metrics = {
                'Accuracy': accuracy_clean,
                'Precision': precision_clean,
                'Recall': recall_clean,
                'F1-Score': f1_clean,
                'ROC-AUC': roc_auc_clean
            }
            
            print(f"{'Métrica':<15} {'Original':<12} {'Limpo':<12} {'Diferença':<12} {'% Mudança':<12}")
            print("-"*65)
            
            for metric in original_metrics.keys():
                orig = original_metrics[metric]
                clean = clean_metrics[metric]
                diff = clean - orig
                pct_change = (diff / orig) * 100
                
                print(f"{metric:<15} {orig:<12.4f} {clean:<12.4f} {diff:<+12.4f} {pct_change:<+12.2f}%")
            
            print()
            
            # 9. Matriz de confusão
            print("="*100)
            print("MATRIZ DE CONFUSÃO - TESTE LIMPO")
            print("="*100)
            print()
            
            cm_clean = confusion_matrix(y_test_clean, y_pred_clean)
            
            print("Matriz de confusão:")
            print(cm_clean)
            print()
            
            tn, fp, fn, tp = cm_clean.ravel()
            print(f"True Negatives:  {tn} ({tn/(tn+fp)*100:.2f}% dos negativos)")
            print(f"False Positives: {fp} ({fp/(tn+fp)*100:.2f}% dos negativos)")
            print(f"False Negatives: {fn} ({fn/(tp+fn)*100:.2f}% dos positivos)")
            print(f"True Positives:  {tp} ({tp/(tp+fn)*100:.2f}% dos positivos)")
            print()
            
            # 10. Relatório de classificação
            print("="*100)
            print("RELATÓRIO DE CLASSIFICAÇÃO DETALHADO")
            print("="*100)
            print()
            print(classification_report(y_test_clean, y_pred_clean, 
                                       target_names=['Não-Fraude', 'Fraude']))
            
            # 11. Salvar resultados
            results_path = base_path / "test/results"
            
            # Salvar matriz de confusão como imagem
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_clean, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            plt.title('Confusion Matrix - Clean Test Set (No Duplicates)', fontsize=14, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            plt.savefig(results_path / 'confusion_matrix_clean_test.png', dpi=300)
            print(f"✓ Matriz de confusão salva em: {results_path / 'confusion_matrix_clean_test.png'}")
            
            # Salvar relatório
            report_path = results_path / 'clean_test_evaluation_report.txt'
            with open(report_path, 'w') as f:
                f.write("AVALIAÇÃO NO TESTE LIMPO (SEM DUPLICADOS)\n")
                f.write("="*100 + "\n\n")
                f.write(f"Teste original: {len(test_df)} exemplos\n")
                f.write(f"Duplicados removidos: {duplicates_count} exemplos\n")
                f.write(f"Teste limpo: {len(test_clean_df)} exemplos\n\n")
                
                f.write("MÉTRICAS NO TESTE LIMPO:\n\n")
                for metric, value in clean_metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")
                
                f.write("\n\nCOMPARAÇÃO COM TESTE ORIGINAL:\n\n")
                f.write(f"{'Métrica':<15} {'Original':<12} {'Limpo':<12} {'Diferença':<12}\n")
                f.write("-"*55 + "\n")
                
                for metric in original_metrics.keys():
                    orig = original_metrics[metric]
                    clean = clean_metrics[metric]
                    diff = clean - orig
                    f.write(f"{metric:<15} {orig:<12.4f} {clean:<12.4f} {diff:<+12.4f}\n")
                
                f.write("\n\nMATRIZ DE CONFUSÃO:\n\n")
                f.write(f"TN: {tn}, FP: {fp}\n")
                f.write(f"FN: {fn}, TP: {tp}\n\n")
                
                f.write("\nRELATÓRIO DE CLASSIFICAÇÃO:\n\n")
                f.write(classification_report(y_test_clean, y_pred_clean, 
                                             target_names=['Não-Fraude', 'Fraude']))
            
            print(f"✓ Relatório salvo em: {report_path}")
            print()
            
            # 12. Conclusão
            print("="*100)
            print("CONCLUSÃO")
            print("="*100)
            print()
            
            if f1_clean >= 0.95:
                print("✅ EXCELENTE: F1-Score no teste limpo >= 0.95")
                print(f"   O modelo mantém performance excepcional ({f1_clean:.4f}) mesmo sem duplicados.")
            elif f1_clean >= 0.90:
                print("✅ MUITO BOM: F1-Score no teste limpo >= 0.90")
                print(f"   O modelo mantém alta performance ({f1_clean:.4f}) mesmo sem duplicados.")
            else:
                print("⚠️  ATENÇÃO: F1-Score no teste limpo < 0.90")
                print(f"   Performance caiu para {f1_clean:.4f} sem duplicados.")
            
            print()
            
            diff_f1 = f1_clean - original_metrics['F1-Score']
            if abs(diff_f1) < 0.01:
                print(f"✅ Diferença desprezível ({diff_f1:+.4f})")
                print("   O data leakage teve impacto mínimo nas métricas.")
            elif diff_f1 < 0:
                print(f"⚠️  Queda de {abs(diff_f1):.4f} no F1-Score")
                print("   O data leakage teve algum impacto, mas modelo ainda é robusto.")
            else:
                print(f"✅ Melhora de {diff_f1:+.4f} no F1-Score")
                print("   Teste limpo tem melhor representatividade.")
            
            print()
            print("="*100)
            
        else:
            print(f"   ❌ Modelo não encontrado")
            print("   Execute o treinamento primeiro ou forneça o caminho correto do modelo.")
    else:
        print(f"   ❌ Vetorizador não encontrado")
        print("   Execute a vetorização primeiro ou forneça o caminho correto do vetorizador.")

if __name__ == "__main__":
    evaluate_without_duplicates()
