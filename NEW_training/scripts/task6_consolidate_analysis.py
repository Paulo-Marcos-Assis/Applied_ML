#!/usr/bin/env python3
"""
Task 6 — Consolidação e Análise (18 combinações)
- Carrega resultados dos 3 classificadores
- Gera CONSOLIDACAO_FINAL.md com análise comparativa
- Identifica melhor modelo para Task 7
"""

import json
import os
from datetime import datetime

BASE_DIR = "/home/paulo/CascadeProjects/Applied_ML"
RESULTS_DIR = os.path.join(BASE_DIR, "NEW_training/training/results")

def load_all_results():
    """Carrega os 3 JSONs consolidados (NB, SVM, RF)"""
    results = []
    
    # Naive Bayes
    with open(os.path.join(RESULTS_DIR, "naive_bayes/naive_bayes_results.json")) as f:
        nb_results = json.load(f)
        for r in nb_results:
            r['classifier_name'] = 'Naive Bayes'
            r['classifier_abbr'] = 'NB'
        results.extend(nb_results)
    
    # SVM
    with open(os.path.join(RESULTS_DIR, "svm/svm_results.json")) as f:
        svm_results = json.load(f)
        for r in svm_results:
            r['classifier_name'] = 'SVM'
            r['classifier_abbr'] = 'SVM'
        results.extend(svm_results)
    
    # Random Forest
    with open(os.path.join(RESULTS_DIR, "random_forest/random_forest_results.json")) as f:
        rf_results = json.load(f)
        for r in rf_results:
            r['classifier_name'] = 'Random Forest'
            r['classifier_abbr'] = 'RF'
        results.extend(rf_results)
    
    return results


def generate_consolidacao_md(results):
    """Gera CONSOLIDACAO_FINAL.md"""
    
    # Ordena por F1 decrescente
    results_sorted = sorted(results, key=lambda r: r.get('f1', 0), reverse=True)
    best = results_sorted[0]
    
    lines = []
    lines.append("# Consolidação Final — 18 Combinações (Vetorização × Classificador)\n\n")
    lines.append("## Resumo Executivo\n\n")
    lines.append(f"Este documento consolida os resultados de **18 combinações** de vetorizações e classificadores para detecção de fraudes em notícias, avaliadas no conjunto de desenvolvimento (9.163 exemplos, 1,54% positivos).\n\n")
    lines.append(f"**Melhor modelo identificado:** {best['vectorization'].upper()} + {best['classifier_name']} (F1-Score: {best['f1']:.4f})\n\n")
    lines.append("---\n\n")
    
    # Tabela completa
    lines.append("## 1. Resultados Completos — Todas as 18 Combinações\n\n")
    lines.append("### 1.1 Tabela Geral Ordenada por F1-Score (Dev Set)\n\n")
    lines.append("| Rank | Vetorização | Classificador | F1 | Precision | Recall | CV F1 | PR-AUC | ROC-AUC | Accuracy |\n")
    lines.append("|------|-------------|---------------|-----|-----------|--------|-------|--------|---------|----------|\n")
    
    for i, r in enumerate(results_sorted, 1):
        vec = r['vectorization']
        clf = r['classifier_abbr']
        f1 = r.get('f1', 0)
        prec = r.get('precision', 0)
        rec = r.get('recall', 0)
        cv_f1 = r.get('cv_f1', 0)
        pr_auc = r.get('pr_auc', 0)
        roc_auc = r.get('roc_auc', 0)
        acc = r.get('accuracy', 0)
        
        f1_str = f"**{f1:.4f}**" if i == 1 else f"{f1:.4f}"
        lines.append(f"| {i} | {vec} | {clf} | {f1_str} | {prec:.4f} | {rec:.4f} | {cv_f1:.4f} | {pr_auc:.4f} | {roc_auc:.4f} | {acc:.4f} |\n")
    
    lines.append("\n")
    
    # Top 5
    lines.append("### 1.2 Top 5 Modelos\n\n")
    for i, r in enumerate(results_sorted[:5], 1):
        vec = r['vectorization']
        clf = r['classifier_name']
        f1 = r.get('f1', 0)
        prec = r.get('precision', 0)
        rec = r.get('recall', 0)
        pr_auc = r.get('pr_auc', 0)
        
        lines.append(f"{i}. **{vec.upper()} + {clf}: {f1:.4f}**\n")
        lines.append(f"   - Precision: {prec:.4f}\n")
        lines.append(f"   - Recall: {rec:.4f}\n")
        lines.append(f"   - PR-AUC: {pr_auc:.4f}\n")
        lines.append(f"\n")
    
    lines.append("---\n\n")
    
    # Análise por classificador
    lines.append("## 2. Análise por Classificador\n\n")
    
    for clf_name in ['Naive Bayes', 'SVM', 'Random Forest']:
        clf_results = [r for r in results_sorted if r['classifier_name'] == clf_name]
        best_clf = clf_results[0]
        worst_clf = clf_results[-1]
        avg_f1 = sum(r.get('f1', 0) for r in clf_results) / len(clf_results)
        
        lines.append(f"### 2.{['Naive Bayes', 'SVM', 'Random Forest'].index(clf_name) + 1} {clf_name}\n\n")
        lines.append(f"**Melhor:** {best_clf['vectorization']} (F1={best_clf['f1']:.4f})\n\n")
        lines.append(f"**Pior:** {worst_clf['vectorization']} (F1={worst_clf['f1']:.4f})\n\n")
        lines.append(f"**Média F1:** {avg_f1:.4f}\n\n")
        
        lines.append("| Vetorização | F1 | Precision | Recall | PR-AUC |\n")
        lines.append("|-------------|-----|-----------|--------|--------|\n")
        for r in clf_results:
            lines.append(f"| {r['vectorization']} | {r.get('f1', 0):.4f} | {r.get('precision', 0):.4f} | {r.get('recall', 0):.4f} | {r.get('pr_auc', 0):.4f} |\n")
        lines.append("\n")
    
    lines.append("---\n\n")
    
    # Análise por vetorização
    lines.append("## 3. Análise por Vetorização\n\n")
    
    vecs = ['tfidf', 'fasttext', 'bert_base', 'bert_large', 'albertina_base', 'albertina_large']
    for vec in vecs:
        vec_results = [r for r in results_sorted if r['vectorization'] == vec]
        best_vec = vec_results[0]
        worst_vec = vec_results[-1]
        
        lines.append(f"### 3.{vecs.index(vec) + 1} {vec.upper()}\n\n")
        lines.append(f"**Melhor classificador:** {best_vec['classifier_name']} (F1={best_vec['f1']:.4f})\n\n")
        lines.append(f"**Pior classificador:** {worst_vec['classifier_name']} (F1={worst_vec['f1']:.4f})\n\n")
        
        lines.append("| Classificador | F1 | Precision | Recall | CV F1 |\n")
        lines.append("|---------------|-----|-----------|--------|-------|\n")
        for r in vec_results:
            lines.append(f"| {r['classifier_name']} | {r.get('f1', 0):.4f} | {r.get('precision', 0):.4f} | {r.get('recall', 0):.4f} | {r.get('cv_f1', 0):.4f} |\n")
        lines.append("\n")
    
    lines.append("---\n\n")
    
    # Conclusões
    lines.append("## 4. Conclusões e Recomendações\n\n")
    lines.append(f"### 4.1 Melhor Modelo\n\n")
    lines.append(f"**{best['vectorization'].upper()} + {best['classifier_name']}** será usado na Task 7 (avaliação final no test set).\n\n")
    lines.append(f"- **F1 no Dev:** {best['f1']:.4f}\n")
    lines.append(f"- **Precision:** {best['precision']:.4f}\n")
    lines.append(f"- **Recall:** {best['recall']:.4f}\n")
    lines.append(f"- **CV F1 (generalização estimada):** {best['cv_f1']:.4f}\n")
    lines.append(f"- **PR-AUC:** {best['pr_auc']:.4f}\n")
    lines.append(f"- **ROC-AUC:** {best['roc_auc']:.4f}\n")
    lines.append(f"- **Melhores hiperparâmetros:** {json.dumps(best.get('best_params', {}))}\n\n")
    
    lines.append("### 4.2 Observações Gerais\n\n")
    lines.append("1. **TF-IDF domina:** As 2 melhores combinações usam TF-IDF (SVM e RF)\n")
    lines.append("2. **SVM vs RF:** SVM ligeiramente superior ao RF em TF-IDF, mas RF mais robusto em embeddings\n")
    lines.append("3. **Naive Bayes fraco:** Sem `class_weight`, NB é catastroficamente inferior (melhor F1=0.592 vs 0.708 do SVM)\n")
    lines.append("4. **Embeddings sofrem com desbalanceamento:** BERT e Albertina têm F1 consistentemente menor que TF-IDF\n")
    lines.append("5. **Dataset desbalanceado (1,54% pos):** Modelos sem compensação de classe (NB) falham drasticamente\n\n")
    
    lines.append("### 4.3 Próximos Passos\n\n")
    lines.append("- **Task 7:** Avaliar melhor modelo no test set (11.454 exemplos, isolado)\n")
    lines.append("- **Análise de threshold:** Curva PR no dev para otimizar threshold (0.5 pode ser inadequado)\n")
    lines.append("- **ConvergenceWarning (SVM):** Se necessário, re-executar BERT-Large e Albertina-Large com `StandardScaler` ou `max_iter=50000`\n")
    lines.append("- **Oversampling (NB):** Testar SMOTE nas combinações densas se recall for crítico\n\n")
    
    lines.append("---\n\n")
    lines.append(f"**Gerado em:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
    
    return ''.join(lines)


def main():
    print("=" * 70)
    print("TASK 6 — CONSOLIDAÇÃO E ANÁLISE (18 combinações)")
    print("=" * 70)
    
    print("\nCarregando resultados...")
    results = load_all_results()
    print(f"  ✓ {len(results)} combinações carregadas")
    
    print("\nGerando CONSOLIDACAO_FINAL.md...")
    consolidacao_md = generate_consolidacao_md(results)
    
    out_path = os.path.join(RESULTS_DIR, "CONSOLIDACAO_FINAL.md")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(consolidacao_md)
    
    print(f"  ✓ Salvo em: {out_path}")
    
    # Identifica melhor modelo
    best = max(results, key=lambda r: r.get('f1', 0))
    print(f"\n{'=' * 70}")
    print(f"MELHOR MODELO: {best['vectorization'].upper()} + {best['classifier_name']}")
    print(f"F1 no Dev: {best['f1']:.4f}")
    print(f"{'=' * 70}")
    
    print("\nTASK 6 CONCLUÍDA")


if __name__ == "__main__":
    main()
