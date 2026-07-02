#!/usr/bin/env python3
"""
Funções compartilhadas para Task 5 — geração de relatórios consolidados.
"""

import os
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save_comparison_csv(all_results, out_path):
    cols = ['vectorization', 'cv_f1', 'f1', 'precision', 'recall',
            'pr_auc', 'roc_auc', 'accuracy']
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        writer.writeheader()
        for r in all_results:
            row = {k: r.get(k, '') for k in cols}
            for k in cols:
                if isinstance(row[k], float):
                    row[k] = f"{row[k]:.4f}"
            writer.writerow(row)


def save_comparison_png(all_results, out_path, classifier_name):
    vecs = [r['vectorization'] for r in all_results]
    f1_vals = [r.get('f1', 0) for r in all_results]
    prec_vals = [r.get('precision', 0) for r in all_results]
    rec_vals = [r.get('recall', 0) for r in all_results]

    x = np.arange(len(vecs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, f1_vals, width, label='F1', color='#2196F3')
    ax.bar(x, prec_vals, width, label='Precision', color='#4CAF50')
    ax.bar(x + width, rec_vals, width, label='Recall', color='#FF9800')

    ax.set_ylabel('Score')
    ax.set_title(f'{classifier_name} — Comparação entre Vetorizações (Dev Set)')
    ax.set_xticks(x)
    ax.set_xticklabels(vecs, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)

    for i, v in enumerate(f1_vals):
        ax.text(i - width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def save_explicacao_md(all_results, classifier_name, classifier_desc,
                       config_str, out_path):
    best = max(all_results, key=lambda r: r.get('f1', 0))

    lines = []
    lines.append(f"# Relatório do Experimento: {classifier_name}\n")
    lines.append(f"## 1. Objetivo\n")
    lines.append(f"Avaliar o impacto de diferentes técnicas de vetorização na performance do classificador {classifier_name} para detecção de fraudes em notícias. Dataset desbalanceado (1,53% positivos).\n")
    lines.append(f"## 2. Metodologia\n")
    lines.append(f"### 2.1 Divisão dos Dados\n")
    lines.append(f"- **Treino:** 36.652 exemplos (562 positivos, 36.090 negativos)\n")
    lines.append(f"- **Desenvolvimento:** 9.163 exemplos (141 positivos, 9.022 negativos)\n")
    lines.append(f"- **Teste:** 11.454 exemplos (isolado, não usado nesta fase)\n")
    lines.append(f"### 2.2 Classificador\n")
    lines.append(f"{classifier_desc}\n")
    lines.append(f"**Configuração:**\n")
    lines.append(f"```python\n{config_str}\n```\n")
    lines.append(f"### 2.3 Protocolo\n")
    lines.append(f"1. GridSearchCV com stratified 5-fold no treino (F1 como scoring)\n")
    lines.append(f"2. Retreina com melhores hiperparâmetros no treino inteiro\n")
    lines.append(f"3. Avalia no dev set\n")
    lines.append(f"4. Teste permanece isolado (Task 7)\n")
    lines.append(f"## 3. Resultados\n")
    lines.append(f"### 3.1 Tabela Comparativa (Dev Set)\n")
    lines.append(f"| Vetorização | CV F1 | F1 | Precision | Recall | PR-AUC | ROC-AUC | Accuracy |\n")
    lines.append(f"|-------------|-------|----|-----------|--------|--------|---------|----------|\n")
    for r in all_results:
        lines.append(
            f"| {r['vectorization']} | {r.get('cv_f1', 0):.4f} | "
            f"{r.get('f1', 0):.4f} | {r.get('precision', 0):.4f} | "
            f"{r.get('recall', 0):.4f} | {r.get('pr_auc', 0):.4f} | "
            f"{r.get('roc_auc', 0):.4f} | {r.get('accuracy', 0):.4f} |\n"
        )
    lines.append(f"\n**Melhor Vetorização: {best['vectorization']} (F1: {best.get('f1', 0):.4f})**\n")
    lines.append(f"### 3.2 Melhores Hiperparâmetros por Vetorização\n")
    lines.append(f"| Vetorização | Best Params |\n")
    lines.append(f"|-------------|-------------|\n")
    for r in all_results:
        params_str = json.dumps(r.get('best_params', {}))
        lines.append(f"| {r['vectorization']} | {params_str} |\n")
    lines.append(f"\n## 4. Análise\n")
    lines.append(f"- **Melhor combinação:** {best['vectorization']} + {classifier_name}\n")
    lines.append(f"- **F1 no dev:** {best.get('f1', 0):.4f}\n")
    lines.append(f"- **Precision:** {best.get('precision', 0):.4f}\n")
    lines.append(f"- **Recall:** {best.get('recall', 0):.4f}\n")
    lines.append(f"- **CV F1 (estimativa de generalização):** {best.get('cv_f1', 0):.4f}\n")

    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def generate_consolidated_reports(all_results, results_dir, classifier_name,
                                  classifier_desc, config_str,
                                  consolidated_json_name):
    classifier_slug = classifier_name.lower().replace(' ', '_')
    classifier_dir = os.path.join(results_dir, classifier_slug)
    os.makedirs(classifier_dir, exist_ok=True)

    consolidated_path = os.path.join(classifier_dir, consolidated_json_name)
    with open(consolidated_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    save_comparison_csv(
        all_results,
        os.path.join(classifier_dir, f"{classifier_slug}_comparison.csv")
    )
    save_comparison_png(
        all_results,
        os.path.join(classifier_dir, f"{classifier_slug}_comparison.png"),
        classifier_name
    )
    save_explicacao_md(
        all_results, classifier_name, classifier_desc, config_str,
        os.path.join(classifier_dir, "EXPLICACAO_EXPERIMENTO.md")
    )
