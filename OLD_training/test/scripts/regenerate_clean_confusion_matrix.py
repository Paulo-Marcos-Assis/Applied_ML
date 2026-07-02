#!/usr/bin/env python3
"""
Regenerar matriz de confusão do test set limpo com rótulos padronizados
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Matriz de confusão do test limpo (334 exemplos)
# Valores extraídos do FINAL_TEST_REPORT.md
cm_clean = [
    [159, 0],   # True Negative, False Positive
    [7, 168]    # False Negative, True Positive
]

# Criar figura com rótulos padronizados
plt.figure(figsize=(8, 6))
sns.heatmap(cm_clean, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['Negative', 'Positive'],
           yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - Clean Test Set (No Duplicates)', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()

# Salvar
output_path = Path("/home/paulo/CascadeProjects/Applied_ML/test/results/confusion_matrix_clean_test.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Matriz de confusão regenerada: {output_path}")
print(f"  - Rótulos padronizados: ['Negative', 'Positive']")
print(f"  - Total: 334 exemplos (159 TN, 0 FP, 7 FN, 168 TP)")
