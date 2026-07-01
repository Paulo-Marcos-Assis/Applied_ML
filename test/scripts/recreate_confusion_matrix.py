"""
Recria a matriz de confusão com título atualizado
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Dados da matriz de confusão
cm = np.array([[156, 3],
               [7, 168]])

# Criar figura
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['Non-Fraud', 'Fraud'],
           yticklabels=['Non-Fraud', 'Fraud'])
plt.title('Confusion Matrix - Test Set')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

# Salvar
output_path = Path("/home/paulo/CascadeProjects/Applied_ML/test/results/confusion_matrix_clean_test.png")
plt.savefig(output_path, dpi=300)
print(f"Matriz de confusão salva em: {output_path}")
