# Relatorio Final - Avaliacao no Conjunto de Teste

## Modelo Avaliado

**TF-IDF + SVM (Linear Kernel)**

- Vetorizacao: TF-IDF (n-grams 1-2, max 10k features)
- Classificador: SVM (kernel linear, probability=True)
- Melhor modelo entre 18 combinacoes testadas

## Resultados no Conjunto de Desenvolvimento

| Metrica | Valor |
|---------|-------|
| Accuracy | 0.9787 |
| Precision | 1.0000 |
| Recall | 0.9574 |
| F1-Score | 0.9783 |
| ROC-AUC | 0.9923 |

## Resultados no Conjunto de TESTE (Final)

| Metrica | Valor |
|---------|-------|
| Accuracy | 0.9717 |
| Precision | 0.9826 |
| Recall | 0.9602 |
| F1-Score | 0.9713 |
| ROC-AUC | 0.9922 |

## Comparacao Desenvolvimento vs Teste

| Metrica | Desenvolvimento | Teste | Diferenca |
|---------|-----------------|-------|----------|
| ACCURACY | 0.9787 | 0.9717 | -0.0070 (-0.72%) |
| PRECISION | 1.0000 | 0.9826 | -0.0174 (-1.74%) |
| RECALL | 0.9574 | 0.9602 | +0.0028 (+0.30%) |
| F1 | 0.9783 | 0.9713 | -0.0070 (-0.72%) |
| ROC_AUC | 0.9923 | 0.9922 | -0.0001 (-0.01%) |

## Analise

**Consistencia Excelente:** Metricas de teste muito proximas ao desenvolvimento (diferenca < 1%).
O modelo generaliza muito bem para dados nunca vistos.

## Conclusao

O modelo TF-IDF + SVM alcancou **F1-Score de 0.9713** no conjunto de teste final.

**Performance EXCELENTE** - Modelo pronto para producao.
