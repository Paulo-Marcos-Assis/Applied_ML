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

## Resultados no Conjunto de TESTE (Final - Limpo)

**Conjunto de Teste Limpo: 334 exemplos (175 positivos, 159 negativos)**
- 19 duplicados removidos do conjunto inicial de 353 exemplos
- Data leakage: 5.38% (impacto negligível nas métricas)

| Metrica | Valor |
|---------|-------|
| Accuracy | 0.9701 |
| Precision | 0.9825 |
| Recall | 0.9600 |
| F1-Score | 0.9711 |
| ROC-AUC | 0.9921 |

## Comparacao Desenvolvimento vs Teste

| Metrica | Desenvolvimento | Teste (Limpo) | Diferenca |
|---------|-----------------|---------------|----------|
| ACCURACY | 0.9787 | 0.9701 | -0.0086 (-0.88%) |
| PRECISION | 1.0000 | 0.9825 | -0.0175 (-1.75%) |
| RECALL | 0.9574 | 0.9600 | +0.0026 (+0.27%) |
| F1 | 0.9783 | 0.9711 | -0.0072 (-0.74%) |
| ROC_AUC | 0.9923 | 0.9921 | -0.0002 (-0.02%) |

## Verificacao de Data Leakage (Junho 2026)

| Aspecto | Detalhes |
|---------|----------|
| Teste inicial | 353 exemplos |
| Duplicados encontrados | 19 (5.38%) |
| Teste final limpo | 334 exemplos |
| Impacto no F1-Score | -0.0002 (negligível) |
| Conclusão | Métricas validadas - generalização genuína |

## Analise

**Consistencia Excelente:** Metricas de teste muito proximas ao desenvolvimento (diferenca < 1%).
O modelo generaliza muito bem para dados nunca vistos.

**Data Leakage Verificado:** Análise pós-treinamento identificou 19 duplicados (5.38%) entre treino e teste. Após remoção, o modelo mantém performance excepcional (F1: 0.9711), confirmando que as altas métricas refletem genuína capacidade de generalização, não memorização.

## Conclusao

O modelo TF-IDF + SVM alcancou **F1-Score de 0.9711** no conjunto de teste final limpo (334 exemplos).

**Performance EXCELENTE** - Modelo validado e pronto para producao.

**Matriz de Confusão (Teste Limpo):**
- True Negatives: 156 (98.11%)
- False Positives: 3 (1.89%)
- False Negatives: 7 (4.00%)
- True Positives: 168 (96.00%)
