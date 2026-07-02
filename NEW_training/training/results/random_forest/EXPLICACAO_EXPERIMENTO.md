# Relatório do Experimento: Random Forest
## 1. Objetivo
Avaliar o impacto de diferentes técnicas de vetorização na performance do classificador Random Forest para detecção de fraudes em notícias. Dataset desbalanceado (1,53% positivos).
## 2. Metodologia
### 2.1 Divisão dos Dados
- **Treino:** 36.652 exemplos (562 positivos, 36.090 negativos)
- **Desenvolvimento:** 9.163 exemplos (141 positivos, 9.022 negativos)
- **Teste:** 11.454 exemplos (isolado, não usado nesta fase)
### 2.2 Classificador
Random Forest é um ensemble de árvores de decisão.
Reduz overfitting combinando múltiplas árvores com amostragem aleatória.
class_weight='balanced': ajusta pesos inversamente proporcionais à frequência das classes.
predict_proba fornece probabilidades para PR-AUC e ROC-AUC.
**Configuração:**
```python
RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
Grid: n_estimators=[100, 200] x max_depth=[20, 50, None]
CV: StratifiedKFold(5, shuffle=True, random_state=42)
Scoring: f1
```
### 2.3 Protocolo
1. GridSearchCV com stratified 5-fold no treino (F1 como scoring)
2. Retreina com melhores hiperparâmetros no treino inteiro
3. Avalia no dev set
4. Teste permanece isolado (Task 7)
## 3. Resultados
### 3.1 Tabela Comparativa (Dev Set)
| Vetorização | CV F1 | F1 | Precision | Recall | PR-AUC | ROC-AUC | Accuracy |
|-------------|-------|----|-----------|--------|--------|---------|----------|
| tfidf | 0.6851 | 0.6741 | 0.7054 | 0.6454 | 0.7251 | 0.9875 | 0.9904 |
| fasttext | 0.5650 | 0.5329 | 0.4611 | 0.6312 | 0.5441 | 0.9812 | 0.9830 |
| bert_base | 0.6195 | 0.6061 | 0.6504 | 0.5674 | 0.6475 | 0.9847 | 0.9887 |
| bert_large | 0.5571 | 0.4883 | 0.7222 | 0.3688 | 0.5713 | 0.9700 | 0.9881 |
| albertina_base | 0.5194 | 0.4854 | 0.7692 | 0.3546 | 0.5575 | 0.9426 | 0.9884 |
| albertina_large | 0.4077 | 0.3596 | 0.8649 | 0.2270 | 0.5824 | 0.9572 | 0.9876 |

**Melhor Vetorização: tfidf (F1: 0.6741)**
### 3.2 Melhores Hiperparâmetros por Vetorização
| Vetorização | Best Params |
|-------------|-------------|
| tfidf | {"max_depth": 20, "n_estimators": 100} |
| fasttext | {"max_depth": 20, "n_estimators": 100} |
| bert_base | {"max_depth": 20, "n_estimators": 200} |
| bert_large | {"max_depth": 20, "n_estimators": 100} |
| albertina_base | {"max_depth": 20, "n_estimators": 200} |
| albertina_large | {"max_depth": 20, "n_estimators": 200} |

## 4. Análise
- **Melhor combinação:** tfidf + Random Forest
- **F1 no dev:** 0.6741
- **Precision:** 0.7054
- **Recall:** 0.6454
- **CV F1 (estimativa de generalização):** 0.6851
