# Relatório do Experimento: Naive Bayes
## 1. Objetivo
Avaliar o impacto de diferentes técnicas de vetorização na performance do classificador Naive Bayes para detecção de fraudes em notícias. Dataset desbalanceado (1,53% positivos).
## 2. Metodologia
### 2.1 Divisão dos Dados
- **Treino:** 36.652 exemplos (562 positivos, 36.090 negativos)
- **Desenvolvimento:** 9.163 exemplos (141 positivos, 9.022 negativos)
- **Teste:** 11.454 exemplos (isolado, não usado nesta fase)
### 2.2 Classificador
Naive Bayes é um classificador probabilístico baseado no teorema de Bayes.
Assume independência entre features (forte assunção, raramente verdadeira).
- MultinomialNB: para TF-IDF (valores não-negativos)
- GaussianNB: para embeddings densos (BERT, Albertina, FastText)
**Sem ajuste de classe** (class_prior=None) — baseline fraco esperado em dataset desbalanceado.
Pendência: testar oversampling (SMOTE) se recall ficar muito baixo.
**Configuração:**
```python
MultinomialNB(alpha=grid)  # TF-IDF
GaussianNB(var_smoothing=grid)  # dense
Grid: alpha=[0.1, 0.5, 1.0] / var_smoothing=[1e-9, 1e-7, 1e-5]
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
| tfidf | 0.6209 | 0.5919 | 0.5278 | 0.6738 | 0.6294 | 0.9836 | 0.9857 |
| fasttext | 0.1370 | 0.1357 | 0.0734 | 0.8936 | 0.0835 | 0.8939 | 0.8248 |
| bert_base | 0.2914 | 0.2921 | 0.1751 | 0.8794 | 0.1909 | 0.9527 | 0.9344 |
| bert_large | 0.1017 | 0.0954 | 0.0511 | 0.7163 | 0.0486 | 0.8125 | 0.7910 |
| albertina_base | 0.1136 | 0.0829 | 0.0541 | 0.1773 | 0.0229 | 0.5641 | 0.9396 |
| albertina_large | 0.1982 | 0.1891 | 0.1126 | 0.5887 | 0.1071 | 0.8880 | 0.9223 |

**Melhor Vetorização: tfidf (F1: 0.5919)**
### 3.2 Melhores Hiperparâmetros por Vetorização
| Vetorização | Best Params |
|-------------|-------------|
| tfidf | {"alpha": 0.5} |
| fasttext | {"var_smoothing": 1e-09} |
| bert_base | {"var_smoothing": 1e-09} |
| bert_large | {"var_smoothing": 1e-09} |
| albertina_base | {"var_smoothing": 1e-05} |
| albertina_large | {"var_smoothing": 1e-09} |

## 4. Análise
- **Melhor combinação:** tfidf + Naive Bayes
- **F1 no dev:** 0.5919
- **Precision:** 0.5278
- **Recall:** 0.6738
- **CV F1 (estimativa de generalização):** 0.6209
