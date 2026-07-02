# Consolidação Final — 18 Combinações (Vetorização × Classificador)

## Resumo Executivo

Este documento consolida os resultados de **18 combinações** de vetorizações e classificadores para detecção de fraudes em notícias, avaliadas no conjunto de desenvolvimento (9.163 exemplos, 1,54% positivos).

**Melhor modelo identificado:** TFIDF + SVM (F1-Score: 0.7075)

---

## 1. Resultados Completos — Todas as 18 Combinações

### 1.1 Tabela Geral Ordenada por F1-Score (Dev Set)

| Rank | Vetorização | Classificador | F1 | Precision | Recall | CV F1 | PR-AUC | ROC-AUC | Accuracy |
|------|-------------|---------------|-----|-----------|--------|-------|--------|---------|----------|
| 1 | tfidf | SVM | **0.7075** | 0.6797 | 0.7376 | 0.7213 | 0.7665 | 0.9843 | 0.9906 |
| 2 | tfidf | RF | 0.6741 | 0.7054 | 0.6454 | 0.6851 | 0.7251 | 0.9875 | 0.9904 |
| 3 | albertina_large | SVM | 0.6190 | 0.5333 | 0.7376 | 0.6309 | 0.7328 | 0.9741 | 0.9860 |
| 4 | bert_base | RF | 0.6061 | 0.6504 | 0.5674 | 0.6195 | 0.6475 | 0.9847 | 0.9887 |
| 5 | bert_base | SVM | 0.5994 | 0.5049 | 0.7376 | 0.6090 | 0.6654 | 0.9794 | 0.9848 |
| 6 | tfidf | NB | 0.5919 | 0.5278 | 0.6738 | 0.6209 | 0.6294 | 0.9836 | 0.9857 |
| 7 | bert_large | SVM | 0.5831 | 0.5584 | 0.6099 | 0.6492 | 0.6700 | 0.9693 | 0.9866 |
| 8 | fasttext | RF | 0.5329 | 0.4611 | 0.6312 | 0.5650 | 0.5441 | 0.9812 | 0.9830 |
| 9 | bert_large | RF | 0.4883 | 0.7222 | 0.3688 | 0.5571 | 0.5713 | 0.9700 | 0.9881 |
| 10 | albertina_base | RF | 0.4854 | 0.7692 | 0.3546 | 0.5194 | 0.5575 | 0.9426 | 0.9884 |
| 11 | albertina_base | SVM | 0.4363 | 0.2997 | 0.8014 | 0.4593 | 0.6965 | 0.9641 | 0.9681 |
| 12 | albertina_large | RF | 0.3596 | 0.8649 | 0.2270 | 0.4077 | 0.5824 | 0.9572 | 0.9876 |
| 13 | fasttext | SVM | 0.3112 | 0.1875 | 0.9149 | 0.3234 | 0.5662 | 0.9823 | 0.9377 |
| 14 | bert_base | NB | 0.2921 | 0.1751 | 0.8794 | 0.2914 | 0.1909 | 0.9527 | 0.9344 |
| 15 | albertina_large | NB | 0.1891 | 0.1126 | 0.5887 | 0.1982 | 0.1071 | 0.8880 | 0.9223 |
| 16 | fasttext | NB | 0.1357 | 0.0734 | 0.8936 | 0.1370 | 0.0835 | 0.8939 | 0.8248 |
| 17 | bert_large | NB | 0.0954 | 0.0511 | 0.7163 | 0.1017 | 0.0486 | 0.8125 | 0.7910 |
| 18 | albertina_base | NB | 0.0829 | 0.0541 | 0.1773 | 0.1136 | 0.0229 | 0.5641 | 0.9396 |

### 1.2 Top 5 Modelos

1. **TFIDF + SVM: 0.7075**
   - Precision: 0.6797
   - Recall: 0.7376
   - PR-AUC: 0.7665

2. **TFIDF + Random Forest: 0.6741**
   - Precision: 0.7054
   - Recall: 0.6454
   - PR-AUC: 0.7251

3. **ALBERTINA_LARGE + SVM: 0.6190**
   - Precision: 0.5333
   - Recall: 0.7376
   - PR-AUC: 0.7328

4. **BERT_BASE + Random Forest: 0.6061**
   - Precision: 0.6504
   - Recall: 0.5674
   - PR-AUC: 0.6475

5. **BERT_BASE + SVM: 0.5994**
   - Precision: 0.5049
   - Recall: 0.7376
   - PR-AUC: 0.6654

---

## 2. Análise por Classificador

### 2.1 Naive Bayes

**Melhor:** tfidf (F1=0.5919)

**Pior:** albertina_base (F1=0.0829)

**Média F1:** 0.2312

| Vetorização | F1 | Precision | Recall | PR-AUC |
|-------------|-----|-----------|--------|--------|
| tfidf | 0.5919 | 0.5278 | 0.6738 | 0.6294 |
| bert_base | 0.2921 | 0.1751 | 0.8794 | 0.1909 |
| albertina_large | 0.1891 | 0.1126 | 0.5887 | 0.1071 |
| fasttext | 0.1357 | 0.0734 | 0.8936 | 0.0835 |
| bert_large | 0.0954 | 0.0511 | 0.7163 | 0.0486 |
| albertina_base | 0.0829 | 0.0541 | 0.1773 | 0.0229 |

### 2.2 SVM

**Melhor:** tfidf (F1=0.7075)

**Pior:** fasttext (F1=0.3112)

**Média F1:** 0.5428

| Vetorização | F1 | Precision | Recall | PR-AUC |
|-------------|-----|-----------|--------|--------|
| tfidf | 0.7075 | 0.6797 | 0.7376 | 0.7665 |
| albertina_large | 0.6190 | 0.5333 | 0.7376 | 0.7328 |
| bert_base | 0.5994 | 0.5049 | 0.7376 | 0.6654 |
| bert_large | 0.5831 | 0.5584 | 0.6099 | 0.6700 |
| albertina_base | 0.4363 | 0.2997 | 0.8014 | 0.6965 |
| fasttext | 0.3112 | 0.1875 | 0.9149 | 0.5662 |

### 2.3 Random Forest

**Melhor:** tfidf (F1=0.6741)

**Pior:** albertina_large (F1=0.3596)

**Média F1:** 0.5244

| Vetorização | F1 | Precision | Recall | PR-AUC |
|-------------|-----|-----------|--------|--------|
| tfidf | 0.6741 | 0.7054 | 0.6454 | 0.7251 |
| bert_base | 0.6061 | 0.6504 | 0.5674 | 0.6475 |
| fasttext | 0.5329 | 0.4611 | 0.6312 | 0.5441 |
| bert_large | 0.4883 | 0.7222 | 0.3688 | 0.5713 |
| albertina_base | 0.4854 | 0.7692 | 0.3546 | 0.5575 |
| albertina_large | 0.3596 | 0.8649 | 0.2270 | 0.5824 |

---

## 3. Análise por Vetorização

### 3.1 TFIDF

**Melhor classificador:** SVM (F1=0.7075)

**Pior classificador:** Naive Bayes (F1=0.5919)

| Classificador | F1 | Precision | Recall | CV F1 |
|---------------|-----|-----------|--------|-------|
| SVM | 0.7075 | 0.6797 | 0.7376 | 0.7213 |
| Random Forest | 0.6741 | 0.7054 | 0.6454 | 0.6851 |
| Naive Bayes | 0.5919 | 0.5278 | 0.6738 | 0.6209 |

### 3.2 FASTTEXT

**Melhor classificador:** Random Forest (F1=0.5329)

**Pior classificador:** Naive Bayes (F1=0.1357)

| Classificador | F1 | Precision | Recall | CV F1 |
|---------------|-----|-----------|--------|-------|
| Random Forest | 0.5329 | 0.4611 | 0.6312 | 0.5650 |
| SVM | 0.3112 | 0.1875 | 0.9149 | 0.3234 |
| Naive Bayes | 0.1357 | 0.0734 | 0.8936 | 0.1370 |

### 3.3 BERT_BASE

**Melhor classificador:** Random Forest (F1=0.6061)

**Pior classificador:** Naive Bayes (F1=0.2921)

| Classificador | F1 | Precision | Recall | CV F1 |
|---------------|-----|-----------|--------|-------|
| Random Forest | 0.6061 | 0.6504 | 0.5674 | 0.6195 |
| SVM | 0.5994 | 0.5049 | 0.7376 | 0.6090 |
| Naive Bayes | 0.2921 | 0.1751 | 0.8794 | 0.2914 |

### 3.4 BERT_LARGE

**Melhor classificador:** SVM (F1=0.5831)

**Pior classificador:** Naive Bayes (F1=0.0954)

| Classificador | F1 | Precision | Recall | CV F1 |
|---------------|-----|-----------|--------|-------|
| SVM | 0.5831 | 0.5584 | 0.6099 | 0.6492 |
| Random Forest | 0.4883 | 0.7222 | 0.3688 | 0.5571 |
| Naive Bayes | 0.0954 | 0.0511 | 0.7163 | 0.1017 |

### 3.5 ALBERTINA_BASE

**Melhor classificador:** Random Forest (F1=0.4854)

**Pior classificador:** Naive Bayes (F1=0.0829)

| Classificador | F1 | Precision | Recall | CV F1 |
|---------------|-----|-----------|--------|-------|
| Random Forest | 0.4854 | 0.7692 | 0.3546 | 0.5194 |
| SVM | 0.4363 | 0.2997 | 0.8014 | 0.4593 |
| Naive Bayes | 0.0829 | 0.0541 | 0.1773 | 0.1136 |

### 3.6 ALBERTINA_LARGE

**Melhor classificador:** SVM (F1=0.6190)

**Pior classificador:** Naive Bayes (F1=0.1891)

| Classificador | F1 | Precision | Recall | CV F1 |
|---------------|-----|-----------|--------|-------|
| SVM | 0.6190 | 0.5333 | 0.7376 | 0.6309 |
| Random Forest | 0.3596 | 0.8649 | 0.2270 | 0.4077 |
| Naive Bayes | 0.1891 | 0.1126 | 0.5887 | 0.1982 |

---

## 4. Conclusões e Recomendações

### 4.1 Melhor Modelo

**TFIDF + SVM** será usado na Task 7 (avaliação final no test set).

- **F1 no Dev:** 0.7075
- **Precision:** 0.6797
- **Recall:** 0.7376
- **CV F1 (generalização estimada):** 0.7213
- **PR-AUC:** 0.7665
- **ROC-AUC:** 0.9843
- **Melhores hiperparâmetros:** {"C": 1}

### 4.2 Observações Gerais

1. **TF-IDF domina:** As 2 melhores combinações usam TF-IDF (SVM e RF)
2. **SVM vs RF:** SVM ligeiramente superior ao RF em TF-IDF, mas RF mais robusto em embeddings
3. **Naive Bayes fraco:** Sem `class_weight`, NB é catastroficamente inferior (melhor F1=0.592 vs 0.708 do SVM)
4. **Embeddings sofrem com desbalanceamento:** BERT e Albertina têm F1 consistentemente menor que TF-IDF
5. **Dataset desbalanceado (1,54% pos):** Modelos sem compensação de classe (NB) falham drasticamente

### 4.3 Próximos Passos

- **Task 7:** Avaliar melhor modelo no test set (11.454 exemplos, isolado)
- **Análise de threshold:** Curva PR no dev para otimizar threshold (0.5 pode ser inadequado)
- **ConvergenceWarning (SVM):** Se necessário, re-executar BERT-Large e Albertina-Large com `StandardScaler` ou `max_iter=50000`
- **Oversampling (NB):** Testar SMOTE nas combinações densas se recall for crítico

---

**Gerado em:** 02/07/2026 14:21:58
