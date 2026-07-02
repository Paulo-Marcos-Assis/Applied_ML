# Pipeline Desbalanceado — Cronologia de Execução

## Estrutura de Diretórios (Dados)

```
Applied_ML/
├── dataset/
│   └── cleaned_data_no_bias/
│       └── POSITIVE_DF_COMPANIES_REDUZIDO.csv          (881 exemplos, 100% positivos)
│
└── NEW_training/
    ├── NEGATIVES_CONSOLIDATED.csv                      (56.390 exemplos, 100% negativos)
    ├── new_dataset/
    │   └── negatives_SAFE_CLEAN.csv                    (55.595 exemplos, 100% negativos)
    │
    ├── FOR_TRAINING/
    │   ├── CONSOLIDATED_IMBALANCED.csv                 (57.271 exemplos — 1,54% pos / 98,46% neg)
    │   ├── DISTRIBUTION_REPORT.txt                     (relatório de distribuição)
    │   ├── train.csv                                   (36.652 exemplos — 1,53% pos / 98,47% neg)
    │   ├── dev.csv                                     (9.163 exemplos — 1,54% pos / 98,46% neg)
    │   ├── Pre_processed_for_Sparse/
    │   │   ├── train_preprocessed.csv                  (pesado — para TF-IDF/FastText)
    │   │   └── dev_preprocessed.csv                    (pesado — para TF-IDF/FastText)
    │   └── Pre_processed_for_Embeddings/
    │       ├── train_bert.csv                          (leve — para BERT/Albertina)
    │       └── dev_bert.csv                            (leve — para BERT/Albertina)
    │
    ├── FOR_TEST/
    │   ├── test.csv                                    (11.454 exemplos — 1,54% pos / 98,46% neg)
    │   ├── Pre_processed_for_Sparse/
    │   │   └── test_preprocessed.csv                   (pesado — para TF-IDF/FastText)
    │   └── Pre_processed_for_Embeddings/
    │       └── test_bert.csv                           (leve — para BERT/Albertina)
    │
    ├── scripts/
    │   ├── task1_consolidate.py                        (Task 1 — consolidação)
    │   ├── task2_split.py                              (Task 2 — split estratificado)
    │   ├── task3_preprocess.py                         (Task 3 — pré-processamento dual)
    │   ├── task4a_tfidf_fasttext.py                    (Task 4a — TF-IDF + FastText)
    │   ├── task4b_bert.py                                (Task 4b — BERT-Base + BERT-Large)
    │   ├── task4c_albertina.py                         (Task 4c — Albertina-Base)
    │   ├── task4d_albertina_large.py                   (Task 4d — Albertina-Large)
    │   ├── task5a_naive_bayes.py                       (Task 5a — Naive Bayes)
    │   ├── task5b_svm.py                               (Task 5b — SVM LinearSVC)
    │   └── task5c_random_forest.py                     (Task 5c — Random Forest)
    │
    ├── vectorization/
    │   ├── tfidf/                                       (matrizes esparsas + vectorizer.pkl)
    │   ├── fasttext/                                    (embeddings densos 300d)
    │   ├── bert_base/                                   (embeddings densos 768d)
    │   ├── bert_large/                                  (embeddings densos 1024d)
    │   ├── albertina_base/                              (embeddings densos 768d)
    │   └── albertina_large/                             (embeddings densos 1536d)
    │
    ├── training/
    │   └── results/                                     (18 combinações: 6 vetorizações × 3 classificadores)
    │
    └── new_chronology/
        └── PROGRESS.md                                 (este arquivo)
```

---

## Task 1 — Padronização e Consolidação

**Script:** `NEW_training/scripts/task1_consolidate.py`

**O que fez:**
- Carregou 881 positivos de `dataset/cleaned_data_no_bias/POSITIVE_DF_COMPANIES_REDUZIDO.csv` → selecionou `url, title, text`, adicionou `label=1`, extraiu `portal` da URL
- Carregou 56.390 negativos de `NEW_training/NEGATIVES_CONSOLIDATED.csv` → padronizou `label` para `0` (unificando `0` e `negative`)
- Consolidou em schema: `url, title, text, label, portal, date_publication`
- Gerou relatório de distribuição

**Resultado:**
- Total: 57.271 exemplos
- Positivos: 881 (1,54%)
- Negativos: 56.390 (98,46%)
- Razão de desbalanceamento: 1:64

**Arquivos gerados:**
- `NEW_training/FOR_TRAINING/CONSOLIDATED_IMBALANCED.csv`
- `NEW_training/FOR_TRAINING/DISTRIBUTION_REPORT.txt`

**Árvore da Task 1:**
```
NEW_training/
├── scripts/
│   └── task1_consolidate.py                            (script de consolidação)
└── FOR_TRAINING/
    ├── CONSOLIDATED_IMBALANCED.csv                     (57.271 exemplos — 1,54% pos / 98,46% neg)
    └── DISTRIBUTION_REPORT.txt                         (relatório de distribuição)
```

---

## Task 2 — Split Estratificado 80/20

**Script:** `NEW_training/scripts/task2_split.py`

**O que fez:**
- Stratify apenas por `label` (sem portal, para evitar "portal = fraude")
- Split 1: 80% treino+dev / 20% teste
- Split 2: do treino+dev, 80% treino / 20% dev
- Deduplicação por URL entre splits (0 overlaps)
- Teste isolado em `FOR_TEST/`

**Resultado:**

| Split | Total | Positivos | % Pos | Negativos |
|-------|-------|-----------|-------|-----------|
| Train | 36.652 | 562 | 1,53% | 36.090 |
| Dev | 9.163 | 141 | 1,54% | 9.022 |
| Test | 11.454 | 176 | 1,54% | 11.278 |

**Arquivos gerados:**
- `NEW_training/FOR_TRAINING/train.csv`
- `NEW_training/FOR_TRAINING/dev.csv`
- `NEW_training/FOR_TEST/test.csv`

**Árvore da Task 2:**
```
NEW_training/
├── scripts/
│   └── task2_split.py                                  (script de split estratificado)
├── FOR_TRAINING/
│   ├── train.csv                                       (36.652 exemplos — 1,53% pos / 98,47% neg)
│   └── dev.csv                                         (9.163 exemplos — 1,54% pos / 98,46% neg)
└── FOR_TEST/
    └── test.csv                                        (11.454 exemplos — 1,54% pos / 98,46% neg)
```

---

## Task 3 — Pré-processamento Dual

**Script:** `NEW_training/scripts/task3_preprocess.py`

**Por que dois tipos de pré-processamento?**

**PESADO (TF-IDF, FastText):** Modelos clássicos baseados em Bag-of-Words e embeddings estáticos funcionam melhor com texto normalizado. Remover acentos, pontuação, stopwords e converter para lowercase reduz a dimensionalidade do vocabulário e evita que variações superficiais da mesma palavra (ex: "Fraude" vs "fraude" vs "FRAUDE") sejam tratadas como tokens distintos. Isso é especialmente importante para TF-IDF, que conta frequências de termos.

**LEVE (BERT, Albertina):** Modelos contextuais pré-treinados (BERTimbau, Albertina) foram treinados **com** acentos, maiúsculas e pontuação. Esses modelos usam tokenizadores WordPiece/BPE que aprendem subpalavras considerando acentuação e capitalização. Remover esses elementos degrada a representação — o modelo perde informação linguística real (ex: diferença entre "Banco" instituição e "banco" assento). Preservar o texto original maximiza o aproveitamento do pré-treinamento.

**Referência:** `chronology/04_Vectorization.md` — Seção "Questão Fundamental: Pré-processamento vs Modelos Pré-treinados"

**O que fez:**
- **Pesado:** removeu HTML, URLs, pontuação, acentos, stopwords; lowercase; concatenou `title + text` → `processed_text`
- **Leve:** removeu apenas URLs e normalizou espaços; preservou acentos, maiúsculas, pontuação; concatenou `title + text` → `processed_text`
- Aplicado a train, dev, e test **separadamente**
- Vetorização (fit/transform) fica para Task 4 — vectorizer fitado **apenas no treino**

**⚠️ Correção de Data Leakage vs pipeline anterior:**
No pipeline anterior, o TF-IDF foi `fit_transform` em todos os 1.410 exemplos antes do split treino/dev. Isso significa que o vocabulário e pesos IDF "viram" o dev, inflando potencialmente as métricas. No novo pipeline, o split é feito antes (Task 2) e o vectorizer será fitado apenas no treino (Task 4). Ver nota detalhada em `chronology/04_Vectorization.md`.

**Arquivos gerados:**
- `NEW_training/FOR_TRAINING/Pre_processed_for_Sparse/train_preprocessed.csv`
- `NEW_training/FOR_TRAINING/Pre_processed_for_Sparse/dev_preprocessed.csv`
- `NEW_training/FOR_TRAINING/Pre_processed_for_Embeddings/train_bert.csv`
- `NEW_training/FOR_TRAINING/Pre_processed_for_Embeddings/dev_bert.csv`
- `NEW_training/FOR_TEST/Pre_processed_for_Sparse/test_preprocessed.csv`
- `NEW_training/FOR_TEST/Pre_processed_for_Embeddings/test_bert.csv`

**Árvore da Task 3:**
```
NEW_training/
├── scripts/
│   └── task3_preprocess.py                              (script de pré-processamento dual)
├── FOR_TRAINING/
│   ├── Pre_processed_for_Sparse/
│   │   ├── train_preprocessed.csv                       (36.652 — pesado, para TF-IDF/FastText)
│   │   └── dev_preprocessed.csv                         (9.163 — pesado, para TF-IDF/FastText)
│   └── Pre_processed_for_Embeddings/
│       ├── train_bert.csv                               (36.652 — leve, para BERT/Albertina)
│       └── dev_bert.csv                                 (9.163 — leve, para BERT/Albertina)
└── FOR_TEST/
    ├── Pre_processed_for_Sparse/
    │   └── test_preprocessed.csv                        (11.454 — pesado, para TF-IDF/FastText)
    └── Pre_processed_for_Embeddings/
        └── test_bert.csv                                (11.454 — leve, para BERT/Albertina)
```

---

## Task 4 — Vetorização (6 técnicas)

**Scripts:**
- `NEW_training/scripts/task4a_tfidf_fasttext.py` (TF-IDF + FastText)
- `NEW_training/scripts/task4b_bert.py` (BERT-Base + BERT-Large)
- `NEW_training/scripts/task4c_albertina.py` (Albertina-Base — separado por erro de dtype BFloat16 no DeBERTa)
- `NEW_training/scripts/task4d_albertina_large.py` (Albertina-Large — modelo correto: `PORTULAN/albertina-900m-portuguese-ptbr-encoder-brwac`)

**Prevenção de Data Leakage:**
- TF-IDF: `fit_transform` **apenas no treino**; dev e test recebem apenas `transform()`
- FastText: pesos TF-IDF derivados do vectorizer fitado no treino
- BERT e Albertina: modelos pré-treinados, não fitam nos dados (apenas extração de embeddings)

**Parâmetros TF-IDF:**
- `ngram_range=(1,2)`, `min_df=2`, `max_df=0.9`, `max_features=10000`, `norm='l2'`
- `min_df=2` mantido para consistência com pipeline anterior

**Modelos utilizados:**

| Técnica | Modelo | Dim | Tipo |
|---------|--------|-----|------|
| TF-IDF | sklearn TfidfVectorizer | 10.000 | Esparsa |
| FastText | cc.pt.300.vec (TF-IDF weighted) | 300 | Densa |
| BERT-Base | neuralmind/bert-base-portuguese-cased | 768 | Densa |
| BERT-Large | neuralmind/bert-large-portuguese-cased | 1024 | Densa |
| Albertina-Base | PORTULAN/albertina-ptbr-base | 768 | Densa |
| Albertina-Large | PORTULAN/albertina-900m-portuguese-ptbr-encoder-brwac | 1536 | Densa |

**Resultado — Shapes e Distribuição:**

| Técnica | Train | Dev | Test | Dim |
|---------|-------|-----|------|-----|
| TF-IDF | 36.652 | 9.163 | 11.454 | 10.000 |
| FastText | 36.652 | 9.163 | 11.454 | 300 |
| BERT-Base | 36.652 | 9.163 | 11.454 | 768 |
| BERT-Large | 36.652 | 9.163 | 11.454 | 1024 |
| Albertina-Base | 36.652 | 9.163 | 11.454 | 768 |
| Albertina-Large | 36.652 | 9.163 | 11.454 | 1536 |

Labels em todos os splits: train (pos=562, neg=36.090) | dev (pos=141, neg=9.022) | test (pos=176, neg=11.278)

**Arquivos gerados por técnica:**
- TF-IDF: `train_sparse.npz`, `dev_sparse.npz`, `test_sparse.npz`, `vectorizer.pkl`, `labels_{train,dev,test}.npy`
- Demais: `train_embeddings.npy`, `dev_embeddings.npy`, `test_embeddings.npy`, `labels_{train,dev,test}.npy`

**Árvore da Task 4:**
```
NEW_training/
├── scripts/
│   ├── task4a_tfidf_fasttext.py                        (TF-IDF fit no treino + FastText TF-IDF weighted)
│   ├── task4b_bert.py                                  (BERT-Base + BERT-Large)
│   ├── task4c_albertina.py                             (Albertina-Base — float32 fix)
│   └── task4d_albertina_large.py                       (Albertina-Large — modelo 900m)
└── vectorization/
    ├── tfidf/
    │   ├── train_sparse.npz                             (36.652 × 10.000 — esparsa)
    │   ├── dev_sparse.npz                               (9.163 × 10.000 — esparsa)
    │   ├── test_sparse.npz                              (11.454 × 10.000 — esparsa)
    │   ├── vectorizer.pkl                               (TfidfVectorizer fitado no treino)
    │   ├── labels_train.npy                             (562 pos / 36.090 neg)
    │   ├── labels_dev.npy                               (141 pos / 9.022 neg)
    │   └── labels_test.npy                              (176 pos / 11.278 neg)
    ├── fasttext/
    │   ├── train_embeddings.npy                         (36.652 × 300 — densa)
    │   ├── dev_embeddings.npy                           (9.163 × 300 — densa)
    │   ├── test_embeddings.npy                          (11.454 × 300 — densa)
    │   └── labels_{train,dev,test}.npy
    ├── bert_base/
    │   ├── train_embeddings.npy                         (36.652 × 768 — densa)
    │   ├── dev_embeddings.npy                           (9.163 × 768 — densa)
    │   ├── test_embeddings.npy                          (11.454 × 768 — densa)
    │   └── labels_{train,dev,test}.npy
    ├── bert_large/
    │   ├── train_embeddings.npy                         (36.652 × 1024 — densa)
    │   ├── dev_embeddings.npy                           (9.163 × 1024 — densa)
    │   ├── test_embeddings.npy                          (11.454 × 1024 — densa)
    │   └── labels_{train,dev,test}.npy
    ├── albertina_base/
    │   ├── train_embeddings.npy                         (36.652 × 768 — densa)
    │   ├── dev_embeddings.npy                           (9.163 × 768 — densa)
    │   ├── test_embeddings.npy                          (11.454 × 768 — densa)
    │   └── labels_{train,dev,test}.npy
    └── albertina_large/
        ├── train_embeddings.npy                         (36.652 × 1536 — densa)
        ├── dev_embeddings.npy                           (9.163 × 1536 — densa)
        ├── test_embeddings.npy                          (11.454 × 1536 — densa)
        └── labels_{train,dev,test}.npy
```

---

## Task 5 — Treinamento (18 combinações: 6 vetorizações × 3 classificadores)

**Scripts:**
- `NEW_training/scripts/task5a_naive_bayes.py` (Naive Bayes — 6 combinações)
- `NEW_training/scripts/task5b_svm.py` (SVM LinearSVC — 6 combinações)
- `NEW_training/scripts/task5c_random_forest.py` (Random Forest — 6 combinações)

**Diferença vs pipeline anterior:**
- Pipeline anterior: **sem tuning** (requisito acadêmico, default params)
- Novo pipeline: **Nível 2 — tuning leve** com GridSearchCV (curated grid) + stratified 5-fold no treino
- Dataset 30x maior (36.652 vs 1.128) sustenta tuning sem overfit grave

**Classificadores e configurações:**

| Classificador | Variante | `class_weight` | Grid | Combinações |
|---------------|----------|----------------|------|-------------|
| Naive Bayes | MNB (TF-IDF) + GNB (dense) | ❌ Sem ajuste (baseline fraco) | `alpha=[0.1, 0.5, 1.0]` / `var_smoothing=[1e-9, 1e-7, 1e-5]` | 6 |
| SVM | LinearSVC | `balanced` | `C=[0.1, 1, 10]` | 6 |
| Random Forest | RandomForestClassifier | `balanced` | `n_estimators=[100, 200]` × `max_depth=[20, 50, None]` | 6 |

**MNB vs GNB — divisão técnica (não escolha):**
- MultinomialNB: apenas TF-IDF (exige valores não-negativos)
- GaussianNB: apenas embeddings densos (MNB quebra com valores negativos de BERT/Albertina)

**Naive Bayes sem ajuste de classe:**
- Opção C (baseline fraco) — sem `class_prior` nem oversampling
- Esperado: recall baixo (modelo tenderá a prever negativo)
- Pendência: testar oversampling (SMOTE) se recall ficar muito baixo

**Métricas por combinação:**

| Métrica | Papel |
|---------|-------|
| **F1-Score** | Primária (seleção) |
| **Precision** | Secundária (custo de falso positivo) |
| **Recall** | Secundária (custo de falso negativo) |
| **PR-AUC** | Secundária (robustez em desbalanceamento) |
| **ROC-AUC** | Reportada (comparação com pipeline anterior) |
| **Accuracy** | Reportada (referência apenas) |

**Protocolo de avaliação:**
```
Para cada uma das 18 combinações:
  1. GridSearchCV com stratified 5-fold no TREINO (36.652)
     → encontra melhores hiperparâmetros (F1 como scoring)
     → reporta CV mean ± std
  2. Retreina com melhores hiperparâmetros no TREINO inteiro
  3. Avalia no DEV (9.163) — métricas finais
  4. Salva: model.pkl, best_params.json, classification_report.txt, confusion_matrix.png
  5. TEST (11.454) permanece isolado — só na Task 7
```

**Outputs por combinação:**
- `best_params.json` — melhores hiperparâmetros do GridSearchCV
- `model.pkl` — modelo treinado com melhores hiperparâmetros
- `classification_report.txt` — relatório completo no dev set
- `confusion_matrix.png` — matriz de confusão no dev set

**Arquivos consolidados:**
- `naive_bayes_results.json` — 6 combinações NB
- `svm_results.json` — 6 combinações SVM
- `random_forest_results.json` — 6 combinações RF

**Resultados Consolidados (Dev Set — 9.163 exemplos):**

| Rank | Vetorização | Classificador | F1 | Precision | Recall | CV F1 | PR-AUC | ROC-AUC |
|------|-------------|---------------|-----|-----------|--------|-------|--------|---------|
| 1 | TF-IDF | SVM | **0.708** | 0.680 | 0.738 | 0.721 | 0.767 | 0.984 |
| 2 | TF-IDF | RF | 0.674 | 0.705 | 0.645 | 0.685 | 0.725 | 0.987 |
| 3 | Albertina-Large | SVM | 0.619 | 0.533 | 0.738 | 0.631 | 0.733 | 0.974 |
| 4 | BERT-Base | RF | 0.606 | 0.650 | 0.567 | 0.620 | 0.647 | 0.985 |
| 5 | BERT-Base | SVM | 0.599 | 0.505 | 0.738 | 0.609 | 0.665 | 0.979 |
| 6 | TF-IDF | NB | 0.592 | 0.528 | 0.674 | 0.621 | 0.629 | 0.984 |
| 7 | BERT-Large | SVM | 0.583 | 0.558 | 0.610 | 0.649 | 0.670 | 0.969 |
| 8 | FastText | RF | 0.533 | 0.461 | 0.631 | 0.565 | 0.544 | 0.981 |
| 9 | BERT-Large | RF | 0.488 | 0.722 | 0.369 | 0.557 | 0.571 | 0.970 |
| 10 | Albertina-Base | RF | 0.485 | 0.769 | 0.355 | 0.519 | 0.558 | 0.943 |
| 11 | Albertina-Base | SVM | 0.436 | 0.300 | 0.801 | 0.459 | 0.697 | 0.964 |
| 12 | Albertina-Large | RF | 0.360 | 0.865 | 0.227 | 0.408 | 0.582 | 0.957 |
| 13 | FastText | SVM | 0.311 | 0.188 | 0.915 | 0.323 | 0.566 | 0.982 |
| 14 | BERT-Base | NB | 0.292 | 0.175 | 0.879 | 0.291 | 0.191 | 0.953 |
| 15 | Albertina-Large | NB | 0.189 | 0.113 | 0.589 | 0.198 | 0.107 | 0.888 |
| 16 | FastText | NB | 0.136 | 0.073 | 0.894 | 0.137 | 0.084 | 0.894 |
| 17 | BERT-Large | NB | 0.095 | 0.051 | 0.716 | 0.102 | 0.049 | 0.813 |
| 18 | Albertina-Base | NB | 0.083 | 0.054 | 0.177 | 0.114 | 0.023 | 0.564 |

**Observações:**
- **TF-IDF + SVM** lidera (F1=0.708) — melhor combinação
- SVM e RF superiores ao NB em todas as vetorizações
- NB sem `class_weight` catastroficamente fraco (melhor F1=0.592)
- Embeddings densos sofrem mais com desbalanceamento que TF-IDF
- ConvergenceWarning no SVM: BERT-Large e Albertina-Large (não convergiu em 10K iter, mas impacto <3% no F1)

**Árvore da Task 5 (estrutura final):**
```
NEW_training/
├── scripts/
│   ├── task5a_naive_bayes.py                           (MNB + GNB, sem ajuste de classe)
│   ├── task5b_svm.py                                   (LinearSVC, class_weight=balanced)
│   ├── task5c_random_forest.py                         (RandomForest, class_weight=balanced)
│   └── task5_utils.py                                  (funções compartilhadas de relatório)
└── training/
    └── results/
        ├── tfidf/
        │   ├── naive_bayes/                             (best_params.json, model.pkl, report, cm.png)
        │   ├── svm/
        │   └── random_forest/
        ├── fasttext/
        │   ├── naive_bayes/
        │   ├── svm/
        │   └── random_forest/
        ├── bert_base/
        │   ├── naive_bayes/
        │   ├── svm/
        │   └── random_forest/
        ├── bert_large/
        │   ├── naive_bayes/
        │   ├── svm/
        │   └── random_forest/
        ├── albertina_base/
        │   ├── naive_bayes/
        │   ├── svm/
        │   └── random_forest/
        ├── albertina_large/
        │   ├── naive_bayes/
        │   ├── svm/
        │   └── random_forest/
        ├── naive_bayes/                                 (consolidado por classificador)
        │   ├── EXPLICACAO_EXPERIMENTO.md
        │   ├── naive_bayes_comparison.csv
        │   ├── naive_bayes_comparison.png
        │   ├── naive_bayes_execution_log.txt
        │   └── naive_bayes_results.json
        ├── svm/
        │   ├── EXPLICACAO_EXPERIMENTO.md
        │   ├── svm_comparison.csv
        │   ├── svm_comparison.png
        │   ├── svm_execution_log.txt
        │   └── svm_results.json
        └── random_forest/
            ├── EXPLICACAO_EXPERIMENTO.md
            ├── random_forest_comparison.csv
            ├── random_forest_comparison.png
            ├── random_forest_execution_log.txt
            └── random_forest_results.json
```

**Status:** ✅ CONCLUÍDA (02/07/2026)

---

## Pontos Finais a Resolver (pendentes)

1. **Threshold ótimo** — Analisar curva PR no Dev para ajustar threshold (0,5 pode ser inadequado para desbalanceamento)
2. **Near-duplicatas entre splits** — Verificar similaridade de texto entre treino/teste (notícias republicadas em portais diferentes)
3. **min_df no TF-IDF** — Mantido `min_df=2` para consistência com pipeline anterior. Se após Task 5 o recall estiver baixo, testar `min_df=1` como experimento para preservar termos raros de fraude em dataset desbalanceado
4. **Oversampling para Naive Bayes** — NB executado sem ajuste de classe (baseline fraco). Se recall ficar muito baixo, testar oversampling (SMOTE) nas 5 combinações densas (GNB) como alternativa ao `class_prior`
