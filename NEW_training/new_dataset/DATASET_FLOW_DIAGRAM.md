# Dataset Flow Diagram — NEW_training Pipeline

## Visão Geral

Pipeline completo de preparação de dados para detecção de fraudes em notícias, desde consolidação até splits estratificados para treinamento.

---

## Diagrama de Fluxo

```mermaid
flowchart TD
    %% ===== FASE 1: CONSOLIDAÇÃO =====
    subgraph CONSOLIDACAO["TASK 1 — CONSOLIDAÇÃO (Task 1)"]
        A1["Dataset Positivos<br/>cleaned_data_no_bias/<br/>POSITIVE_DF_COMPANIES_REDUZIDO.csv<br/>881 exemplos (100% fraudes)"]
        A2["Dataset Negativos<br/>new_dataset/<br/>negatives_SAFE_CLEAN.csv<br/>55.595 exemplos (100% não-fraudes)"]
        A3["Consolidação<br/>task1_consolidate.py<br/>Merge + shuffle + dedup por URL"]
        A4["CONSOLIDATED_IMBALANCED.csv<br/>57.271 exemplos<br/>881 pos (1,54%) + 56.390 neg (98,46%)"]
        
        A1 --> A3
        A2 --> A3
        A3 --> A4
    end

    %% ===== FASE 2: SPLIT 1 (TREINO+DEV vs TEST) =====
    subgraph SPLIT1["TASK 2 — SPLIT ESTRATIFICADO (80/20)"]
        B1["CONSOLIDATED_IMBALANCED.csv<br/>57.271 exemplos<br/>881 pos (1,54%) / 56.390 neg (98,46%)"]
        B2["Split 1: 80/20<br/>Stratified by label<br/>random_state=42"]
        B3["TREINO+DEV<br/>45.815 exemplos (80%)<br/>703 pos (1,53%) / 45.112 neg (98,47%)"]
        B4["TEST (ISOLADO)<br/>11.454 exemplos (20%)<br/>176 pos (1,54%) / 11.278 neg (98,46%)"]
        
        B1 --> B2
        B2 --> B3
        B2 --> B4
    end

    %% ===== FASE 3: SPLIT 2 (TREINO vs DEV) =====
    subgraph SPLIT2["SPLIT 2 — TREINO vs DEV (80/20)"]
        C1["TREINO+DEV<br/>45.815 exemplos<br/>703 pos / 45.112 neg"]
        C2["Split 2: 80/20<br/>Stratified by label<br/>random_state=42"]
        C3["TREINO FINAL<br/>36.652 exemplos (80%)<br/>562 pos (1,53%) / 36.090 neg (98,47%)"]
        C4["DEV (VALIDAÇÃO)<br/>9.163 exemplos (20%)<br/>141 pos (1,54%) / 9.022 neg (98,46%)"]
        
        C1 --> C2
        C2 --> C3
        C2 --> C4
    end

    %% ===== FASE 4: PRÉ-PROCESSAMENTO =====
    subgraph PREPROC["TASK 3 — PRÉ-PROCESSAMENTO DUAL"]
        D1["TREINO: 36.652<br/>DEV: 9.163<br/>TEST: 11.454"]
        D2["Pipeline Pesado<br/>Tokenização + Lemmatização<br/>Stopwords + Normalização"]
        D3["Pipeline Leve<br/>Apenas normalização básica<br/>Preserva estrutura original"]
        D4["Pre_processed_for_Sparse/<br/>train_preprocessed.csv<br/>dev_preprocessed.csv<br/>test_preprocessed.csv"]
        D5["Pre_processed_for_Embeddings/<br/>train_bert.csv<br/>dev_bert.csv<br/>test_bert.csv"]
        
        D1 --> D2
        D1 --> D3
        D2 --> D4
        D3 --> D5
    end

    %% ===== FASE 5: VETORIZAÇÃO =====
    subgraph VETORIZATION["TASK 4 — VETORIZAÇÃO (6 técnicas)"]
        E1["Sparse (TF-IDF)<br/>Pre_processed_for_Sparse/"]
        E2["Dense (Embeddings)<br/>Pre_processed_for_Embeddings/"]
        E3["TF-IDF<br/>train: 36.652 × 10.000<br/>dev: 9.163 × 10.000<br/>test: 11.454 × 10.000"]
        E4["FastText<br/>train: 36.652 × 300<br/>dev: 9.163 × 300<br/>test: 11.454 × 300"]
        E5["BERT-Base<br/>train: 36.652 × 768<br/>dev: 9.163 × 768<br/>test: 11.454 × 768"]
        E6["BERT-Large<br/>train: 36.652 × 1024<br/>dev: 9.163 × 1024<br/>test: 11.454 × 1024"]
        E7["Albertina-Base<br/>train: 36.652 × 768<br/>dev: 9.163 × 768<br/>test: 11.454 × 768"]
        E8["Albertina-Large<br/>train: 36.652 × 1536<br/>dev: 9.163 × 1536<br/>test: 11.454 × 1536"]
        
        E1 --> E3
        E2 --> E4
        E2 --> E5
        E2 --> E6
        E2 --> E7
        E2 --> E8
    end

    %% ===== FASE 6: TREINAMENTO =====
    subgraph TRAINING["TASK 5 — TREINAMENTO (18 combinações)"]
        F1["6 Vetorizações<br/>× 3 Classificadores<br/>= 18 Combinações"]
        F2["Naive Bayes<br/>6 combinações<br/>Sem class_weight"]
        F3["SVM LinearSVC<br/>6 combinações<br/>class_weight=balanced"]
        F4["Random Forest<br/>6 combinações<br/>class_weight=balanced"]
        F5["GridSearchCV 5-fold<br/>no TREINO (36.652)<br/>Avaliação no DEV (9.163)"]
        F6["TEST (11.454)<br/>PERMANECE ISOLADO<br/>Não usado no treinamento"]
        
        F1 --> F2
        F1 --> F3
        F1 --> F4
        F2 --> F5
        F3 --> F5
        F4 --> F5
        F5 -.-> F6
    end

    %% ===== CONEXÕES ENTRE FASES =====
    A4 --> B1
    B3 --> C1
    B4 --> D1
    C3 --> D1
    C4 --> D1
    D4 --> E1
    D5 --> E2
    E3 --> F1
    E4 --> F1
    E5 --> F1
    E6 --> F1
    E7 --> F1
    E8 --> F1

    %% ===== ESTILO =====
    classDef consolidacao fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
    classDef split fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef preproc fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef vetorization fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    classDef training fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef isolado fill:#ffebee,stroke:#d32f2f,stroke-width:3px,stroke-dasharray: 5 5

    class A1,A2,A3,A4 consolidacao
    class B1,B2,B3,B4,C1,C2,C3,C4 split
    class D1,D2,D3,D4,D5 preproc
    class E1,E2,E3,E4,E5,E6,E7,E8 vetorization
    class F1,F2,F3,F4,F5 training
    class B4,F6 isolado
```

---

## Resumo Quantitativo

### Distribuição Final dos Dados

| Split | Total | Positivos | % Pos | Negativos | % Neg | Uso |
|-------|-------|-----------|-------|-----------|-------|-----|
| **TREINO** | 36.652 | 562 | 1,53% | 36.090 | 98,47% | GridSearchCV 5-fold + retreino |
| **DEV** | 9.163 | 141 | 1,54% | 9.022 | 98,46% | Avaliação de modelos (Task 5) |
| **TEST** | 11.454 | 176 | 1,54% | 11.278 | 98,46% | **ISOLADO** — Task 7 apenas |
| **TOTAL** | 57.269 | 879 | 1,54% | 56.390 | 98,46% | - |

**Nota:** 2 exemplos perdidos na deduplicação entre splits (57.271 → 57.269)

### Proporções dos Splits

```
CONSOLIDATED (57.271)
    ├── TREINO+DEV (45.815) ——— 80%
    │   ├── TREINO (36.652) ——— 80% do TREINO+DEV
    │   └── DEV (9.163) ———————— 20% do TREINO+DEV
    └── TEST (11.454) ——————————— 20% (ISOLADO)
```

### Vetorizações Geradas

| Técnica | Tipo | Dimensões | Train | Dev | Test | Status |
|---------|------|-----------|-------|-----|------|--------|
| TF-IDF | Sparse | 10.000 | 36.652 | 9.163 | 11.454 | ✅ |
| FastText | Dense | 300 | 36.652 | 9.163 | 11.454 | ✅ |
| BERT-Base | Dense | 768 | 36.652 | 9.163 | 11.454 | ✅ |
| BERT-Large | Dense | 1.024 | 36.652 | 9.163 | 11.454 | ✅ |
| Albertina-Base | Dense | 768 | 36.652 | 9.163 | 11.454 | ✅ |
| Albertina-Large | Dense | 1.536 | 36.652 | 9.163 | 11.454 | ✅ |

### Combinações Treinadas (Task 5)

| Classificador | Vetorizações | class_weight | Total Combinações |
|---------------|--------------|--------------|-------------------|
| Naive Bayes | 6 | ❌ Sem ajuste | 6 |
| SVM LinearSVC | 6 | ✅ balanced | 6 |
| Random Forest | 6 | ✅ balanced | 6 |
| **TOTAL** | - | - | **18** |

**Melhor modelo (Task 6):** TF-IDF + SVM (F1=0.7075 no Dev)

---

## Estrutura de Diretórios

```
NEW_training/
├── FOR_TRAINING/                                       (Dados de treinamento)
│   ├── CONSOLIDATED_IMBALANCED.csv                     (57.271 — origem)
│   ├── DISTRIBUTION_REPORT.txt                         (relatório de distribuição)
│   ├── train.csv                                       (36.652 — treino)
│   ├── dev.csv                                         (9.163 — validação)
│   ├── Pre_processed_for_Sparse/
│   │   ├── train_preprocessed.csv                      (pesado — TF-IDF/FastText)
│   │   └── dev_preprocessed.csv
│   └── Pre_processed_for_Embeddings/
│       ├── train_bert.csv                              (leve — BERT/Albertina)
│       └── dev_bert.csv
│
├── FOR_TEST/                                           (Dados de teste — ISOLADO)
│   ├── test.csv                                        (11.454 — teste)
│   ├── Pre_processed_for_Sparse/
│   │   └── test_preprocessed.csv
│   └── Pre_processed_for_Embeddings/
│       └── test_bert.csv
│
├── vectorization/                                      (Vetorizações prontas)
│   ├── tfidf/
│   │   ├── train_sparse.npz                            (36.652 × 10.000)
│   │   ├── dev_sparse.npz                              (9.163 × 10.000)
│   │   ├── test_sparse.npz                             (11.454 × 10.000)
│   │   └── labels_{train,dev,test}.npy
│   ├── fasttext/
│   │   ├── train_embeddings.npy                        (36.652 × 300)
│   │   ├── dev_embeddings.npy                          (9.163 × 300)
│   │   ├── test_embeddings.npy                         (11.454 × 300)
│   │   └── labels_{train,dev,test}.npy
│   ├── bert_base/                                      (36.652/9.163/11.454 × 768)
│   ├── bert_large/                                     (36.652/9.163/11.454 × 1.024)
│   ├── albertina_base/                                 (36.652/9.163/11.454 × 768)
│   └── albertina_large/                                (36.652/9.163/11.454 × 1.536)
│
└── training/
    └── results/                                        (Resultados de treinamento)
        ├── tfidf/
        │   ├── naive_bayes/                            (model.pkl, best_params.json, report, cm)
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
        ├── naive_bayes/                                (consolidado por classificador)
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
        ├── random_forest/
        │   ├── EXPLICACAO_EXPERIMENTO.md
        │   ├── random_forest_comparison.csv
        │   ├── random_forest_comparison.png
        │   ├── random_forest_execution_log.txt
        │   └── random_forest_results.json
        └── CONSOLIDACAO_FINAL.md                       (Task 6 — análise dos 18 resultados)
```

---

## Características do Dataset

### Desbalanceamento

- **Classe positiva (fraudes):** 1,54% (879 de 57.269)
- **Classe negativa (não-fraudes):** 98,46% (56.390 de 57.269)
- **Razão:** ~1:64 (1 fraude para cada 64 não-fraudes)

### Estratificação

- ✅ Proporção mantida entre splits (1,53-1,54% em todos)
- ✅ Sem vazamento (deduplicação por URL entre splits)
- ✅ Shuffle com random_state=42 (reprodutível)

### Isolamento do Test Set

- ❌ **NÃO usado** em Task 1-5 (consolidação, split, pré-processamento, vetorização, treinamento)
- ❌ **NÃO usado** em GridSearchCV ou tuning de hiperparâmetros
- ❌ **NÃO usado** em seleção de modelos (Task 6)
- ✅ **SERÁ usado** apenas na Task 7 (avaliação final do melhor modelo)

---

**Gerado em:** 02/07/2026
