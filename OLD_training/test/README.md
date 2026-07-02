# Avaliacao Final no Conjunto de Teste

Esta pasta contem **tudo** que foi usado e gerado na avaliacao final do modelo TF-IDF + SVM
no conjunto de teste reservado. Nao e necessario procurar em outros diretorios do repositorio
para reproduzir ou entender os resultados.

---

## Modelo Avaliado

**TF-IDF + SVM (kernel linear)**

- Melhor entre 18 combinacoes testadas (NB, RF, SVM x TF-IDF, BERT, Albertina, FastText, Word2Vec)
- F1-Score no conjunto de desenvolvimento: **0.9783**
- F1-Score no conjunto de TESTE (limpo, 334 exemplos): **0.9711**
- Data leakage verificado: 19 duplicados removidos (5.38%), impacto negligível

---

## Estrutura de Pastas

```
test/
├── README.md                          <- este arquivo
│
├── results/                           <- outputs gerados pela avaliacao final
│   ├── FINAL_TEST_REPORT.md           <- relatorio comparando dev vs teste
│   ├── test_classification_report.txt <- precision/recall/f1 por classe
│   ├── test_confusion_matrix.png      <- matriz de confusao (imagem)
│   └── test_results_20260519_181436.pkl <- metricas salvas em pickle
│
├── dataset/                           <- dado de teste utilizado
│   ├── TEST_PREPROCESSED.csv          <- 353 exemplos pre-processados (original, com duplicados)
│   └── TEST_CLEAN_NO_DUPLICATES.csv   <- 334 exemplos limpos (19 duplicados removidos)
│                                         Origem: dataset/cleaned_data_no_bias/FOR_TRAINING/Pre_processed_for_Sparse/
│
├── model/                             <- artefatos do modelo final
│   ├── svm_tf_idf_20260519_174152.pkl           <- classificador SVM treinado
│   ├── tfidf_ngram12_mindf2_maxdf09_maxfeat10k_20260518_220950_vectorizer.pkl  <- vetorizador TF-IDF fitado no treino
│   └── vectorized_data/               <- dados de teste ja vetorizados (prontos para inferencia)
│       ├── test_tfidf_20260519_180924_matrix.npz    <- matriz TF-IDF (353 x 10000)
│       ├── test_tfidf_20260519_180924_labels.npy    <- labels ground-truth
│       └── test_tfidf_20260519_180924_metadata.pkl  <- metadados da vetorizacao
│
└── scripts/                           <- scripts usados na avaliacao
    ├── vectorize_test_dataset.py      <- vetoriza TEST_PREPROCESSED.csv usando o vetorizador do treino
    └── evaluate_best_model_on_test.py <- carrega modelo + dados vetorizados, gera metricas e relatorios
```

---

## Como Reproduzir a Avaliacao

Partindo do zero (sem usar os dados ja vetorizados):

```bash
# 1. Vetorizar o conjunto de teste
python test/scripts/vectorize_test_dataset.py

# 2. Avaliar o modelo (ajuste os caminhos em main() se necessario)
python test/scripts/evaluate_best_model_on_test.py
```

Ou, se quiser usar diretamente os dados ja vetorizados em `model/vectorized_data/`:
carregar a matriz `.npz` e o `.pkl` do modelo SVM e chamar `model.predict()`.

---

## Resultados Finais

**Teste Limpo (334 exemplos, sem duplicados):**

| Metrica   | Desenvolvimento | Teste (Limpo) | Diferenca |
|-----------|-----------------|---------------|-----------|
| Accuracy  | 0.9787          | 0.9701        | -0.88%    |
| Precision | 1.0000          | 0.9825        | -1.75%    |
| Recall    | 0.9574          | 0.9600        | +0.27%    |
| F1-Score  | 0.9783          | 0.9711        | -0.74%    |
| ROC-AUC   | 0.9923          | 0.9921        | -0.02%    |

**Verificacao de Data Leakage (Junho 2026):**
- Teste inicial: 353 exemplos
- Duplicados removidos: 19 (5.38%)
- Teste final: 334 exemplos
- Impacto no F1-Score: -0.0002 (negligível)

**Conclusao:** Diferenca < 1% entre dev e teste — modelo generaliza excelentemente.
As altas métricas refletem genuína capacidade de generalização, não memorização.

---

## Origem dos Arquivos (no repositorio original)

| Arquivo em `test/`                        | Origem no repositorio                                              |
|-------------------------------------------|--------------------------------------------------------------------|
| `model/svm_tf_idf_*.pkl`                  | `training/results/svm_vectorizations/`                            |
| `model/tfidf_*_vectorizer.pkl`            | `vectorization/tf_idf/`                                           |
| `model/vectorized_data/`                  | `vectorization/tf_idf/test/`                                      |
| `dataset/TEST_PREPROCESSED.csv`           | `dataset/cleaned_data_no_bias/FOR_TRAINING/Pre_processed_for_Sparse/` |
| `scripts/vectorize_test_dataset.py`       | `scripts/`                                                        |
| `scripts/evaluate_best_model_on_test.py`  | `training/`                                                       |
