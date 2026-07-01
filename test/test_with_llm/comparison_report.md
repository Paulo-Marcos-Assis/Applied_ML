# Comparacao: TF-IDF + SVM vs LLM

**Data:** 08/06/2026 18:02  
**Dataset:** `TEST_PREPROCESSED.csv` — 353 exemplos (mesmo conjunto da avaliacao do ML)  
**Modelo LLM:** `qwen2.5:7b`  
**Servidor Ollama:** `http://localhost:11434`  

> **Nota:** O texto enviado a LLM e o texto pre-processado para vetorizacao esparsa
> (stopwords removidas). Isso pode penalizar a LLM vs uso de texto bruto.

---

## Metricas Comparativas

| Metrica    | TF-IDF + SVM | LLM (qwen2.5-7b) |
|------------|-------------|-----------------|
| Accuracy   |       0.9717 |       0.7559 |
| Precision  |       0.9826 |       0.9894 |
| Recall     |       0.9602 |       0.5314 |
| F1         |       0.9713 |       0.6914 |
| ROC-AUC    | 0.9922 | N/A (sem probabilidades) |

## Detalhes da Avaliacao LLM

- Exemplos avaliados: 340 / 353
- Erros / timeouts: 13
- Tempo total de inferencia: 12493.9s (208.2 min)
- Tempo medio por exemplo: 36.7s

## Classification Report — TF-IDF + SVM

```
              precision    recall  f1-score   support

    Negative     0.9613    0.9831    0.9721       177
    Positive     0.9826    0.9602    0.9713       176

    accuracy                         0.9717       353
   macro avg     0.9719    0.9716    0.9717       353
weighted avg     0.9719    0.9717    0.9717       353

```

## Classification Report — LLM (qwen2.5-7b)

```
              precision    recall  f1-score   support

    Negative     0.6667    0.9939    0.7981       165
    Positive     0.9894    0.5314    0.6914       175

    accuracy                         0.7559       340
   macro avg     0.8280    0.7627    0.7448       340
weighted avg     0.8328    0.7559    0.7432       340

```

## Analise

O modelo ML **superou** a LLM em F1-Score: **0.9713** vs 0.6914 (delta = -0.2799 (28.8% de diferenca)).

## Extracoes da LLM (noticias classificadas como fraude)

- Noticias classificadas como fraude pela LLM: **94**
- Com empresas identificadas: 83
- Com tipos de fraude identificados: 94
- Ver detalhes: `llm_fraud_extractions.csv`

## Arquivos Gerados

- `comparison_confusion_matrices.png` — matrizes de confusao lado a lado
- `llm_predictions.csv` — todas as predicoes (LLM + ML + ground truth + extracoes)
- `llm_fraud_extractions.csv` — apenas fraudes detectadas pela LLM com extracoes
- `llm_predictions_checkpoint.json` — checkpoint com predicoes brutas da LLM
