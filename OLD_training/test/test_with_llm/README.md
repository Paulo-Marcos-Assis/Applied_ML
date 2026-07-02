# Comparacao: TF-IDF + SVM vs LLM

Avalia o modelo ML treinado e uma LLM no **mesmo conjunto de teste reservado** (353 exemplos),
gerando metricas comparativas, matrizes de confusao e um CSV com todas as predicoes.

---

## Script

### `test_llm_vs_ml.py`

Novo script criado para esta comparacao. Nao e uma copia dos scripts de `processors/` —
eles operam sobre arquivos JSON de noticias scraped e classificam em 4 categorias.
Este script foi adaptado para:
- Entrada: `TEST_PREPROCESSED.csv` com ground truth (`label`)
- Tarefa: classificacao **binaria** (1 = fraude, 0 = nao-fraude)
- Comparacao direta entre ML e LLM nas mesmas metricas

Infraestrutura Ollama reutilizada de `processors/classify_news_triple.py`:
conexao via `langchain_ollama`, timeout via `signal.SIGALRM`, parse JSON robusto
(remove blocos `<think>` do qwen3 e delimitadores de codigo).

---

## Como Usar

```bash
# Execucao completa (chama a LLM para todos os 353 exemplos)
python test/test_with_llm/test_llm_vs_ml.py

# Retomar de checkpoint (caso o processo tenha sido interrompido)
python test/test_with_llm/test_llm_vs_ml.py --resume

# Apenas gerar relatorio (requer checkpoint existente)
python test/test_with_llm/test_llm_vs_ml.py --report-only
```

### Variaveis de ambiente

| Variavel       | Padrao                     | Descricao                    |
|----------------|----------------------------|------------------------------|
| `OLLAMA_HOST`  | `http://localhost:11434`   | URL do servidor Ollama       |
| `OLLAMA_MODEL` | `qwen3:8b`                 | Modelo a usar                |

Exemplos:
```bash
# Usar servidor remoto com modelo diferente
OLLAMA_HOST=https://ollama.ceos.ufsc.br OLLAMA_MODEL=gpt-oss:20b python test_llm_vs_ml.py

# Usar outro modelo local
OLLAMA_MODEL=llama3.1:8b python test_llm_vs_ml.py
```

---

## Entradas

| Arquivo | Descricao |
|---------|-----------|
| `../dataset/TEST_PREPROCESSED.csv` | 353 exemplos com `title_clean`, `text_clean`, `label` |
| `../model/svm_tf_idf_*.pkl` | Classificador SVM treinado |
| `../model/tfidf_*_vectorizer.pkl` | Vetorizador TF-IDF fitado no treino |

> **Nota sobre o texto:** O CSV contem texto pre-processado para vetorizacao esparsa
> (stopwords removidas). Esse e o mesmo texto usado para avaliar o ML, garantindo
> comparacao justa — mas pode penalizar ligeiramente a LLM em relacao ao uso de texto bruto.

---

## Saidas Geradas

| Arquivo | Descricao |
|---------|-----------|
| `llm_predictions_checkpoint.json` | Checkpoint com predicoes brutas da LLM (permite retomar) |
| `llm_predictions.csv` | Predicoes de ambos os modelos + ground truth por exemplo |
| `comparison_confusion_matrices.png` | Matrizes de confusao lado a lado |
| `comparison_report.md` | Relatorio completo com metricas, classification reports e analise |

---

## Referencia: Desempenho do Modelo ML no Teste

| Metrica   | Valor  |
|-----------|--------|
| Accuracy  | 0.9717 |
| Precision | 0.9826 |
| Recall    | 0.9602 |
| F1-Score  | 0.9713 |
| ROC-AUC   | 0.9922 |

Ver relatorio completo em `../results/FINAL_TEST_REPORT.md`.
