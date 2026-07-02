# Consolidacao Final: Comparacao de 18 Combinacoes

## Resumo Executivo

Este documento consolida os resultados de **18 combinacoes** de vetorizacoes e classificadores para deteccao de fraudes em noticias, testadas no conjunto de desenvolvimento (282 exemplos).

**Melhor modelo identificado: TF-IDF + SVM (F1-Score: 0.9783)**

## 1. Resultados Completos - Todas as 18 Combinacoes

### 1.1 Tabela Geral Ordenada por F1-Score

| Rank | Vetorizacao | Classificador | F1-Score | Accuracy | Precision | Recall | ROC-AUC |
|------|-------------|---------------|----------|----------|-----------|--------|---------|
| 1 | TF-IDF | SVM | **0.9783** | 0.9787 | 1.0000 | 0.9574 | 0.9923 |
| 2 | TF-IDF | Naive Bayes | 0.9565 | 0.9574 | 0.9778 | 0.9362 | 0.9809 |
| 3 | TF-IDF | Random Forest | 0.9537 | 0.9539 | 0.9571 | 0.9504 | 0.9901 |
| 4 | BERT-Large | SVM | 0.9416 | 0.9433 | 0.9699 | 0.9149 | 0.9874 |
| 5 | BERT-Base | SVM | 0.9324 | 0.9326 | 0.9357 | 0.9291 | 0.9783 |
| 6 | FastText | Random Forest | 0.9304 | 0.9326 | 0.9621 | 0.9007 | 0.9794 |
| 7 | BERT-Base | Random Forest | 0.9203 | 0.9220 | 0.9407 | 0.9007 | 0.9749 |
| 8 | FastText | SVM | 0.9170 | 0.9184 | 0.9338 | 0.9007 | 0.9671 |
| 9 | Albertina-Large | SVM | 0.9158 | 0.9184 | 0.9470 | 0.8865 | 0.9689 |
| 10 | Albertina-Base | SVM | 0.9143 | 0.9149 | 0.9209 | 0.9078 | 0.9652 |
| 11 | Albertina-Base | Random Forest | 0.9124 | 0.9149 | 0.9398 | 0.8865 | 0.9632 |
| 12 | BERT-Large | Random Forest | 0.8978 | 0.9007 | 0.9248 | 0.8723 | 0.9658 |
| 13 | BERT-Base | Naive Bayes | 0.8788 | 0.8865 | 0.9431 | 0.8227 | 0.9316 |
| 14 | FastText | Naive Bayes | 0.8633 | 0.8652 | 0.8759 | 0.8511 | 0.9381 |
| 15 | Albertina-Large | Random Forest | 0.8273 | 0.8298 | 0.8394 | 0.8156 | 0.9284 |
| 16 | Albertina-Large | Naive Bayes | 0.7647 | 0.7730 | 0.7939 | 0.7376 | 0.8285 |
| 17 | BERT-Large | Naive Bayes | 0.7333 | 0.7447 | 0.7674 | 0.7021 | 0.7938 |
| 18 | Albertina-Base | Naive Bayes | 0.3483 | 0.5887 | 0.8378 | 0.2199 | 0.6027 |

### 1.2 Top 5 Modelos

1. **TF-IDF + SVM: 0.9783**
   - Precision perfeita (100%)
   - Recall excelente (95.74%)
   - Nenhum falso positivo

2. **TF-IDF + Naive Bayes: 0.9565**
   - Baseline original
   - Performance muito solida
   - Rapido e eficiente

3. **TF-IDF + Random Forest: 0.9537**
   - Ligeiramente inferior ao baseline
   - Metricas balanceadas
   - Robusto

4. **BERT-Large + SVM: 0.9416**
   - Melhor embedding
   - SVM aproveita bem BERT-Large
   - Precision alta (96.99%)

5. **BERT-Base + SVM: 0.9324**
   - Segunda melhor embedding
   - Performance consistente
   - Boa alternativa

## 2. Analise por Vetorizacao

### 2.1 TF-IDF (Melhor Vetorizacao)

| Classificador | F1-Score | Ranking Geral |
|---------------|----------|---------------|
| SVM | 0.9783 | 1º |
| Naive Bayes | 0.9565 | 2º |
| Random Forest | 0.9537 | 3º |

**Conclusao:** TF-IDF e a melhor vetorizacao para TODOS os classificadores.

### 2.2 BERT-Large

| Classificador | F1-Score | Ranking Geral |
|---------------|----------|---------------|
| SVM | 0.9416 | 4º |
| Random Forest | 0.8978 | 12º |
| Naive Bayes | 0.7333 | 17º |

**Conclusao:** BERT-Large precisa de SVM para brilhar. Com Naive Bayes, performance e pessima.

### 2.3 BERT-Base

| Classificador | F1-Score | Ranking Geral |
|---------------|----------|---------------|
| SVM | 0.9324 | 5º |
| Random Forest | 0.9203 | 7º |
| Naive Bayes | 0.8788 | 13º |

**Conclusao:** BERT-Base tem performance mais consistente que BERT-Large, mas ainda precisa de SVM.

### 2.4 FastText

| Classificador | F1-Score | Ranking Geral |
|---------------|----------|---------------|
| Random Forest | 0.9304 | 6º |
| SVM | 0.9170 | 8º |
| Naive Bayes | 0.8633 | 14º |

**Conclusao:** FastText e a UNICA vetorizacao onde Random Forest vence SVM.

### 2.5 Albertina-Large

| Classificador | F1-Score | Ranking Geral |
|---------------|----------|---------------|
| SVM | 0.9158 | 9º |
| Random Forest | 0.8273 | 15º |
| Naive Bayes | 0.7647 | 16º |

**Conclusao:** Albertina-Large tem performance irregular. Melhor com SVM.

### 2.6 Albertina-Base

| Classificador | F1-Score | Ranking Geral |
|---------------|----------|---------------|
| SVM | 0.9143 | 10º |
| Random Forest | 0.9124 | 11º |
| Naive Bayes | 0.3483 | 18º (PIOR) |

**Conclusao:** Albertina-Base e catastrofico com Naive Bayes (F1: 0.3483). SVM e RF resgatam a performance.

## 3. Analise por Classificador

### 3.1 SVM (Melhor Classificador)

| Vetorizacao | F1-Score | Ranking Geral |
|-------------|----------|---------------|
| TF-IDF | 0.9783 | 1º |
| BERT-Large | 0.9416 | 4º |
| BERT-Base | 0.9324 | 5º |
| FastText | 0.9170 | 8º |
| Albertina-Large | 0.9158 | 9º |
| Albertina-Base | 0.9143 | 10º |

**Media SVM: 0.9349**

**Conclusao:** SVM e o classificador mais robusto. Funciona bem com TODAS as vetorizacoes.

### 3.2 Random Forest

| Vetorizacao | F1-Score | Ranking Geral |
|-------------|----------|---------------|
| TF-IDF | 0.9537 | 3º |
| FastText | 0.9304 | 6º |
| BERT-Base | 0.9203 | 7º |
| Albertina-Base | 0.9124 | 11º |
| BERT-Large | 0.8978 | 12º |
| Albertina-Large | 0.8273 | 15º |

**Media Random Forest: 0.9070**

**Conclusao:** Random Forest e intermediario. Melhor com FastText, pior com Albertina-Large.

### 3.3 Naive Bayes

| Vetorizacao | F1-Score | Ranking Geral |
|-------------|----------|---------------|
| TF-IDF | 0.9565 | 2º |
| BERT-Base | 0.8788 | 13º |
| FastText | 0.8633 | 14º |
| Albertina-Large | 0.7647 | 16º |
| BERT-Large | 0.7333 | 17º |
| Albertina-Base | 0.3483 | 18º |

**Media Naive Bayes: 0.7575**

**Conclusao:** Naive Bayes e excelente APENAS com TF-IDF. Pessimo com embeddings densos.

## 4. Insights Principais

### 4.1 TF-IDF Domina

- **TF-IDF ocupa os 3 primeiros lugares**
- TF-IDF + SVM e imbativel (F1: 0.9783)
- TF-IDF funciona bem com TODOS os classificadores

### 4.2 SVM e o Classificador Universal

- **SVM vence em 5 de 6 vetorizacoes**
- Unica excecao: FastText (RF vence por 0.0134)
- SVM resgata Albertina-Base (de 0.3483 para 0.9143)

### 4.3 Naive Bayes e Especialista em TF-IDF

- Excelente com TF-IDF (F1: 0.9565)
- Catastrofico com Albertina-Base (F1: 0.3483)
- Nao funciona com embeddings densos

### 4.4 Random Forest e Intermediario

- Melhor que NB, pior que SVM (na maioria dos casos)
- Unica vitoria: FastText (F1: 0.9304)
- Sofre com alta dimensionalidade (TF-IDF)

### 4.5 BERT-Large Precisa de SVM

- Com SVM: 4º lugar (F1: 0.9416)
- Com NB: 17º lugar (F1: 0.7333)
- Diferenca de 0.2083 (28.4%)

### 4.6 Albertina-Base e Incompativel com Naive Bayes

- Com NB: ULTIMO lugar (F1: 0.3483)
- Com SVM: 10º lugar (F1: 0.9143)
- Melhoria de 162.5%

## 5. Recomendacoes

### 5.1 Para Producao

**Modelo escolhido: TF-IDF + SVM**
- F1-Score: 0.9783
- Precision: 100% (nenhum falso positivo)
- Recall: 95.74%
- ROC-AUC: 0.9923

**Justificativa:**
- Melhor performance geral
- Precision perfeita (critico para deteccao de fraude)
- Rapido para treinar e prever
- Interpretavel (features sao palavras)

### 5.2 Alternativas

**Se precisar de velocidade:**
- TF-IDF + Naive Bayes (F1: 0.9565)
- Muito rapido, performance excelente

**Se precisar de embeddings:**
- BERT-Large + SVM (F1: 0.9416)
- Melhor opcao com embeddings contextuais

**Se precisar de robustez:**
- TF-IDF + Random Forest (F1: 0.9537)
- Ensemble reduz variancia

### 5.3 O que Evitar

**Nunca usar:**
- Albertina-Base + Naive Bayes (F1: 0.3483)
- BERT-Large + Naive Bayes (F1: 0.7333)
- Qualquer embedding denso com Naive Bayes

## 6. Proximos Passos

### 6.1 Avaliacao no Conjunto de Teste

**Modelo a ser testado: TF-IDF + SVM**

**Conjunto de teste:**
- Localizado em: `/dataset/cleaned_data_no_bias/FOR_TRAINING/Pre_processed_for_Sparse/TEST_PREPROCESSED.csv`
- Ainda nao vetorizado
- Completamente isolado (nunca visto)

**Passos:**
1. Vetorizar conjunto de teste com TF-IDF
2. Carregar modelo TF-IDF + SVM treinado
3. Fazer predicoes no teste
4. Calcular metricas finais
5. Gerar relatorio final

### 6.2 Possivel Tuning de Hiperparametros

**Se metricas no teste forem insatisfatorias:**
- Otimizar parametro C do SVM (regularizacao)
- Testar diferentes kernels (RBF, polinomial)
- Grid search com validacao cruzada

**Nota:** Tuning deve ser feito APENAS se necessario, pois baseline ja e excelente.

## 7. Estrutura de Arquivos Gerados

```
training/results/
├── baseline_naive_bayes/
│   ├── EXPLICACAO_BASELINE.md
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── naive_bayes_baseline_*.pkl
│
├── naive_bayes_vectorizations/
│   ├── EXPLICACAO_EXPERIMENTO.md
│   ├── naive_bayes_vectorizations_comparison.png
│   ├── naive_bayes_vectorizations_comparison.csv
│   ├── fasttext/
│   │   ├── confusion_matrix.png
│   │   └── classification_report.txt
│   ├── bert_base/
│   │   ├── confusion_matrix.png
│   │   └── classification_report.txt
│   └── ... (para cada vetorizacao)
│
├── svm_vectorizations/
│   ├── EXPLICACAO_EXPERIMENTO.md
│   ├── svm_vectorizations_comparison.png
│   ├── svm_vectorizations_comparison.csv
│   ├── tf_idf/
│   │   ├── confusion_matrix.png
│   │   └── classification_report.txt
│   └── ... (para cada vetorizacao)
│
├── random_forest_vectorizations/
│   ├── EXPLICACAO_EXPERIMENTO.md
│   ├── random_forest_vectorizations_comparison.png
│   ├── random_forest_vectorizations_comparison.csv
│   ├── tf_idf/
│   │   ├── confusion_matrix.png
│   │   └── classification_report.txt
│   └── ... (para cada vetorizacao)
│
└── CONSOLIDACAO_FINAL.md (este documento)
```

## 8. Metricas de Validacao Cruzada

### 8.1 TF-IDF + SVM (Melhor Modelo)

**Validacao Cruzada (5-fold no treino):**
- Accuracy: 0.9672 +/- 0.0156
- Precision: 0.9766 +/- 0.0081
- Recall: 0.9574 +/- 0.0378
- F1-Score: 0.9668 +/- 0.0164
- ROC-AUC: 0.9901 +/- 0.0083

**Desenvolvimento:**
- Accuracy: 0.9787
- Precision: 1.0000
- Recall: 0.9574
- F1-Score: 0.9783
- ROC-AUC: 0.9923

**Consistencia:** Metricas de CV e Dev muito proximas (diferenca < 2%)

## 9. Conclusao Final

Apos testar **18 combinacoes** de vetorizacoes e classificadores, o modelo **TF-IDF + SVM** emergiu como claro vencedor com F1-Score de **0.9783** no conjunto de desenvolvimento.

**Principais descobertas:**

1. **TF-IDF e superior a embeddings** para este problema especifico
2. **SVM e o classificador mais robusto** (funciona bem com todas as vetorizacoes)
3. **Naive Bayes e excelente APENAS com TF-IDF**
4. **Random Forest e intermediario** entre NB e SVM
5. **Embeddings densos precisam de SVM** para ter boa performance

**Modelo final recomendado para teste:**
- **Vetorizacao:** TF-IDF (n-grams 1-2, max 10k features)
- **Classificador:** SVM (kernel linear, probability=True)
- **F1-Score esperado no teste:** ~0.97 (baseado em desenvolvimento)

O experimento foi bem-sucedido e todos os outputs estao padronizados e documentados para apresentacao academica.
