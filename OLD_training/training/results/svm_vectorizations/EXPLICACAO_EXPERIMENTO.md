# Relatorio do Experimento: SVM com Multiplas Vetorizacoes

## 1. Objetivo do Experimento

Avaliar o impacto de diferentes tecnicas de vetorizacao na performance do classificador SVM (Support Vector Machine) para deteccao de fraudes em noticias. Este experimento compara TODAS as 6 vetorizacoes: TF-IDF, FastText, BERT-Base, BERT-Large, Albertina-Base e Albertina-Large.

## 2. Metodologia

### 2.1 Divisao dos Dados

**Identica aos experimentos anteriores:**

- **Dados Totais Vetorizados:** 1,410 exemplos (705 positivos, 705 negativos)
- **Conjunto de Treino:** 1,128 exemplos (80%)
- **Conjunto de Desenvolvimento:** 282 exemplos (20%)
- **Conjunto de Teste:** Separado e ainda nao vetorizado

### 2.2 Modelo: Support Vector Machine (SVM)

**O que e SVM?**

SVM e um algoritmo de aprendizado supervisionado que encontra o hiperplano otimo que melhor separa as classes no espaco de features. E considerado um dos classificadores mais poderosos para problemas de classificacao binaria.

**Conceitos principais:**

1. **Hiperplano de Separacao:**
   - Linha (2D), plano (3D) ou hiperplano (>3D) que divide as classes
   - SVM busca o hiperplano com maior margem entre as classes

2. **Margem:**
   - Distancia entre o hiperplano e os pontos mais proximos de cada classe
   - SVM maximiza essa margem para melhor generalizacao

3. **Vetores de Suporte:**
   - Pontos de dados mais proximos ao hiperplano
   - Sao os unicos pontos que definem o hiperplano
   - Dao nome ao algoritmo (Support Vector Machine)

4. **Kernel:**
   - Funcao que transforma dados para espaco de maior dimensao
   - Permite separar dados nao-linearmente separaveis
   - Tipos: linear, polinomial, RBF (radial basis function), sigmoid

**Configuracao utilizada:**

```python
model = SVC(kernel='linear', probability=True, random_state=42)
```

- **Kernel linear:** Assume separacao linear no espaco de features
- **probability=True:** Habilita calculo de probabilidades (necessario para ROC-AUC)
- **random_state=42:** Garante reproducibilidade

**Por que kernel linear?**

1. **Alta dimensionalidade:** TF-IDF tem 10,000 features, embeddings tem 300-1536
2. **Dados ja em espaco de alta dimensao:** Kernel nao-linear pode causar overfitting
3. **Interpretabilidade:** Kernel linear e mais interpretavel
4. **Eficiencia:** Mais rapido que kernels nao-lineares

**Diferenca entre SVM e Naive Bayes:**

| Aspecto | SVM | Naive Bayes |
|---------|-----|-------------|
| Tipo | Discriminativo | Generativo |
| Objetivo | Maximizar margem | Calcular probabilidades |
| Assumpcoes | Nenhuma sobre distribuicao | Independencia de features |
| Complexidade | O(n^2) a O(n^3) | O(n) |
| Performance | Melhor para dados complexos | Melhor para dados simples |
| Overfitting | Menos propenso | Mais propenso |

### 2.3 Processo de Treinamento

**Identico aos experimentos anteriores:**

1. **Carregamento dos dados vetorizados**
2. **Split treino/desenvolvimento (80/20)**
3. **Validacao cruzada 5-fold no treino**
4. **Retreinamento em todos os dados de treino**
5. **Avaliacao no desenvolvimento**

**Codigo principal:**

```python
# Linhas 96-112: Treinamento e avaliacao
def train_and_evaluate_svm(X_train, y_train, X_dev, y_dev, vec_name):
    # Configuracao do modelo
    model = SVC(kernel='linear', probability=True, random_state=42)
    
    # Validacao cruzada no treino
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    for metric in scoring_metrics:
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric)
        cv_results[metric] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
    
    # Treinar modelo final no treino
    model.fit(X_train, y_train)
    
    # Avaliar no desenvolvimento
    y_pred = model.predict(X_dev)
    y_prob = model.predict_proba(X_dev)[:, 1]
    
    # Calcular metricas
    dev_metrics = {
        'accuracy': accuracy_score(y_dev, y_pred),
        'precision': precision_score(y_dev, y_pred),
        'recall': recall_score(y_dev, y_pred),
        'f1': f1_score(y_dev, y_pred),
        'roc_auc': roc_auc_score(y_dev, y_prob)
    }
    
    return model, cv_results, dev_metrics
```

## 3. Resultados

### 3.1 Tabela Comparativa (Conjunto de Desenvolvimento)

| Vetorizacao | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------------|----------|-----------|--------|----------|---------|
| **TF-IDF** | **0.9787** | **1.0000** | **0.9574** | **0.9783** | 0.9923 |
| BERT-Large | 0.9433 | 0.9699 | 0.9149 | 0.9416 | **0.9874** |
| BERT-Base | 0.9326 | 0.9357 | 0.9291 | 0.9324 | 0.9783 |
| Albertina-Large | 0.9184 | 0.9470 | 0.8865 | 0.9158 | 0.9689 |
| FastText | 0.9184 | 0.9338 | 0.9007 | 0.9170 | 0.9671 |
| Albertina-Base | 0.9149 | 0.9209 | 0.9078 | 0.9143 | 0.9652 |

**Melhor Vetorizacao: TF-IDF (F1-Score: 0.9783)**

### 3.2 Resultados Detalhados por Vetorizacao

#### 3.2.1 TF-IDF (F1: 0.9783) - MELHOR

**Validacao Cruzada:**
- Accuracy: 0.9672 +/- 0.0156
- Precision: 0.9766 +/- 0.0081
- Recall: 0.9574 +/- 0.0378
- F1-Score: 0.9668 +/- 0.0164
- ROC-AUC: 0.9901 +/- 0.0083

**Desenvolvimento:**
- Accuracy: 0.9787
- Precision: 1.0000 (PERFEITO!)
- Recall: 0.9574
- F1-Score: 0.9783
- ROC-AUC: 0.9923

**Analise:**
- Performance excepcional
- Precision perfeita (100%) - nenhum falso positivo
- Recall excelente (95.74%)
- Melhor combinacao ate agora (supera baseline NB)
- SVM aproveita muito bem features esparsas do TF-IDF

#### 3.2.2 BERT-Large (F1: 0.9416) - 2º LUGAR

**Validacao Cruzada:**
- Accuracy: 0.9390 +/- 0.0256
- Precision: 0.9584 +/- 0.0259
- Recall: 0.9186 +/- 0.0499
- F1-Score: 0.9379 +/- 0.0273
- ROC-AUC: 0.9844 +/- 0.0134

**Desenvolvimento:**
- Accuracy: 0.9433
- Precision: 0.9699
- Recall: 0.9149
- F1-Score: 0.9416
- ROC-AUC: 0.9874

**Analise:**
- Excelente performance com SVM
- MUITO melhor que com Naive Bayes (F1: 0.7333)
- SVM consegue aproveitar embeddings densos
- BERT-Large finalmente mostra seu potencial

#### 3.2.3 BERT-Base (F1: 0.9324) - 3º LUGAR

**Validacao Cruzada:**
- Accuracy: 0.9284 +/- 0.0218
- Precision: 0.9375 +/- 0.0235
- Recall: 0.9186 +/- 0.0499
- F1-Score: 0.9274 +/- 0.0236
- ROC-AUC: 0.9791 +/- 0.0148

**Desenvolvimento:**
- Accuracy: 0.9326
- Precision: 0.9357
- Recall: 0.9291
- F1-Score: 0.9324
- ROC-AUC: 0.9783

**Analise:**
- Performance solida
- Melhor que com Naive Bayes (F1: 0.8788)
- Recall melhor que BERT-Large
- SVM aproveita bem embeddings contextuais

#### 3.2.4 Albertina-Large (F1: 0.9158) - 4º LUGAR

**Validacao Cruzada:**
- Accuracy: 0.9158 +/- 0.0471
- Precision: 0.9200 +/- 0.0392
- Recall: 0.9115 +/- 0.0962
- F1-Score: 0.9149 +/- 0.0510
- ROC-AUC: 0.9685 +/- 0.0267

**Desenvolvimento:**
- Accuracy: 0.9184
- Precision: 0.9470
- Recall: 0.8865
- F1-Score: 0.9158
- ROC-AUC: 0.9689

**Analise:**
- MUITO melhor que com Naive Bayes (F1: 0.7647)
- SVM resolveu problema de Albertina
- Alta variancia no recall (±0.0962)
- Performance aceitavel

#### 3.2.5 FastText (F1: 0.9170) - 5º LUGAR

**Validacao Cruzada:**
- Accuracy: 0.9169 +/- 0.0239
- Precision: 0.9175 +/- 0.0285
- Recall: 0.9168 +/- 0.0395
- F1-Score: 0.9169 +/- 0.0250
- ROC-AUC: 0.9698 +/- 0.0197

**Desenvolvimento:**
- Accuracy: 0.9184
- Precision: 0.9338
- Recall: 0.9007
- F1-Score: 0.9170
- ROC-AUC: 0.9671

**Analise:**
- Melhor que com Naive Bayes (F1: 0.8633)
- Performance consistente
- SVM aproveita melhor embeddings FastText

#### 3.2.6 Albertina-Base (F1: 0.9143) - 6º LUGAR

**Validacao Cruzada:**
- Accuracy: 0.8989 +/- 0.0253
- Precision: 0.8870 +/- 0.0578
- Recall: 0.9168 +/- 0.0763
- F1-Score: 0.9006 +/- 0.0264
- ROC-AUC: 0.9635 +/- 0.0227

**Desenvolvimento:**
- Accuracy: 0.9149
- Precision: 0.9209
- Recall: 0.9078
- F1-Score: 0.9143
- ROC-AUC: 0.9652

**Analise:**
- DRAMATICAMENTE melhor que com Naive Bayes (F1: 0.3483)
- SVM resgatou Albertina-Base
- Performance agora aceitavel
- Ainda o pior entre os embeddings

### 3.3 Comparacao: SVM vs Naive Bayes

| Vetorizacao | SVM F1 | NB F1 | Diferenca | Melhoria |
|-------------|--------|-------|-----------|----------|
| TF-IDF | 0.9783 | 0.9565 | +0.0218 | +2.3% |
| BERT-Large | 0.9416 | 0.7333 | +0.2083 | +28.4% |
| BERT-Base | 0.9324 | 0.8788 | +0.0536 | +6.1% |
| Albertina-Large | 0.9158 | 0.7647 | +0.1511 | +19.8% |
| FastText | 0.9170 | 0.8633 | +0.0537 | +6.2% |
| Albertina-Base | 0.9143 | 0.3483 | +0.5660 | +162.5% |

**Conclusoes:**

1. **SVM e SUPERIOR a Naive Bayes em TODAS as vetorizacoes**
2. **Maior melhoria:** Albertina-Base (+162.5%)
3. **Menor melhoria:** TF-IDF (+2.3%)
4. **Embeddings se beneficiam MUITO mais do SVM**

### 3.4 Comparacao com Baseline

| Modelo | F1-Score | Diferenca vs Baseline |
|--------|----------|----------------------|
| **TF-IDF + SVM** | **0.9783** | **+0.0218 (+2.3%)** |
| TF-IDF + NB (Baseline) | 0.9565 | - |
| BERT-Large + SVM | 0.9416 | -0.0149 (-1.6%) |
| BERT-Base + SVM | 0.9324 | -0.0241 (-2.5%) |

**NOVO MELHOR MODELO: TF-IDF + SVM (F1: 0.9783)**

## 4. Analise Critica

### 4.1 Por que SVM superou Naive Bayes?

**Razoes principais:**

1. **SVM e mais poderoso:**
   - Nao assume independencia de features
   - Pode capturar relacoes complexas entre features
   - Maximiza margem de separacao

2. **SVM funciona bem com alta dimensionalidade:**
   - TF-IDF: 10,000 features
   - Embeddings: 300-1536 features
   - SVM nao sofre com "curse of dimensionality"

3. **SVM aproveita melhor embeddings densos:**
   - Naive Bayes assume distribuicao normal (Gaussian)
   - SVM nao faz assumpcoes sobre distribuicao
   - Embeddings podem ter distribuicoes complexas

4. **Kernel linear e adequado:**
   - Dados ja em espaco de alta dimensao
   - Separacao linear e suficiente
   - Evita overfitting

### 4.2 Por que TF-IDF ainda e o melhor?

**Mesmo com SVM, TF-IDF supera embeddings:**

1. **Features esparsas e discriminativas:**
   - Palavras-chave especificas de fraude
   - Presenca/ausencia de termos importantes
   - SVM encontra hiperplano otimo nesse espaco

2. **Alta dimensionalidade (10,000) e vantagem:**
   - Mais features = mais informacao
   - SVM lida bem com esparsidade
   - Embeddings comprimem informacao (300-1536 dim)

3. **Precision perfeita (100%):**
   - Nenhum falso positivo
   - SVM encontrou separacao quase perfeita
   - Margem maxima entre classes

### 4.3 Por que BERT-Large superou BERT-Base com SVM?

**Inversao em relacao a Naive Bayes:**

- Com NB: BERT-Base (0.8788) > BERT-Large (0.7333)
- Com SVM: BERT-Large (0.9416) > BERT-Base (0.9324)

**Razoes:**

1. **SVM aproveita maior capacidade:**
   - BERT-Large tem 1024 dimensoes vs 768 do Base
   - Mais informacao semantica
   - SVM consegue usar essa informacao extra

2. **Nao sofre com overfitting:**
   - SVM maximiza margem
   - Regularizacao implicita
   - Dataset pequeno nao e problema

3. **Embeddings mais ricos:**
   - 24 camadas vs 12
   - Representacoes mais abstratas
   - SVM captura padroes complexos

### 4.4 Resgate do Albertina

**Albertina-Base teve melhoria dramatica:**
- NB: F1 = 0.3483 (pior)
- SVM: F1 = 0.9143 (aceitavel)
- Melhoria: +162.5%

**Por que?**

1. **Problema era o Gaussian NB:**
   - Assumia distribuicao normal
   - Albertina nao seguia essa distribuicao
   - SVM nao faz essa assumpcao

2. **SVM e mais robusto:**
   - Funciona com qualquer distribuicao
   - Apenas busca hiperplano otimo
   - Nao precisa estimar parametros de distribuicao

## 5. Limitacoes do Experimento

1. **Apenas kernel linear testado:**
   - Kernels nao-lineares (RBF, polinomial) podem ter resultados diferentes
   - Podem capturar relacoes nao-lineares
   - Mas podem causar overfitting com dataset pequeno

2. **Sem tuning de hiperparametros:**
   - Parametro C (regularizacao) nao foi otimizado
   - Gamma (para kernels nao-lineares) nao foi testado
   - Performance pode melhorar com tuning

3. **Tempo de treinamento:**
   - SVM e mais lento que Naive Bayes
   - Complexidade O(n^2) a O(n^3)
   - Pode ser problema para datasets grandes

## 6. Proximos Passos

**Testar Random Forest com todas as vetorizacoes:**
- Algoritmo ensemble (combina multiplas arvores)
- Pode capturar relacoes nao-lineares
- Menos propenso a overfitting
- Fornece importancia de features

**Hipotese:** Random Forest pode ter performance intermediaria entre NB e SVM.

## 7. Arquivos Gerados

- `svm_vectorizations_comparison_20260519_174152.csv` - Tabela comparativa
- `svm_vectorizations_comparison_20260519_174152.png` - Grafico comparativo
- `svm_tf_idf_20260519_174152.pkl` - Modelo TF-IDF
- `svm_fasttext_20260519_174152.pkl` - Modelo FastText
- `svm_bert_base_20260519_174152.pkl` - Modelo BERT-Base
- `svm_bert_large_20260519_174152.pkl` - Modelo BERT-Large
- `svm_albertina_base_20260519_174152.pkl` - Modelo Albertina-Base
- `svm_albertina_large_20260519_174152.pkl` - Modelo Albertina-Large
- `svm_all_vectorizations_metadata_20260519_174152.pkl` - Metadados completos
- `EXPLICACAO_EXPERIMENTO.md` - Este relatorio

## 8. Conclusao

O experimento demonstrou que **SVM e SUPERIOR a Naive Bayes para TODAS as vetorizacoes**, com melhorias variando de +2.3% (TF-IDF) ate +162.5% (Albertina-Base).

**Novo melhor modelo: TF-IDF + SVM (F1: 0.9783)**
- Precision perfeita (100%)
- Recall excelente (95.74%)
- Supera baseline Naive Bayes em +2.3%

**Licoes aprendidas:**

1. **SVM aproveita melhor embeddings densos que Naive Bayes**
2. **TF-IDF continua sendo a melhor vetorizacao, mesmo com SVM**
3. **BERT-Large finalmente mostra seu potencial com SVM**
4. **Albertina foi resgatado pelo SVM (de 0.3483 para 0.9143)**
5. **Kernel linear e suficiente para este problema**

**Recomendacao:** TF-IDF + SVM e o novo baseline a ser superado. Random Forest sera testado a seguir para verificar se consegue melhorar ainda mais.
