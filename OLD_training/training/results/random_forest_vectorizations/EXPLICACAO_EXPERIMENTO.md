# Relatorio do Experimento: Random Forest com Multiplas Vetorizacoes

## 1. Objetivo do Experimento

Avaliar o impacto de diferentes tecnicas de vetorizacao na performance do classificador Random Forest para deteccao de fraudes em noticias. Este experimento compara TODAS as 6 vetorizacoes: TF-IDF, FastText, BERT-Base, BERT-Large, Albertina-Base e Albertina-Large.

## 2. Metodologia

### 2.1 Divisao dos Dados

**Identica aos experimentos anteriores:**

- **Dados Totais Vetorizados:** 1,410 exemplos (705 positivos, 705 negativos)
- **Conjunto de Treino:** 1,128 exemplos (80%)
- **Conjunto de Desenvolvimento:** 282 exemplos (20%)
- **Conjunto de Teste:** Separado e ainda nao vetorizado

### 2.2 Modelo: Random Forest

**O que e Random Forest?**

Random Forest e um algoritmo ensemble que combina multiplas arvores de decisao para fazer predicoes mais robustas e precisas. E um dos algoritmos mais populares em machine learning devido a sua versatilidade e performance.

**Conceitos principais:**

1. **Ensemble Learning:**
   - Combina predicoes de multiplas arvores de decisao
   - Cada arvore vota na classe final
   - Decisao por maioria (classificacao) ou media (regressao)

2. **Bootstrap Aggregating (Bagging):**
   - Cada arvore e treinada em amostra aleatoria dos dados (com reposicao)
   - Reduz variancia e evita overfitting
   - Aumenta robustez do modelo

3. **Feature Randomness:**
   - Cada split em cada arvore considera apenas subconjunto aleatorio de features
   - Decorrelaciona as arvores
   - Melhora diversidade do ensemble

4. **Vantagens:**
   - Menos propenso a overfitting que arvores individuais
   - Lida bem com features nao-lineares
   - Fornece importancia de features
   - Robusto a outliers
   - Funciona bem com alta dimensionalidade

**Configuracao utilizada:**

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
```

- **n_estimators=100:** 100 arvores no ensemble
- **random_state=42:** Garante reproducibilidade
- **Demais parametros:** Valores padrao do scikit-learn

**Parametros padrao importantes:**
- **max_features='sqrt':** Cada split considera sqrt(n_features) features aleatorias
- **max_depth=None:** Arvores crescem ate folhas puras
- **min_samples_split=2:** Minimo de amostras para fazer split
- **min_samples_leaf=1:** Minimo de amostras em folha

**Diferenca entre Random Forest e outros classificadores:**

| Aspecto | Random Forest | SVM | Naive Bayes |
|---------|---------------|-----|-------------|
| Tipo | Ensemble (arvores) | Discriminativo | Generativo |
| Nao-linearidade | Sim (nativo) | Sim (com kernel) | Nao |
| Interpretabilidade | Media (importancia) | Baixa | Alta |
| Overfitting | Baixo | Medio | Alto |
| Velocidade | Media | Lenta | Rapida |
| Alta dimensao | Boa | Excelente | Boa |

### 2.3 Processo de Treinamento

**Identico aos experimentos anteriores:**

1. **Carregamento dos dados vetorizados**
2. **Split treino/desenvolvimento (80/20)**
3. **Validacao cruzada 5-fold no treino**
4. **Retreinamento em todos os dados de treino**
5. **Avaliacao no desenvolvimento**
6. **Geracao de relatorios individuais (matriz de confusao e classification report)**

**Codigo principal:**

```python
# Linhas 91-117: Treinamento e avaliacao
def train_and_evaluate_random_forest(X_train, y_train, X_dev, y_dev, vec_name):
    # Configuracao do modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
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
| **TF-IDF** | **0.9539** | 0.9571 | **0.9504** | **0.9537** | **0.9901** |
| FastText | 0.9326 | **0.9621** | 0.9007 | 0.9304 | 0.9794 |
| BERT-Base | 0.9220 | 0.9407 | 0.9007 | 0.9203 | 0.9749 |
| Albertina-Base | 0.9149 | 0.9398 | 0.8865 | 0.9124 | 0.9632 |
| BERT-Large | 0.9007 | 0.9248 | 0.8723 | 0.8978 | 0.9658 |
| Albertina-Large | 0.8298 | 0.8394 | 0.8156 | 0.8273 | 0.9284 |

**Melhor Vetorizacao: TF-IDF (F1-Score: 0.9537)**

### 3.2 Resultados Detalhados por Vetorizacao

#### 3.2.1 TF-IDF (F1: 0.9537) - MELHOR

**Validacao Cruzada:**
- Accuracy: 0.9549 +/- 0.0191
- Precision: 0.9584 +/- 0.0162
- Recall: 0.9504 +/- 0.0352
- F1-Score: 0.9543 +/- 0.0195
- ROC-AUC: 0.9905 +/- 0.0072

**Desenvolvimento:**
- Accuracy: 0.9539
- Precision: 0.9571
- Recall: 0.9504
- F1-Score: 0.9537
- ROC-AUC: 0.9901

**Analise:**
- Performance excelente e consistente
- Metricas balanceadas (precision e recall similares)
- Melhor ROC-AUC entre todas as vetorizacoes
- TF-IDF continua sendo a melhor escolha

#### 3.2.2 FastText (F1: 0.9304) - 2º LUGAR

**Validacao Cruzada:**
- Accuracy: 0.9328 +/- 0.0282
- Precision: 0.9609 +/- 0.0269
- Recall: 0.9009 +/- 0.0517
- F1-Score: 0.9299 +/- 0.0298
- ROC-AUC: 0.9806 +/- 0.0159

**Desenvolvimento:**
- Accuracy: 0.9326
- Precision: 0.9621 (MELHOR)
- Recall: 0.9007
- F1-Score: 0.9304
- ROC-AUC: 0.9794

**Analise:**
- Segunda melhor performance
- Precision mais alta (96.21%)
- Modelo conservador (menos falsos positivos)
- Melhor que com SVM (F1: 0.9170)

#### 3.2.3 BERT-Base (F1: 0.9203) - 3º LUGAR

**Validacao Cruzada:**
- Accuracy: 0.9273 +/- 0.0377
- Precision: 0.9513 +/- 0.0318
- Recall: 0.9008 +/- 0.0561
- F1-Score: 0.9252 +/- 0.0397
- ROC-AUC: 0.9759 +/- 0.0224

**Desenvolvimento:**
- Accuracy: 0.9220
- Precision: 0.9407
- Recall: 0.9007
- F1-Score: 0.9203
- ROC-AUC: 0.9749

**Analise:**
- Performance solida
- Pior que com SVM (F1: 0.9324)
- Random Forest nao aproveitou tao bem BERT quanto SVM

#### 3.2.4 Albertina-Base (F1: 0.9124) - 4º LUGAR

**Validacao Cruzada:**
- Accuracy: 0.8715 +/- 0.0537
- Precision: 0.9094 +/- 0.0600
- Recall: 0.8248 +/- 0.0851
- F1-Score: 0.8644 +/- 0.0585
- ROC-AUC: 0.9424 +/- 0.0376

**Desenvolvimento:**
- Accuracy: 0.9149
- Precision: 0.9398
- Recall: 0.8865
- F1-Score: 0.9124
- ROC-AUC: 0.9632

**Analise:**
- Performance aceitavel
- Melhor que com Naive Bayes (F1: 0.3483)
- Similar ao SVM (F1: 0.9143)
- Alta variancia na validacao cruzada

#### 3.2.5 BERT-Large (F1: 0.8978) - 5º LUGAR

**Validacao Cruzada:**
- Accuracy: 0.9123 +/- 0.0497
- Precision: 0.9247 +/- 0.0242
- Recall: 0.8973 +/- 0.0817
- F1-Score: 0.9105 +/- 0.0530
- ROC-AUC: 0.9707 +/- 0.0243

**Desenvolvimento:**
- Accuracy: 0.9007
- Precision: 0.9248
- Recall: 0.8723
- F1-Score: 0.8978
- ROC-AUC: 0.9658

**Analise:**
- Performance inferior ao BERT-Base
- Pior que com SVM (F1: 0.9416)
- Possivel overfitting (modelo muito complexo)
- Alta variancia

#### 3.2.6 Albertina-Large (F1: 0.8273) - PIOR

**Validacao Cruzada:**
- Accuracy: 0.8600 +/- 0.0603
- Precision: 0.8796 +/- 0.0650
- Recall: 0.8353 +/- 0.0975
- F1-Score: 0.8560 +/- 0.0636
- ROC-AUC: 0.9379 +/- 0.0392

**Desenvolvimento:**
- Accuracy: 0.8298
- Precision: 0.8394
- Recall: 0.8156
- F1-Score: 0.8273
- ROC-AUC: 0.9284

**Analise:**
- Pior performance entre todas
- Melhor que com Naive Bayes (F1: 0.7647)
- Pior que com SVM (F1: 0.9158)
- Muito instavel (alta variancia)

### 3.3 Comparacao: Random Forest vs SVM vs Naive Bayes

| Vetorizacao | RF F1 | SVM F1 | NB F1 | Melhor |
|-------------|-------|--------|-------|--------|
| TF-IDF | 0.9537 | **0.9783** | 0.9565 | SVM |
| FastText | 0.9304 | 0.9170 | 0.8633 | **RF** |
| BERT-Base | 0.9203 | **0.9324** | 0.8788 | SVM |
| BERT-Large | 0.8978 | **0.9416** | 0.7333 | SVM |
| Albertina-Base | 0.9124 | **0.9143** | 0.3483 | SVM |
| Albertina-Large | 0.8273 | **0.9158** | 0.7647 | SVM |

**Conclusoes:**

1. **SVM e superior em 5 de 6 vetorizacoes**
2. **Random Forest vence apenas com FastText**
3. **TF-IDF + SVM continua sendo o melhor (F1: 0.9783)**
4. **Random Forest e intermediario entre NB e SVM**

### 3.4 Comparacao com Baseline

| Modelo | F1-Score | Diferenca vs Baseline |
|--------|----------|----------------------|
| **TF-IDF + SVM** | **0.9783** | **+0.0218 (+2.3%)** |
| TF-IDF + NB (Baseline) | 0.9565 | - |
| TF-IDF + RF | 0.9537 | -0.0028 (-0.3%) |

**Random Forest com TF-IDF e ligeiramente inferior ao baseline Naive Bayes**

## 4. Analise Critica

### 4.1 Por que Random Forest nao superou SVM?

**Razoes principais:**

1. **SVM e mais adequado para alta dimensionalidade:**
   - TF-IDF: 10,000 features esparsas
   - SVM maximiza margem no espaco de alta dimensao
   - Random Forest pode se perder em muitas features

2. **Random Forest sofre com esparsidade:**
   - TF-IDF e muito esparso (maioria zeros)
   - Arvores de decisao preferem features densas
   - Splits em features esparsas sao menos informativos

3. **SVM e mais eficiente com kernel linear:**
   - Separacao linear e suficiente para este problema
   - Random Forest precisa de muitas arvores para aproximar hiperplano linear
   - SVM encontra diretamente o hiperplano otimo

4. **Embeddings densos favorecem SVM:**
   - SVM nao faz assumpcoes sobre distribuicao
   - Random Forest pode criar splits subotimos
   - SVM encontra margem maxima globalmente

### 4.2 Por que Random Forest venceu com FastText?

**FastText + RF (0.9304) > FastText + SVM (0.9170):**

1. **Embeddings FastText sao mais adequados para arvores:**
   - 300 dimensoes densas e balanceadas
   - Nao tao alta dimensionalidade quanto TF-IDF
   - Arvores conseguem fazer splits informativos

2. **Random Forest captura nao-linearidades:**
   - FastText pode ter relacoes nao-lineares entre dimensoes
   - Random Forest explora essas relacoes naturalmente
   - SVM linear pode perder alguns padroes

3. **Ensemble reduz variancia:**
   - FastText pode ter ruido
   - 100 arvores suavizam predicoes
   - SVM e mais sensivel a ruido

### 4.3 Por que TF-IDF ainda e o melhor?

**Mesmo com Random Forest, TF-IDF supera embeddings:**

1. **Features esparsas e discriminativas:**
   - Palavras-chave especificas de fraude
   - Presenca/ausencia de termos importantes
   - Random Forest identifica features mais importantes

2. **Importancia de features:**
   - Random Forest pode ranquear features
   - Palavras mais discriminativas recebem maior peso
   - Embeddings comprimem informacao

3. **Robustez do ensemble:**
   - 100 arvores votam
   - Decisao mais robusta que arvore unica
   - Reduz overfitting

### 4.4 Ranking Final dos Classificadores

**Para TF-IDF (melhor vetorizacao):**
1. SVM: F1 = 0.9783 (MELHOR)
2. Naive Bayes: F1 = 0.9565
3. Random Forest: F1 = 0.9537

**Para embeddings (media):**
1. SVM: Melhor em 5 de 6 vetorizacoes
2. Random Forest: Melhor em 1 de 6 (FastText)
3. Naive Bayes: Pior em todas

## 5. Limitacoes do Experimento

1. **Apenas 100 arvores testadas:**
   - Mais arvores (200, 500) podem melhorar performance
   - Mas aumentam tempo de treinamento

2. **Sem tuning de hiperparametros:**
   - max_depth, min_samples_split, max_features nao foram otimizados
   - Performance pode melhorar com tuning

3. **Dataset pequeno (1,128 treino):**
   - Random Forest se beneficia de mais dados
   - Com mais dados, pode superar SVM

## 6. Estrutura de Outputs Gerados

**Para cada vetorizacao, foram gerados:**
- Matriz de confusao individual (PNG)
- Classification report individual (TXT)
- Modelo treinado (PKL)

**Outputs gerais:**
- Grafico comparativo entre vetorizacoes (PNG)
- Tabela comparativa (CSV)
- Metadados completos (PKL)
- Documento explicativo (MD)

**Exemplo de estrutura:**
```
results/random_forest_vectorizations/
├── EXPLICACAO_EXPERIMENTO.md
├── random_forest_vectorizations_comparison.png
├── random_forest_vectorizations_comparison.csv
├── tf_idf/
│   ├── confusion_matrix.png
│   └── classification_report.txt
├── fasttext/
│   ├── confusion_matrix.png
│   └── classification_report.txt
└── ... (para cada vetorizacao)
```

## 7. Arquivos Gerados

- `random_forest_vectorizations_comparison_20260519_175551.csv` - Tabela comparativa
- `random_forest_vectorizations_comparison_20260519_175551.png` - Grafico comparativo
- `random_forest_tf_idf_20260519_175552.pkl` - Modelo TF-IDF
- `random_forest_fasttext_20260519_175552.pkl` - Modelo FastText
- `random_forest_bert_base_20260519_175552.pkl` - Modelo BERT-Base
- `random_forest_bert_large_20260519_175552.pkl` - Modelo BERT-Large
- `random_forest_albertina_base_20260519_175552.pkl` - Modelo Albertina-Base
- `random_forest_albertina_large_20260519_175552.pkl` - Modelo Albertina-Large
- `random_forest_all_vectorizations_metadata_20260519_175552.pkl` - Metadados completos
- Subdiretorios individuais com matrizes de confusao e classification reports
- `EXPLICACAO_EXPERIMENTO.md` - Este relatorio

## 8. Conclusao

O experimento demonstrou que **Random Forest tem performance intermediaria entre Naive Bayes e SVM**, sendo superior ao Naive Bayes mas inferior ao SVM na maioria dos casos.

**Melhor modelo Random Forest: TF-IDF + RF (F1: 0.9537)**
- Ligeiramente inferior ao baseline Naive Bayes (-0.3%)
- Significativamente inferior ao SVM (-2.5%)

**Licoes aprendidas:**

1. **SVM continua sendo o melhor classificador para este problema**
2. **Random Forest vence apenas com FastText (embeddings densos de media dimensao)**
3. **TF-IDF e a melhor vetorizacao para todos os classificadores**
4. **Random Forest sofre com alta dimensionalidade e esparsidade do TF-IDF**
5. **Para embeddings, SVM e consistentemente superior**

**Ranking Final (18 combinacoes):**
1. **TF-IDF + SVM: F1 = 0.9783** (CAMPEAO)
2. TF-IDF + Naive Bayes: F1 = 0.9565
3. TF-IDF + Random Forest: F1 = 0.9537

**Recomendacao final:** TF-IDF + SVM e o modelo escolhido para avaliacao no conjunto de teste final.
