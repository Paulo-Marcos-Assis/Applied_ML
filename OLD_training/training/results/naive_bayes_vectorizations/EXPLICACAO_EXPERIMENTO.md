# Relatorio do Experimento: Naive Bayes com Multiplas Vetorizacoes

## 1. Objetivo do Experimento

Avaliar o impacto de diferentes tecnicas de vetorizacao na performance do classificador Naive Bayes para deteccao de fraudes em noticias. Este experimento compara 5 vetorizacoes: FastText, BERT-Base, BERT-Large, Albertina-Base e Albertina-Large.

**NOTA:** TF-IDF + Naive Bayes ja foi testado no baseline (F1-Score: 0.9565).

## 2. Metodologia

### 2.1 Divisao dos Dados

**Identica ao baseline para todos os experimentos:**

- **Dados Totais Vetorizados:** 1,410 exemplos (705 positivos, 705 negativos)
- **Conjunto de Treino:** 1,128 exemplos (80%)
- **Conjunto de Desenvolvimento:** 282 exemplos (20%)
- **Conjunto de Teste:** Separado e ainda nao vetorizado

### 2.2 Vetorizacoes Testadas

#### 2.2.1 FastText (300 dimensoes)

**O que e FastText?**
- Modelo de embeddings desenvolvido pelo Facebook
- Aprende representacoes de palavras baseadas em subpalavras (n-gramas de caracteres)
- Vantagem: Lida bem com palavras fora do vocabulario (OOV)
- Pre-treinado em corpus portugues (Common Crawl)

**Configuracao:**
- Modelo: `cc.pt.300.bin` (Common Crawl Portuguese)
- Dimensoes: 300
- Ponderacao: TF-IDF weighted (media ponderada dos embeddings das palavras)
- Matriz final: (1,410 exemplos, 300 features)

**Como funciona:**
1. Cada palavra do documento e convertida em vetor de 300 dimensoes
2. Vetores sao ponderados pelo peso TF-IDF da palavra
3. Media ponderada gera vetor final do documento

**Codigo de carregamento:**
```python
# Linha 215: Configuracao FastText
'FastText': {
    'dir': os.path.join(BASE_DIR, 'vectorization/fasttext/outputs'),
    'is_dense': True  # Embeddings sao densos (nao esparsos)
}
```

#### 2.2.2 BERT-Base (768 dimensoes)

**O que e BERT?**
- Bidirectional Encoder Representations from Transformers
- Modelo de linguagem contextual (considera contexto bidirecional)
- Captura relacoes semanticas complexas
- Pre-treinado em portugues

**Configuracao:**
- Modelo: `bert-base-portuguese-cased`
- Dimensoes: 768
- Estrategia: [CLS] token embedding (representacao do documento inteiro)
- Matriz final: (1,410 exemplos, 768 features)

**Como funciona:**
1. Documento e tokenizado com tokenizador BERT
2. Passa pela rede neural BERT (12 camadas)
3. Token [CLS] no inicio captura representacao do documento completo
4. Vetor de 768 dimensoes representa o documento

**Codigo de carregamento:**
```python
# Linha 218: Configuracao BERT-Base
'BERT-Base': {
    'dir': os.path.join(BASE_DIR, 'vectorization/bert_base'),
    'is_dense': True
}
```

#### 2.2.3 BERT-Large (1024 dimensoes)

**O que e BERT-Large?**
- Versao maior do BERT com mais parametros
- 24 camadas (vs 12 do Base)
- Mais capacidade de capturar padroes complexos
- Mais lento e exige mais recursos

**Configuracao:**
- Modelo: `bert-large-portuguese-cased`
- Dimensoes: 1024
- Estrategia: [CLS] token embedding
- Matriz final: (1,410 exemplos, 1024 features)

**Diferenca para BERT-Base:**
- Mais camadas (24 vs 12)
- Mais dimensoes (1024 vs 768)
- Mais parametros (~340M vs ~110M)

#### 2.2.4 Albertina-Base (768 dimensoes)

**O que e Albertina?**
- Modelo BERT treinado especificamente em portugues brasileiro
- Desenvolvido pela Universidade de Lisboa
- Corpus: BrWaC (Brazilian Web as Corpus)
- Otimizado para portugues do Brasil

**Configuracao:**
- Modelo: `albertina-100m-portuguese-ptbr-encoder`
- Dimensoes: 768
- Estrategia: [CLS] token embedding
- Matriz final: (1,410 exemplos, 768 features)

**Diferenca para BERT:**
- Treinado exclusivamente em portugues brasileiro
- Corpus mais especifico (web brasileira)

#### 2.2.5 Albertina-Large (1536 dimensoes)

**O que e Albertina-Large?**
- Versao maior do Albertina
- Mais parametros (~900M)
- Maior capacidade de representacao

**Configuracao:**
- Modelo: `albertina-900m-portuguese-ptbr-encoder-brwac`
- Dimensoes: 1536
- Estrategia: [CLS] token embedding
- Matriz final: (1,410 exemplos, 1536 features)

### 2.3 Modelo: Gaussian Naive Bayes

**Por que Gaussian ao inves de Multinomial?**

- **Multinomial NB:** Para dados esparsos e discretos (TF-IDF)
- **Gaussian NB:** Para dados densos e continuos (embeddings)

**Codigo que escolhe o tipo:**
```python
# Linhas 76-80: Escolha do tipo de Naive Bayes
if is_dense:
    model = GaussianNB()
    print("Usando GaussianNB (dados densos - embeddings)")
else:
    model = MultinomialNB(alpha=1.0)
    print("Usando MultinomialNB (dados esparsos)")
```

**Como funciona Gaussian NB:**
- Assume que features seguem distribuicao normal (Gaussiana)
- Para cada classe, calcula media e variancia de cada feature
- Usa essas estatisticas para calcular probabilidades

### 2.4 Processo de Treinamento

**Identico ao baseline:**

1. **Carregamento dos dados vetorizados**
2. **Split treino/desenvolvimento (80/20)**
3. **Validacao cruzada 5-fold no treino**
4. **Retreinamento em todos os dados de treino**
5. **Avaliacao no desenvolvimento**

**Codigo principal:**
```python
# Linhas 248-268: Loop principal
for vec_name, vec_config in vectorizations.items():
    # 1. Carregar dados
    X, y = load_vectorization_data(vec_config['dir'], vec_name)
    
    # 2. Split treino/desenvolvimento
    X_train, X_dev, y_train, y_dev = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Treinar e avaliar
    model, cv_results, dev_metrics = train_and_evaluate_naive_bayes(
        X_train, y_train, X_dev, y_dev, vec_name, vec_config['is_dense']
    )
    
    # 4. Armazenar resultados
    results[vec_name] = {
        'model': model,
        'cv_results': cv_results,
        'dev_metrics': dev_metrics
    }
```

## 3. Resultados

### 3.1 Tabela Comparativa (Conjunto de Desenvolvimento)

| Vetorizacao | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------------|----------|-----------|--------|----------|---------|
| **BERT-Base** | **0.8865** | **0.9431** | 0.8227 | **0.8788** | 0.9316 |
| FastText | 0.8652 | 0.8759 | **0.8511** | 0.8633 | **0.9381** |
| Albertina-Large | 0.7730 | 0.7939 | 0.7376 | 0.7647 | 0.8285 |
| BERT-Large | 0.7447 | 0.7674 | 0.7021 | 0.7333 | 0.7938 |
| Albertina-Base | 0.5887 | 0.8378 | 0.2199 | 0.3483 | 0.6027 |

**Melhor Vetorizacao: BERT-Base (F1-Score: 0.8788)**

### 3.2 Resultados Detalhados por Vetorizacao

#### 3.2.1 FastText (F1: 0.8633)

**Validacao Cruzada:**
- Accuracy: 0.8626 +/- 0.0358
- Precision: 0.8645 +/- 0.0311
- Recall: 0.8600 +/- 0.0501
- F1-Score: 0.8621 +/- 0.0373
- ROC-AUC: 0.9226 +/- 0.0479

**Desenvolvimento:**
- Accuracy: 0.8652
- Precision: 0.8759
- Recall: 0.8511
- F1-Score: 0.8633
- ROC-AUC: 0.9381

**Analise:**
- Performance solida e consistente
- Melhor recall entre os embeddings (0.8511)
- Baixa variancia na validacao cruzada
- Boa capacidade de generalizacao

#### 3.2.2 BERT-Base (F1: 0.8788) - MELHOR

**Validacao Cruzada:**
- Accuracy: 0.8945 +/- 0.0437
- Precision: 0.9438 +/- 0.0653
- Recall: 0.8405 +/- 0.0693
- F1-Score: 0.8884 +/- 0.0463
- ROC-AUC: 0.9446 +/- 0.0497

**Desenvolvimento:**
- Accuracy: 0.8865
- Precision: 0.9431 (MELHOR)
- Recall: 0.8227
- F1-Score: 0.8788 (MELHOR)
- ROC-AUC: 0.9316

**Analise:**
- Melhor F1-Score entre os embeddings
- Precision excelente (94.31%)
- Recall um pouco menor que FastText
- Modelo mais conservador (menos falsos positivos)

#### 3.2.3 BERT-Large (F1: 0.7333)

**Validacao Cruzada:**
- Accuracy: 0.8032 +/- 0.0399
- Precision: 0.8145 +/- 0.0403
- Recall: 0.7855 +/- 0.0545
- F1-Score: 0.7996 +/- 0.0423
- ROC-AUC: 0.8480 +/- 0.0623

**Desenvolvimento:**
- Accuracy: 0.7447
- Precision: 0.7674
- Recall: 0.7021
- F1-Score: 0.7333
- ROC-AUC: 0.7938

**Analise:**
- Performance INFERIOR ao BERT-Base
- Possivel overfitting (CV melhor que Dev)
- Modelo muito complexo para dataset pequeno
- 1024 dimensoes podem ser excessivas

#### 3.2.4 Albertina-Base (F1: 0.3483) - PIOR

**Validacao Cruzada:**
- Accuracy: 0.6055 +/- 0.0245
- Precision: 0.8828 +/- 0.0745
- Recall: 0.2446 +/- 0.0672
- F1-Score: 0.3815 +/- 0.0777
- ROC-AUC: 0.6097 +/- 0.0393

**Desenvolvimento:**
- Accuracy: 0.5887
- Precision: 0.8378
- Recall: 0.2199 (PIOR)
- F1-Score: 0.3483 (PIOR)
- ROC-AUC: 0.6027

**Analise:**
- Performance muito baixa
- Recall extremamente baixo (22%)
- Modelo muito conservador (classifica quase tudo como negativo)
- Precision alta mas inutilizada pelo recall baixo
- Possivel incompatibilidade entre Albertina e Gaussian NB

#### 3.2.5 Albertina-Large (F1: 0.7647)

**Validacao Cruzada:**
- Accuracy: 0.7882 +/- 0.0716
- Precision: 0.7983 +/- 0.0541
- Recall: 0.7714 +/- 0.1343
- F1-Score: 0.7834 +/- 0.0843
- ROC-AUC: 0.8365 +/- 0.0616

**Desenvolvimento:**
- Accuracy: 0.7730
- Precision: 0.7939
- Recall: 0.7376
- F1-Score: 0.7647
- ROC-AUC: 0.8285

**Analise:**
- Melhor que Albertina-Base mas ainda fraco
- Alta variancia no recall (±0.1343)
- Modelo instavel
- Melhor que BERT-Large mas pior que BERT-Base

### 3.3 Comparacao com Baseline TF-IDF

| Modelo | F1-Score | Diferenca vs Baseline |
|--------|----------|----------------------|
| **TF-IDF + NB (Baseline)** | **0.9565** | - |
| BERT-Base + NB | 0.8788 | -0.0777 (-8.1%) |
| FastText + NB | 0.8633 | -0.0932 (-9.7%) |
| Albertina-Large + NB | 0.7647 | -0.1918 (-20.0%) |
| BERT-Large + NB | 0.7333 | -0.2232 (-23.3%) |
| Albertina-Base + NB | 0.3483 | -0.6082 (-63.6%) |

**CONCLUSAO IMPORTANTE:** TF-IDF continua sendo a MELHOR vetorizacao para Naive Bayes!

## 4. Analise Critica

### 4.1 Por que TF-IDF superou os embeddings?

**Razoes principais:**

1. **Naive Bayes e otimizado para TF-IDF:**
   - Assume independencia entre features
   - TF-IDF gera features independentes (palavras)
   - Embeddings tem features correlacionadas (dimensoes semanticas)

2. **Gaussian NB e mais simples que Multinomial NB:**
   - Gaussian assume distribuicao normal
   - Embeddings podem nao seguir distribuicao normal
   - Multinomial e mais adequado para contagens (TF-IDF)

3. **Dimensionalidade:**
   - TF-IDF: 10,000 features esparsas (maioria zeros)
   - Embeddings: 300-1536 features densas (todas preenchidas)
   - Naive Bayes funciona melhor com esparsidade

4. **Natureza do problema:**
   - Deteccao de fraude pode depender de palavras-chave especificas
   - TF-IDF captura bem presenca/ausencia de palavras
   - Embeddings capturam semantica, que pode ser menos relevante

### 4.2 Por que BERT-Base superou BERT-Large?

**Overfitting em modelos grandes:**
- BERT-Large tem 1024 dimensoes (vs 768 do Base)
- Dataset pequeno (1,128 treino)
- Modelo muito complexo para poucos dados
- Gaussian NB nao consegue estimar bem 1024 distribuicoes normais

**Regra geral:** Modelos maiores precisam de mais dados

### 4.3 Por que Albertina teve performance tao baixa?

**Hipoteses:**

1. **Incompatibilidade com Gaussian NB:**
   - Embeddings Albertina podem ter distribuicao diferente
   - Gaussian NB assume normalidade

2. **Dominio especifico:**
   - Albertina treinado em web brasileira geral
   - Dataset de noticias pode ter vocabulario diferente

3. **Recall extremamente baixo:**
   - Modelo classifica quase tudo como negativo
   - Possivel problema de calibracao

### 4.4 Quando usar cada vetorizacao?

**TF-IDF:**
- Melhor para Naive Bayes
- Rapido e eficiente
- Interpretavel
- **Recomendado para este problema**

**FastText:**
- Bom para palavras fora do vocabulario
- Mais robusto que TF-IDF para erros ortograficos
- Segunda melhor opcao

**BERT-Base:**
- Melhor para capturar semantica
- Pode ser melhor com classificadores mais complexos (SVM, Random Forest)
- Terceira opcao

**BERT-Large / Albertina:**
- Nao recomendados para dataset pequeno
- Podem ser uteis com mais dados

## 5. Limitacoes do Experimento

1. **Dataset pequeno (1,128 treino):**
   - Embeddings precisam de mais dados
   - Gaussian NB pode nao estimar bem distribuicoes

2. **Naive Bayes e modelo simples:**
   - Nao aproveita toda capacidade dos embeddings
   - Classificadores mais complexos podem ter resultados diferentes

3. **Apenas uma metrica de agregacao:**
   - Embeddings usam media ponderada TF-IDF
   - Outras estrategias (max pooling, concatenacao) nao foram testadas

## 6. Proximos Passos

**Testar embeddings com classificadores mais complexos:**
1. SVM com todas as vetorizacoes
2. Random Forest com todas as vetorizacoes

**Hipotese:** Embeddings podem ter melhor performance com SVM ou Random Forest, que capturam relacoes nao-lineares.

## 7. Arquivos Gerados

- `naive_bayes_vectorizations_comparison_20260519_173212.csv` - Tabela comparativa
- `naive_bayes_vectorizations_comparison_20260519_173212.png` - Grafico comparativo
- `naive_bayes_fasttext_20260519_173213.pkl` - Modelo FastText
- `naive_bayes_bert-base_20260519_173213.pkl` - Modelo BERT-Base
- `naive_bayes_bert-large_20260519_173213.pkl` - Modelo BERT-Large
- `naive_bayes_albertina-base_20260519_173213.pkl` - Modelo Albertina-Base
- `naive_bayes_albertina-large_20260519_173213.pkl` - Modelo Albertina-Large
- `naive_bayes_all_vectorizations_metadata_20260519_173213.pkl` - Metadados completos
- `EXPLICACAO_EXPERIMENTO.md` - Este relatorio

## 8. Conclusao

O experimento demonstrou que **TF-IDF continua sendo a melhor vetorizacao para Naive Bayes** (F1: 0.9565), superando todos os embeddings testados. Entre os embeddings, **BERT-Base teve melhor performance** (F1: 0.8788), seguido por FastText (F1: 0.8633).

**Licoes aprendidas:**
1. Naive Bayes e otimizado para features esparsas e independentes (TF-IDF)
2. Embeddings densos nao sao bem aproveitados por Gaussian NB
3. Modelos maiores (BERT-Large, Albertina-Large) sofrem com dataset pequeno
4. Albertina teve problemas de compatibilidade com Gaussian NB

**Recomendacao:** Manter TF-IDF + Naive Bayes como baseline. Testar embeddings com classificadores mais sofisticados (SVM, Random Forest) que podem aproveitar melhor suas capacidades semanticas.
