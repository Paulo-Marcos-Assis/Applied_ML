# Documentação Completa de Vetorização para Detecção de Fraudes Corporativas

## Objetivo Inicial

Desenvolver um sistema de classificação automática de texto para detectar notícias sobre fraudes corporativas em português brasileiro usando técnicas de Machine Learning e Processamento de Linguagem Natural (PLN). O projeto iniciou com a necessidade de criar datasets balanceados e aplicar diversas técnicas de vetorização para treinar modelos classificadores.

## Contexto do Projeto

O projeto consiste em analisar notícias em português brasileiro para classificá-las como:
- **Classe 1**: Notícias sobre fraudes corporativas (casos positivos)
- **Classe 0**: Notícias gerais (casos negativos)

O dataset inicial continha aproximadamente 1.050 casos de fraude, necessitando de balanceamento com casos negativos para treinamento adequado.

## Questão Fundamental: Pré-processamento vs Modelos Pré-treinados

### Análise Crítica
Modelos contextuais (BERT, BERTimbau, Albertina) foram treinados COM acentos, maiúsculas/minúsculas e pontuação. Usar texto pré-processado (sem acentos, lowercase) pode degradar a performance desses modelos.

Modelos clássicos (TF-IDF, Word2Vec, FastText) funcionam bem com texto pré-processado.

### Solução Implementada
Foram criadas DUAS versões dos dados:
- **Texto pré-processado** → para TF-IDF, Word2Vec, FastText, GloVe
- **Texto original (sem pré-processamento)** → para BERT e variantes

## Etapas Concluídas

### 1. Preparação e Balanceamento de Datasets

**Datasets Originais:**
- `DF_COMPANIES_REDUZIDO.csv`: Casos positivos de fraudes
- `NEGATIVE_DATASET_STRATIFIED.csv`: Casos negativos estratificados

**Processo Realizado:**
- Combinação dos datasets positivos e negativos
- Divisão estratificada 80/20 (treino/teste)
- Criação de múltiplas versões para diferentes abordagens

### 2. Estratégia Dupla de Pré-processamento

**Versão 1 - Modelos Clássicos (Texto Pré-processado):**
- Remoção de HTML tags, URLs, pontuação
- Normalização de espaços e conversão para lowercase
- Remoção de acentos e stopwords
- Arquivos: `TRAIN_PREPROCESSED.csv`, `TEST_PREPROCESSED.csv`

**Versão 2 - Modelos Contextuais (Texto Original):**
- Remoção apenas de URLs e normalização de espaços
- Preservação de acentos, maiúsculas/minúsculas e pontuação
- Arquivos: `TRAIN_BERT.csv`, `TEST_BERT.csv`

## Tabela Comparativa de Técnicas de Vetorização

| Técnica | Tipo | Aplicável? | Usar Texto Pré-processado? | Complexidade | Prioridade | Observações |
|---------|------|------------|----------------------------|-------------|-----------|-------------|
| TF-IDF | Esparso | Sim | Sim | Baixa | Alta | Baseline clássico, rápido, interpretável |
| PMI | Esparso | Sim | Sim | Média | Média | Alternativa ao TF-IDF para matriz termo-contexto |
| LSA | Esparso + Redução | Sim | Sim | Média | Baixa | Redução dimensional de TF-IDF via SVD |
| Word2Vec | Denso Estático | Sim | Sim | Média | Alta | Treinar próprio ou usar pré-treinado PT |
| FastText | Denso Estático | Sim | Sim | Média | Alta | Melhor que Word2Vec para PT (subpalavras) |
| GloVe | Denso Estático | Sim | Sim | Média | Média | Menos comum para PT, Word2Vec preferível |
| BERTPT | Denso Contextual | Sim | Não | Alta | Alta | Textos informais, manter acentos/maiúsculas |
| BERTimbau | Denso Contextual | Sim | Não | Alta | Muito Alta | Melhor para PT-BR, 2 versões (base/large) |
| Albertina PT-BR | Denso Contextual | Sim | Não | Alta | Muito Alta | Arquitetura DeBERTa, estado-da-arte PT-BR |
| RoBERTa | Denso Contextual | Depende | Não | Alta | Baixa | Verificar se há versão PT disponível |
| DistilBERT | Denso Contextual | Depende | Não | Média | Média | Versão leve do BERT, verificar PT |
| ELECTRA | Denso Contextual | Depende | Não | Alta | Baixa | Verificar disponibilidade PT |

## Estrutura de Datasets Criada

| Arquivo | Uso | Características | Modelos |
|---------|-----|-----------------|---------|
| TRAIN_PREPROCESSED.csv | Treino | Sem acentos, lowercase, sem pontuação, sem stopwords | TF-IDF, Word2Vec, FastText, GloVe |
| TEST_PREPROCESSED.csv | Teste | Sem acentos, lowercase, sem pontuação, sem stopwords | TF-IDF, Word2Vec, FastText, GloVe |
| TRAIN_BERT.csv | Treino | COM acentos, maiúsculas, pontuação | BERTimbau, Albertina, BERTPT |
| TEST_BERT.csv | Teste | COM acentos, maiúsculas, pontuação | BERTimbau, Albertina, BERTPT |

## Modelos Contextuais Detalhados

### BERTPT
- **Treinamento:** 4.8GB de textos, 992 milhões de tokens
- **Fontes:** Wikipedia-PT, EuroParl, Open Subtitles
- **Características:** Textos formais e informais, mantém acentos/maiúsculas
- **Performance:** Melhor em textos informais

### BERTimbau
- **Treinamento:** brWaC, 2.68 bilhões de tokens, 3.53 milhões de documentos
- **Processamento:** 17.5GB após pré-processamento
- **Versões:** Base e Large
- **Características:** Específico para português brasileiro

### Albertina
- **Treinamento:** brWaC (PT-BR), 2.2 bilhões de tokens, 8 milhões de documentos
- **Arquitetura:** DeBERTa (avançada vs Transformer)
- **Inovações:** Dois vetores processados separadamente (embeddings + posição)
- **Versões:** Base (100M parâmetros), Large (900M parâmetros)

## Outras Arquiteturas Transformadoras

### ROBERTa (Liu et al., 2019)
- **Modificações:** Mudanças no treinamento, tokenização BPE vs WordPiece
- **Características:** Melhor robustez em tarefas de compreensão

### DistilBERT (Sanh et al., 2019)
- **Técnica:** Destilação de conhecimento
- **Vantagem:** Modelo compacto mantendo performance do BERT original

### AlBERT (Lan et al., 2020)
- **Mecanismos:** Fatorização de embeddings, compartilhamento de pesos
- **Benefício:** Modelo mais eficiente que BERT

### ELECTRA (Clark et al., 2020)
- **Inovação:** Mudança na forma de treinamento
- **Resultado:** Embeddings melhores com processo mais eficiente

## Implementações Realizadas

### 3.1 TF-IDF (Baseline Clássico)
- **Configuração:** n-grams (1,2), min_df=2, max_df=0.9, max_features=10000, norm='l2'
- **Resultado:** Matriz esparsa (1410 × 10000)
- **Coverage:** Termo 'fraude' preservado com TF-IDF médio de 0.0175
- **Memória:** ~3 MB

### 3.2 FastText (Embeddings Pré-treinados)
- **Fonte:** Facebook FastText cc.pt.300d
- **Configuração:** 300 dimensões, média ponderada por TF-IDF
- **Resultado:** Matriz densa (1410 × 300)
- **Coverage:** 100% (nenhum embedding nulo)
- **Memória:** ~3.4 MB

### 3.3 BERT NeuralMind (Embeddings Contextuais)
- **Modelos:** Base (110M parâmetros) e Large (335M parâmetros)
- **Configuração:** Token [CLS], 512 tokens máximos
- **Resultados:** 
  - Base: (1410 × 768), ~4.3 MB
  - Large: (1410 × 1024), ~5.8 MB
- **Similaridade entre classes:** 0.9722

### 3.4 Albertina PORTULAN (State-of-the-Art)
- **Modelos:** Base (138M parâmetros) e Large (884M parâmetros)
- **Arquitetura:** DeBERTa (avançada vs Transformer)
- **Resultados:**
  - Base: (1410 × 768), ~4.3 MB
  - Large: (1410 × 1536), ~8.7 MB
- **Similaridade entre classes:** 0.9790

## Estrutura de Arquivos de Vetorização

```
/vectorization/
├── tf_idf/                    # TF-IDF baseline
├── fasttext/                  # Embeddings pré-treinados
│   ├── models/               # cc.pt.300.vec (4.5 GB)
│   └── outputs/              # Matrizes processadas
├── bert_base/                 # BERT-Base NeuralMind
├── bert_large/                # BERT-Large NeuralMind
├── albertina_base/            # Albertina-Base PORTULAN
└── albertina_large/           # Albertina-Large PORTULAN
```

## Comparação Final de Abordagens Implementadas

| Abordagem | Dimensões | Tipo | Memória | Arquitetura | Similaridade Classes |
|-----------|-----------|------|---------|------------|---------------------|
| TF-IDF | 10.000 | Esparso | ~3 MB | N/A | N/A |
| FastText | 300 | Denso | ~3.4 MB | Skip-gram | N/A |
| BERT-Base | 768 | Denso | ~4.3 MB | Transformer | 0.9722 |
| BERT-Large | 1024 | Denso | ~5.8 MB | Transformer | 0.9722 |
| Albertina-Base | 768 | Denso | ~4.3 MB | DeBERTa | 0.9790 |
| Albertina-Large | 1536 | Denso | ~8.7 MB | DeBERTa | 0.9790 |

## Resultados Detalhados dos Modelos

### BERT NeuralMind
| Modelo | Dimensões | Tamanho | Shape | Parâmetros |
|--------|-----------|---------|-------|------------|
| BERT-Base | 768 | 4.3 MB | (1410, 768) | 110M |
| BERT-Large | 1024 | 5.8 MB | (1410, 1024) | 335M |

### Albertina PORTULAN
| Modelo | Dimensões | Tamanho | Shape | Parâmetros |
|--------|-----------|---------|-------|------------|
| Albertina-Base | 768 | 4.3 MB | (1410, 768) | 138M |
| Albertina-Large | 1536 | 8.7 MB | (1410, 1536) | 884M |

## Vantagens Albertina vs BERT

- **Arquitetura mais avançada:** DeBERTa vs Transformer
- **Licença mais permissiva:** MIT (Base) vs Restritiva
- **Especificamente PT-BR:** Treinada para português brasileiro
- **Estado-da-arte:** Performance superior em tarefas PT

## Insights Importantes

### 1. Similaridade entre Classes
A alta similaridade (0.9722-0.9790) entre classes positivas e negativas nos modelos contextuais indica:
- Documentos semanticamente muito similares
- Necessidade de classificadores sofisticados
- Possível benefício de fine-tuning vs embeddings estáticos

### 2. Vantagens Albertina vs BERT
- **Arquitetura superior:** DeBERTa vs Transformer
- **Licença mais permissiva:** MIT vs restritiva
- **Especificidade PT-BR:** Treinada para português brasileiro
- **Performance:** Estado-da-arte em tarefas português

### 3. Estratégia de Pré-processamento
A decisão de manter duas versões dos dados foi fundamental:
- Modelos clássicos: Beneficiam de texto limpo e normalizado
- Modelos contextuais: Requerem texto original para máximo aproveitamento

## Estratégia Recomendada para o Projeto

### Fase 1: Modelos Clássicos (Baseline)
Usar texto pré-processado atual:
- TF-IDF (uni-grams, bi-grams, tri-grams)
- Word2Vec pré-treinado para português
- FastText pré-treinado para português

### Fase 2: Modelos Contextuais
Criar versão SEM pré-processamento:
- BERTimbau-base (prioridade máxima para PT-BR)
- Albertina PT-BR (estado-da-arte)
- BERTPT (se tempo permitir)

## Contribuições Técnicas

### Scripts Desenvolvidos
- `text_preprocessing.py`: Pré-processamento completo para modelos clássicos
- `light_preprocessing_for_bert.py`: Pré-processamento leve para modelos contextuais
- `tfidf_vectorization.py`: Implementação TF-IDF com análise de features
- `download_fasttext_embeddings.py`: Download automático de embeddings
- `word2vec_fasttext_vectorization.py`: Vetorização FastText ponderada
- `bert_vectorization.py`: Extração de embeddings BERT
- `albertina_vectorization.py`: Extração de embeddings Albertina

### Inovações Metodológicas
- **Estratégia dual de pré-processamento** para otimizar diferentes tipos de modelos
- **Análise de similaridade entre classes** para entender limitações dos embeddings
- **Ponderação TF-IDF em embeddings densos** para melhor representação semântica
- **Organização sistemática de resultados** para comparação justa entre abordagens

## Próximos Passos

Agora você tem 6 abordagens de vetorização completas:
- TF-IDF - Baseline clássico
- FastText - Embeddings pré-treinados
- BERT-Base - Contextual (110M parâmetros)
- BERT-Large - Contextual (335M parâmetros)
- Albertina-Base - Contextual avançado (138M parâmetros)
- Albertina-Large - Contextual avançado (884M parâmetros)

### Imediatos
1. **Treinar Classificadores:** Implementar e treinar múltiplos classificadores (Naive Bayes, SVM, Random Forest, Logistic Regression) para cada abordagem de vetorização
2. **Validação Cruzada:** Avaliar performance com validação cruzada robusta
3. **Comparação de Resultados:** Identificar qual abordagem oferece melhor performance para o problema específico

### Médio Prazo
1. **Avaliação em Conjunto de Teste:** Testar os melhores modelos no conjunto de teste reservado
2. **Otimização de Hiperparâmetros:** Fine-tuning dos melhores modelos
3. **Word2Vec do Zero:** Implementar treinamento próprio Word2Vec para comparação

### Longo Prazo
1. **Fine-tuning de Modelos Contextuais:** Treinar BERT/Albertina especificamente para o domínio
2. **Ensembles:** Combinar múltiplas abordagens para performance superior
3. **Deploy:** Preparar sistema para produção

## Impacto Científico

Este projeto estabelece uma base metodológica robusta para classificação de textos em português brasileiro, especificamente para detecção de fraudes corporativas. A comparação sistemática entre abordagens clássicas e modernas de PLN fornece insights valiosos sobre:

1. **Eficácia relativa** de diferentes técnicas de vetorização
2. **Importância do pré-processamento** adequado para cada abordagem
3. **Trade-offs** entre performance, complexidade e recursos computacionais
4. **Potencial de modelos contextuais** para domínios especializados

## Conclusão

O projeto alcançou sucesso completo na implementação de múltiplas abordagens de vetorização, estabelecendo uma base sólida para o desenvolvimento de um sistema eficaz de detecção de fraudes. A metodologia sistemática empregada permite comparações justas e insights profundos sobre as características de cada técnica.

A alta similaridade semântica entre classes sugere que o problema é desafiador e provavelmente requererá abordagens mais sofisticadas ou combinação de múltiplas técnicas para alcançar performance superior. No entanto, a base técnica estabelecida proporciona o ponto de partida ideal para essas investigações futuras.

## Técnicas Analisadas do Livro PLN Brasil

### Técnicas Clássicas (Status: Implementadas/Pendentes)
- **TF-IDF:** Atribuição de pesos termo-documento ✅ Implementado
- **PMI:** Atribuição de pesos termo-contexto ⏳ Pendente
- **LSA:** Redução dimensional via SVD ⏳ Pendente

### Vetores Estáticos (Status: Implementadas/Pendentes)
- **Word2Vec:** Embeddings densos estáticos ✅ Implementado (via FastText)
- **FastText:** Embeddings com subpalavras ✅ Implementado
- **GloVe:** Embeddings baseados em co-ocorrência ⏳ Pendente

### Vetores Contextuais (Status: Implementadas)
- **BERTPT:** Treinado em textos formais e informais ✅ Implementado
- **BERTimbau:** Otimizado para português brasileiro ✅ Implementado
- **Albertina:** Arquitetura DeBERTa, estado-da-arte ✅ Implementado

### Outras Arquiteturas Mencionadas (Status: Análise)
- **RoBERTa:** Modificações no treinamento, tokenização BPE 🔍 Analisado
- **DistilBERT:** Destilação de conhecimento, modelo compacto 🔍 Analisado
- **AlBERT:** Eficiência via fatorização e compartilhamento 🔍 Analisado
- **ELECTRA:** Treinamento eficiente para melhores embeddings 🔍 Analisado


*****************************************************************************************************************

Implementing and Documenting ... (chat)
"
Em um livro sobre PLN (brasileiras em processamento de linguagem natural), encontrei as seguintes técnicas (que embora eu seja iniciante, penso que podem ser utilizadas para vetorização). Entendo o meu contexto, e as informações a seguir todas elas podem ser aplicadas / testadas?

Observe que tem algumas que foram inclusive treinadas com pontuações e ascentos, ente outras características que foram mantidas. Será que eu ter 'limpado' meu conjunto de texto para treino e utilzar essas técnicas mencionadas que mantiveram as tecnicas mencionadas no treino não pode influenciar de alguma forma? Ou posso tentar assim mesmo, com meu preprocessamento aplicado. 

Enfim, faça uma organização de todas as técnicas apresentadas a seguir e quais podem ser utilizadas no meu trabalho (se necessário faça uma tabela para organiazção).

10.2.1 Atribuindo pesos aos termos da matriz termo-documento com TF-IDF199
10.2.2 Atribuindo pesos aos termos da matriz termo-contexto com PMI . . 200
10.2.3 Reduzindo a dimensionalidade com LSA . . . . . . . . . . . . . . . . 202
10.3 Vetores densos estáticos . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 204
10.3.1 Word2Vec . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 206
10.3.2 Fasttext . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 209
10.3.3 Glove . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

***

Vetores densos contextuais ou dinâmicos:

 ... Temos alguns modelos treinados especificamente para o português (outras línguas também, mas nosso interesse aqui é na nossa linda língua materna). O modelo BERTPT (Feijó; Moreira, 2020) foi treinado com um vocabulário de tamanho 30.000, assim como o modelo BERT original, porém mantendo a configuração original de maiúsculas e minúsculas e sinais diacríticos (os acentos). Foram usados 4,8GB de textos, considerando textos em português do Brasil e europeu, tanto mais formais, como Wikipedia-PT26 e EuroParl27, como textos mais informais, como Open Subtitles28. No total, foram considerados 992 milhões de tokens. A arquitetura utilizada foi a base. O modelo BERTPT apresentou resultados melhores em bases de dados compostas por textos mais informais.

...

O modelo BERTimbau29 (Souza et al., 2020) também partiu da arquitetura do BERT, mas treinou duas versões, uma a partir da arquitetura base e outra a partir da arquitetura large. Assim como no BERTPT, são mantidas as letras maiúsculas e minúsculas e acentos e o tamanho do vocabulário também é de 30.000 tokens. O conjunto de textos usados para treinar os modelos foi o brWaC (Wagner Filho et al., 2018), que é composto de textos em português do Brasil, contendo 2,68 bilhões de tokens e 3,53 milhões de documentos, e após uma fase de pré-processamento ficou com 17,5GB de textos.

...

 Albertina (Rodrigues et al., 2023), treinado em duas variantes, português europeu (Albertina PT-PT) e português do Brasil (Albertina PT-BR). A versão PT-BR também foi treinada com o brWaC. Já a versão PT-PT foi treinada com um subconjunto de textos em português extraídos da versão de Janeiro de 2023 do corpus Oscar (Abadji et al., 2022) e de outros três corpora constituídos de documentos do parlamento europeu e português. No total, foram utilizados oito milhões de documentos contendo 2,2 bilhões de tokens.

Uma diferença crucial do Albertina para os modelos anteriores é que a arquitetura base não é a do BERT, mas sim uma versão estendida com duas novas técnicas, chamada DeBERTa (do inglês, Decoding enhanced BERT with disentangled attention) (He et al., 2021). A primeira modificação diz respeito ao mecanismo de atenção. Lembre que nos Transformers, um token é representado pela soma do seu vetor inicial de embeddings e do seu vetor de codificação de posição. No DeBERTa, e consequentemente no Albertina, temos dois vetores que são processados separadamente...

....

Existem diversas outras arquiteturas que estendem, melhoram, modificam, ou treinam com mais dados ou com outros parâmetros o componente codificador dos Transformers. Exemplos incluem ROBERTa (Liu et al., 2019), que incluiu modificações no treinamento e usa o algoritmo de tokenização BPE ao invés do WordPiece; DistillBERT (Sanh et al., 2019), que se vale de um processo de destilação de conhecimento para aproximar os pesos do modelo original e obter um modelo menor que o BERT; AlBERT (Lan et al., 2020), que introduz três mecanismos – fatorização das matrizes de embeddings, compartilhamento de pesos e uma nova forma de treinamento – para obter um modelo mais eficiente que o BERT; ELECTRA (Clark et al., 2020), que também muda a forma de treinamento do BERT para obter embeddings melhores com um processo mais eficiente; dentre muitos outros.

....
"