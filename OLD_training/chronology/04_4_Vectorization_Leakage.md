
## ⚠️ Nota de possível Data Leakage (identificado posteriormente) em @04_Vectorization.md

Na abordagem anterior, a vetorização TF-IDF foi feita **antes** do split treino/dev:

1. TRAIN.csv (1.410 exemplos) → TfidfVectorizer.fit_transform() nos 1.410 inteiros
2. Depois, nos scripts de treino → train_test_split divide os 1.410 em 1.128 treino + 282 dev
3. O vectorizer foi **fitado em 282 exemplos que viraram dev** → o vocabulário e os pesos IDF "viram" o dev

**Impacto:** O modelo teve acesso indireto às palavras e frequências do conjunto de dev durante a vetorização. Isso pode inflar levemente as métricas de avaliação no dev, pois o IDF já "conhece" os termos do dev.

**Para TF-IDF e FastText** (que usa pesos TF-IDF): o vectorizer deveria ser fitado **apenas no treino**, e depois transform no dev/test.

**Para BERT e Albertina:** não há problema, pois são modelos pré-treinados que não fitam nos dados (apenas extraem embeddings).

**Correção no novo pipeline (NEW_training):** o split treino/dev é feito **antes** da vetorização (Task 2 → Task 4). O vectorizer é fitado apenas no treino, e dev/test recebem apenas transform().
