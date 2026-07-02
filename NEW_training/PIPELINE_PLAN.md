# Pipeline Desbalanceado — Plano de Tarefas

## Contexto

Professor questionou o treino com dataset 50/50 balanceado: na realidade, a maioria das notícias não é fraude. Objetivo: retreinar com dados desbalanceados (~1,5% positivos vs ~98,5% negativos).

## Dados Disponíveis

### Positivos (fraude = 1)
- **Arquivo:** `/home/paulo/CascadeProjects/Applied_ML/dataset/cleaned_data_no_bias/POSITIVE_DF_COMPANIES_REDUZIDO.csv`
- **Linhas:** 881
- **Colunas:** `Unnamed: 0, url, Unnamed: 2, title, Unnamed: 4, text, Unnamed: 6, empresa(s), Unnamed: 8, pessoa(s), Unnamed: 10, fraude, Unnamed: 12, date_publication`
- **Label implícito:** todos são fraude (adicionar coluna `label=1`)

### Negativos (não-fraude = 0)
- **Arquivo:** `/home/paulo/CascadeProjects/Applied_ML/NEW_training/NEGATIVES_CONSOLIDATED.csv`
- **Linhas:** 56.390
- **Colunas:** `url, title, text, label, portal`
- **Labels:** `0` (55.595 rows — ndmais + nsctotal) | `negative` (795 rows — BBC, Gazeta, CartaCapital, etc.)
- **Portais:** ndmais.com.br (45.036), www.nsctotal.com.br (10.559), www.bbc.com (382), www.gazetadopovo.com.br (155), www.cartacapital.com.br (138), demais (120)

### Negativos já limpos (apenas label=0, ndmais+nsctotal)
- **Arquivo:** `/home/paulo/CascadeProjects/Applied_ML/NEW_training/new_dataset/negatives_SAFE_CLEAN.csv`
- **Linhas:** 55.595
- **Script de extração:** `/home/paulo/CascadeProjects/Applied_ML/NEW_training/new_dataset/extract_fraud_negatives_SAFE.py`

### Pipeline original (referência)
- **Flow diagram:** `/home/paulo/CascadeProjects/Applied_ML/FLOW_DIAGRAM.md`
- **Scripts de pré-processamento:** `/home/paulo/CascadeProjects/Applied_ML/processors/overall/`
- **Scripts de vetorização:** `/home/paulo/CascadeProjects/Applied_ML/scripts/`
- **Scripts de treino:** `/home/paulo/CascadeProjects/Applied_ML/training/`
- **Scripts de teste:** `/home/paulo/CascadeProjects/Applied_ML/test/scripts/`
- **Dataset de treino atual (balanceado):** `/home/paulo/CascadeProjects/Applied_ML/dataset/cleaned_data_no_bias/FOR_TRAINING/`

## Tasks

### Task 1 — Padronização e Consolidação
- Normalizar positivo: selecionar `url, title, text, date_publication`, adicionar `label=1`, extrair `portal` da URL
- Normalizar negativo: padronizar `label` para `0` (unificar `0` e `negative`), adicionar `date_publication` se possível
- Consolidar em `/home/paulo/CascadeProjects/Applied_ML/NEW_training/FOR_TRAINING/CONSOLIDATED_IMBALANCED.csv`
- Schema final: `url, title, text, label, portal, date_publication`
- Gerar relatório de distribuição

### Task 2 — Split Estratificado 80/20
- Split por `label` + `portal` (mantendo proporção ~1,5%:98,5%)
- Treino+Dev (80%) → dividir em Treino (80%) / Dev (20%)
- Teste (20%)
- Deduplicação por URL entre splits
- Salvar: `train.csv`, `dev.csv`, `test.csv` em `NEW_training/FOR_TRAINING/`

### Task 3 — Pré-processamento Dual
- **Pesado** (sem acentos, lowercase, sem stopwords) → `*_PREPROCESSED.csv` (para TF-IDF, FastText)
- **Leve** (preserva acentos/maiúsculas) → `*_BERT.csv` (para BERT, Albertina)
- Reaproveitar lógica de `processors/overall/` e `scripts/light_preprocessing_for_bert.py`

### Task 4 — Vetorização (6 técnicas)
- TF-IDF (n-grams 1-2, 10k features)
- FastText (cc.pt.300d, média ponderada TF-IDF)
- BERT-Base (NeuralMind, 768 dim)
- BERT-Large (NeuralMind, 1.024 dim)
- Albertina-Base (PORTULAN, 768 dim)
- Albertina-Large (PORTULAN, 1.536 dim)
- Salvar em `NEW_training/vectorization/`

### Task 5 — Treinamento (18 combinações)
- 3 classificadores × 6 vetorizações = 18
- **Ajuste anti-desbalanceamento:** `class_weight='balanced'` (SVM, RF), `ComplementNB` ou `class_prior` (NB)
- 5-fold CV estratificado no treino
- Avaliação no Dev: F1, Precision, Recall, ROC-AUC, PR-AUC
- Sem tuning (requisito acadêmico)
- Salvar em `NEW_training/training/results/`

### Task 6 — Análise e Seleção
- Consolidar 18 resultados
- Critério: F1 no Dev (considerar PR-AUC — mais informativo em desbalanceamento)
- Comparar com pipeline anterior (balanceado)
- Gerar `NEW_training/training/results/CONSOLIDACAO_FINAL.md`

### Task 7 — Avaliação Final no Teste
- Vetorizar teste com técnica vencedora
- Avaliar: F1, Precision, Recall, ROC-AUC, PR-AUC, matriz de confusão
- Verificar data leakage por URL
- Gap Dev→Teste
- Gerar `NEW_training/test/results/FINAL_TEST_REPORT.md`

### Task 8 — Comparação e Documentação
- Tabela: balanceado vs desbalanceado (F1, Precision, Recall, ROC-AUC)
- Análise de impacto por vetorização/classificador
- Atualizar `FLOW_DIAGRAM.md`
- Relatório para o professor

## Exclusão
- **Fase 10 (Vinculação de Empresas)** — excluída deste plano por enquanto
