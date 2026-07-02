# Companies Linkage - Vinculação de Empresas

**Data de Criação:** 8 de Junho de 2026  
**Objetivo:** Sistema de vinculação automática entre empresas mencionadas em notícias de fraude e dataset oficial de estabelecimentos

---

## 📁 Estrutura da Pasta

```
companies_linkage/
├── README.md                          # Este arquivo
├── requirements_matching.txt          # Dependências Python
│
├── scripts/                           # Scripts de processamento
│   ├── match_companies_tfidf.py       # Script principal de matching
│   └── analyze_matching_results.py   # Análise comparativa de resultados
│
├── docs/                              # Documentação
│   └── 07_Company_Linking.md          # Documentação completa da metodologia
│
└── results/                           # Resultados gerados
    └── linked_companies/              # Outputs do matching
        ├── company_matches_tfidf_*.json    # Resultados TF-IDF
        ├── company_matches_fuzzy_*.json    # Resultados Fuzzy
        ├── company_matches_hybrid_*.json   # Resultados Híbrido
        ├── company_matches_*.csv           # CSVs para análise
        └── stats_*.txt                     # Estatísticas por método
```

---

## 🎯 Descrição

Este módulo implementa três abordagens de matching textual para vincular empresas mencionadas em notícias com o dataset oficial de estabelecimentos (193k empresas):

1. **TF-IDF + Cosine Similarity** - Matching semântico via character n-grams
2. **Fuzzy String Matching** - Similaridade Levenshtein (Token Sort Ratio)
3. **Híbrido** - Combinação ponderada (60% TF-IDF + 40% Fuzzy)

---

## 🚀 Como Usar

### **1. Instalar Dependências**
```bash
pip install -r requirements_matching.txt
```

### **2. Executar Matching**
```bash
# Ativar ambiente virtual
source .venv/bin/activate

# Executar matching (testa os 3 métodos)
python3 scripts/match_companies_tfidf.py
```

### **3. Analisar Resultados**
```bash
# Gera comparação entre métodos
python3 scripts/analyze_matching_results.py
```

---

## 📊 Datasets

### **Input**
- **Notícias:** `test/dataset/TEST_CLEAN_NO_DUPLICATES_WITH_MATCHES.csv`
  - Coluna: `empresa(s)` - Empresas mencionadas
  - Total: 175 notícias, 343 menções

- **Estabelecimentos:** `dataset/raw_companies_hmg/estabelecimento.csv`
  - Coluna: `nome_fantasia` - Nome fantasia da empresa
  - Total: 193.032 estabelecimentos

### **Output**
- **JSON:** Resultados completos com metadados
- **CSV:** Formato tabular para análise
- **TXT:** Estatísticas resumidas

---

## 📈 Resultados

**Taxa de Sucesso:** ~98.3%  
**Matches Exatos (score > 0.9):** 71 empresas  
**Empresas Vinculadas:** 337/343  
**Não Encontradas:** 6

### **Exemplos de Matches Perfeitos**
- "Companhia Catarinense de Águas e Saneamento (Casan)" → Score: 1.000
- "Qualidade Mineração" → Score: 1.000
- "Lojas Americanas" → Score: 1.000
- "Consigaz" → Score: 1.000

---

## 🔧 Configurações

Parâmetros principais em `match_companies_tfidf.py`:

```python
# TF-IDF
TFIDF_NGRAM_RANGE = (2, 4)      # Character n-grams
COSINE_THRESHOLD = 0.3          # Similaridade mínima

# Fuzzy
FUZZY_THRESHOLD = 70            # Score mínimo (0-100)

# Híbrido
TFIDF_WEIGHT = 0.6              # Peso TF-IDF
FUZZY_WEIGHT = 0.4              # Peso Fuzzy
```

---

## 📚 Documentação Completa

Ver `docs/07_Company_Linking.md` para:
- Metodologia detalhada
- Exemplos de código
- Análise de casos problemáticos
- Próximos passos (LLM, microserviço, etc.)

---

## 🚀 Próximos Passos

### **Curto Prazo**
- [ ] Implementar desambiguação via LLM
- [ ] Adicionar filtros geográficos (município)
- [ ] Adicionar filtros setoriais (CNAE)

### **Médio Prazo**
- [ ] Substituir CSV por PostgreSQL
- [ ] Implementar RabbitMQ
- [ ] Dockerizar como microserviço
- [ ] Integrar com main-server

---

## 👤 Autor

**Cascade AI Assistant**  
Sessão: 8 de Junho de 2026
