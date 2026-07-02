# 📰 Collector Notícias - Clean Version

Sistema limpo e organizado para scraping de notícias de portais brasileiros.

## 📁 Estrutura

```
collector_noticias/
├── main.py                  # Collector principal (main-server version)
├── tools.py                 # Ferramentas de parsing
├── crawler_configs.json     # Configurações de portais
├── requirements.txt         # Dependências Python
├── scrape_bbc.py           # Scraper standalone para BBC Portuguese
├── scraped/                # Notícias scrapadas
│   ├── bbcportuguese/      # BBC Portuguese (26 JSONs)
│   ├── jornalconexao/      # Jornal Conexão (40 JSONs)
│   ├── olharsc/            # Olhar SC (100 JSONs)
│   └── iclnoticias/        # ICL Notícias (10 JSONs)
└── README.md               # Este arquivo
```

## 🚀 Uso Rápido

### **Scraping BBC Portuguese**

```bash
# Instalar dependências
pip install requests beautifulsoup4

# Scrappear TODAS as páginas disponíveis (auto-detect com busca binária)
python3 scrape_bbc.py

# Scrappear número específico de páginas (desabilita auto-detect)
python3 scrape_bbc.py --max-pages 50 --no-auto-detect

# Continuar de uma página específica
python3 scrape_bbc.py --start-page 10
```

**Características**:
- ✅ **Busca binária automática** para encontrar número máximo de páginas
- ✅ Detecta artigos já scrapados (não duplica)
- ✅ Rate limiting automático (0.5s entre artigos, 1s entre páginas)
- ✅ Salva incrementalmente
- ✅ Resumo ao final

**Algoritmo de Busca Binária**:
1. **Busca exponencial**: Testa páginas 1, 2, 4, 8, 16... até encontrar página sem artigos
2. **Busca binária**: Refina o intervalo para encontrar a última página com artigos
3. **Resultado**: Número exato de páginas disponíveis no site

---

## 📊 Dados Disponíveis

| Portal | Artigos | Descrição |
|--------|---------|-----------|
| **bbcportuguese** | 26 | Notícias gerais internacionais (BBC) |
| **olharsc** | 100 | Notícias gerais locais SC |
| **jornalconexao** | 40 | Notícias gerais locais SC |
| **iclnoticias** | 10 | Notícias locais SC |
| **TOTAL** | **176** | Notícias scrapadas |

---

## 🔧 Configuração de Portais

O arquivo `crawler_configs.json` contém configurações para 10 portais:
- ndmais
- nsc
- jornalconexao
- olharsc
- tvbv
- portaldelicitacao
- ocpnews
- iclnoticias
- g1sc
- **bbcportuguese** (novo)

---

## 📝 Formato dos JSONs

Cada notícia é salva como:

```json
{
  "url": "https://www.bbc.com/portuguese/articles/...",
  "title": "Título da notícia",
  "text": "Conteúdo completo da notícia...",
  "date_publication": "2026-05-14",
  "portal": "bbcportuguese",
  "scraped_at": "2026-05-14T14:30:00"
}
```

---

## 🎯 Próximos Passos

1. **Scrappear BBC completo**: `python3 scrape_bbc.py`
2. **Filtrar candidatos negativos**: Remover notícias com palavras-chave de fraude
3. **Amostragem estratificada**: Criar dataset balanceado de 1,050 notícias

---

## 📦 Dependências

```bash
pip install -r requirements.txt
```

Principais:
- `requests` - HTTP requests
- `beautifulsoup4` - HTML parsing
- `lxml` - XML/HTML parser

---

## 🐛 Troubleshooting

### Erro: "No articles found"
- Verifique conexão com internet
- BBC pode ter mudado estrutura HTML
- Tente com `--start-page 1`

### Erro: "Permission denied"
- Verifique permissões da pasta `scraped/`
- Execute: `chmod -R 755 scraped/`

---

## 📄 Licença

Parte do projeto de pesquisa sobre detecção de fraudes em licitações públicas de SC.
