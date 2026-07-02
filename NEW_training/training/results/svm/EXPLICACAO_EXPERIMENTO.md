# Relatório do Experimento: SVM
## 1. Objetivo
Avaliar o impacto de diferentes técnicas de vetorização na performance do classificador SVM para detecção de fraudes em notícias. Dataset desbalanceado (1,53% positivos).
## 2. Metodologia
### 2.1 Divisão dos Dados
- **Treino:** 36.652 exemplos (562 positivos, 36.090 negativos)
- **Desenvolvimento:** 9.163 exemplos (141 positivos, 9.022 negativos)
- **Teste:** 11.454 exemplos (isolado, não usado nesta fase)
### 2.2 Classificador
SVM (Support Vector Machine) encontra o hiperplano ótimo que melhor separa as classes.
LinearSVC: kernel linear, mais eficiente para alta dimensionalidade.
class_weight='balanced': ajusta pesos inversamente proporcionais à frequência das classes.
decision_function fornece scores para PR-AUC e ROC-AUC.
**Configuração:**
```python
LinearSVC(class_weight='balanced', dual='auto', max_iter=10000, random_state=42)
Grid: C=[0.1, 1, 10]
CV: StratifiedKFold(5, shuffle=True, random_state=42)
Scoring: f1
```
### 2.3 Protocolo
1. GridSearchCV com stratified 5-fold no treino (F1 como scoring)
2. Retreina com melhores hiperparâmetros no treino inteiro
3. Avalia no dev set
4. Teste permanece isolado (Task 7)
## 3. Resultados
### 3.1 Tabela Comparativa (Dev Set)
| Vetorização | CV F1 | F1 | Precision | Recall | PR-AUC | ROC-AUC | Accuracy |
|-------------|-------|----|-----------|--------|--------|---------|----------|
| tfidf | 0.7213 | 0.7075 | 0.6797 | 0.7376 | 0.7665 | 0.9843 | 0.9906 |
| fasttext | 0.3234 | 0.3112 | 0.1875 | 0.9149 | 0.5662 | 0.9823 | 0.9377 |
| bert_base | 0.6090 | 0.5994 | 0.5049 | 0.7376 | 0.6654 | 0.9794 | 0.9848 |
| bert_large | 0.6492 | 0.5831 | 0.5584 | 0.6099 | 0.6700 | 0.9693 | 0.9866 |
| albertina_base | 0.4593 | 0.4363 | 0.2997 | 0.8014 | 0.6965 | 0.9641 | 0.9681 |
| albertina_large | 0.6309 | 0.6190 | 0.5333 | 0.7376 | 0.7328 | 0.9741 | 0.9860 |

**Melhor Vetorização: tfidf (F1: 0.7075)**
### 3.2 Melhores Hiperparâmetros por Vetorização
| Vetorização | Best Params |
|-------------|-------------|
| tfidf | {"C": 1} |
| fasttext | {"C": 10} |
| bert_base | {"C": 1} |
| bert_large | {"C": 10} |
| albertina_base | {"C": 10} |
| albertina_large | {"C": 0.1} |

## 4. Análise
- **Melhor combinação:** tfidf + SVM
- **F1 no dev:** 0.7075
- **Precision:** 0.6797
- **Recall:** 0.7376
- **CV F1 (estimativa de generalização):** 0.7213
