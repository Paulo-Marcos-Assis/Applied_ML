# Training Methodology - Fraud Detection in News Articles

## Project Overview

The Applied_ML project is a complete Machine Learning system for automatic detection of corporate fraud in Brazilian Portuguese news articles. The objective is to automatically classify journalistic articles as fraud-related or not, using advanced Natural Language Processing techniques.

### Previous Phases Summary

**Phase 1: Initial Large-Scale Collection (Jan–Feb 2026)**
- 108,790 candidate articles from 5 portals screened via `gpt-oss:20b` (Ollama)
- 1,569 fraud-related cases detected; 845 with identified companies
- 270 articles manually added from curated governmental/judicial sources ("983 test set")
- Consolidation + deduplication → **1,050 articles** (`DF_COMPANIES_26_02.csv`, 16 portals)

**Phase 2: Complementary Collection (May 16–19, 2026)**
- 1,177/1,180 articles processed (99.7%) via `qwen3:8b` (Ollama)
- 80 frauds identified (26 corporate, 54 general)
- 530 hard negatives + 196 pure negatives categorized
- Entity extraction (companies, people, fraud types)

**Phase 3: Data Cleaning and Preparation**
- Critical bias identified: Master/Vorcaro (20.57% in CSV, up to 81.1% in JSONs)
- Bias reduction to 5.87% (85.6% average reduction across datasets)
- Enrichment: +47 fraud JSONs integrated into positive set
- Final dataset: 882 positive + 882 negative = **1,764 articles** (33 portals)
- Perfect 1:1 balance; +106% people diversity, +47% portal coverage

**Phase 4: Text Vectorization**
- Dual strategy implemented:
  - Classical models: Preprocessed text (no accents, lowercase)
  - Contextual models: Original text (with accents, uppercase)
- 6 vectorization techniques:
  - **Classical:** TF-IDF (10,000 dimensions), FastText (300 dimensions)
  - **Contextual:** BERT-Base (768 dim), BERT-Large (1,024 dim), Albertina-Base (768 dim), Albertina-Large (1,536 dim)
- High semantic similarity between classes (0.9722-0.9790) indicates challenging problem

---

## Training Phase Requirements

### Academic Requirements
The training phase must fulfill the following requirements:
1. **Establish and justify a performance baseline** based on literature or domain knowledge
2. **Train multiple models** (minimum 2-3 different algorithms)
3. **Iterate using data preparation techniques** to improve performance
4. **Report results clearly** with appropriate performance metrics
5. **No hyperparameter tuning or regularization** in this phase (establish honest baselines)

### Selected Models for Evaluation
- **Naive Bayes** (MultinomialNB for sparse data, GaussianNB for dense embeddings)
- **Support Vector Machine (SVM)** with linear kernel
- **Random Forest** with 100 estimators

### Baseline Strategy
**Baseline:** TF-IDF + Naive Bayes
- Justification: Most common baseline in NLP text classification literature
- Fast to train and interpret
- Establishes minimum expected performance
- Literature suggests F1-Score of 0.65-0.70 for similar fraud detection tasks

---

## Data Split Strategy

### Dataset Distribution

**Total Dataset:** 1,764 articles (882 positive, 882 negative - perfectly balanced)

**First Split: Training Pool vs Test Set**
- **Training + Development Pool (80%):** 1,410 articles
  - Positive: 705 articles
  - Negative: 705 articles
- **Test Set (20%):** 353 articles initial (isolated until final evaluation)
  - Initial: 176 positive, 177 negative
  - **After duplicate removal: 334 articles (175 positive, 159 negative)**
  - 19 duplicates removed (5.38% data leakage)
  - Location (clean): `/test/dataset/TEST_CLEAN_NO_DUPLICATES.csv`
  - Location (sparse/FastText): `/dataset/cleaned_data_no_bias/FOR_TRAINING/Pre_processed_for_Sparse/TEST_PREPROCESSED.csv`
  - Location (BERT/Albertina): `/dataset/cleaned_data_no_bias/FOR_TRAINING/Pre_processed_for_Embeddings/TEST_BERT.csv`

**Second Split: Training vs Development**
- **Training Set (80% of 1,410):** 1,128 articles
  - Positive: 564 articles
  - Negative: 564 articles
  - Usage: Model training + Cross-validation
- **Development Set (20% of 1,410):** 282 articles
  - Positive: 141 articles
  - Negative: 141 articles
  - Usage: Model selection and performance evaluation

### Cross-Validation Configuration
- **Type:** Stratified 5-Fold Cross-Validation
- **Applied to:** Training set only (1,128 articles)
- **Configuration per fold:**
  - Training: ~902 articles (80% of 1,128)
  - Validation: ~226 articles (20% of 1,128)
- **Characteristics:**
  - Maintains 50/50 class proportion in each fold
  - Shuffle: True with random_state=42 for reproducibility
  - Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - Result: Mean ± standard deviation across 5 iterations

### Complete Pipeline & Data Flow Diagram

```
════════════════════════════════════════════════════════════════════
FASE 1: Coleta Inicial (Jan–Fev 2026)
════════════════════════════════════════════════════════════════════
108,790 artigos (5 portais) → LLM gpt-oss:20b (Ollama)
    |
    +→ 1,569 fraudes detectadas
    +→ 845 artigos com empresas identificadas
    |
    +→ ADIÇÃO MANUAL: "983 test set"
    |   +→ 270 artigos de portais governamentais/jurídicos
    |       fraud_news_FROM_983_pt1.csv (89) + pt2.csv (181)
    |       Portais: MPSC, Polícia Civil, TCE-SC, outros
    |
    +→ Consolidação: 845 + 270 = 1,115 → deduplicação −65
    +→ 1,050 artigos → DF_COMPANIES_26_02.csv (16 portais)

════════════════════════════════════════════════════════════════════
FASE 2: Coleta Complementar (Maio 2026)
════════════════════════════════════════════════════════════════════
1,180 artigos (8 portais) → LLM qwen3:8b (Ollama)
    |
    +→ 1,177 processados (99.7%)
    +→ 80 fraudes (26 corporate + 54 general)  → JSONs fraud
    +→ 530 hard negatives + 196 pure negatives → JSONs neg

════════════════════════════════════════════════════════════════════
FASE 3: Limpeza de Bias (Master/Vorcaro)
════════════════════════════════════════════════════════════════════
1,050 (CSV) + 1,182 (JSONs pré-existentes) = 2,232 artigos
    |
    +→ CSV:   1,050 → 886  (−164, bias: 20.57% → 5.87%)
    +→ JSONs: 1,182 → 1,036 (−146 arquivos)
         Corporate Fraud:  37 →  8 (−29)
         General Fraud:    66 → 39 (−27)
         Hard Negatives:  700 → 653 (−47)
         Pure Negatives:  379 → 336 (−43)

════════════════════════════════════════════════════════════════════
FASE 4: Enriquecimento do Conjunto Positivo
════════════════════════════════════════════════════════════════════
886 (CSV limpo) + 47 (JSONs fraud limpos: 8 corp + 39 gen)
    |
    +→ Deduplicação por URL: −47 duplicatas
    +→ 882 artigos positivos únicos (22 portais)

════════════════════════════════════════════════════════════════════
FASE 5: Criação do Conjunto Negativo
════════════════════════════════════════════════════════════════════
989 disponíveis (653 hard negatives + 336 pure negatives)
    |
    +→ Amostragem estratificada
    +→ 882 artigos negativos (15 portais)
         583 hard negatives (66.1%) + 299 pure negatives (33.9%)

════════════════════════════════════════════════════════════════════
FASE 6: Dataset Final → Splits
════════════════════════════════════════════════════════════════════
1,764 artigos (882 pos + 882 neg) | 33 portais | Jan 2020–Jul 2025
    |
    +-- 1,410 (80%) → Training + Development
    |       |
    |       +-- 1,128 (80%) → TRAINING
    |       |       |
    |       |       +-- 5-Fold Cross-Validation (stratified)
    |       |           +-- Fold 1: ~902 train / ~226 val
    |       |           +-- Fold 2: ~902 train / ~226 val
    |       |           +-- Fold 3: ~902 train / ~226 val
    |       |           +-- Fold 4: ~902 train / ~226 val
    |       |           +-- Fold 5: ~902 train / ~226 val
    |       |
    |       +-- 282 (20%) → DEVELOPMENT (model selection)
    |
    +-- 353 (20%) → TEST (initial)
            |
            +→ Data leakage verification (Jun 8, 2026)
            +→ 19 duplicatas removidas (5.38%)
            +→ 334 artigos → TEST CLEAN
                   175 positivos + 159 negativos
════════════════════════════════════════════════════════════════════
```

### Dataset Summary Table

| Dataset | Articles | Positive | Negative | % of Total | Purpose |
|---------|----------|----------|----------|------------|---------|
| Total | 1,764 | 882 | 882 | 100% | - |
| Training | 1,128 | 564 | 564 | 64% | Model training + CV |
| Development | 282 | 141 | 141 | 16% | Model selection |
| Test (initial) | 353 | 176 | 177 | 20% | Final evaluation |
| **Test (clean)** | **334** | **175** | **159** | **19%** | **Final (no duplicates)** |
| CV (each fold train) | ~902 | ~451 | ~451 | ~51% | Internal training |
| CV (each fold val) | ~226 | ~113 | ~113 | ~13% | Internal validation |

---

## Experimental Design

### Experiment Structure

The training phase was organized into 4 main experiments to systematically evaluate 18 model combinations:

**1. Baseline Experiment**
- Model: TF-IDF + Naive Bayes
- Purpose: Establish performance baseline
- Script: `baseline_naive_bayes_tfidf.py`

**2. Naive Bayes with Multiple Vectorizations**
- Fixed classifier: Naive Bayes (GaussianNB for embeddings)
- Variable vectorizations: FastText, BERT-Base, BERT-Large, Albertina-Base, Albertina-Large (5 combinations)
- Note: TF-IDF already covered in baseline
- Script: `train_naive_bayes_all_vectorizations.py`

**3. SVM with Multiple Vectorizations**
- Fixed classifier: SVM (linear kernel, probability=True)
- Variable vectorizations: TF-IDF, FastText, BERT-Base, BERT-Large, Albertina-Base, Albertina-Large (6 combinations)
- Script: `train_svm_all_vectorizations.py`

**4. Random Forest with Multiple Vectorizations**
- Fixed classifier: Random Forest (100 estimators)
- Variable vectorizations: TF-IDF, FastText, BERT-Base, BERT-Large, Albertina-Base, Albertina-Large (6 combinations)
- Script: `train_random_forest_all_vectorizations.py`

**Total Combinations:** 1 (baseline) + 5 (NB) + 6 (SVM) + 6 (RF) = 18 combinations

### Evaluation Protocol

For each model combination:
1. **Cross-Validation:** 5-fold stratified CV on training set (1,128 samples)
   - Reports: Mean ± standard deviation for each metric
2. **Final Training:** Retrain on entire training set (1,128 samples)
3. **Development Evaluation:** Evaluate on development set (282 samples)
   - Reports: Single value for each metric
4. **Outputs Generated:**
   - Individual confusion matrix (PNG)
   - Individual classification report (TXT)
   - Trained model (PKL)
   - Execution log (TXT)
5. **Comparative Analysis:** Generate comparison plots and tables across all vectorizations

---

## Results Summary

### Baseline Performance

**TF-IDF + Naive Bayes**
- Cross-Validation (5-fold on training):
  - Accuracy: 0.9575 ± 0.0163
  - Precision: 0.9730 ± 0.0321
  - Recall: 0.9415 ± 0.0345
  - F1-Score: 0.9568 ± 0.0166
  - ROC-AUC: 0.9799 ± 0.0186

- Development Set:
  - Accuracy: 0.9574
  - Precision: 0.9778
  - Recall: 0.9362
  - F1-Score: 0.9565
  - ROC-AUC: 0.9809

**Analysis:** Excellent baseline performance (F1: 0.9565), significantly exceeding literature expectations (0.65-0.70). Minimal difference between CV and development (0.03%) indicates no overfitting.

### Naive Bayes with Multiple Vectorizations

| Vectorization | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Ranking |
|---------------|----------|-----------|--------|----------|---------|---------|
| TF-IDF (baseline) | 0.9574 | 0.9778 | 0.9362 | **0.9565** | 0.9809 | 1st |
| BERT-Base | 0.8865 | 0.9431 | 0.8227 | 0.8788 | 0.9316 | 2nd |
| FastText | 0.8652 | 0.8759 | 0.8511 | 0.8633 | 0.9381 | 3rd |
| Albertina-Large | 0.7730 | 0.7939 | 0.7376 | 0.7647 | 0.8285 | 4th |
| BERT-Large | 0.7447 | 0.7674 | 0.7021 | 0.7333 | 0.7938 | 5th |
| Albertina-Base | 0.5887 | 0.8378 | 0.2199 | 0.3483 | 0.6027 | 6th |

**Key Findings:**
- TF-IDF remains superior for Naive Bayes
- BERT-Base shows best performance among embeddings (F1: 0.8788)
- Albertina-Base struggles with Naive Bayes (very low recall: 0.2199)

### SVM with Multiple Vectorizations

| Vectorization | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Ranking |
|---------------|----------|-----------|--------|----------|---------|---------|
| TF-IDF | 0.9787 | **1.0000** | 0.9574 | **0.9783** | 0.9923 | 1st |
| BERT-Large | 0.9433 | 0.9699 | 0.9149 | 0.9416 | 0.9874 | 2nd |
| BERT-Base | 0.9326 | 0.9357 | 0.9291 | 0.9324 | 0.9783 | 3rd |
| FastText | 0.9184 | 0.9338 | 0.9007 | 0.9170 | 0.9671 | 4th |
| Albertina-Large | 0.9184 | 0.9470 | 0.8865 | 0.9158 | 0.9689 | 5th |
| Albertina-Base | 0.9149 | 0.9209 | 0.9078 | 0.9143 | 0.9652 | 6th |

**Key Findings:**
- **TF-IDF + SVM achieves PERFECT precision (100%)** - no false positives on development set
- Best overall F1-Score: 0.9783
- SVM significantly improves Albertina-Base: from 0.3483 (NB) to 0.9143 (SVM) = +162.5%
- All combinations achieve F1 > 0.91

### Random Forest with Multiple Vectorizations

| Vectorization | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Ranking |
|---------------|----------|-----------|--------|----------|---------|---------|
| TF-IDF | 0.9539 | 0.9571 | 0.9504 | 0.9537 | 0.9901 | 1st |
| FastText | 0.9326 | 0.9621 | 0.9007 | 0.9304 | 0.9794 | 2nd |
| BERT-Base | 0.9220 | 0.9407 | 0.9007 | 0.9203 | 0.9749 | 3rd |
| Albertina-Base | 0.9149 | 0.9398 | 0.8865 | 0.9124 | 0.9632 | 4th |
| BERT-Large | 0.9007 | 0.9248 | 0.8723 | 0.8978 | 0.9658 | 5th |
| Albertina-Large | 0.8298 | 0.8394 | 0.8156 | 0.8273 | 0.9284 | 6th |

**Key Findings:**
- TF-IDF maintains best performance (F1: 0.9537)
- FastText shows strong performance with Random Forest (F1: 0.9304)
- Generally lower performance than SVM but better than Naive Bayes for embeddings

### Top 10 Model Combinations (Overall Ranking)

| Rank | Vectorization | Classifier | F1-Score | Precision | Recall | ROC-AUC |
|------|---------------|------------|----------|-----------|--------|---------|
| 1 | TF-IDF | SVM | **0.9783** | **1.0000** | 0.9574 | 0.9923 |
| 2 | TF-IDF | Naive Bayes | 0.9565 | 0.9778 | 0.9362 | 0.9809 |
| 3 | TF-IDF | Random Forest | 0.9537 | 0.9571 | 0.9504 | 0.9901 |
| 4 | BERT-Large | SVM | 0.9416 | 0.9699 | 0.9149 | 0.9874 |
| 5 | BERT-Base | SVM | 0.9324 | 0.9357 | 0.9291 | 0.9783 |
| 6 | FastText | Random Forest | 0.9304 | 0.9621 | 0.9007 | 0.9794 |
| 7 | BERT-Base | Random Forest | 0.9203 | 0.9407 | 0.9007 | 0.9749 |
| 8 | FastText | SVM | 0.9170 | 0.9338 | 0.9007 | 0.9671 |
| 9 | Albertina-Large | SVM | 0.9158 | 0.9470 | 0.8865 | 0.9689 |
| 10 | Albertina-Base | SVM | 0.9143 | 0.9209 | 0.9078 | 0.9652 |

**Key Insights:**
- TF-IDF dominates top 3 positions across all classifiers
- SVM shows best overall performance
- BERT embeddings perform well with SVM (ranks 4-5)
- FastText is competitive with Random Forest

---

## Final Test Evaluation

### Best Model Selection
Based on development set performance, **TF-IDF + SVM** was selected for final evaluation on the held-out test set.

### Test Set Performance

**Test Set Characteristics (Clean - After Duplicate Removal):**
- **334 articles (175 positive, 159 negative)**
- 19 duplicates removed from initial 353 articles (5.38% data leakage)
- Never seen during training or development
- Vectorized separately after model selection
- Data leakage verification: June 2026

**Results on Clean Test Set:**
- Accuracy: 0.9701
- Precision: 0.9825
- Recall: 0.9600
- F1-Score: 0.9711
- ROC-AUC: 0.9921

**Original Test Set (353 articles, with duplicates):**
- Accuracy: 0.9717
- Precision: 0.9826
- Recall: 0.9602
- F1-Score: 0.9713
- ROC-AUC: 0.9922
- **Difference after duplicate removal: -0.0002 F1-Score (negligible impact)**

### Generalization Analysis

| Metric | Development | Test | Difference | % Change |
|--------|-------------|------|------------|----------|
| F1-Score | 0.9783 | 0.9713 | -0.0070 | -0.7% |
| Accuracy | 0.9787 | 0.9717 | -0.0070 | -0.7% |
| Precision | 1.0000 | 0.9826 | -0.0174 | -1.7% |
| Recall | 0.9574 | 0.9602 | +0.0028 | +0.3% |
| ROC-AUC | 0.9923 | 0.9922 | -0.0001 | -0.01% |

**Analysis:**
- Excellent generalization: < 1% difference between development and test
- F1-Score remains very high (0.9713) on unseen data
- Recall actually improved slightly on test set (+0.3%)
- ROC-AUC virtually identical (0.9923 vs 0.9922)
- No evidence of overfitting

### Error Analysis

**Development Set (282 samples):**
- True Negatives: 141 (100%)
- False Positives: 0 (0%)
- False Negatives: 6 (4.26%)
- True Positives: 135 (95.74%)

**Test Set - Clean (334 samples, no duplicates):**
- True Negatives: 156 (98.11%)
- False Positives: 3 (1.89%)
- False Negatives: 7 (4.00%)
- True Positives: 168 (96.00%)

**Test Set - Original (353 samples, with 19 duplicates):**
- True Negatives: 174 (98.31%)
- False Positives: 3 (1.69%)
- False Negatives: 7 (3.98%)
- True Positives: 169 (96.02%)

**Key Observations:**
- Very few errors overall (< 3% error rate)
- Development: Perfect precision (no false positives)
- Test: Near-perfect precision (98.26%)
- Both sets show high recall (> 95%)
- Error patterns consistent across development and test

---

## Implementation Details

### Scripts Created

1. **baseline_naive_bayes_tfidf.py**
   - Establishes TF-IDF + Naive Bayes baseline
   - 5-fold cross-validation on training set
   - Evaluation on development set
   - Generates confusion matrix, classification report, and model file

2. **train_naive_bayes_all_vectorizations.py**
   - Tests Naive Bayes with 5 vectorizations (excluding TF-IDF)
   - Uses GaussianNB for dense embeddings
   - Generates individual reports for each vectorization
   - Creates comparative plots and tables

3. **train_svm_all_vectorizations.py**
   - Tests SVM with all 6 vectorizations
   - Linear kernel with probability estimates
   - Generates individual reports for each vectorization
   - Creates comparative plots and tables

4. **train_random_forest_all_vectorizations.py**
   - Tests Random Forest with all 6 vectorizations
   - 100 estimators with default parameters
   - Generates individual reports for each vectorization
   - Creates comparative plots and tables

5. **generate_individual_reports.py**
   - Utility script to generate missing individual reports
   - Creates confusion matrices and classification reports
   - Used to standardize outputs across all experiments

6. **evaluate_best_model_on_test.py**
   - Evaluates best model (TF-IDF + SVM) on test set
   - Loads pre-trained model and test data
   - Generates final evaluation report

### Output Structure

```
training/results/
├── baseline_naive_bayes/
│   ├── EXPLICACAO_BASELINE.md
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   ├── execution_log.txt
│   └── naive_bayes_baseline_*.pkl
├── naive_bayes_vectorizations/
│   ├── EXPLICACAO_EXPERIMENTO.md
│   ├── naive_bayes_vectorizations_comparison.png
│   ├── naive_bayes_vectorizations_comparison.csv
│   ├── execution_log.txt
│   ├── fasttext/
│   │   ├── confusion_matrix.png
│   │   └── classification_report.txt
│   ├── bert_base/
│   │   ├── confusion_matrix.png
│   │   └── classification_report.txt
│   └── [similar structure for other vectorizations]
├── svm_vectorizations/
│   ├── EXPLICACAO_EXPERIMENTO.md
│   ├── svm_vectorizations_comparison.png
│   ├── svm_vectorizations_comparison.csv
│   ├── execution_log.txt
│   └── [individual folders for each vectorization]
├── random_forest_vectorizations/
│   ├── EXPLICACAO_EXPERIMENTO.md
│   ├── random_forest_vectorizations_comparison.png
│   ├── random_forest_vectorizations_comparison.csv
│   ├── execution_log.txt
│   └── [individual folders for each vectorization]
└── CONSOLIDACAO_FINAL.md

test/
├── FINAL_TEST_REPORT.md
├── test_confusion_matrix.png
├── test_classification_report.txt
└── test_results_*.pkl
```

### Key Implementation Decisions

1. **Train/Development/Test Split:**
   - Training: 64% (1,128 samples) - for model training and CV
   - Development: 16% (282 samples) - for model selection
   - Test: 20% (353 initial, 334 clean) - for final evaluation only
   - 19 duplicates removed (5.38% data leakage detected)

2. **Cross-Validation:**
   - Stratified 5-fold to maintain class balance
   - Applied only to training set
   - Random state fixed (42) for reproducibility

3. **Model Configurations:**
   - No hyperparameter tuning (as per requirements)
   - Default configurations for all models
   - Focus on establishing honest baselines

4. **Evaluation Metrics:**
   - Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - F1-Score as primary metric (balances precision and recall)
   - Confusion matrices for error analysis

5. **Naive Bayes Variants:**
   - MultinomialNB for sparse data (TF-IDF)
   - GaussianNB for dense embeddings (FastText, BERT, Albertina)

---

## Key Findings and Conclusions

### Main Findings

1. **TF-IDF Superiority:**
   - TF-IDF consistently outperforms embeddings across all classifiers
   - Best results: TF-IDF + SVM (F1: 0.9783)
   - Surprising given the sophistication of BERT/Albertina models

2. **SVM Excellence:**
   - SVM achieves best performance across all vectorizations
   - Perfect precision (100%) on development set with TF-IDF
   - Significantly improves poor-performing combinations (e.g., Albertina-Base)

3. **Embedding Performance:**
   - BERT embeddings perform better than Albertina for this task
   - BERT-Large + SVM achieves F1: 0.9416 (4th overall)
   - Embeddings benefit more from complex classifiers (SVM, RF) than Naive Bayes

4. **Excellent Generalization:**
   - < 1% difference between development and test performance
   - No evidence of overfitting despite high performance
   - Validates data preparation and experimental methodology

5. **Dataset Quality:**
   - Performance significantly exceeds literature expectations (0.65-0.70 vs 0.97)
   - Bias control and balancing strategies were effective
   - High-quality preprocessing enables strong model performance

### Surprising Results

1. **TF-IDF vs Embeddings:**
   - Expected: Contextual embeddings (BERT, Albertina) would outperform TF-IDF
   - Observed: TF-IDF consistently superior
   - Possible reasons:
     - Fraud-specific vocabulary well captured by TF-IDF
     - Embeddings may be too general for this specific domain
     - Dataset size may favor simpler representations

2. **Albertina-Base with Naive Bayes:**
   - Extremely poor performance (F1: 0.3483)
   - Very low recall (0.2199) despite high precision (0.8378)
   - Dramatically improves with SVM (F1: 0.9143)
   - Suggests Naive Bayes assumption violated for Albertina embeddings

3. **Perfect Precision:**
   - TF-IDF + SVM achieved 100% precision on development set
   - Zero false positives across 141 negative samples
   - Rare achievement in real-world classification tasks

### Limitations

1. **Domain Specificity:**
   - Results specific to Brazilian Portuguese fraud detection
   - May not generalize to other languages or fraud types

2. **Dataset Size:**
   - 1,764 total samples relatively small for deep learning
   - May explain why simpler models (TF-IDF) perform well

3. **No Hyperparameter Tuning:**
   - All models use default configurations
   - Performance could potentially improve with optimization

4. **Embedding Models:**
   - Pre-trained embeddings not fine-tuned for fraud detection
   - Domain-specific fine-tuning could improve performance

5. **Class Balance:**
   - Perfect 50/50 balance may not reflect real-world distribution
   - Model performance on imbalanced data unknown

6. **Temporal Aspects:**
   - No consideration of temporal patterns or trends
   - All articles treated as independent samples

---

## Future Work

1. **Hyperparameter Optimization:**
   - Grid search for SVM (C, kernel parameters)
   - Random Forest optimization (n_estimators, max_depth)
   - Could further improve top models

2. **Fine-Tuning Embeddings:**
   - Fine-tune BERT/Albertina on fraud detection corpus
   - Domain-specific training may improve embedding quality
   - Potential to surpass TF-IDF performance

3. **Ensemble Methods:**
   - Combine top models (TF-IDF + SVM, BERT-Large + SVM)
   - Voting or stacking approaches
   - May reduce errors further

4. **Feature Engineering:**
   - Incorporate entity information (companies, people)
   - Add metadata features (source, date, length)
   - Fraud type classification (multi-class)

5. **Imbalanced Data:**
   - Test performance on realistic class distributions
   - Implement cost-sensitive learning
   - Evaluate with different thresholds

6. **Explainability:**
   - SHAP or LIME for model interpretation
   - Identify key fraud indicators
   - Support human review process

7. **Production Deployment:**
   - Real-time classification pipeline
   - Monitoring and retraining strategies
   - Integration with news aggregation systems

8. **Extended Evaluation:**
   - Test on news from different time periods
   - Cross-validation across news sources
   - Robustness to adversarial examples

---

## Conclusion

The training phase successfully established a robust fraud detection system for Brazilian Portuguese news articles. The systematic evaluation of 18 model combinations revealed that **TF-IDF + SVM** achieves exceptional performance (F1: 0.9711 on clean test set) with perfect precision on the development set.

**Data Leakage Verification (June 2026):** Post-training analysis identified 19 duplicate examples (5.38%) between training and test sets. After removing these duplicates, the model maintains exceptional performance (F1: 0.9711), confirming genuine generalization capability with negligible impact from data leakage.

Key achievements:
- Established honest baselines without hyperparameter tuning
- Demonstrated excellent generalization (< 1% dev-test gap)
- Identified TF-IDF superiority over sophisticated embeddings
- Created comprehensive documentation and reproducible methodology
- Generated standardized outputs for all experiments

The project provides a solid foundation for future enhancements and demonstrates that careful data preparation and systematic evaluation can achieve state-of-the-art results even with classical ML approaches.

---

## References

All experimental results, trained models, and detailed documentation are available in:
- `/training/results/` - All training experiments and outputs
- `/test/` - Final test evaluation results
- `/chronology/` - Project documentation and methodology
- IEEE report template with complete results in `/ieee_report_template.md`
