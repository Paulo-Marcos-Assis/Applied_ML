#!/usr/bin/env python3
"""
Task 3 — Pré-processamento Dual
- PESADO: remove HTML, URLs, pontuação, acentos, stopwords, lowercase → TF-IDF, FastText
- LEVE: remove apenas URLs e normaliza espaços, preserva acentos/pontuação/maiúsculas → BERT, Albertina
- Aplicado a train, dev, e test separadamente
- Vetorização (fit/transform) fica para Task 4 — vectorizer fitado APENAS no treino
"""

import pandas as pd
import re
import unicodedata
import os

BASE_DIR = "/home/paulo/CascadeProjects/Applied_ML"
TRAIN_DIR = os.path.join(BASE_DIR, "NEW_training/FOR_TRAINING")
TEST_DIR = os.path.join(BASE_DIR, "NEW_training/FOR_TEST")

SPARSE_TRAIN_OUT = os.path.join(TRAIN_DIR, "Pre_processed_for_Sparse")
SPARSE_TEST_OUT = os.path.join(TEST_DIR, "Pre_processed_for_Sparse")
EMB_TRAIN_OUT = os.path.join(TRAIN_DIR, "Pre_processed_for_Embeddings")
EMB_TEST_OUT = os.path.join(TEST_DIR, "Pre_processed_for_Embeddings")

STOPWORDS_PT = set([
    'a', 'ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo', 'as', 'ate',
    'ate', 'com', 'como', 'da', 'das', 'de', 'dela', 'delas', 'dele', 'deles', 'depois',
    'do', 'dos', 'e', 'ela', 'elas', 'ele', 'eles', 'em', 'entre', 'era', 'eram', 'essa',
    'essas', 'esse', 'esses', 'esta', 'estamos', 'estao', 'estas', 'estava', 'estavam',
    'este', 'esteja', 'estejam', 'estes', 'esteve', 'estive', 'estivemos', 'estiver',
    'estiveram', 'estivesse', 'estivessem', 'estou', 'eu', 'foi', 'fomos', 'for', 'fora',
    'foram', 'fosse', 'fossem', 'fui', 'ha', 'isso', 'isto', 'ja', 'la', 'lhe', 'lhes',
    'lo', 'mais', 'mas', 'me', 'mesmo', 'meu', 'meus', 'minha', 'minhas', 'muito', 'na',
    'nao', 'nas', 'nem', 'no', 'nos', 'nossa', 'nossas', 'nosso', 'nossos', 'num', 'numa',
    'o', 'os', 'ou', 'para', 'pela', 'pelas', 'pelo', 'pelos', 'por', 'qual', 'quando',
    'que', 'quem', 'sao', 'se', 'seja', 'sejam', 'sem', 'sera', 'serao', 'seu', 'seus',
    'so', 'sua', 'suas', 'tambem', 'te', 'tem', 'temos', 'tenho', 'tera', 'terao', 'teu',
    'teus', 'tinha', 'tinham', 'tive', 'tivemos', 'tu', 'tua', 'tuas', 'um', 'uma', 'umas',
    'uns', 'você', 'vocês', 'vos', 'à', 'às'
])


def remove_html(text):
    return re.sub(r'<[^>]+>', '', text)


def remove_urls(text):
    return re.sub(r'http[s]?://\S+|www\.\S+', '', text)


def remove_punctuation(text):
    return re.sub(r'[^\w\s]', ' ', text)


def remove_accents(text):
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join([c for c in nfkd if not unicodedata.combining(c)])


def normalize_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()


def heavy_preprocess(text):
    if pd.isna(text):
        return ''
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = text.lower()
    text = remove_accents(text)
    text = normalize_spaces(text)
    words = [w for w in text.split() if w not in STOPWORDS_PT]
    return ' '.join(words)


def light_preprocess(text):
    if pd.isna(text):
        return ''
    text = remove_urls(text)
    text = normalize_spaces(text)
    return text


def process_file(input_path, output_path, preprocess_fn, label):
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  [{label}] Carregado: {len(df):,} linhas de {input_path}")

    df['processed_text'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).apply(preprocess_fn)

    cols_keep = ['url', 'processed_text', 'label', 'portal']
    for c in cols_keep:
        if c not in df.columns:
            df[c] = ''
    df = df[cols_keep]

    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"  [{label}] Salvo: {output_path} ({len(df):,} linhas)")

    n_pos = (df['label'] == 1).sum()
    n_neg = (df['label'] == 0).sum()
    print(f"  [{label}] pos={n_pos} neg={n_neg}")

    return df


def main():
    print("=" * 70)
    print("TASK 3 — PRÉ-PROCESSAMENTO DUAL")
    print("=" * 70)

    for d in [SPARSE_TRAIN_OUT, SPARSE_TEST_OUT, EMB_TRAIN_OUT, EMB_TEST_OUT]:
        os.makedirs(d, exist_ok=True)

    print("\n--- PRÉ-PROCESSAMENTO PESADO (TF-IDF, FastText) ---")
    process_file(
        os.path.join(TRAIN_DIR, "train.csv"),
        os.path.join(SPARSE_TRAIN_OUT, "train_preprocessed.csv"),
        heavy_preprocess, "TRAIN"
    )
    process_file(
        os.path.join(TRAIN_DIR, "dev.csv"),
        os.path.join(SPARSE_TRAIN_OUT, "dev_preprocessed.csv"),
        heavy_preprocess, "DEV"
    )
    process_file(
        os.path.join(TEST_DIR, "test.csv"),
        os.path.join(SPARSE_TEST_OUT, "test_preprocessed.csv"),
        heavy_preprocess, "TEST"
    )

    print("\n--- PRÉ-PROCESSAMENTO LEVE (BERT, Albertina) ---")
    process_file(
        os.path.join(TRAIN_DIR, "train.csv"),
        os.path.join(EMB_TRAIN_OUT, "train_bert.csv"),
        light_preprocess, "TRAIN"
    )
    process_file(
        os.path.join(TRAIN_DIR, "dev.csv"),
        os.path.join(EMB_TRAIN_OUT, "dev_bert.csv"),
        light_preprocess, "DEV"
    )
    process_file(
        os.path.join(TEST_DIR, "test.csv"),
        os.path.join(EMB_TEST_OUT, "test_bert.csv"),
        light_preprocess, "TEST"
    )

    print("\n" + "=" * 70)
    print("TASK 3 CONCLUÍDA")
    print("=" * 70)
    print("\nNOTA: O vectorizer (TF-IDF) será fitado APENAS no treino na Task 4.")
    print("      Dev e test receberão apenas transform() — sem data leakage.")


if __name__ == "__main__":
    main()
