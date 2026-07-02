#!/usr/bin/env python3
"""
Task 2 — Split Estratificado 80/20
- Split por label apenas (mantendo proporção ~1,5%:98,5%)
- Treino+Dev (80%) → dividir em Treino (80%) / Dev (20%)
- Teste (20%) isolado em FOR_TEST/
- Deduplicação por URL entre splits
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

BASE_DIR = "/home/paulo/CascadeProjects/Applied_ML"
INPUT_FILE = os.path.join(BASE_DIR, "NEW_training/FOR_TRAINING/CONSOLIDATED_IMBALANCED.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "NEW_training/FOR_TRAINING")
TEST_DIR = os.path.join(BASE_DIR, "NEW_training/FOR_TEST")


def main():
    print("=" * 70)
    print("TASK 2 — SPLIT ESTRATIFICADO 80/20")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"Carregado: {len(df):,} exemplos")
    print(f"Positivos: {(df['label']==1).sum():,} ({(df['label']==1).mean()*100:.2f}%)")
    print(f"Negativos: {(df['label']==0).sum():,} ({(df['label']==0).mean()*100:.2f}%)")

    df = df.dropna(subset=['url'])
    df = df.drop_duplicates(subset=['url'], keep='first')
    print(f"Após dedup global: {len(df):,}")

    df_traindev, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    print(f"\nSplit 1 — Treino+Dev: {len(df_traindev):,} | Teste: {len(df_test):,}")

    df_train, df_dev = train_test_split(
        df_traindev, test_size=0.2, random_state=42, stratify=df_traindev['label']
    )
    print(f"Split 2 — Treino: {len(df_train):,} | Dev: {len(df_dev):,}")

    train_urls = set(df_train['url'])
    dev_urls = set(df_dev['url'])
    test_urls = set(df_test['url'])

    overlap_td = train_urls & dev_urls
    overlap_tt = train_urls & test_urls
    overlap_dt = dev_urls & test_urls
    print(f"\nDeduplicação entre splits:")
    print(f"  Train∩Dev: {len(overlap_td)}")
    print(f"  Train∩Test: {len(overlap_tt)}")
    print(f"  Dev∩Test: {len(overlap_dt)}")

    if overlap_td:
        df_dev = df_dev[~df_dev['url'].isin(overlap_td)]
    if overlap_tt:
        df_test = df_test[~df_test['url'].isin(overlap_tt)]
    if overlap_dt:
        df_test = df_test[~df_test['url'].isin(overlap_dt)]

    print(f"\nApós dedup entre splits:")
    print(f"  Train: {len(df_train):,}")
    print(f"  Dev: {len(df_dev):,}")
    print(f"  Test: {len(df_test):,}")

    for name, d in [('TRAIN', df_train), ('DEV', df_dev), ('TEST', df_test)]:
        n_pos = (d['label'] == 1).sum()
        n_neg = (d['label'] == 0).sum()
        pct = (n_pos / len(d)) * 100
        print(f"  {name}: {len(d):,} | pos={n_pos} ({pct:.2f}%) | neg={n_neg} ({100-pct:.2f}%)")

    os.makedirs(TEST_DIR, exist_ok=True)

    df_train.to_csv(os.path.join(TRAIN_DIR, "train.csv"), index=False, encoding='utf-8')
    df_dev.to_csv(os.path.join(TRAIN_DIR, "dev.csv"), index=False, encoding='utf-8')
    df_test.to_csv(os.path.join(TEST_DIR, "test.csv"), index=False, encoding='utf-8')

    print(f"\nArquivos salvos:")
    print(f"  {TRAIN_DIR}/train.csv ({len(df_train):,} linhas)")
    print(f"  {TRAIN_DIR}/dev.csv   ({len(df_dev):,} linhas)")
    print(f"  {TEST_DIR}/test.csv  ({len(df_test):,} linhas)")

    print("\n" + "=" * 70)
    print("TASK 2 CONCLUÍDA")
    print("=" * 70)


if __name__ == "__main__":
    main()
