#!/usr/bin/env python3
"""
Task 1 — Padronização e Consolidação
Normaliza positivos e negativos, consolida em CONSOLIDATED_IMBALANCED.csv
Schema final: url, title, text, label, portal, date_publication
"""

import pandas as pd
import numpy as np
from urllib.parse import urlparse
import os
import sys

BASE_DIR = "/home/paulo/CascadeProjects/Applied_ML"
OUTPUT_DIR = os.path.join(BASE_DIR, "NEW_training/FOR_TRAINING")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "CONSOLIDATED_IMBALANCED.csv")
REPORT_FILE = os.path.join(OUTPUT_DIR, "DISTRIBUTION_REPORT.txt")

POSITIVE_FILE = os.path.join(BASE_DIR, "dataset/cleaned_data_no_bias/POSITIVE_DF_COMPANIES_REDUZIDO.csv")
NEGATIVE_FILE = os.path.join(BASE_DIR, "NEW_training/NEGATIVES_CONSOLIDATED.csv")


def extract_portal(url):
    if pd.isna(url) or not url:
        return "unknown"
    try:
        parsed = urlparse(url)
        return parsed.netloc if parsed.netloc else "unknown"
    except Exception:
        return "unknown"


def normalize_positives():
    print("=" * 70)
    print("NORMALIZANDO POSITIVOS (fraude = 1)")
    print("=" * 70)

    df = pd.read_csv(POSITIVE_FILE)
    print(f"Carregado: {len(df)} linhas")
    print(f"Colunas originais: {df.columns.tolist()}")

    df_norm = pd.DataFrame()
    df_norm['url'] = df['url'].fillna('')
    df_norm['title'] = df['title'].fillna('')
    df_norm['text'] = df['text'].fillna('')
    df_norm['label'] = 1
    df_norm['portal'] = df['url'].apply(extract_portal)
    df_norm['date_publication'] = df['date_publication'].fillna('')

    df_norm = df_norm.drop_duplicates(subset=['url'], keep='first')
    print(f"Após dedup por URL: {len(df_norm)} linhas")
    print(f"Portais: {df_norm['portal'].value_counts().to_dict()}")

    return df_norm


def normalize_negatives():
    print("\n" + "=" * 70)
    print("NORMALIZANDO NEGATIVOS (não-fraude = 0)")
    print("=" * 70)

    df = pd.read_csv(NEGATIVE_FILE, low_memory=False)
    print(f"Carregado: {len(df)} linhas")
    print(f"Colunas originais: {df.columns.tolist()}")
    print(f"Labels originais: {df['label'].value_counts().to_dict()}")

    df_norm = pd.DataFrame()
    df_norm['url'] = df['url'].fillna('')
    df_norm['title'] = df['title'].fillna('')
    df_norm['text'] = df['text'].fillna('')
    df_norm['label'] = 0
    df_norm['portal'] = df['portal'].fillna(df['url'].apply(extract_portal))
    df_norm['date_publication'] = ''

    df_norm = df_norm.drop_duplicates(subset=['url'], keep='first')
    print(f"Após dedup por URL: {len(df_norm)} linhas")
    print(f"Portais: {df_norm['portal'].value_counts().head(10).to_dict()}")

    return df_norm


def generate_report(df_combined, df_pos, df_neg):
    print("\n" + "=" * 70)
    print("GERANDO RELATÓRIO DE DISTRIBUIÇÃO")
    print("=" * 70)

    total = len(df_combined)
    n_pos = len(df_pos)
    n_neg = len(df_neg)
    pct_pos = (n_pos / total) * 100
    pct_neg = (n_neg / total) * 100

    report_lines = [
        "=" * 70,
        "RELATÓRIO DE DISTRIBUIÇÃO — CONSOLIDATED_IMBALANCED.csv",
        "=" * 70,
        "",
        f"Total de exemplos: {total:,}",
        f"Positivos (fraude=1): {n_pos:,} ({pct_pos:.2f}%)",
        f"Negativos (fraude=0): {n_neg:,} ({pct_neg:.2f}%)",
        f"Razão de desbalanceamento: 1:{n_neg/n_pos:.1f}",
        "",
        "Distribuição por portal (top 20):",
    ]

    portal_counts = df_combined.groupby(['portal', 'label']).size().unstack(fill_value=0)
    portal_counts['total'] = portal_counts.sum(axis=1)
    portal_counts = portal_counts.sort_values('total', ascending=False).head(20)

    for portal, row in portal_counts.iterrows():
        pos = int(row.get(1, 0))
        neg = int(row.get(0, 0))
        tot = int(row['total'])
        report_lines.append(f"  {portal:40s}  pos={pos:5d}  neg={neg:6d}  total={tot:6d}")

    report_lines.extend([
        "",
        "Distribuição por label:",
        f"  label=0: {n_neg:,} ({pct_neg:.2f}%)",
        f"  label=1: {n_pos:,} ({pct_pos:.2f}%)",
        "",
        "Schema: url, title, text, label, portal, date_publication",
        "=" * 70,
    ])

    report_text = "\n".join(report_lines)
    print(report_text)

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\nRelatório salvo: {REPORT_FILE}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_pos = normalize_positives()
    df_neg = normalize_negatives()

    print("\n" + "=" * 70)
    print("CONSOLIDANDO")
    print("=" * 70)

    df_combined = pd.concat([df_pos, df_neg], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total consolidado: {len(df_combined):,}")
    print(f"Schema: {df_combined.columns.tolist()}")

    df_combined.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"Salvo: {OUTPUT_FILE}")

    generate_report(df_combined, df_pos, df_neg)

    print("\n" + "=" * 70)
    print("TASK 1 CONCLUÍDA")
    print("=" * 70)


if __name__ == "__main__":
    main()
