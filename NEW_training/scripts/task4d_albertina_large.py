#!/usr/bin/env python3
"""
Task 4d — Vetorização Albertina-Large apenas
- Modelo correto: PORTULAN/albertina-900m-portuguese-ptbr-encoder-brwac
- Força float32 para evitar erro de BFloat16 no DeBERTa
"""

import pandas as pd
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModel

BASE_DIR = "/home/paulo/CascadeProjects/Applied_ML"
TRAIN_DIR = os.path.join(BASE_DIR, "NEW_training/FOR_TRAINING")
TEST_DIR = os.path.join(BASE_DIR, "NEW_training/FOR_TEST")
VEC_DIR = os.path.join(BASE_DIR, "NEW_training/vectorization")

EMB_TRAIN = os.path.join(TRAIN_DIR, "Pre_processed_for_Embeddings/train_bert.csv")
EMB_DEV = os.path.join(TRAIN_DIR, "Pre_processed_for_Embeddings/dev_bert.csv")
EMB_TEST = os.path.join(TEST_DIR, "Pre_processed_for_Embeddings/test_bert.csv")

MODEL_KEY = "albertina_large"
MODEL_NAME = "PORTULAN/albertina-900m-portuguese-ptbr-encoder-brwac"

BATCH_SIZE = 16
MAX_LENGTH = 512


def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    texts = df['processed_text'].fillna('').tolist()
    labels = df['label'].values.astype(int)
    return texts, labels


def extract_embeddings(texts, tokenizer, model, device):
    all_embeddings = []
    model.eval()

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

        if cls_embeddings.dtype != torch.float32:
            cls_embeddings = cls_embeddings.float()

        all_embeddings.append(cls_embeddings.cpu().numpy())

        if (i // BATCH_SIZE) % 10 == 0:
            print(f"    Batch {i//BATCH_SIZE + 1}/{(len(texts) + BATCH_SIZE - 1) // BATCH_SIZE}")

    return np.vstack(all_embeddings)


def main():
    print("=" * 70)
    print(f"TASK 4d — ALBERTINA-LARGE ({MODEL_NAME})")
    print("=" * 70)

    out_dir = os.path.join(VEC_DIR, MODEL_KEY)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"Carregando modelo: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(device)
    print(f"Modelo carregado. Parâmetros: {sum(p.numel() for p in model.parameters()):,}")

    for split_name, file_path in [("train", EMB_TRAIN), ("dev", EMB_DEV), ("test", EMB_TEST)]:
        print(f"\n  Processando {split_name}...")
        texts, labels = load_data(file_path)
        print(f"  {split_name}: {len(texts)} textos")

        embeddings = extract_embeddings(texts, tokenizer, model, device)
        print(f"  Embeddings shape: {embeddings.shape}")

        np.save(os.path.join(out_dir, f"{split_name}_embeddings.npy"), embeddings)
        np.save(os.path.join(out_dir, f"labels_{split_name}.npy"), labels)

    del model
    torch.cuda.empty_cache()
    print(f"\n  Arquivos salvos em: {out_dir}")
    print("\n" + "=" * 70)
    print("TASK 4d CONCLUÍDA")
    print("=" * 70)


if __name__ == "__main__":
    main()
