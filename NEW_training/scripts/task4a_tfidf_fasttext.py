#!/usr/bin/env python3
"""
Task 4a — Vetorização TF-IDF + FastText
- TF-IDF: fitado APENAS no treino, transform em dev e test
- FastText: embeddings cc.pt.300d ponderados por TF-IDF (pesos do treino)
"""

import pandas as pd
import numpy as np
import pickle
import os
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors

BASE_DIR = "/home/paulo/CascadeProjects/Applied_ML"
TRAIN_DIR = os.path.join(BASE_DIR, "NEW_training/FOR_TRAINING")
TEST_DIR = os.path.join(BASE_DIR, "NEW_training/FOR_TEST")
VEC_DIR = os.path.join(BASE_DIR, "NEW_training/vectorization")
FASTTEXT_MODEL = os.path.join(BASE_DIR, "vectorization/fasttext/models/cc.pt.300.vec")

SPARSE_TRAIN = os.path.join(TRAIN_DIR, "Pre_processed_for_Sparse/train_preprocessed.csv")
SPARSE_DEV = os.path.join(TRAIN_DIR, "Pre_processed_for_Sparse/dev_preprocessed.csv")
SPARSE_TEST = os.path.join(TEST_DIR, "Pre_processed_for_Sparse/test_preprocessed.csv")


def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    texts = df['processed_text'].fillna('').values
    labels = df['label'].values.astype(int)
    return texts, labels


def vectorize_tfidf():
    print("\n" + "=" * 70)
    print("TF-IDF VETORIZAÇÃO")
    print("=" * 70)

    out_dir = os.path.join(VEC_DIR, "tfidf")
    os.makedirs(out_dir, exist_ok=True)

    train_texts, train_labels = load_data(SPARSE_TRAIN)
    dev_texts, dev_labels = load_data(SPARSE_DEV)
    test_texts, test_labels = load_data(SPARSE_TEST)

    print(f"Train: {len(train_texts)} | Dev: {len(dev_texts)} | Test: {len(test_texts)}")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=10000,
        norm='l2'
    )

    print("Fitando TF-IDF no TREINO (apenas)...")
    train_matrix = vectorizer.fit_transform(train_texts)
    print(f"Vocabulário: {len(vectorizer.vocabulary_):,} termos")
    print(f"Matriz treino: {train_matrix.shape}")

    print("Transform dev e test...")
    dev_matrix = vectorizer.transform(dev_texts)
    test_matrix = vectorizer.transform(test_texts)
    print(f"Matriz dev: {dev_matrix.shape} | Matriz test: {test_matrix.shape}")

    sparse.save_npz(os.path.join(out_dir, "train_sparse.npz"), train_matrix)
    sparse.save_npz(os.path.join(out_dir, "dev_sparse.npz"), dev_matrix)
    sparse.save_npz(os.path.join(out_dir, "test_sparse.npz"), test_matrix)

    np.save(os.path.join(out_dir, "labels_train.npy"), train_labels)
    np.save(os.path.join(out_dir, "labels_dev.npy"), dev_labels)
    np.save(os.path.join(out_dir, "labels_test.npy"), test_labels)

    with open(os.path.join(out_dir, "vectorizer.pkl"), 'wb') as f:
        pickle.dump(vectorizer, f)

    print(f"Arquivos salvos em: {out_dir}")

    feature_names = vectorizer.get_feature_names_out()
    print(f"\nTop 20 features (por IDF):")
    sorted_idx = np.argsort(vectorizer.idf_)
    for i in sorted_idx[:20]:
        print(f"  {feature_names[i]:30s}  idf={vectorizer.idf_[i]:.4f}")

    fraude_idx = [i for i, f in enumerate(feature_names) if 'fraude' in f.lower()]
    if fraude_idx:
        print(f"\nFeatures com 'fraude': {len(fraude_idx)}")
        for i in fraude_idx[:5]:
            print(f"  {feature_names[i]:30s}  idf={vectorizer.idf_[i]:.4f}")

    return vectorizer, train_matrix, train_labels, dev_matrix, dev_labels, test_matrix, test_labels


def vectorize_fasttext(vectorizer, train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels):
    print("\n" + "=" * 70)
    print("FASTTEXT VETORIZAÇÃO (TF-IDF weighted)")
    print("=" * 70)

    out_dir = os.path.join(VEC_DIR, "fasttext")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Carregando FastText: {FASTTEXT_MODEL}")
    ft_model = KeyedVectors.load_word2vec_format(FASTTEXT_MODEL, binary=False)
    print(f"FastText carregado: {len(ft_model.key_to_index):,} palavras, {ft_model.vector_size} dim")

    feature_names = vectorizer.get_feature_names_out()
    word_to_tfidf = dict(zip(feature_names, vectorizer.idf_))

    dim = ft_model.vector_size

    def texts_to_embeddings(texts, label):
        embeddings = np.zeros((len(texts), dim), dtype=np.float32)
        for idx, text in enumerate(texts):
            words = text.split()
            weighted_vec = np.zeros(dim, dtype=np.float32)
            total_weight = 0.0
            for word in words:
                if word in ft_model.key_to_index and word in word_to_tfidf:
                    weight = word_to_tfidf[word]
                    weighted_vec += weight * ft_model[word]
                    total_weight += weight
            if total_weight > 0:
                weighted_vec /= total_weight
            embeddings[idx] = weighted_vec
        print(f"  {label}: {embeddings.shape}")
        return embeddings

    train_emb = texts_to_embeddings(train_texts, "Train")
    dev_emb = texts_to_embeddings(dev_texts, "Dev")
    test_emb = texts_to_embeddings(test_texts, "Test")

    np.save(os.path.join(out_dir, "train_embeddings.npy"), train_emb)
    np.save(os.path.join(out_dir, "dev_embeddings.npy"), dev_emb)
    np.save(os.path.join(out_dir, "test_embeddings.npy"), test_emb)
    np.save(os.path.join(out_dir, "labels_train.npy"), train_labels)
    np.save(os.path.join(out_dir, "labels_dev.npy"), dev_labels)
    np.save(os.path.join(out_dir, "labels_test.npy"), test_labels)

    print(f"Arquivos salvos em: {out_dir}")

    n_nonzero = np.sum(np.any(train_emb != 0, axis=1))
    print(f"Embeddings não-nulos: {n_nonzero}/{len(train_emb)} ({n_nonzero/len(train_emb)*100:.1f}%)")


def main():
    print("=" * 70)
    print("TASK 4a — TF-IDF + FASTTEXT")
    print("=" * 70)

    train_texts, train_labels = load_data(SPARSE_TRAIN)
    dev_texts, dev_labels = load_data(SPARSE_DEV)
    test_texts, test_labels = load_data(SPARSE_TEST)

    vectorizer, train_matrix, _, dev_matrix, _, test_matrix, _ = vectorize_tfidf()

    vectorize_fasttext(vectorizer, train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels)

    print("\n" + "=" * 70)
    print("TASK 4a CONCLUÍDA — TF-IDF + FastText")
    print("=" * 70)


if __name__ == "__main__":
    main()
