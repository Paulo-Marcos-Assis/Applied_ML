import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from datetime import datetime

def create_output_dir(base_path):
    """Criar diretório de saída se não existir"""
    os.makedirs(base_path, exist_ok=True)
    print(f"✓ Diretório criado/verificado: {base_path}")

def load_preprocessed_data(train_path):
    """Carregar dados pré-processados de treino"""
    print(f"\nCarregando dados de treino: {train_path}")
    df_train = pd.read_csv(train_path)
    print(f"✓ Dados carregados: {len(df_train)} exemplos")
    print(f"  - Positivos: {len(df_train[df_train['label'] == 1])}")
    print(f"  - Negativos: {len(df_train[df_train['label'] == 0])}")
    return df_train

def concatenate_title_text(df, title_col='title_clean', text_col='text_clean'):
    """Concatenar título e texto limpos"""
    print(f"\nConcatenando '{title_col}' + '{text_col}'...")
    df['combined_text'] = df[title_col].fillna('') + ' ' + df[text_col].fillna('')
    
    # Estatísticas
    avg_length = df['combined_text'].str.split().str.len().mean()
    min_length = df['combined_text'].str.split().str.len().min()
    max_length = df['combined_text'].str.split().str.len().max()
    
    print(f"✓ Texto combinado criado")
    print(f"  - Tamanho médio: {avg_length:.1f} palavras")
    print(f"  - Min: {min_length} | Max: {max_length} palavras")
    
    return df

def fit_tfidf_vectorizer(texts, ngram_range=(1,2), min_df=2, max_df=0.9, 
                         max_features=10000, norm='l2'):
    """
    Treinar vetorizador TF-IDF
    
    Parâmetros:
    - ngram_range: (1,2) = unigrams + bigrams
    - min_df: 2 = ignorar termos que aparecem em menos de 2 documentos
    - max_df: 0.9 = ignorar termos que aparecem em mais de 90% dos documentos
    - max_features: 10000 = limitar vocabulário a 10k features
    - norm: 'l2' = normalização euclidiana
    """
    print("\n" + "="*70)
    print("CONFIGURAÇÃO TF-IDF")
    print("="*70)
    print(f"N-grams: {ngram_range}")
    print(f"min_df: {min_df}")
    print(f"max_df: {max_df}")
    print(f"max_features: {max_features}")
    print(f"norm: {norm}")
    print("="*70)
    
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        norm=norm,
        sublinear_tf=True  # Usar escala logarítmica para TF
    )
    
    print("\nTreinando vetorizador TF-IDF...")
    X_tfidf = vectorizer.fit_transform(texts)
    
    print(f"✓ Vetorização concluída!")
    print(f"  - Shape da matriz: {X_tfidf.shape}")
    print(f"  - Vocabulário: {len(vectorizer.vocabulary_)} termos")
    print(f"  - Esparsidade: {(1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])) * 100:.2f}%")
    
    return vectorizer, X_tfidf

def save_vectorizer_and_matrix(vectorizer, X_tfidf, y_train, output_dir, config_name):
    """Salvar vetorizador, matriz e labels"""
    
    # Criar nome base com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"tfidf_{config_name}_{timestamp}"
    
    # Salvar vetorizador
    vectorizer_path = os.path.join(output_dir, f"{base_name}_vectorizer.pkl")
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"\n✓ Vetorizador salvo: {vectorizer_path}")
    
    # Salvar matriz TF-IDF (formato esparso)
    matrix_path = os.path.join(output_dir, f"{base_name}_matrix.npz")
    from scipy.sparse import save_npz
    save_npz(matrix_path, X_tfidf)
    print(f"✓ Matriz TF-IDF salva: {matrix_path}")
    
    # Salvar labels
    labels_path = os.path.join(output_dir, f"{base_name}_labels.npy")
    np.save(labels_path, y_train)
    print(f"✓ Labels salvos: {labels_path}")
    
    # Salvar metadados
    metadata = {
        'timestamp': timestamp,
        'config_name': config_name,
        'n_samples': X_tfidf.shape[0],
        'n_features': X_tfidf.shape[1],
        'vocabulary_size': len(vectorizer.vocabulary_),
        'ngram_range': vectorizer.ngram_range,
        'min_df': vectorizer.min_df,
        'max_df': vectorizer.max_df,
        'max_features': vectorizer.max_features,
        'norm': vectorizer.norm,
        'sparsity': (1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])) * 100
    }
    
    metadata_path = os.path.join(output_dir, f"{base_name}_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Metadados salvos: {metadata_path}")
    
    return base_name

def analyze_top_features(vectorizer, X_tfidf, y_train, top_n=20):
    """Analisar top features por classe"""
    print("\n" + "="*70)
    print("ANÁLISE DE TOP FEATURES POR CLASSE")
    print("="*70)
    
    feature_names = vectorizer.get_feature_names_out()
    
    # Separar por classe
    X_pos = X_tfidf[y_train == 1]
    X_neg = X_tfidf[y_train == 0]
    
    # Calcular média TF-IDF por feature em cada classe
    mean_tfidf_pos = np.asarray(X_pos.mean(axis=0)).flatten()
    mean_tfidf_neg = np.asarray(X_neg.mean(axis=0)).flatten()
    
    # Top features para classe positiva (fraudes)
    top_pos_idx = mean_tfidf_pos.argsort()[-top_n:][::-1]
    print(f"\nTOP {top_n} FEATURES - CLASSE POSITIVA (Fraudes):")
    for i, idx in enumerate(top_pos_idx, 1):
        print(f"{i:2d}. {feature_names[idx]:30s} (TF-IDF médio: {mean_tfidf_pos[idx]:.4f})")
    
    # Top features para classe negativa
    top_neg_idx = mean_tfidf_neg.argsort()[-top_n:][::-1]
    print(f"\nTOP {top_n} FEATURES - CLASSE NEGATIVA (Não-fraudes):")
    for i, idx in enumerate(top_neg_idx, 1):
        print(f"{i:2d}. {feature_names[idx]:30s} (TF-IDF médio: {mean_tfidf_neg[idx]:.4f})")
    
    # Verificar se 'fraude' está no vocabulário
    print("\n" + "="*70)
    print("VERIFICAÇÃO: Termo 'fraude' no vocabulário")
    print("="*70)
    
    fraude_terms = [term for term in feature_names if 'fraud' in term.lower()]
    if fraude_terms:
        print(f"✓ Termos relacionados a 'fraude' encontrados: {len(fraude_terms)}")
        for term in fraude_terms[:10]:
            if term in vectorizer.vocabulary_:
                idx = vectorizer.vocabulary_[term]
                print(f"  - '{term}': TF-IDF médio pos={mean_tfidf_pos[idx]:.4f}, neg={mean_tfidf_neg[idx]:.4f}")
    else:
        print("⚠ Nenhum termo relacionado a 'fraude' encontrado no vocabulário")
        print("  (Pode ter sido removido por max_df ou min_df)")

def main():
    """Função principal"""
    print("\n" + "="*70)
    print("VETORIZAÇÃO TF-IDF - CONJUNTO DE TREINO")
    print("="*70)
    
    # Configurações
    TRAIN_PATH = 'dataset/cleaned_data/FOR_TRAINING/Pre_processed_for_Sparse/TRAIN_PREPROCESSED.csv'
    OUTPUT_DIR = 'vectorization/tf_idf'
    
    # Criar diretório de saída
    create_output_dir(OUTPUT_DIR)
    
    # Carregar dados
    df_train = load_preprocessed_data(TRAIN_PATH)
    
    # Concatenar título + texto
    df_train = concatenate_title_text(df_train)
    
    # Extrair textos e labels
    X_text = df_train['combined_text'].values
    y_train = df_train['label'].values
    
    # Configuração TF-IDF (conforme solicitado pelo usuário)
    config = {
        'ngram_range': (1, 2),  # unigrams + bigrams
        'min_df': 2,            # mínimo 2 documentos
        'max_df': 0.9,          # máximo 90% dos documentos
        'max_features': 10000,  # limite de 10k features
        'norm': 'l2'            # normalização euclidiana
    }
    
    # Treinar vetorizador
    vectorizer, X_tfidf = fit_tfidf_vectorizer(X_text, **config)
    
    # Salvar artefatos
    config_name = "ngram12_mindf2_maxdf09_maxfeat10k"
    base_name = save_vectorizer_and_matrix(vectorizer, X_tfidf, y_train, OUTPUT_DIR, config_name)
    
    # Análise de features
    analyze_top_features(vectorizer, X_tfidf, y_train, top_n=20)
    
    print("\n" + "="*70)
    print("VETORIZAÇÃO TF-IDF CONCLUÍDA COM SUCESSO!")
    print("="*70)
    print(f"\nArquivos salvos em: {OUTPUT_DIR}/")
    print(f"Nome base: {base_name}")
    print("\nPróximos passos:")
    print("1. Treinar classificadores com a matriz TF-IDF")
    print("2. Avaliar performance no conjunto de teste")
    print("3. Comparar com outras configurações de TF-IDF")

if __name__ == "__main__":
    main()
