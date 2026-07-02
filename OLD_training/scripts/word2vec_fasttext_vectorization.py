import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer

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
    """Concatenar título e texto limpos em nova coluna"""
    print(f"\nCriando coluna 'title_text' concatenando '{title_col}' + '{text_col}'...")
    df['title_text'] = df[title_col].fillna('') + ' ' + df[text_col].fillna('')
    
    # Estatísticas
    avg_length = df['title_text'].str.split().str.len().mean()
    min_length = df['title_text'].str.split().str.len().min()
    max_length = df['title_text'].str.split().str.len().max()
    
    print(f"✓ Coluna 'title_text' criada")
    print(f"  - Tamanho médio: {avg_length:.1f} palavras")
    print(f"  - Min: {min_length} | Max: {max_length} palavras")
    
    return df

def load_fasttext_embeddings(model_path):
    """Carregar embeddings FastText pré-treinados"""
    print(f"\nCarregando embeddings FastText: {model_path}")
    
    try:
        # Carregar modelo (limitar para carregamento mais rápido)
        print("Carregando embeddings (isso pode levar alguns minutos)...")
        model = KeyedVectors.load_word2vec_format(model_path, binary=False)
        print(f"✓ Embeddings FastText carregados com sucesso!")
        print(f"  - Vocabulário: {len(model.key_to_index)} palavras")
        print(f"  - Dimensões: {model.vector_size}")
        
        return model
    except Exception as e:
        print(f"❌ Erro ao carregar embeddings: {e}")
        return None

def create_tfidf_weights(texts, max_features=10000):
    """
    Criar pesos TF-IDF para ponderação de word embeddings
    """
    print("\nCriando pesos TF-IDF para ponderação...")
    
    # Usar mesma configuração da vetorização TF-IDF anterior
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=max_features,
        norm='l2',
        sublinear_tf=True
    )
    
    # Fit apenas para obter pesos
    tfidf.fit(texts)
    
    print(f"✓ Pesos TF-IDF criados: {len(tfidf.vocabulary_)} termos")
    return tfidf

def text_to_weighted_embedding(text, embedding_model, tfidf_vectorizer, embedding_dim=300):
    """
    Converter texto para embedding ponderado por TF-IDF
    Ignora palavras desconhecidas (conforme solicitado)
    """
    if pd.isna(text) or not text.strip():
        return np.zeros(embedding_dim)
    
    words = text.split()
    word_embeddings = []
    word_weights = []
    
    for word in words:
        # Ignorar palavras desconhecidas (conforme solicitado)
        if word in embedding_model.key_to_index:
            # Obter embedding
            embedding = embedding_model[word]
            word_embeddings.append(embedding)
            
            # Obter peso TF-IDF
            if word in tfidf_vectorizer.vocabulary_:
                weight = tfidf_vectorizer.idf_[tfidf_vectorizer.vocabulary_[word]]
            else:
                weight = 1.0  # peso neutro
            
            word_weights.append(weight)
    
    # Se não encontrou nenhuma palavra conhecida, retornar vetor zero
    if not word_embeddings:
        return np.zeros(embedding_dim)
    
    # Converter para arrays numpy
    word_embeddings = np.array(word_embeddings)
    word_weights = np.array(word_weights)
    
    # Normalizar pesos
    if word_weights.sum() > 0:
        word_weights = word_weights / word_weights.sum()
    
    # Calcular embedding ponderado
    weighted_embedding = np.average(word_embeddings, axis=0, weights=word_weights)
    
    return weighted_embedding

def vectorize_documents(df, embedding_model, tfidf_vectorizer, embedding_dim=300):
    """
    Vetorizar todos os documentos usando FastText ponderado por TF-IDF
    """
    print(f"\nVetorizando {len(df)} documentos com FastText ponderado...")
    
    # Aplicar vetorização
    embeddings = df['title_text'].apply(
        lambda x: text_to_weighted_embedding(x, embedding_model, tfidf_vectorizer, embedding_dim)
    )
    
    # Converter para matriz numpy
    X_embeddings = np.array(embeddings.tolist())
    
    print(f"✓ Vetorização concluída!")
    print(f"  - Shape da matriz: {X_embeddings.shape}")
    print(f"  - Tipo: {X_embeddings.dtype}")
    
    # Verificar embeddings nulos (documentos sem palavras conhecidas)
    null_embeddings = np.sum(np.all(X_embeddings == 0, axis=1))
    print(f"  - Documentos com embedding nulo: {null_embeddings} ({100*null_embeddings/len(df):.1f}%)")
    
    return X_embeddings

def save_embeddings_artifacts(X_embeddings, y_train, output_dir, config_name):
    """Salvar matriz de embeddings e metadados"""
    
    # Criar nome base com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"fasttext_{config_name}_{timestamp}"
    
    # Salvar matriz de embeddings
    matrix_path = os.path.join(output_dir, f"{base_name}_matrix.npy")
    np.save(matrix_path, X_embeddings)
    print(f"✓ Matriz de embeddings salva: {matrix_path}")
    
    # Salvar labels
    labels_path = os.path.join(output_dir, f"{base_name}_labels.npy")
    np.save(labels_path, y_train)
    print(f"✓ Labels salvos: {labels_path}")
    
    # Salvar metadados
    metadata = {
        'timestamp': timestamp,
        'config_name': config_name,
        'n_samples': X_embeddings.shape[0],
        'n_features': X_embeddings.shape[1],
        'embedding_dim': X_embeddings.shape[1],
        'source': 'FastText cc.pt.300d',
        'weighting': 'TF-IDF weighted average',
        'null_embeddings': int(np.sum(np.all(X_embeddings == 0, axis=1)))
    }
    
    metadata_path = os.path.join(output_dir, f"{base_name}_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Metadados salvos: {metadata_path}")
    
    return base_name

def analyze_embeddings(X_embeddings, y_train, embedding_model):
    """Analisar embeddings gerados"""
    print("\n" + "="*70)
    print("ANÁLISE DOS FASTTEXT EMBEDDINGS")
    print("="*70)
    
    # Estatísticas básicas
    print(f"Estatísticas da matriz de embeddings:")
    print(f"  - Média: {X_embeddings.mean():.4f}")
    print(f"  - Desvio padrão: {X_embeddings.std():.4f}")
    print(f"  - Min: {X_embeddings.min():.4f}")
    print(f"  - Max: {X_embeddings.max():.4f}")
    
    # Verificar palavras mais similares a 'fraude'
    if 'fraude' in embedding_model.key_to_index:
        print(f"\nTop 10 palavras mais similares a 'fraude':")
        similar_words = embedding_model.most_similar('fraude', topn=10)
        for i, (word, similarity) in enumerate(similar_words, 1):
            print(f"  {i:2d}. {word:20s} ({similarity:.4f})")
    else:
        print(f"\n⚠ Palavra 'fraude' não encontrada no vocabulário FastText")
    
    # Separar embeddings por classe
    X_pos = X_embeddings[y_train == 1]
    X_neg = X_embeddings[y_train == 0]
    
    print(f"\nEmbeddings por classe:")
    print(f"  - Classe positiva: média={X_pos.mean():.4f}, std={X_pos.std():.4f}")
    print(f"  - Classe negativa: média={X_neg.mean():.4f}, std={X_neg.std():.4f}")

def main():
    """Função principal"""
    print("\n" + "="*70)
    print("VETORIZAÇÃO FASTTEXT - CONJUNTO DE TREINO")
    print("="*70)
    
    # Configurações
    TRAIN_PATH = 'dataset/cleaned_data/FOR_TRAINING/Pre_processed_for_Sparse/TRAIN_PREPROCESSED.csv'
    OUTPUT_DIR = 'vectorization/fasttext'
    EMBEDDING_DIM = 300
    MODEL_PATH = 'vectorization/word2vec/models/cc.pt.300.vec'
    
    # Criar diretório de saída
    create_output_dir(OUTPUT_DIR)
    
    # Carregar dados
    df_train = load_preprocessed_data(TRAIN_PATH)
    
    # Criar coluna concatenada title_text
    df_train = concatenate_title_text(df_train)
    
    # Verificar se embeddings foram baixados
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Embeddings FastText não encontrados: {MODEL_PATH}")
        print("Execute primeiro: python scripts/download_fasttext_embeddings.py")
        return
    
    # Carregar embeddings FastText
    embedding_model = load_fasttext_embeddings(MODEL_PATH)
    if not embedding_model:
        return
    
    # Criar pesos TF-IDF
    tfidf_vectorizer = create_tfidf_weights(df_train['title_text'])
    
    # Vetorizar documentos
    X_embeddings = vectorize_documents(df_train, embedding_model, tfidf_vectorizer, EMBEDDING_DIM)
    
    # Extrair labels
    y_train = df_train['label'].values
    
    # Salvar artefatos
    config_name = "cc_pt_300d_tfidf_weighted"
    base_name = save_embeddings_artifacts(X_embeddings, y_train, OUTPUT_DIR, config_name)
    
    # Análise dos embeddings
    analyze_embeddings(X_embeddings, y_train, embedding_model)
    
    print("\n" + "="*70)
    print("VETORIZAÇÃO FASTTEXT CONCLUÍDA COM SUCESSO!")
    print("="*70)
    print(f"\nArquivos salvos em: {OUTPUT_DIR}/")
    print(f"Nome base: {base_name}")
    print("\nPróximos passos:")
    print("1. Treinar classificadores com a matriz FastText")
    print("2. Comparar performance com TF-IDF")
    print("3. Experimentar treinar Word2Vec do zero")

if __name__ == "__main__":
    main()
