import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

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

def download_nilc_embeddings(output_dir):
    """
    Baixar NILC Word Embeddings pré-treinados
    NOTA: Esta função simula o download. Na prática, você precisará baixar manualmente
    """
    print("\n" + "="*70)
    print("EMBEDDINGS NILC WORD2VEC")
    print("="*70)
    print("NOTA: Você precisará baixar manualmente os embeddings NILC:")
    print("1. Acesse: http://www.nilc.icmc.usp.br/embeddings")
    print("2. Baixe 'Word2Vec (CBOW)' - 300 dimensões")
    print("3. Salve como: word2vec_cbow_s300.zip")
    print("4. Extraia para: vectorization/word2vec/word2vec_cbow_s300.txt")
    print("="*70)
    
    # Criar diretório
    word2vec_dir = os.path.join(output_dir, 'models')
    os.makedirs(word2vec_dir, exist_ok=True)
    
    # Caminho esperado
    model_path = os.path.join(word2vec_dir, 'word2vec_cbow_s300.txt')
    
    if os.path.exists(model_path):
        print(f"✓ Embeddings NILC encontrados: {model_path}")
        return model_path
    else:
        print(f"⚠ Embeddings NILC NÃO encontrados: {model_path}")
        print("Por favor, baixe os embeddings NILC e salve no local acima.")
        return None

def load_word2vec_model(model_path):
    """Carregar modelo Word2Vec pré-treinado"""
    print(f"\nCarregando modelo Word2Vec: {model_path}")
    
    try:
        # Carregar modelo
        model = KeyedVectors.load_word2vec_format(model_path, binary=False)
        print(f"✓ Modelo carregado com sucesso!")
        print(f"  - Vocabulário: {len(model.key_to_index)} palavras")
        print(f"  - Dimensões: {model.vector_size}")
        
        return model
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
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
    
    # Fit apenas para obter pesos (não vamos usar a matriz TF-IDF diretamente)
    tfidf.fit(texts)
    
    print(f"✓ Pesos TF-IDF criados: {len(tfidf.vocabulary_)} termos")
    return tfidf

def text_to_weighted_embedding(text, word2vec_model, tfidf_vectorizer, embedding_dim=300):
    """
    Converter texto para embedding ponderado por TF-IDF
    
    Parâmetros:
    - text: texto pré-processado
    - word2vec_model: modelo Word2Vec carregado
    - tfidf_vectorizer: vetorizador TF-IDF para pesos
    - embedding_dim: dimensão dos embeddings (300)
    """
    if pd.isna(text) or not text.strip():
        return np.zeros(embedding_dim)
    
    words = text.split()
    word_embeddings = []
    word_weights = []
    
    for word in words:
        # Verificar se palavra está no vocabulário Word2Vec
        if word in word2vec_model.key_to_index:
            # Obter embedding
            embedding = word2vec_model[word]
            word_embeddings.append(embedding)
            
            # Obter peso TF-IDF (se palavra no vocabulário TF-IDF)
            if word in tfidf_vectorizer.vocabulary_:
                weight = tfidf_vectorizer.idf_[tfidf_vectorizer.vocabulary_[word]]
            else:
                weight = 1.0  # peso neutro se não tiver TF-IDF
            
            word_weights.append(weight)
    
    # Se não encontrou nenhuma palavra conhecida
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

def vectorize_documents(df, word2vec_model, tfidf_vectorizer, embedding_dim=300):
    """
    Vetorizar todos os documentos usando Word2Vec ponderado por TF-IDF
    """
    print(f"\nVetorizando {len(df)} documentos com Word2Vec ponderado...")
    
    # Aplicar vetorização
    embeddings = df['title_text'].apply(
        lambda x: text_to_weighted_embedding(x, word2vec_model, tfidf_vectorizer, embedding_dim)
    )
    
    # Converter para matriz numpy
    X_word2vec = np.array(embeddings.tolist())
    
    print(f"✓ Vetorização concluída!")
    print(f"  - Shape da matriz: {X_word2vec.shape}")
    print(f"  - Tipo: {X_word2vec.dtype}")
    
    # Verificar embeddings nulos (documentos sem palavras conhecidas)
    null_embeddings = np.sum(np.all(X_word2vec == 0, axis=1))
    print(f"  - Documentos com embedding nulo: {null_embeddings} ({100*null_embeddings/len(df):.1f}%)")
    
    return X_word2vec

def save_word2vec_artifacts(X_word2vec, y_train, output_dir, config_name):
    """Salvar matriz Word2Vec e metadados"""
    
    # Criar nome base com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"word2vec_{config_name}_{timestamp}"
    
    # Salvar matriz Word2Vec
    matrix_path = os.path.join(output_dir, f"{base_name}_matrix.npy")
    np.save(matrix_path, X_word2vec)
    print(f"✓ Matriz Word2Vec salva: {matrix_path}")
    
    # Salvar labels
    labels_path = os.path.join(output_dir, f"{base_name}_labels.npy")
    np.save(labels_path, y_train)
    print(f"✓ Labels salvos: {labels_path}")
    
    # Salvar metadados
    metadata = {
        'timestamp': timestamp,
        'config_name': config_name,
        'n_samples': X_word2vec.shape[0],
        'n_features': X_word2vec.shape[1],
        'embedding_dim': X_word2vec.shape[1],
        'source': 'NILC Word2Vec CBOW 300d',
        'weighting': 'TF-IDF weighted average',
        'null_embeddings': int(np.sum(np.all(X_word2vec == 0, axis=1)))
    }
    
    metadata_path = os.path.join(output_dir, f"{base_name}_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Metadados salvos: {metadata_path}")
    
    return base_name

def analyze_embeddings(X_word2vec, y_train, word2vec_model):
    """Analisar embeddings gerados"""
    print("\n" + "="*70)
    print("ANÁLISE DOS WORD EMBEDDINGS")
    print("="*70)
    
    # Estatísticas básicas
    print(f"Estatísticas da matriz de embeddings:")
    print(f"  - Média: {X_word2vec.mean():.4f}")
    print(f"  - Desvio padrão: {X_word2vec.std():.4f}")
    print(f"  - Min: {X_word2vec.min():.4f}")
    print(f"  - Max: {X_word2vec.max():.4f}")
    
    # Verificar palavras mais similares a 'fraude'
    if 'fraude' in word2vec_model.key_to_index:
        print(f"\nTop 10 palavras mais similares a 'fraude':")
        similar_words = word2vec_model.most_similar('fraude', topn=10)
        for i, (word, similarity) in enumerate(similar_words, 1):
            print(f"  {i:2d}. {word:20s} ({similarity:.4f})")
    else:
        print(f"\n⚠ Palavra 'fraude' não encontrada no vocabulário Word2Vec")
    
    # Separar embeddings por classe
    X_pos = X_word2vec[y_train == 1]
    X_neg = X_word2vec[y_train == 0]
    
    print(f"\nEmbeddings por classe:")
    print(f"  - Classe positiva: média={X_pos.mean():.4f}, std={X_pos.std():.4f}")
    print(f"  - Classe negativa: média={X_neg.mean():.4f}, std={X_neg.std():.4f}")

def main():
    """Função principal"""
    print("\n" + "="*70)
    print("VETORIZAÇÃO WORD2VEC - CONJUNTO DE TREINO")
    print("="*70)
    
    # Configurações
    TRAIN_PATH = 'dataset/cleaned_data/FOR_TRAINING/Pre_processed_for_Sparse/TRAIN_PREPROCESSED.csv'
    OUTPUT_DIR = 'vectorization/word2vec'
    EMBEDDING_DIM = 300
    
    # Criar diretório de saída
    create_output_dir(OUTPUT_DIR)
    
    # Carregar dados
    df_train = load_preprocessed_data(TRAIN_PATH)
    
    # Criar coluna concatenada title_text
    df_train = concatenate_title_text(df_train)
    
    # Baixar/carregar embeddings NILC
    model_path = download_nilc_embeddings(OUTPUT_DIR)
    if not model_path:
        print("\n❌ Execute o download dos embeddings NILC e execute novamente.")
        return
    
    # Carregar modelo Word2Vec
    word2vec_model = load_word2vec_model(model_path)
    if not word2vec_model:
        return
    
    # Criar pesos TF-IDF
    tfidf_vectorizer = create_tfidf_weights(df_train['title_text'])
    
    # Vetorizar documentos
    X_word2vec = vectorize_documents(df_train, word2vec_model, tfidf_vectorizer, EMBEDDING_DIM)
    
    # Extrair labels
    y_train = df_train['label'].values
    
    # Salvar artefatos
    config_name = "nilc_cbow_300d_tfidf_weighted"
    base_name = save_word2vec_artifacts(X_word2vec, y_train, OUTPUT_DIR, config_name)
    
    # Análise dos embeddings
    analyze_embeddings(X_word2vec, y_train, word2vec_model)
    
    print("\n" + "="*70)
    print("VETORIZAÇÃO WORD2VEC CONCLUÍDA COM SUCESSO!")
    print("="*70)
    print(f"\nArquivos salvos em: {OUTPUT_DIR}/")
    print(f"Nome base: {base_name}")
    print("\nPróximos passos:")
    print("1. Treinar classificadores com a matriz Word2Vec")
    print("2. Comparar performance com TF-IDF")
    print("3. Experimentar FastText embeddings")

if __name__ == "__main__":
    main()
