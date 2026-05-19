import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from scipy import sparse
from datetime import datetime

def load_tfidf_vectorizer(tfidf_dir):
    """Carregar vetorizador TF-IDF salvo do treino"""
    print(f"\nCarregando vetorizador TF-IDF de: {tfidf_dir}")
    
    # Encontrar arquivo do vetorizador
    vectorizer_files = [f for f in os.listdir(tfidf_dir) if f.endswith('_vectorizer.pkl')]
    if not vectorizer_files:
        raise FileNotFoundError(f"Nenhum vetorizador encontrado em {tfidf_dir}")
    
    vectorizer_path = os.path.join(tfidf_dir, vectorizer_files[0])
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    print(f"✓ Vetorizador carregado: {vectorizer_files[0]}")
    print(f"  - Features: {len(vectorizer.get_feature_names_out())}")
    print(f"  - N-gram range: {vectorizer.ngram_range}")
    
    return vectorizer

def load_test_data(test_path):
    """Carregar dados de teste pré-processados"""
    print(f"\nCarregando dados de teste: {test_path}")
    df_test = pd.read_csv(test_path)
    print(f"✓ Dados de teste carregados: {len(df_test)} exemplos")
    print(f"  - Positivos: {len(df_test[df_test['label'] == 1])}")
    print(f"  - Negativos: {len(df_test[df_test['label'] == 0])}")
    return df_test

def concatenate_test_text(df, title_col='title_clean', text_col='text_clean'):
    """Concatenar título e texto do teste"""
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

def vectorize_test_data(vectorizer, test_texts):
    """Vetorizar dados de teste usando vetorizador existente"""
    print(f"\nVetorizando {len(test_texts)} textos de teste...")
    
    # Transformar textos (sem fit - usa vocabulário do treino)
    X_test = vectorizer.transform(test_texts)
    
    print(f"✓ Dados de teste vetorizados!")
    print(f"  - Shape da matriz: {X_test.shape}")
    print(f"  - Tipo: {type(X_test)}")
    print(f"  - Densidade: {X_test.nnz / (X_test.shape[0] * X_test.shape[1]):.6f}")
    
    return X_test

def save_test_vectorization(X_test, y_test, output_dir, vectorizer_info):
    """Salvar matriz de teste e labels"""
    
    # Criar nome base com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"test_tfidf_{timestamp}"
    
    # Salvar matriz TF-IDF
    matrix_path = os.path.join(output_dir, f"{base_name}_matrix.npz")
    sparse.save_npz(matrix_path, X_test)
    print(f"✓ Matriz de teste salva: {matrix_path}")
    
    # Salvar labels
    labels_path = os.path.join(output_dir, f"{base_name}_labels.npy")
    np.save(labels_path, y_test)
    print(f"✓ Labels de teste salvos: {labels_path}")
    
    # Salvar metadados
    metadata = {
        'timestamp': timestamp,
        'dataset_type': 'test',
        'n_samples': X_test.shape[0],
        'n_features': X_test.shape[1],
        'vectorizer_config': vectorizer_info,
        'source': 'TEST_PREPROCESSED.csv',
        'preprocessing': 'text_clean + title_clean',
        'purpose': 'Final evaluation of best model'
    }
    
    metadata_path = os.path.join(output_dir, f"{base_name}_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Metadados salvos: {metadata_path}")
    
    return base_name

def analyze_test_vectorization(X_test, y_test):
    """Analisar características da vetorização de teste"""
    print("\n" + "="*70)
    print("ANÁLISE DA VETORIZAÇÃO DE TESTE")
    print("="*70)
    
    # Estatísticas básicas
    print(f"Estatísticas da matriz de teste:")
    print(f"  - Amostras: {X_test.shape[0]}")
    print(f"  - Features: {X_test.shape[1]}")
    print(f"  - Elementos não-zero: {X_test.nnz}")
    print(f"  - Densidade: {X_test.nnz / (X_test.shape[0] * X_test.shape[1]):.6f}")
    print(f"  - Sparsity: {(1 - X_test.nnz / (X_test.shape[0] * X_test.shape[1])) * 100:.2f}%")
    
    # Média de features por documento
    avg_features = X_test.nnz / X_test.shape[0]
    print(f"  - Média de features por documento: {avg_features:.1f}")
    
    # Distribuição de classes
    n_pos = np.sum(y_test == 1)
    n_neg = np.sum(y_test == 0)
    print(f"\nDistribuição de classes no teste:")
    print(f"  - Positivos: {n_pos} ({n_pos/len(y_test)*100:.1f}%)")
    print(f"  - Negativos: {n_neg} ({n_neg/len(y_test)*100:.1f}%)")
    
    return avg_features

def main():
    """Função principal para vetorizar conjunto de teste"""
    print("\n" + "="*70)
    print("VETORIZAÇÃO DO CONJUNTO DE TESTE - TF-IDF")
    print("="*70)
    
    # Configurações
    TFIDF_DIR = 'vectorization/tf_idf'
    TEST_PATH = 'dataset/cleaned_data_no_bias/FOR_TRAINING/Pre_processed_for_Sparse/TEST_PREPROCESSED.csv'
    OUTPUT_DIR = 'vectorization/tf_idf/test'
    
    # Criar diretório de saída
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Diretório de saída criado: {OUTPUT_DIR}")
    
    # Carregar vetorizador do treino
    vectorizer = load_tfidf_vectorizer(TFIDF_DIR)
    vectorizer_info = {
        'ngram_range': vectorizer.ngram_range,
        'min_df': vectorizer.min_df,
        'max_df': vectorizer.max_df,
        'max_features': vectorizer.max_features,
        'norm': vectorizer.norm,
        'vocabulary_size': len(vectorizer.get_feature_names_out())
    }
    
    # Carregar dados de teste
    df_test = load_test_data(TEST_PATH)
    
    # Concatenar título e texto
    df_test = concatenate_test_text(df_test)
    
    # Vetorizar dados de teste
    X_test = vectorize_test_data(vectorizer, df_test['combined_text'])
    
    # Extrair labels
    y_test = df_test['label'].values
    
    # Analisar vetorização
    avg_features = analyze_test_vectorization(X_test, y_test)
    
    # Salvar resultados
    base_name = save_test_vectorization(X_test, y_test, OUTPUT_DIR, vectorizer_info)
    
    print("\n" + "="*70)
    print("VETORIZAÇÃO DE TESTE CONCLUÍDA")
    print("="*70)
    print(f"\nArquivos salvos em: {OUTPUT_DIR}/")
    print(f"Nome base: {base_name}")
    print(f"\nArquivos criados:")
    print(f"  - {base_name}_matrix.npz (matriz TF-IDF)")
    print(f"  - {base_name}_labels.npy (labels)")
    print(f"  - {base_name}_metadata.pkl (metadados)")
    
    print(f"\nPróximos passos:")
    print(f"1. Carregar modelo TF-IDF + SVM treinado")
    print(f"2. Fazer predições na matriz de teste")
    print(f"3. Calcular métricas finais")
    print(f"4. Gerar relatório final de avaliação")
    
    print(f"\nCaminho da matriz de teste para uso:")
    print(f"📁 {OUTPUT_DIR}/{base_name}_matrix.npz")
    
    return OUTPUT_DIR, base_name

if __name__ == "__main__":
    output_dir, base_name = main()
