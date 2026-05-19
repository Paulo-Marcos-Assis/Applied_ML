import pandas as pd
import re
from bs4 import BeautifulSoup
from unidecode import unidecode
import nltk
from nltk.corpus import stopwords

def download_nltk_resources():
    """Download necessary NLTK resources"""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

def remove_html_tags(text):
    """Remove HTML tags including script and style elements"""
    if pd.isna(text):
        return ""
    soup = BeautifulSoup(text, 'html.parser')
    for script in soup(["script", "style"]):
        script.decompose()
    return soup.get_text()

def remove_urls(text):
    """Remove URLs (http/https)"""
    if pd.isna(text):
        return ""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(url_pattern, '', text)

def remove_punctuation(text):
    """Remove punctuation and special characters, keeping only letters, numbers and spaces"""
    if pd.isna(text):
        return ""
    return re.sub(r'[^\w\s]', ' ', text)

def normalize_spaces(text):
    """Convert multiple spaces to single space and strip"""
    if pd.isna(text):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def to_lowercase(text):
    """Convert text to lowercase"""
    if pd.isna(text):
        return ""
    return text.lower()

def remove_accents(text):
    """Remove accents and normalize to ASCII"""
    if pd.isna(text):
        return ""
    return unidecode(text)

def remove_stopwords_pt(text):
    """Remove Portuguese stopwords"""
    if pd.isna(text):
        return ""
    stop_words = set(stopwords.words('portuguese'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def preprocess_text(text):
    """Apply all preprocessing steps in sequence"""
    # 1. Limpeza Básica
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = normalize_spaces(text)
    
    # 2. Normalização
    text = to_lowercase(text)
    text = remove_accents(text)
    
    # 3. Stopwords
    text = remove_stopwords_pt(text)
    
    # Final cleanup
    text = normalize_spaces(text)
    
    return text

def preprocess_dataset(input_file, output_file):
    """Preprocess a dataset CSV file"""
    print(f"Carregando dataset: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Total de linhas: {len(df)}")
    print(f"Colunas: {df.columns.tolist()}")
    
    # Apply preprocessing to title and text columns
    print("Aplicando pré-processamento em 'title'...")
    df['title_clean'] = df['title'].apply(preprocess_text)
    
    print("Aplicando pré-processamento em 'text'...")
    df['text_clean'] = df['text'].apply(preprocess_text)
    
    # Save preprocessed dataset
    print(f"Salvando dataset processado: {output_file}")
    df.to_csv(output_file, index=False)
    
    print(f"✓ Dataset processado salvo com sucesso!")
    print(f"  - Linhas: {len(df)}")
    print(f"  - Colunas: {df.columns.tolist()}")
    
    return df

if __name__ == "__main__":
    # Download NLTK resources
    print("Baixando recursos NLTK...")
    download_nltk_resources()
    
    # Process TRAIN dataset
    print("\n" + "="*60)
    print("PROCESSANDO DATASET DE TREINO")
    print("="*60)
    train_df = preprocess_dataset(
        'dataset/cleaned_data/FOR_TRAINING/TRAIN.csv',
        'dataset/cleaned_data/FOR_TRAINING/TRAIN_PREPROCESSED.csv'
    )
    
    # Process TEST dataset
    print("\n" + "="*60)
    print("PROCESSANDO DATASET DE TESTE")
    print("="*60)
    test_df = preprocess_dataset(
        'dataset/cleaned_data/FOR_TRAINING/TEST.csv',
        'dataset/cleaned_data/FOR_TRAINING/TEST_PREPROCESSED.csv'
    )
    
    # Show examples
    print("\n" + "="*60)
    print("EXEMPLOS DE PRÉ-PROCESSAMENTO")
    print("="*60)
    
    print("\n--- EXEMPLO 1 (TREINO) ---")
    idx = 0
    print(f"\nTítulo original:\n{train_df.iloc[idx]['title'][:200]}...")
    print(f"\nTítulo processado:\n{train_df.iloc[idx]['title_clean'][:200]}...")
    
    print(f"\nTexto original:\n{train_df.iloc[idx]['text'][:300]}...")
    print(f"\nTexto processado:\n{train_df.iloc[idx]['text_clean'][:300]}...")
    
    print("\n" + "="*60)
    print("PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print("="*60)
