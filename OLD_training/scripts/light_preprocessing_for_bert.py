import pandas as pd
import re

def remove_urls(text):
    """Remove URLs mantendo todo o resto"""
    if pd.isna(text):
        return ""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(url_pattern, '', text)

def normalize_spaces(text):
    """Normalizar espaços múltiplos para espaço único"""
    if pd.isna(text):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def light_preprocess_for_bert(text):
    """
    Limpeza LEVE para modelos BERT
    MANTÉM: acentos, maiúsculas/minúsculas, pontuação
    REMOVE: URLs, espaços excessivos
    """
    text = remove_urls(text)
    text = normalize_spaces(text)
    return text

def preprocess_dataset_for_bert(input_file, output_file):
    """Preprocessar dataset com limpeza leve para BERT"""
    print(f"Carregando dataset: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Total de linhas: {len(df)}")
    print(f"Colunas: {df.columns.tolist()}")
    
    # Aplicar limpeza leve
    print("Aplicando limpeza leve em 'title' (mantendo acentos, maiúsculas, pontuação)...")
    df['title_bert'] = df['title'].apply(light_preprocess_for_bert)
    
    print("Aplicando limpeza leve em 'text' (mantendo acentos, maiúsculas, pontuação)...")
    df['text_bert'] = df['text'].apply(light_preprocess_for_bert)
    
    # Salvar dataset
    print(f"Salvando dataset: {output_file}")
    df.to_csv(output_file, index=False)
    
    print(f"✓ Dataset salvo com sucesso!")
    print(f"  - Linhas: {len(df)}")
    print(f"  - Colunas: {df.columns.tolist()}")
    
    return df

if __name__ == "__main__":
    # Processar TRAIN dataset
    print("\n" + "="*70)
    print("PROCESSANDO DATASET DE TREINO PARA MODELOS BERT")
    print("="*70)
    train_df = preprocess_dataset_for_bert(
        'dataset/cleaned_data/FOR_TRAINING/TRAIN.csv',
        'dataset/cleaned_data/FOR_TRAINING/TRAIN_BERT.csv'
    )
    
    # Processar TEST dataset
    print("\n" + "="*70)
    print("PROCESSANDO DATASET DE TESTE PARA MODELOS BERT")
    print("="*70)
    test_df = preprocess_dataset_for_bert(
        'dataset/cleaned_data/FOR_TRAINING/TEST.csv',
        'dataset/cleaned_data/FOR_TRAINING/TEST_BERT.csv'
    )
    
    # Mostrar exemplos
    print("\n" + "="*70)
    print("EXEMPLOS DE LIMPEZA LEVE PARA BERT")
    print("="*70)
    
    print("\n--- EXEMPLO 1 ---")
    idx = 1
    print(f"\nTítulo original:")
    print(train_df.iloc[idx]['title'])
    print(f"\nTítulo limpo (BERT):")
    print(train_df.iloc[idx]['title_bert'])
    
    print(f"\nTexto original (primeiros 300 chars):")
    print(train_df.iloc[idx]['text'][:300])
    print(f"\nTexto limpo BERT (primeiros 300 chars):")
    print(train_df.iloc[idx]['text_bert'][:300])
    
    # Verificar que manteve características importantes
    print("\n" + "="*70)
    print("VERIFICAÇÃO: CARACTERÍSTICAS MANTIDAS")
    print("="*70)
    
    has_accents = train_df['text_bert'].str.contains('[áàâãéêíóôõúüç]', regex=True, na=False).sum()
    has_uppercase = train_df['text_bert'].str.contains('[A-Z]', regex=True, na=False).sum()
    has_punctuation = train_df['text_bert'].str.contains('[.,!?;:]', regex=True, na=False).sum()
    
    print(f"✓ Textos com acentos: {has_accents}/{len(train_df)}")
    print(f"✓ Textos com maiúsculas: {has_uppercase}/{len(train_df)}")
    print(f"✓ Textos com pontuação: {has_punctuation}/{len(train_df)}")
    
    print("\n" + "="*70)
    print("LIMPEZA LEVE CONCLUÍDA - DATASETS PRONTOS PARA BERT!")
    print("="*70)
