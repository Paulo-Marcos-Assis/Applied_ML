import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def create_output_dir(base_path):
    """Criar diretório de saída se não existir"""
    os.makedirs(base_path, exist_ok=True)
    print(f"✓ Diretório criado/verificado: {base_path}")

def load_bert_data(train_path):
    """Carregar dados BERT (com acentos e pontuação)"""
    print(f"\nCarregando dados BERT: {train_path}")
    df_train = pd.read_csv(train_path)
    print(f"✓ Dados carregados: {len(df_train)} exemplos")
    print(f"  - Positivos: {len(df_train[df_train['label'] == 1])}")
    print(f"  - Negativos: {len(df_train[df_train['label'] == 0])}")
    return df_train

def concatenate_title_text_bert(df, title_col='title_bert', text_col='text_bert'):
    """Concatenar título e texto BERT em nova coluna"""
    print(f"\nCriando coluna 'title_text_bert' concatenando '{title_col}' + '{text_col}'...")
    df['title_text_bert'] = df[title_col].fillna('') + ' ' + df[text_col].fillna('')
    
    # Estatísticas
    avg_length = df['title_text_bert'].str.split().str.len().mean()
    min_length = df['title_text_bert'].str.split().str.len().min()
    max_length = df['title_text_bert'].str.split().str.len().max()
    
    print(f"✓ Coluna 'title_text_bert' criada")
    print(f"  - Tamanho médio: {avg_length:.1f} palavras")
    print(f"  - Min: {min_length} | Max: {max_length} palavras")
    
    return df

def load_bert_model(model_name):
    """Carregar modelo e tokenizer BERT"""
    print(f"\nCarregando modelo BERT: {model_name}")
    
    try:
        # Carregar tokenizer e modelo
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Verificar GPU disponível
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        print(f"✓ Modelo BERT carregado com sucesso!")
        print(f"  - Device: {device}")
        print(f"  - Modelo: {model.config.model_type}")
        print(f"  - Camadas: {model.config.num_hidden_layers}")
        print(f"  - Dimensão: {model.config.hidden_size}")
        
        return tokenizer, model, device
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return None, None, None

def tokenize_texts(texts, tokenizer, max_length=512):
    """Tokenizar textos para BERT"""
    print(f"\nTokenizando {len(texts)} textos...")
    
    # Tokenizar
    encodings = tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    print(f"✓ Tokenização concluída!")
    print(f"  - Input IDs shape: {encodings['input_ids'].shape}")
    print(f"  - Attention mask shape: {encodings['attention_mask'].shape}")
    
    return encodings

def get_bert_embeddings(model, encodings, device, batch_size=16):
    """Extrair embeddings BERT (CLS token)"""
    print(f"\nExtraindo embeddings BERT...")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Total exemplos: {len(encodings['input_ids'])}")
    
    # Criar dataset
    dataset = TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask']
    )
    
    # Criar dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Extrair embeddings
    all_embeddings = []
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (batch_input_ids, batch_attention_mask) in enumerate(tqdm(dataloader)):
            # Mover para device
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            
            # Forward pass
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask
            )
            
            # Usar embedding do token [CLS] (primeiro token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().numpy())
            
            # Progresso
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processados: {(batch_idx + 1) * batch_size}/{len(encodings['input_ids'])}")
    
    # Concatenar todos os embeddings
    embeddings = np.concatenate(all_embeddings, axis=0)
    
    print(f"✓ Embeddings BERT extraídos!")
    print(f"  - Shape final: {embeddings.shape}")
    print(f"  - Tipo: {embeddings.dtype}")
    
    return embeddings

def save_bert_artifacts(X_bert, y_train, output_dir, model_name):
    """Salvar matriz BERT e metadados"""
    
    # Criar nome base com timestamp e modelo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split('/')[-1]  # Ex: 'bert-base-portuguese-cased'
    base_name = f"bert_{model_short}_{timestamp}"
    
    # Salvar matriz BERT
    matrix_path = os.path.join(output_dir, f"{base_name}_matrix.npy")
    np.save(matrix_path, X_bert)
    print(f"✓ Matriz BERT salva: {matrix_path}")
    
    # Salvar labels
    labels_path = os.path.join(output_dir, f"{base_name}_labels.npy")
    np.save(labels_path, y_train)
    print(f"✓ Labels salvos: {labels_path}")
    
    # Salvar metadados
    metadata = {
        'timestamp': timestamp,
        'model_name': model_name,
        'model_short': model_short,
        'n_samples': X_bert.shape[0],
        'n_features': X_bert.shape[1],
        'embedding_dim': X_bert.shape[1],
        'source': 'BERT contextual embeddings',
        'pooling_strategy': 'CLS_token',
        'max_length': 512,
        'batch_size': 16
    }
    
    metadata_path = os.path.join(output_dir, f"{base_name}_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Metadados salvos: {metadata_path}")
    
    return base_name

def analyze_bert_embeddings(X_bert, y_train, model_name):
    """Analisar embeddings BERT gerados"""
    print("\n" + "="*70)
    print(f"ANÁLISE DOS EMBEDDINGS BERT - {model_name}")
    print("="*70)
    
    # Estatísticas básicas
    print(f"Estatísticas da matriz de embeddings:")
    print(f"  - Média: {X_bert.mean():.4f}")
    print(f"  - Desvio padrão: {X_bert.std():.4f}")
    print(f"  - Min: {X_bert.min():.4f}")
    print(f"  - Max: {X_bert.max():.4f}")
    
    # Separar embeddings por classe
    X_pos = X_bert[y_train == 1]
    X_neg = X_bert[y_train == 0]
    
    print(f"\nEmbeddings por classe:")
    print(f"  - Classe positiva: média={X_pos.mean():.4f}, std={X_pos.std():.4f}")
    print(f"  - Classe negativa: média={X_neg.mean():.4f}, std={X_neg.std():.4f}")
    
    # Similaridade média entre classes
    from sklearn.metrics.pairwise import cosine_similarity
    pos_mean = X_pos.mean(axis=0, keepdims=True)
    neg_mean = X_neg.mean(axis=0, keepdims=True)
    similarity = cosine_similarity(pos_mean, neg_mean)[0,0]
    
    print(f"\nSimilaridade coseno entre médias das classes: {similarity:.4f}")
    print(f"  - Valores próximos de 1 = classes similares")
    print(f"  - Valores próximos de 0 = classes diferentes")

def vectorize_bert_model(model_name, output_dir):
    """Função principal para vetorizar com um modelo BERT específico"""
    print("\n" + "="*70)
    print(f"VETORIZAÇÃO BERT - {model_name}")
    print("="*70)
    
    # Configurações
    BERT_TRAIN_PATH = 'dataset/cleaned_data_no_bias/FOR_TRAINING/Pre_processed_for_Embeddings/TRAIN_BERT.csv'
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    
    # Criar diretório de saída
    create_output_dir(output_dir)
    
    # Carregar dados BERT
    df_train = load_bert_data(BERT_TRAIN_PATH)
    
    # Criar coluna concatenada
    df_train = concatenate_title_text_bert(df_train)
    
    # Carregar modelo BERT
    tokenizer, model, device = load_bert_model(model_name)
    if not tokenizer or not model:
        return None
    
    # Tokenizar textos
    encodings = tokenize_texts(df_train['title_text_bert'], tokenizer, MAX_LENGTH)
    
    # Extrair embeddings
    X_bert = get_bert_embeddings(model, encodings, device, BATCH_SIZE)
    
    # Extrair labels
    y_train = df_train['label'].values
    
    # Salvar artefatos
    base_name = save_bert_artifacts(X_bert, y_train, output_dir, model_name)
    
    # Análise dos embeddings
    analyze_bert_embeddings(X_bert, y_train, model_name)
    
    print("\n" + "="*70)
    print(f"VETORIZAÇÃO BERT CONCLUÍDA - {model_name}")
    print("="*70)
    print(f"\nArquivos salvos em: {output_dir}/")
    print(f"Nome base: {base_name}")
    
    return base_name

def main():
    """Função principal - vetorizar ambos os modelos BERT"""
    print("\n" + "="*70)
    print("VETORIZAÇÃO BERT - MODELOS NEURALMIND")
    print("="*70)
    
    # Modelos BERT para vetorizar
    bert_models = [
        {
            'name': 'neuralmind/bert-base-portuguese-cased',
            'output_dir': 'vectorization/bert_base',
            'description': 'BERT-Base (12 camadas, 110M parâmetros)'
        },
        {
            'name': 'neuralmind/bert-large-portuguese-cased',
            'output_dir': 'vectorization/bert_large',
            'description': 'BERT-Large (24 camadas, 335M parâmetros)'
        }
    ]
    
    # Instalar dependências se necessário
    try:
        import transformers
        import torch
    except ImportError:
        print("Instalando dependências...")
        os.system("pip install transformers torch")
    
    # Vetorizar cada modelo
    results = {}
    
    for model_config in bert_models:
        print(f"\n{'='*20} {model_config['description']} {'='*20}")
        
        base_name = vectorize_bert_model(
            model_config['name'],
            model_config['output_dir']
        )
        
        if base_name:
            results[model_config['name']] = {
                'base_name': base_name,
                'output_dir': model_config['output_dir'],
                'description': model_config['description']
            }
    
    # Resumo final
    print("\n" + "="*70)
    print("RESUMO DA VETORIZAÇÃO BERT")
    print("="*70)
    
    for model_name, info in results.items():
        print(f"\n✅ {model_name}")
        print(f"   - Base name: {info['base_name']}")
        print(f"   - Output dir: {info['output_dir']}")
        print(f"   - Description: {info['description']}")
    
    print(f"\nTotal de modelos processados: {len(results)}")
    print("\nPróximos passos:")
    print("1. Treinar classificadores com embeddings BERT")
    print("2. Comparar performance com TF-IDF e FastText")
    print("3. Experimentar Albertina (se disponível)")

if __name__ == "__main__":
    main()
