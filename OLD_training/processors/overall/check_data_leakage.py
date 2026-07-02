"""
Script para verificar data leakage entre conjunto de treino e teste
Compara URLs, títulos e textos para identificar possíveis duplicações
"""

import pandas as pd
import numpy as np
from pathlib import Path

def check_leakage():
    """Verifica se há data leakage entre treino e teste"""
    
    print("="*80)
    print("VERIFICAÇÃO DE DATA LEAKAGE")
    print("="*80)
    print()
    
    # Caminhos dos arquivos
    base_path = Path("/home/paulo/CascadeProjects/Applied_ML/dataset/cleaned_data_no_bias/FOR_TRAINING")
    
    # Carregar dados de treino (original, antes do split)
    train_dev_path = base_path / "TRAIN.csv"
    test_path = base_path / "TEST.csv"
    
    print(f"Carregando dados de treino/desenvolvimento: {train_dev_path}")
    train_dev_df = pd.read_csv(train_dev_path)
    print(f"  - Tamanho: {len(train_dev_df)} exemplos")
    print(f"  - Colunas: {list(train_dev_df.columns)}")
    print()
    
    print(f"Carregando dados de teste: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"  - Tamanho: {len(test_df)} exemplos")
    print(f"  - Colunas: {list(test_df.columns)}")
    print()
    
    # Verificar distribuição de classes
    print("-"*80)
    print("DISTRIBUIÇÃO DE CLASSES")
    print("-"*80)
    print("\nTreino/Desenvolvimento:")
    print(train_dev_df['label'].value_counts())
    print(f"Proporção: {train_dev_df['label'].value_counts(normalize=True).to_dict()}")
    
    print("\nTeste:")
    print(test_df['label'].value_counts())
    print(f"Proporção: {test_df['label'].value_counts(normalize=True).to_dict()}")
    print()
    
    # 1. Verificar duplicação por URL
    print("-"*80)
    print("1. VERIFICAÇÃO DE DUPLICAÇÃO POR URL")
    print("-"*80)
    
    if 'url' in train_dev_df.columns and 'url' in test_df.columns:
        train_urls = set(train_dev_df['url'].dropna())
        test_urls = set(test_df['url'].dropna())
        
        common_urls = train_urls.intersection(test_urls)
        
        print(f"URLs únicas no treino: {len(train_urls)}")
        print(f"URLs únicas no teste: {len(test_urls)}")
        print(f"URLs em comum (LEAKAGE): {len(common_urls)}")
        
        if len(common_urls) > 0:
            print("\n⚠️  ALERTA: URLs duplicadas encontradas!")
            print(f"Percentual de leakage: {len(common_urls)/len(test_urls)*100:.2f}% do conjunto de teste")
            print("\nPrimeiras 10 URLs duplicadas:")
            for i, url in enumerate(list(common_urls)[:10], 1):
                print(f"  {i}. {url}")
        else:
            print("\n✓ Nenhuma URL duplicada encontrada")
    else:
        print("Coluna 'url' não encontrada nos datasets")
    print()
    
    # 2. Verificar duplicação por título
    print("-"*80)
    print("2. VERIFICAÇÃO DE DUPLICAÇÃO POR TÍTULO")
    print("-"*80)
    
    if 'title' in train_dev_df.columns and 'title' in test_df.columns:
        train_titles = set(train_dev_df['title'].dropna())
        test_titles = set(test_df['title'].dropna())
        
        common_titles = train_titles.intersection(test_titles)
        
        print(f"Títulos únicos no treino: {len(train_titles)}")
        print(f"Títulos únicos no teste: {len(test_titles)}")
        print(f"Títulos em comum (LEAKAGE): {len(common_titles)}")
        
        if len(common_titles) > 0:
            print("\n⚠️  ALERTA: Títulos duplicados encontrados!")
            print(f"Percentual de leakage: {len(common_titles)/len(test_titles)*100:.2f}% do conjunto de teste")
            print("\nPrimeiros 10 títulos duplicados:")
            for i, title in enumerate(list(common_titles)[:10], 1):
                print(f"  {i}. {title[:100]}...")
        else:
            print("\n✓ Nenhum título duplicado encontrado")
    else:
        print("Coluna 'title' não encontrada nos datasets")
    print()
    
    # 3. Verificar duplicação por texto completo
    print("-"*80)
    print("3. VERIFICAÇÃO DE DUPLICAÇÃO POR TEXTO COMPLETO")
    print("-"*80)
    
    if 'text' in train_dev_df.columns and 'text' in test_df.columns:
        train_texts = set(train_dev_df['text'].dropna())
        test_texts = set(test_df['text'].dropna())
        
        common_texts = train_texts.intersection(test_texts)
        
        print(f"Textos únicos no treino: {len(train_texts)}")
        print(f"Textos únicos no teste: {len(test_texts)}")
        print(f"Textos em comum (LEAKAGE): {len(common_texts)}")
        
        if len(common_texts) > 0:
            print("\n⚠️  ALERTA: Textos duplicados encontrados!")
            print(f"Percentual de leakage: {len(common_texts)/len(test_texts)*100:.2f}% do conjunto de teste")
            print("\nPrimeiros 5 textos duplicados (primeiros 200 caracteres):")
            for i, text in enumerate(list(common_texts)[:5], 1):
                print(f"  {i}. {str(text)[:200]}...")
                print()
        else:
            print("\n✓ Nenhum texto duplicado encontrado")
    else:
        print("Coluna 'text' não encontrada nos datasets")
    print()
    
    # 4. Verificar se há índices duplicados
    print("-"*80)
    print("4. VERIFICAÇÃO DE ÍNDICES")
    print("-"*80)
    
    if 'Unnamed: 0' in train_dev_df.columns and 'Unnamed: 0' in test_df.columns:
        train_indices = set(train_dev_df['Unnamed: 0'])
        test_indices = set(test_df['Unnamed: 0'])
        
        common_indices = train_indices.intersection(test_indices)
        
        print(f"Índices no treino: {len(train_indices)}")
        print(f"Índices no teste: {len(test_indices)}")
        print(f"Índices em comum: {len(common_indices)}")
        
        if len(common_indices) > 0:
            print("\n⚠️  ALERTA: Índices duplicados encontrados!")
            print(f"Isso pode indicar que os mesmos registros estão em ambos os conjuntos")
        else:
            print("\n✓ Nenhum índice duplicado encontrado")
    print()
    
    # 5. Resumo final
    print("="*80)
    print("RESUMO DA VERIFICAÇÃO")
    print("="*80)
    
    leakage_found = False
    
    if 'url' in train_dev_df.columns and 'url' in test_df.columns:
        if len(common_urls) > 0:
            print(f"❌ URLs duplicadas: {len(common_urls)} ({len(common_urls)/len(test_urls)*100:.2f}% do teste)")
            leakage_found = True
        else:
            print("✓ URLs: OK")
    
    if 'title' in train_dev_df.columns and 'title' in test_df.columns:
        if len(common_titles) > 0:
            print(f"❌ Títulos duplicados: {len(common_titles)} ({len(common_titles)/len(test_titles)*100:.2f}% do teste)")
            leakage_found = True
        else:
            print("✓ Títulos: OK")
    
    if 'text' in train_dev_df.columns and 'text' in test_df.columns:
        if len(common_texts) > 0:
            print(f"❌ Textos duplicados: {len(common_texts)} ({len(common_texts)/len(test_texts)*100:.2f}% do teste)")
            leakage_found = True
        else:
            print("✓ Textos: OK")
    
    print()
    if leakage_found:
        print("⚠️  DATA LEAKAGE DETECTADO!")
        print("Os conjuntos de treino e teste contêm exemplos duplicados.")
        print("Isso pode explicar as métricas excepcionalmente altas no conjunto de teste.")
    else:
        print("✅ NENHUM DATA LEAKAGE DETECTADO")
        print("Os conjuntos de treino e teste são completamente independentes.")
        print("As altas métricas no teste refletem genuína capacidade de generalização.")
    
    print()
    print("="*80)
    
    # Salvar relatório
    report_path = Path("/home/paulo/CascadeProjects/Applied_ML/test/results/data_leakage_report.txt")
    with open(report_path, 'w') as f:
        f.write("RELATÓRIO DE VERIFICAÇÃO DE DATA LEAKAGE\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data: {pd.Timestamp.now()}\n\n")
        f.write(f"Treino/Dev: {len(train_dev_df)} exemplos\n")
        f.write(f"Teste: {len(test_df)} exemplos\n\n")
        
        if 'url' in train_dev_df.columns:
            f.write(f"URLs duplicadas: {len(common_urls)}\n")
        if 'title' in train_dev_df.columns:
            f.write(f"Títulos duplicados: {len(common_titles)}\n")
        if 'text' in train_dev_df.columns:
            f.write(f"Textos duplicados: {len(common_texts)}\n")
        
        f.write(f"\nData leakage detectado: {'SIM' if leakage_found else 'NÃO'}\n")
    
    print(f"Relatório salvo em: {report_path}")

if __name__ == "__main__":
    check_leakage()
