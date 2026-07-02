"""
Análise detalhada dos casos duplicados entre treino e teste
"""

import pandas as pd
from pathlib import Path

def analyze_duplicates():
    """Analisa em detalhes os casos duplicados"""
    
    base_path = Path("/home/paulo/CascadeProjects/Applied_ML/dataset/cleaned_data_no_bias/FOR_TRAINING")
    
    train_dev_df = pd.read_csv(base_path / "TRAIN.csv")
    test_df = pd.read_csv(base_path / "TEST.csv")
    
    print("="*100)
    print("ANÁLISE DETALHADA DE DUPLICADOS")
    print("="*100)
    print()
    
    # Encontrar textos duplicados
    train_texts = set(train_dev_df['text'].dropna())
    test_texts = set(test_df['text'].dropna())
    common_texts = train_texts.intersection(test_texts)
    
    print(f"Total de textos duplicados: {len(common_texts)}")
    print()
    
    # Para cada texto duplicado, mostrar informações
    duplicates_info = []
    
    for i, text in enumerate(common_texts, 1):
        # Encontrar no treino
        train_match = train_dev_df[train_dev_df['text'] == text]
        # Encontrar no teste
        test_match = test_df[test_df['text'] == text]
        
        if len(train_match) > 0 and len(test_match) > 0:
            train_label = train_match.iloc[0]['label']
            test_label = test_match.iloc[0]['label']
            train_title = train_match.iloc[0]['title']
            test_title = test_match.iloc[0]['title']
            
            duplicates_info.append({
                'id': i,
                'train_label': train_label,
                'test_label': test_label,
                'labels_match': train_label == test_label,
                'train_title': str(train_title)[:100] if pd.notna(train_title) else 'N/A',
                'test_title': str(test_title)[:100] if pd.notna(test_title) else 'N/A',
                'text_preview': str(text)[:200]
            })
            
            print(f"Duplicado #{i}")
            print(f"  Título (treino): {train_title}")
            print(f"  Título (teste):  {test_title}")
            print(f"  Label (treino): {train_label} | Label (teste): {test_label} | Match: {train_label == test_label}")
            print(f"  Texto (preview): {str(text)[:200]}...")
            print()
    
    # Estatísticas
    df_dup = pd.DataFrame(duplicates_info)
    
    print("="*100)
    print("ESTATÍSTICAS DOS DUPLICADOS")
    print("="*100)
    print()
    print(f"Total de duplicados: {len(df_dup)}")
    print(f"Labels coincidem: {df_dup['labels_match'].sum()} ({df_dup['labels_match'].sum()/len(df_dup)*100:.1f}%)")
    print(f"Labels diferem: {(~df_dup['labels_match']).sum()} ({(~df_dup['labels_match']).sum()/len(df_dup)*100:.1f}%)")
    print()
    
    print("Distribuição de labels nos duplicados:")
    print("\nNo conjunto de treino:")
    print(df_dup['train_label'].value_counts())
    print("\nNo conjunto de teste:")
    print(df_dup['test_label'].value_counts())
    print()
    
    # Impacto nas métricas
    print("="*100)
    print("IMPACTO POTENCIAL NAS MÉTRICAS")
    print("="*100)
    print()
    print(f"Tamanho do conjunto de teste: {len(test_df)}")
    print(f"Exemplos duplicados no teste: {len(common_texts)}")
    print(f"Percentual de duplicação: {len(common_texts)/len(test_df)*100:.2f}%")
    print()
    print("Se o modelo 'memorizou' esses exemplos durante o treino:")
    print(f"  - Até {len(common_texts)} exemplos poderiam ser classificados corretamente por memorização")
    print(f"  - Isso representa {len(common_texts)/len(test_df)*100:.2f}% do conjunto de teste")
    print(f"  - Accuracy inflada em até: {len(common_texts)/len(test_df)*100:.2f} pontos percentuais")
    print()
    
    # Análise de títulos duplicados
    print("="*100)
    print("ANÁLISE DE TÍTULOS DUPLICADOS")
    print("="*100)
    print()
    
    train_titles = set(train_dev_df['title'].dropna())
    test_titles = set(test_df['title'].dropna())
    common_titles = train_titles.intersection(test_titles)
    
    print(f"Total de títulos duplicados: {len(common_titles)}")
    print()
    
    # Verificar se títulos duplicados são apenas títulos genéricos
    generic_titles = [
        'Junte-se a nós!',
        'Mônica Bergamo',
        'Sobre a Gazeta do Povo'
    ]
    
    generic_count = sum(1 for title in common_titles if any(g in title for g in generic_titles))
    print(f"Títulos genéricos/navegação: {generic_count}")
    print(f"Títulos de conteúdo real: {len(common_titles) - generic_count}")
    print()
    
    # Salvar relatório detalhado
    report_path = Path("/home/paulo/CascadeProjects/Applied_ML/test/results/duplicates_detailed_analysis.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ANÁLISE DETALHADA DE DUPLICADOS\n")
        f.write("="*100 + "\n\n")
        
        for dup in duplicates_info:
            f.write(f"Duplicado #{dup['id']}\n")
            f.write(f"  Título (treino): {dup['train_title']}\n")
            f.write(f"  Título (teste):  {dup['test_title']}\n")
            f.write(f"  Label (treino): {dup['train_label']} | Label (teste): {dup['test_label']}\n")
            f.write(f"  Labels coincidem: {dup['labels_match']}\n")
            f.write(f"  Texto: {dup['text_preview']}...\n")
            f.write("\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("RESUMO\n")
        f.write("="*100 + "\n\n")
        f.write(f"Total de duplicados: {len(df_dup)}\n")
        f.write(f"Percentual do teste: {len(common_texts)/len(test_df)*100:.2f}%\n")
        f.write(f"Labels coincidem: {df_dup['labels_match'].sum()}\n")
        f.write(f"Labels diferem: {(~df_dup['labels_match']).sum()}\n")
    
    print(f"Relatório detalhado salvo em: {report_path}")
    
    # Criar CSV com duplicados
    csv_path = Path("/home/paulo/CascadeProjects/Applied_ML/test/results/duplicates_list.csv")
    df_dup.to_csv(csv_path, index=False)
    print(f"CSV com duplicados salvo em: {csv_path}")

if __name__ == "__main__":
    analyze_duplicates()
