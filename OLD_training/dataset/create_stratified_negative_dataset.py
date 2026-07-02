#!/usr/bin/env python3
"""
Cria dataset negativo estratificado baseado no dataset positivo final
"""

import pandas as pd
import json
import random
from pathlib import Path
from collections import defaultdict, Counter

def load_negative_jsons(clean_path):
    """Carrega todos os JSONs negativos"""
    
    categories = {
        'hard_negatives_may': 'Hard Negatives',
        'pure_negatives_may': 'Pure Negatives'
    }
    
    all_negatives = []
    
    for category_dir, category_name in categories.items():
        json_path = Path(clean_path) / category_dir
        
        if not json_path.exists():
            continue
        
        json_files = list(json_path.glob("*.json"))
        print(f"📂 Carregando {category_name}: {len(json_files)} arquivos")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extrair portal/domínio
                url = data.get('url', '')
                if '://' in url:
                    portal = url.split('://')[1].split('/')[0]
                else:
                    portal = 'unknown'
                
                # Calcular tamanho do texto
                text = data.get('text', '')
                title = data.get('title', '')
                full_text = f"{title} {text}"
                word_count = len(full_text.split())
                
                all_negatives.append({
                    'file': json_file.name,
                    'title': title,
                    'text': text,
                    'url': url,
                    'portal': portal,
                    'word_count': word_count,
                    'category': category_name,
                    'full_text': full_text
                })
                
            except Exception as e:
                print(f"  ⚠️ Erro em {json_file.name}: {e}")
    
    print(f"✅ Total de negativas carregadas: {len(all_negatives)}")
    return all_negatives

def analyze_positive_distribution(csv_file):
    """Analisa distribuição do dataset positivo para estratificação"""
    
    print("📊 Analisando distribuição do dataset positivo...")
    
    df = pd.read_csv(csv_file)
    
    # Análise de portais
    portal_counts = Counter()
    for url in df['url'].dropna():
        if '://' in url:
            portal = url.split('://')[1].split('/')[0]
            portal_counts[portal] += 1
    
    # Análise de tamanho de textos
    word_counts = []
    for text in df['text'].dropna():
        word_counts.append(len(str(text).split()))
    
    # Definir faixas de tamanho
    if word_counts:
        word_counts.sort()
        q25 = word_counts[len(word_counts)//4]
        q50 = word_counts[len(word_counts)//2]
        q75 = word_counts[3*len(word_counts)//4]
        
        size_bins = [
            (0, q25, 'Pequeno'),
            (q25, q50, 'Médio-Pequeno'),
            (q50, q75, 'Médio-Grande'),
            (q75, float('inf'), 'Grande')
        ]
    else:
        size_bins = [(0, float('inf'), 'Todos')]
    
    print(f"  📈 Portais identificados: {len(portal_counts)}")
    print(f"  📏 Faixas de tamanho definidas: {len(size_bins)}")
    
    return portal_counts, size_bins, len(df)

def stratified_sampling(negatives, target_size, portal_counts, size_bins):
    """Realiza amostragem estratificada"""
    
    print(f"🎯 Iniciando amostragem estratificada para {target_size} exemplos...")
    
    # Calcular proporções dos portais no positivo
    total_positives = sum(portal_counts.values())
    portal_proportions = {portal: count/total_positives for portal, count in portal_counts.items()}
    
    # Agrupar negativas por portal e tamanho
    negative_groups = defaultdict(lambda: defaultdict(list))
    
    for neg in negatives:
        # Determinar faixa de tamanho
        size_category = 'Todos'
        for min_size, max_size, label in size_bins:
            if min_size <= neg['word_count'] < max_size:
                size_category = label
                break
        
        negative_groups[neg['portal']][size_category].append(neg)
    
    # Calcular quotas por portal
    selected_negatives = []
    remaining_quota = target_size
    
    # Para cada portal, selecionar exemplos proporcionalmente
    for portal, proportion in portal_proportions.items():
        if remaining_quota <= 0:
            break
        
        quota = int(target_size * proportion)
        if portal in negative_groups:
            portal_negatives = negative_groups[portal]
            
            # Distribuir quota pelas faixas de tamanho
            portal_selected = []
            
            # Tentar distribuir igualmente pelas faixas
            available_sizes = list(portal_negatives.keys())
            if available_sizes:
                per_size_quota = max(1, quota // len(available_sizes))
                
                for size_cat in available_sizes:
                    if len(portal_selected) >= quota:
                        break
                    
                    size_negatives = portal_negatives[size_cat]
                    to_select = min(per_size_quota, len(size_negatives), quota - len(portal_selected))
                    
                    # Seleção aleatória dentro da faixa
                    if to_select > 0:
                        selected = random.sample(size_negatives, to_select)
                        portal_selected.extend(selected)
                
                selected_negatives.extend(portal_selected)
                remaining_quota -= len(portal_selected)
    
    # Se ainda faltar exemplos, completar aleatoriamente
    if len(selected_negatives) < target_size:
        needed = target_size - len(selected_negatives)
        
        # Coletar todos os não selecionados
        all_selected_files = {neg['file'] for neg in selected_negatives}
        remaining_negatives = [neg for neg in negatives if neg['file'] not in all_selected_files]
        
        if remaining_negatives:
            additional = random.sample(remaining_negatives, min(needed, len(remaining_negatives)))
            selected_negatives.extend(additional)
    
    print(f"✅ Amostragem concluída: {len(selected_negatives)} exemplos selecionados")
    
    # Análise da amostra
    sample_portals = Counter(neg['portal'] for neg in selected_negatives)
    sample_categories = Counter(neg['category'] for neg in selected_negatives)
    
    print(f"📊 Distribuição da amostra:")
    print(f"  Hard Negatives: {sample_categories['Hard Negatives']}")
    print(f"  Pure Negatives: {sample_categories['Pure Negatives']}")
    print(f"  Portais representados: {len(sample_portals)}")
    
    return selected_negatives

def create_negative_csv(selected_negatives, output_file):
    """Cria CSV com dataset negativo"""
    
    print(f"💾 Criando CSV negativo: {output_file}")
    
    negative_data = []
    for neg in selected_negatives:
        negative_data.append({
            'url': neg['url'],
            'title': neg['title'],
            'text': neg['text'],
            'empresas': '',
            'pessoas': '',
            'fraude': '',
            'date_publication': '01/01/2024',
            'source_file': neg['file'],
            'category': neg['category'],
            'portal': neg['portal'],
            'word_count': neg['word_count'],
            'label': 'negative'
        })
    
    df_neg = pd.DataFrame(negative_data)
    df_neg.to_csv(output_file, index=False)
    
    print(f"✅ CSV negativo salvo: {len(df_neg)} linhas")
    return df_neg

def identify_unused_files(all_negatives, selected_negatives, clean_path):
    """Identifica arquivos não utilizados"""
    
    print("🔍 Identificando arquivos não utilizados...")
    
    selected_files = {neg['file'] for neg in selected_negatives}
    unused_files = []
    
    for neg in all_negatives:
        if neg['file'] not in selected_files:
            unused_files.append({
                'file': neg['file'],
                'category': neg['category'],
                'portal': neg['portal'],
                'word_count': neg['word_count'],
                'title': neg['title'][:100] + "..." if len(neg['title']) > 100 else neg['title']
            })
    
    # Salvar lista de não utilizados
    unused_file = f"{clean_path}/unused_negative_files.json"
    with open(unused_file, 'w', encoding='utf-8') as f:
        json.dump(unused_files, f, ensure_ascii=False, indent=2)
    
    print(f"📋 Arquivos não utilizados: {len(unused_files)}")
    print(f"💾 Lista salva em: {unused_file}")
    
    # Estatísticas por categoria
    unused_by_category = Counter(item['category'] for item in unused_files)
    print(f"📊 Não utilizados por categoria:")
    for category, count in unused_by_category.items():
        print(f"  {category}: {count}")
    
    return unused_files

def main():
    """Função principal"""
    
    print("🎯 CRIANDO DATASET NEGATIVO ESTRATIFICADO")
    print("=" * 60)
    
    # Paths
    clean_path = "/home/paulo/CascadeProjects/Applied_ML/dataset/cleaned"
    positive_csv = f"{clean_path}/DF_COMPANIES_CLEAN_II.csv"
    negative_csv = f"{clean_path}/NEGATIVE_DATASET_STRATIFIED.csv"
    
    # Configurar seed para reprodutibilidade
    random.seed(42)
    
    # Carregar negativas
    all_negatives = load_negative_jsons(clean_path)
    
    # Analisar distribuição positiva
    portal_counts, size_bins, positive_size = analyze_positive_distribution(positive_csv)
    
    # Definir tamanho do dataset negativo
    target_size = positive_size  # Igual ao número de positivos
    print(f"🎯 Tamanho alvo do dataset negativo: {target_size}")
    
    # Amostragem estratificada
    selected_negatives = stratified_sampling(all_negatives, target_size, portal_counts, size_bins)
    
    # Criar CSV negativo
    negative_df = create_negative_csv(selected_negatives, negative_csv)
    
    # Identificar arquivos não utilizados
    unused_files = identify_unused_files(all_negatives, selected_negatives, clean_path)
    
    # Resumo final
    print("\n" + "="*60)
    print("📊 RESUMO FINAL")
    print("="*60)
    print(f"Dataset Positivo: {positive_size} exemplos")
    print(f"Dataset Negativo: {len(selected_negatives)} exemplos")
    print(f"Arquivos Não Utilizados: {len(unused_files)}")
    print(f"Balanceamento: 1:1 ({positive_size}:{len(selected_negatives)})")
    print(f"\n✅ Datasets prontos para treinamento!")
    print(f"📁 Positivo: {positive_csv}")
    print(f"📁 Negativo: {negative_csv}")

if __name__ == "__main__":
    main()
