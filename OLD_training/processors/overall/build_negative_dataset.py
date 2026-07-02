"""
Script para construção do dataset negativo para classificação de fraudes.

Implementa a estratégia de:
1. Filtrar candidatos negativos (excluindo notícias com keywords de fraude)
2. Amostragem estratificada por portal (mantendo proporções similares ao dataset positivo)
3. Análise exploratória comparativa
4. Geração de amostra para validação manual

Baseado em best practices de ML para classificação binária balanceada.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set
from urllib.parse import urlparse
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
import numpy as np


FRAUD_KEYWORDS = [
    'fraude', 'corrupção', 'corrupcao', 'superfaturamento', 'licitação', 'licitacao',
    'propina', 'desvio', 'lavagem de dinheiro', 'cartel',
    'direcionamento', 'organização criminosa', 'organizacao criminosa', 
    'mpsc', 'ministério público', 'ministerio publico',
    'operação policial', 'operacao policial', 'gaeco', 'geac',
    'peculato', 'improbidade', 'malversação', 'malversacao',
    'esquema', 'investigação', 'investigacao', 'denúncia', 'denuncia',
    'condenação', 'condenacao', 'indiciamento', 'prisão preventiva',
    'busca e apreensão', 'busca e apreensao', 'mandado de prisão',
    'crime contra administração pública', 'crime contra administracao publica'
]

EXCLUDE_PORTALS = [
    'mpsc.mp.br',
    'pc.sc.gov.br',
    'tjsc.jus.br',
    'intranet.mpsc.mp.br',
    'portal.mpsc.mp.br'
]

MIN_WORDS = 50
MAX_WORDS = 3000

SAMPLING_STRATEGY = {
    'ndmais': 0.439,
    'iclnoticias': 0.361,
    'nsctotal': 0.083,
    'g1': 0.060,
    'jornalconexao': 0.036,
    'outros': 0.021
}


def extract_portal(url: str) -> str:
    """Extrai o nome do portal da URL."""
    if not url:
        return 'unknown'
    
    domain = urlparse(url).netloc.lower()
    domain = domain.replace('www.', '')
    
    if 'ndmais' in domain:
        return 'ndmais'
    elif 'iclnoticias' in domain:
        return 'iclnoticias'
    elif 'nsctotal' in domain or 'nsc' in domain:
        return 'nsctotal'
    elif 'g1.globo' in domain:
        return 'g1'
    elif 'jornalconexao' in domain:
        return 'jornalconexao'
    elif 'olharsc' in domain:
        return 'olharsc'
    elif 'bbc.com' in domain:
        return 'bbc'
    elif 'cartacapital' in domain:
        return 'cartacapital'
    elif 'folha' in domain:
        return 'folha'
    elif 'gazeta' in domain:
        return 'gazeta'
    else:
        return 'outros'


def contains_fraud_keywords(text: str, title: str) -> bool:
    """Verifica se o texto ou título contém palavras-chave de fraude."""
    combined = f"{text} {title}".lower()
    
    for keyword in FRAUD_KEYWORDS:
        if keyword in combined:
            return True
    return False


def is_excluded_portal(url: str) -> bool:
    """Verifica se a URL é de um portal excluído."""
    url_lower = url.lower()
    for portal in EXCLUDE_PORTALS:
        if portal in url_lower:
            return True
    return False


def count_words(text: str) -> int:
    """Conta o número de palavras no texto."""
    if not text:
        return 0
    return len(text.split())


def load_fraud_files(fraud_csv_path: str) -> Set[str]:
    """Carrega a lista de arquivos que são fraudes conhecidas."""
    df = pd.read_csv(fraud_csv_path)
    
    fraud_urls = set()
    if 'url' in df.columns:
        fraud_urls = set(df['url'].dropna())
    
    return fraud_urls


def filter_negative_candidates(
    scraped_dir: str,
    fraud_csv_path: str,
    output_csv: str = None
) -> pd.DataFrame:
    """
    Filtra JSONs que são candidatos a notícias negativas.
    
    Args:
        scraped_dir: Diretório com os JSONs scrapados
        fraud_csv_path: Path do CSV com fraudes conhecidas
        output_csv: Path opcional para salvar os candidatos
        
    Returns:
        DataFrame com candidatos a negativas
    """
    print("=" * 80)
    print("ETAPA 1: FILTRAGEM DE CANDIDATOS NEGATIVOS")
    print("=" * 80)
    
    fraud_urls = load_fraud_files(fraud_csv_path)
    print(f"\n✓ Carregadas {len(fraud_urls)} URLs de fraudes conhecidas")
    
    scraped_path = Path(scraped_dir)
    all_jsons = list(scraped_path.rglob('*.json'))
    print(f"✓ Encontrados {len(all_jsons)} arquivos JSON")
    
    candidates = []
    stats = {
        'total': len(all_jsons),
        'fraud_known': 0,
        'fraud_keywords': 0,
        'excluded_portal': 0,
        'too_short': 0,
        'too_long': 0,
        'valid': 0,
        'errors': 0
    }
    
    print("\n⏳ Processando arquivos...")
    for i, json_file in enumerate(all_jsons, 1):
        if i % 100 == 0:
            print(f"  Processados: {i}/{len(all_jsons)} ({i/len(all_jsons)*100:.1f}%)")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            url = data.get('url', '')
            
            if url in fraud_urls:
                stats['fraud_known'] += 1
                continue
            
            if is_excluded_portal(url):
                stats['excluded_portal'] += 1
                continue
            
            text = data.get('text', '')
            title = data.get('title', '')
            
            if contains_fraud_keywords(text, title):
                stats['fraud_keywords'] += 1
                continue
            
            word_count = count_words(text)
            
            if word_count < MIN_WORDS:
                stats['too_short'] += 1
                continue
            
            if word_count > MAX_WORDS:
                stats['too_long'] += 1
                continue
            
            portal = extract_portal(url)
            
            candidates.append({
                'file': str(json_file),
                'url': url,
                'title': title,
                'text': text,
                'word_count': word_count,
                'portal': portal,
                'date_publication': data.get('date_publication', ''),
                'scraped_at': data.get('scraped_at', '')
            })
            stats['valid'] += 1
            
        except Exception as e:
            stats['errors'] += 1
            continue
    
    print(f"\n✓ Processamento concluído!")
    print("\n" + "=" * 80)
    print("ESTATÍSTICAS DE FILTRAGEM")
    print("=" * 80)
    print(f"Total de arquivos:              {stats['total']:>6}")
    print(f"Fraudes conhecidas (excluídas): {stats['fraud_known']:>6}")
    print(f"Com keywords de fraude:         {stats['fraud_keywords']:>6}")
    print(f"Portais excluídos:              {stats['excluded_portal']:>6}")
    print(f"Muito curtos (< {MIN_WORDS} palavras):  {stats['too_short']:>6}")
    print(f"Muito longos (> {MAX_WORDS} palavras): {stats['too_long']:>6}")
    print(f"Erros de leitura:               {stats['errors']:>6}")
    print(f"{'=' * 80}")
    print(f"✓ CANDIDATOS VÁLIDOS:           {stats['valid']:>6}")
    print("=" * 80)
    
    df_candidates = pd.DataFrame(candidates)
    
    if output_csv and len(df_candidates) > 0:
        df_candidates.to_csv(output_csv, index=False)
        print(f"\n✓ Candidatos salvos em: {output_csv}")
    
    return df_candidates


def stratified_sampling(
    candidates_df: pd.DataFrame,
    n_samples: int = 1050,
    strategy: Dict[str, float] = SAMPLING_STRATEGY,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Realiza amostragem estratificada por portal.
    
    Args:
        candidates_df: DataFrame com candidatos
        n_samples: Número total de amostras desejadas
        strategy: Dicionário com proporções por portal
        random_state: Seed para reprodutibilidade
        
    Returns:
        DataFrame com amostras selecionadas
    """
    print("\n" + "=" * 80)
    print("ETAPA 2: AMOSTRAGEM ESTRATIFICADA POR PORTAL")
    print("=" * 80)
    
    print(f"\nObjetivo: {n_samples} notícias negativas")
    print("\nDistribuição de candidatos por portal:")
    portal_counts = candidates_df['portal'].value_counts()
    for portal, count in portal_counts.items():
        print(f"  {portal:15s}: {count:>5} notícias")
    
    samples = []
    total_sampled = 0
    
    print("\n" + "-" * 80)
    print("Amostragem por portal:")
    print("-" * 80)
    
    for portal, proportion in strategy.items():
        target_count = int(n_samples * proportion)
        
        if portal == 'outros':
            portal_candidates = candidates_df[
                ~candidates_df['portal'].isin(['ndmais', 'iclnoticias', 'nsctotal', 'g1', 'jornalconexao'])
            ]
        else:
            portal_candidates = candidates_df[
                candidates_df['portal'].str.contains(portal, case=False, na=False)
            ]
        
        available = len(portal_candidates)
        
        if available >= target_count:
            sample = portal_candidates.sample(n=target_count, random_state=random_state)
            sampled = target_count
        else:
            sample = portal_candidates
            sampled = available
        
        samples.append(sample)
        total_sampled += sampled
        
        status = "✓" if sampled == target_count else "⚠"
        print(f"{status} {portal:15s}: {sampled:>4}/{target_count:>4} (disponíveis: {available:>4})")
    
    df_samples = pd.concat(samples, ignore_index=True)
    
    if total_sampled < n_samples:
        print(f"\n⚠ ATENÇÃO: Apenas {total_sampled}/{n_samples} amostras obtidas")
        print(f"  Faltam {n_samples - total_sampled} amostras")
        
        remaining_candidates = candidates_df[~candidates_df.index.isin(df_samples.index)]
        if len(remaining_candidates) > 0:
            additional = min(n_samples - total_sampled, len(remaining_candidates))
            extra_samples = remaining_candidates.sample(n=additional, random_state=random_state)
            df_samples = pd.concat([df_samples, extra_samples], ignore_index=True)
            print(f"  ✓ Adicionadas {additional} amostras aleatórias de outros portais")
    
    print("\n" + "=" * 80)
    print(f"✓ TOTAL AMOSTRADO: {len(df_samples)} notícias")
    print("=" * 80)
    
    return df_samples


def create_validation_sample(
    negatives_df: pd.DataFrame,
    n_validation: int = 100,
    output_csv: str = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Cria amostra para validação manual.
    
    Args:
        negatives_df: DataFrame com negativas
        n_validation: Número de amostras para validação
        output_csv: Path para salvar a amostra
        random_state: Seed para reprodutibilidade
        
    Returns:
        DataFrame com amostra de validação
    """
    print("\n" + "=" * 80)
    print("ETAPA 3: CRIAÇÃO DE AMOSTRA PARA VALIDAÇÃO MANUAL")
    print("=" * 80)
    
    validation_sample = negatives_df.sample(n=min(n_validation, len(negatives_df)), random_state=random_state)
    
    validation_output = validation_sample[['url', 'title', 'portal', 'word_count']].copy()
    validation_output['is_fraud'] = ''
    validation_output['notes'] = ''
    
    if output_csv:
        validation_output.to_csv(output_csv, index=False)
        print(f"\n✓ Amostra de validação salva em: {output_csv}")
        print(f"  Total: {len(validation_output)} notícias")
        print(f"\n  Instruções:")
        print(f"  1. Abra o arquivo CSV")
        print(f"  2. Para cada notícia, preencha a coluna 'is_fraud':")
        print(f"     - 'sim' se for fraude")
        print(f"     - 'nao' se não for fraude")
        print(f"  3. Use a coluna 'notes' para observações")
    
    return validation_output


def exploratory_analysis(
    positives_csv: str,
    negatives_df: pd.DataFrame,
    output_dir: str = None
):
    """
    Realiza análise exploratória comparativa entre positivas e negativas.
    
    Args:
        positives_csv: Path do CSV com notícias positivas (fraudes)
        negatives_df: DataFrame com notícias negativas
        output_dir: Diretório para salvar gráficos
    """
    print("\n" + "=" * 80)
    print("ETAPA 4: ANÁLISE EXPLORATÓRIA COMPARATIVA")
    print("=" * 80)
    
    df_pos = pd.read_csv(positives_csv)
    
    if 'text' in df_pos.columns:
        df_pos['word_count'] = df_pos['text'].apply(lambda x: count_words(str(x)) if pd.notna(x) else 0)
    
    if 'url' in df_pos.columns:
        df_pos['portal'] = df_pos['url'].apply(extract_portal)
    
    print("\n" + "-" * 80)
    print("COMPARAÇÃO DE DISTRIBUIÇÕES")
    print("-" * 80)
    
    print("\n1. TAMANHO DOS TEXTOS (palavras)")
    print(f"   Positivas: média={df_pos['word_count'].mean():.1f}, mediana={df_pos['word_count'].median():.1f}, std={df_pos['word_count'].std():.1f}")
    print(f"   Negativas: média={negatives_df['word_count'].mean():.1f}, mediana={negatives_df['word_count'].median():.1f}, std={negatives_df['word_count'].std():.1f}")
    
    ks_stat, p_value = ks_2samp(df_pos['word_count'], negatives_df['word_count'])
    print(f"\n   Teste Kolmogorov-Smirnov:")
    print(f"   - Estatística: {ks_stat:.4f}")
    print(f"   - P-value: {p_value:.4f}")
    if p_value > 0.05:
        print(f"   ✓ Distribuições similares (p > 0.05)")
    else:
        print(f"   ⚠ Distribuições diferentes (p < 0.05)")
    
    print("\n2. DISTRIBUIÇÃO POR PORTAL")
    print("\n   Positivas:")
    pos_portal_dist = df_pos['portal'].value_counts(normalize=True) * 100
    for portal, pct in pos_portal_dist.head(10).items():
        print(f"   {portal:15s}: {pct:>5.1f}%")
    
    print("\n   Negativas:")
    neg_portal_dist = negatives_df['portal'].value_counts(normalize=True) * 100
    for portal, pct in neg_portal_dist.head(10).items():
        print(f"   {portal:15s}: {pct:>5.1f}%")
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].hist([df_pos['word_count'], negatives_df['word_count']], 
                        bins=30, label=['Positivas', 'Negativas'], alpha=0.7)
        axes[0, 0].set_xlabel('Número de Palavras')
        axes[0, 0].set_ylabel('Frequência')
        axes[0, 0].set_title('Distribuição de Tamanho dos Textos')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].boxplot([df_pos['word_count'], negatives_df['word_count']], 
                           labels=['Positivas', 'Negativas'])
        axes[0, 1].set_ylabel('Número de Palavras')
        axes[0, 1].set_title('Boxplot - Tamanho dos Textos')
        axes[0, 1].grid(True, alpha=0.3)
        
        portal_comparison = pd.DataFrame({
            'Positivas': pos_portal_dist.head(8),
            'Negativas': neg_portal_dist.head(8)
        }).fillna(0)
        
        portal_comparison.plot(kind='bar', ax=axes[1, 0], alpha=0.8)
        axes[1, 0].set_xlabel('Portal')
        axes[1, 0].set_ylabel('Percentual (%)')
        axes[1, 0].set_title('Distribuição por Portal (Top 8)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        summary_data = pd.DataFrame({
            'Classe': ['Positivas', 'Negativas'],
            'Total': [len(df_pos), len(negatives_df)],
            'Média Palavras': [df_pos['word_count'].mean(), negatives_df['word_count'].mean()],
            'Mediana Palavras': [df_pos['word_count'].median(), negatives_df['word_count'].median()]
        })
        
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=summary_data.values,
                                 colLabels=summary_data.columns,
                                 cellLoc='center',
                                 loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1, 1].set_title('Resumo Estatístico')
        
        plt.tight_layout()
        plot_path = output_path / 'exploratory_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Gráficos salvos em: {plot_path}")
        plt.close()
    
    print("\n" + "=" * 80)


def main():
    """Função principal - executa todo o pipeline."""
    
    base_dir = Path(__file__).parent.parent
    scraped_dir = base_dir / 'collector_noticias' / 'scraped'
    fraud_csv = base_dir / 'dataset' / 'DF_COMPANIES_CLEAN.csv'
    output_dir = base_dir / 'dataset' / 'negative_dataset'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("CONSTRUÇÃO DO DATASET NEGATIVO PARA CLASSIFICAÇÃO DE FRAUDES")
    print("=" * 80)
    print(f"\nDiretório de notícias scrapadas: {scraped_dir}")
    print(f"CSV de fraudes conhecidas:       {fraud_csv}")
    print(f"Diretório de saída:              {output_dir}")
    
    candidates_csv = output_dir / 'negative_candidates.csv'
    df_candidates = filter_negative_candidates(
        scraped_dir=str(scraped_dir),
        fraud_csv_path=str(fraud_csv),
        output_csv=str(candidates_csv)
    )
    
    if len(df_candidates) == 0:
        print("\n❌ ERRO: Nenhum candidato válido encontrado!")
        return
    
    negatives_csv = output_dir / 'negative_samples_1050.csv'
    df_negatives = stratified_sampling(
        candidates_df=df_candidates,
        n_samples=1050,
        random_state=42
    )
    
    df_negatives.to_csv(negatives_csv, index=False)
    print(f"\n✓ Dataset negativo salvo em: {negatives_csv}")
    
    validation_csv = output_dir / 'validation_sample_100.csv'
    create_validation_sample(
        negatives_df=df_negatives,
        n_validation=100,
        output_csv=str(validation_csv),
        random_state=42
    )
    
    exploratory_analysis(
        positives_csv=str(fraud_csv),
        negatives_df=df_negatives,
        output_dir=str(output_dir)
    )
    
    print("\n" + "=" * 80)
    print("✓ PIPELINE CONCLUÍDO COM SUCESSO!")
    print("=" * 80)
    print("\nArquivos gerados:")
    print(f"  1. {candidates_csv}")
    print(f"     → Todos os candidatos negativos filtrados")
    print(f"\n  2. {negatives_csv}")
    print(f"     → Dataset final com 1,050 negativas (amostragem estratificada)")
    print(f"\n  3. {validation_csv}")
    print(f"     → Amostra de 100 notícias para validação manual")
    print(f"\n  4. {output_dir / 'exploratory_analysis.png'}")
    print(f"     → Gráficos de análise exploratória")
    
    print("\n" + "=" * 80)
    print("PRÓXIMOS PASSOS RECOMENDADOS")
    print("=" * 80)
    print("\n1. Validação Manual:")
    print(f"   - Abra: {validation_csv}")
    print(f"   - Revise as 100 notícias")
    print(f"   - Marque se alguma é fraude (coluna 'is_fraud')")
    
    print("\n2. Construção do Dataset Final:")
    print(f"   - Combine positivas (1,050) + negativas (1,050)")
    print(f"   - Crie coluna 'label': 1 para fraude, 0 para não-fraude")
    print(f"   - Faça split: 70% treino / 15% validação / 15% teste")
    
    print("\n3. Treinamento:")
    print(f"   - Use validação cruzada estratificada (k=5)")
    print(f"   - Monitore métricas por portal")
    print(f"   - Analise erros de classificação")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
