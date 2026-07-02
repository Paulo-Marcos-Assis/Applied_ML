#!/usr/bin/env python3
"""
Script para analisar e comparar resultados dos diferentes métodos de matching.
Gera relatório comparativo entre TF-IDF, Fuzzy e Hybrid.
"""

import pandas as pd
import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_latest_results(output_dir: str = "dataset/linked_companies"):
    """Carrega os resultados mais recentes de cada método."""
    output_path = Path(output_dir)
    
    results = {}
    for method in ['tfidf', 'fuzzy', 'hybrid']:
        # Busca arquivo JSON mais recente para cada método
        pattern = str(output_path / f"company_matches_{method}_*.json")
        files = sorted(glob.glob(pattern), reverse=True)
        
        if files:
            with open(files[0], 'r', encoding='utf-8') as f:
                results[method] = json.load(f)
            print(f"✓ Carregado: {files[0]}")
        else:
            print(f"✗ Não encontrado: {method}")
    
    return results


def compare_methods(results_dict):
    """Compara métricas entre os métodos."""
    
    print("\n" + "="*80)
    print("COMPARAÇÃO ENTRE MÉTODOS")
    print("="*80 + "\n")
    
    comparison = []
    
    for method, data in results_dict.items():
        stats = data['statistics']
        
        total_mentioned = stats['total_companies_mentioned']
        matched = stats['companies_matched']
        not_matched = stats['companies_not_matched']
        exact = stats['exact_matches']
        multiple = stats['multiple_matches']
        
        match_rate = (matched / total_mentioned * 100) if total_mentioned > 0 else 0
        exact_rate = (exact / matched * 100) if matched > 0 else 0
        
        comparison.append({
            'Método': method.upper(),
            'Empresas Mencionadas': total_mentioned,
            'Vinculadas': matched,
            'Não Vinculadas': not_matched,
            'Taxa Sucesso (%)': round(match_rate, 1),
            'Matches Exatos': exact,
            'Taxa Exatos (%)': round(exact_rate, 1),
            'Múltiplos Candidatos': multiple
        })
    
    df_comparison = pd.DataFrame(comparison)
    print(df_comparison.to_string(index=False))
    print("\n" + "="*80 + "\n")
    
    return df_comparison


def analyze_score_distribution(results_dict):
    """Analisa distribuição de scores por método."""
    
    print("\n" + "="*80)
    print("DISTRIBUIÇÃO DE SCORES")
    print("="*80 + "\n")
    
    for method, data in results_dict.items():
        scores = []
        
        for result in data['results']:
            for match in result['matches']:
                if match['candidatos']:
                    # Pega score do melhor candidato
                    scores.append(match['melhor_score'])
        
        if scores:
            df_scores = pd.Series(scores)
            print(f"\n{method.upper()}:")
            print(f"  Média: {df_scores.mean():.3f}")
            print(f"  Mediana: {df_scores.median():.3f}")
            print(f"  Desvio Padrão: {df_scores.std():.3f}")
            print(f"  Mínimo: {df_scores.min():.3f}")
            print(f"  Máximo: {df_scores.max():.3f}")
            
            # Distribuição por faixas
            print(f"\n  Distribuição:")
            print(f"    Score >= 0.9 (Excelente): {(df_scores >= 0.9).sum()} ({(df_scores >= 0.9).sum()/len(df_scores)*100:.1f}%)")
            print(f"    0.7 <= Score < 0.9 (Bom): {((df_scores >= 0.7) & (df_scores < 0.9)).sum()} ({((df_scores >= 0.7) & (df_scores < 0.9)).sum()/len(df_scores)*100:.1f}%)")
            print(f"    0.5 <= Score < 0.7 (Regular): {((df_scores >= 0.5) & (df_scores < 0.7)).sum()} ({((df_scores >= 0.5) & (df_scores < 0.7)).sum()/len(df_scores)*100:.1f}%)")
            print(f"    Score < 0.5 (Fraco): {(df_scores < 0.5).sum()} ({(df_scores < 0.5).sum()/len(df_scores)*100:.1f}%)")
    
    print("\n" + "="*80 + "\n")


def find_best_matches(results_dict, top_n=10):
    """Encontra os melhores matches de cada método."""
    
    print("\n" + "="*80)
    print(f"TOP {top_n} MELHORES MATCHES POR MÉTODO")
    print("="*80 + "\n")
    
    for method, data in results_dict.items():
        matches_list = []
        
        for result in data['results']:
            for match in result['matches']:
                if match['candidatos']:
                    best = match['candidatos'][0]
                    matches_list.append({
                        'empresa_mencionada': match['empresa_mencionada'],
                        'nome_fantasia': best['nome_fantasia'],
                        'score': match['melhor_score'],
                        'title': result['title'][:60]
                    })
        
        # Ordena por score
        matches_list.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n{method.upper()} - Top {top_n}:")
        print("-" * 80)
        
        for i, m in enumerate(matches_list[:top_n], 1):
            print(f"{i}. '{m['empresa_mencionada']}' → '{m['nome_fantasia']}'")
            print(f"   Score: {m['score']:.3f} | Notícia: {m['title']}...")
            print()


def find_problematic_cases(results_dict):
    """Identifica casos problemáticos (não encontrados ou score baixo)."""
    
    print("\n" + "="*80)
    print("CASOS PROBLEMÁTICOS")
    print("="*80 + "\n")
    
    for method, data in results_dict.items():
        not_found = []
        low_score = []
        
        for result in data['results']:
            for match in result['matches']:
                empresa = match['empresa_mencionada']
                
                if not match['candidatos']:
                    not_found.append({
                        'empresa': empresa,
                        'title': result['title'][:60]
                    })
                elif match['melhor_score'] < 0.5:
                    low_score.append({
                        'empresa': empresa,
                        'match': match['candidatos'][0]['nome_fantasia'],
                        'score': match['melhor_score'],
                        'title': result['title'][:60]
                    })
        
        print(f"\n{method.upper()}:")
        print(f"  Não encontrados: {len(not_found)}")
        print(f"  Score baixo (< 0.5): {len(low_score)}")
        
        if not_found:
            print(f"\n  Exemplos não encontrados:")
            for item in not_found[:5]:
                print(f"    - '{item['empresa']}' | {item['title']}...")
        
        if low_score:
            print(f"\n  Exemplos score baixo:")
            for item in low_score[:5]:
                print(f"    - '{item['empresa']}' → '{item['match']}' (score: {item['score']:.3f})")


def generate_comparison_chart(df_comparison, output_dir="dataset/linked_companies"):
    """Gera gráfico comparativo entre métodos."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Gráfico 1: Taxa de Sucesso
    ax1 = axes[0]
    ax1.bar(df_comparison['Método'], df_comparison['Taxa Sucesso (%)'], color=['#3498db', '#e74c3c', '#2ecc71'])
    ax1.set_ylabel('Taxa de Sucesso (%)')
    ax1.set_title('Taxa de Sucesso por Método')
    ax1.set_ylim(0, 100)
    for i, v in enumerate(df_comparison['Taxa Sucesso (%)']):
        ax1.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
    
    # Gráfico 2: Matches Exatos
    ax2 = axes[1]
    ax2.bar(df_comparison['Método'], df_comparison['Matches Exatos'], color=['#3498db', '#e74c3c', '#2ecc71'])
    ax2.set_ylabel('Número de Matches Exatos')
    ax2.set_title('Matches Exatos (Score > 0.9)')
    for i, v in enumerate(df_comparison['Matches Exatos']):
        ax2.text(i, v + 1, str(v), ha='center', fontweight='bold')
    
    # Gráfico 3: Vinculadas vs Não Vinculadas
    ax3 = axes[2]
    x = range(len(df_comparison))
    width = 0.35
    ax3.bar([i - width/2 for i in x], df_comparison['Vinculadas'], width, label='Vinculadas', color='#2ecc71')
    ax3.bar([i + width/2 for i in x], df_comparison['Não Vinculadas'], width, label='Não Vinculadas', color='#e74c3c')
    ax3.set_ylabel('Número de Empresas')
    ax3.set_title('Empresas Vinculadas vs Não Vinculadas')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_comparison['Método'])
    ax3.legend()
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "comparison_chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico salvo: {output_path}")
    
    plt.close()


def main():
    """Função principal."""
    
    print("\n" + "="*80)
    print("ANÁLISE DE RESULTADOS DE MATCHING")
    print("="*80)
    
    # Carrega resultados
    results = load_latest_results()
    
    if not results:
        print("\n✗ Nenhum resultado encontrado!")
        return
    
    # Compara métodos
    df_comparison = compare_methods(results)
    
    # Analisa distribuição de scores
    analyze_score_distribution(results)
    
    # Melhores matches
    find_best_matches(results, top_n=10)
    
    # Casos problemáticos
    find_problematic_cases(results)
    
    # Gera gráfico
    try:
        generate_comparison_chart(df_comparison)
    except Exception as e:
        print(f"\n⚠ Erro ao gerar gráfico: {e}")
    
    # Salva comparação em CSV
    output_path = Path("dataset/linked_companies") / "methods_comparison.csv"
    df_comparison.to_csv(output_path, index=False)
    print(f"\n✓ Comparação salva: {output_path}")
    
    print("\n" + "="*80)
    print("ANÁLISE CONCLUÍDA")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
