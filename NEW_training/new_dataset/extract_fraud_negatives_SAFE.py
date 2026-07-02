#!/usr/bin/env python3
"""
VERSÃO SEGURA: Extrai apenas notícias que foram PROCESSADAS pelo LLM

PROBLEMA CRÍTICO RESOLVIDO:
- O diretório ndmais tem 191.745 JSONs, mas apenas 78.550 foram processados
- Os 113.195 não processados podem conter fraudes
- Incluí-los como "negativas" contaminaria o dataset de treino

SOLUÇÃO:
- Usar APENAS arquivos que sabemos com certeza que foram processados
- Para ndmais: processar apenas os primeiros 78.550 arquivos (sorted)
- Para outros portais: verificar se total_processed == total de JSONs no diretório
"""

import json
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm

# Configuração
SOURCE_DIR = Path("/home/paulo/projects/old_main/PAULO")
OUTPUT_DIR = Path("/home/paulo/CascadeProjects/Applied_ML/NEW_training/new_dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Arquivos de entrada
RESULT_FILES = {
    'g1sc_iclnoticias': SOURCE_DIR / 'fraud_detection_g1sc_iclnoticias_results.json',
    'jornalconexao': SOURCE_DIR / 'fraud_detection_jornalconexao_results.json',
    'agoralaguna': SOURCE_DIR / 'fraud_detection_agoralaguna_results.json',
    'nsc_consolidado': SOURCE_DIR / 'fraud_detection_nsc_consolidado_results.json',
    'ndmais': SOURCE_DIR / 'fraud_detection_ndmais_results.json'
}

# Diretórios de JSONs originais
JSON_DIRS = {
    'g1sc_iclnoticias': Path("/home/paulo/projects/main-server/collector/noticias/downloaded_news/g1sc_iclnoticias_combined"),
    'jornalconexao': Path("/home/paulo/projects/main-server/collector/noticias/downloaded_news/jornalconexao"),
    'agoralaguna': Path("/home/paulo/projects/main-server/collector/noticias/downloaded_news/agoralaguna"),
    'nsc_consolidado': Path("/home/paulo/projects/old_main/collector/noticias/downloaded_news/nsc_consolidado"),
    'ndmais': Path("/home/paulo/Documentos/NEWS_ARTICLES/ndmais_articles_json")
}


def get_processed_files_list(portal_name, result_file, json_dir):
    """
    Reconstrói lista de arquivos que foram REALMENTE processados pelo LLM.
    
    CRÍTICO: Garante que não incluímos arquivos não processados como negativas.
    """
    with open(result_file) as f:
        data = json.load(f)
    
    total_processed = data.get('total_processed', 0)
    fraud_files = set(entry['file'] for entry in data.get('fraud_news', []))
    
    # Listar todos os JSONs do diretório (sorted, como o main.py faz)
    all_json_files = sorted(list(json_dir.glob("*.json")))
    
    print(f"\n{'='*70}")
    print(f"VERIFICAÇÃO DE SEGURANÇA: {portal_name}")
    print(f"{'='*70}")
    print(f"Total processado pelo LLM: {total_processed:,}")
    print(f"Total de JSONs no diretório: {len(all_json_files):,}")
    print(f"Fraudes detectadas: {len(fraud_files):,}")
    
    # VERIFICAÇÃO CRÍTICA
    if len(all_json_files) > total_processed:
        print(f"⚠️  ATENÇÃO: {len(all_json_files) - total_processed:,} arquivos NÃO foram processados!")
        print(f"   Usando apenas os primeiros {total_processed:,} arquivos (sorted)")
        
        # SOLUÇÃO SEGURA: Usar apenas os primeiros N arquivos (sorted)
        # O main.py processa em ordem: sorted(list(input_path.glob("*.json")))
        processed_files = all_json_files[:total_processed]
    else:
        print(f"✓ Todos os arquivos foram processados")
        processed_files = all_json_files
    
    return processed_files, fraud_files


def extract_safe_negatives(portal_name, result_file, json_dir):
    """Extrai negativas APENAS de arquivos processados pelo LLM"""
    
    if not json_dir.exists():
        print(f"✗ ERRO: Diretório não encontrado: {json_dir}")
        return []
    
    # Obter lista segura de arquivos processados
    processed_files, fraud_files = get_processed_files_list(portal_name, result_file, json_dir)
    
    negatives = []
    fraud_count = 0
    
    print(f"\nExtraindo negativas de {len(processed_files):,} arquivos processados...")
    
    for json_file in tqdm(processed_files, desc="Processando"):
        # Pular fraudes
        if json_file.name in fraud_files:
            fraud_count += 1
            continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                news_data = json.load(f)
            
            title = news_data.get("title", "")
            text = news_data.get("text", "")
            url = news_data.get("url", "")
            
            # Validar conteúdo mínimo
            if not title or not text or len(text.split()) < 50:
                continue
            
            negatives.append({
                'file': json_file.name,
                'title': title,
                'url': url,
                'text': text,
                'portal': urlparse(url).netloc if url else portal_name,
                'label': 0,
                'source_portal': portal_name
            })
        
        except Exception as e:
            continue
    
    print(f"✓ Negativas extraídas: {len(negatives):,}")
    print(f"✓ Fraudes puladas: {fraud_count:,}")
    print(f"✓ Total verificado: {len(negatives) + fraud_count:,}")
    
    return negatives


def extract_safe_positives(portal_name, result_file, json_dir):
    """Extrai fraudes"""
    
    with open(result_file) as f:
        data = json.load(f)
    
    fraud_news = data.get('fraud_news', [])
    
    print(f"\n{'='*70}")
    print(f"Extraindo FRAUDES: {portal_name}")
    print(f"{'='*70}")
    print(f"Total de fraudes: {len(fraud_news):,}")
    
    positives = []
    
    for entry in tqdm(fraud_news, desc="Processando fraudes"):
        file_name = entry['file']
        json_path = json_dir / file_name
        
        if not json_path.exists():
            continue
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                news_data = json.load(f)
            
            title = news_data.get("title", entry.get('title', ''))
            text = news_data.get("text", '')
            url = news_data.get("url", entry.get('url', ''))
            
            analysis = entry.get('analysis', {})
            
            positives.append({
                'file': file_name,
                'title': title,
                'url': url,
                'text': text,
                'portal': urlparse(url).netloc if url else portal_name,
                'companies': '; '.join(analysis.get('companies_involved', [])),
                'people': '; '.join(analysis.get('people_involved', [])),
                'fraud_types': '; '.join(analysis.get('fraud_types', [])),
                'confidence': analysis.get('confidence', ''),
                'label': 1,
                'source_portal': portal_name
            })
        
        except Exception as e:
            continue
    
    print(f"✓ Fraudes extraídas: {len(positives):,}")
    
    return positives


def main():
    print("="*70)
    print("EXTRAÇÃO SEGURA DE DADOS (APENAS PROCESSADOS PELO LLM)")
    print("="*70)
    print()
    
    all_negatives = []
    all_positives = []
    
    # Processar cada portal
    for portal_name, result_file in RESULT_FILES.items():
        if not result_file.exists():
            print(f"✗ Arquivo não encontrado: {result_file}")
            continue
        
        json_dir = JSON_DIRS.get(portal_name)
        if not json_dir:
            print(f"✗ Diretório JSON não configurado para: {portal_name}")
            continue
        
        # Extrair negativas (SAFE)
        negatives = extract_safe_negatives(portal_name, result_file, json_dir)
        all_negatives.extend(negatives)
        
        # Extrair positivas
        positives = extract_safe_positives(portal_name, result_file, json_dir)
        all_positives.extend(positives)
    
    # Salvar CSVs
    print(f"\n{'='*70}")
    print("SALVANDO RESULTADOS")
    print(f"{'='*70}")
    
    # Negativas
    df_negatives = pd.DataFrame(all_negatives)
    output_negatives = OUTPUT_DIR / "negatives_SAFE.csv"
    df_negatives.to_csv(output_negatives, index=False, encoding='utf-8')
    print(f"✓ Negativas salvas: {output_negatives}")
    print(f"  Total: {len(df_negatives):,} artigos")
    
    # Positivas
    df_positives = pd.DataFrame(all_positives)
    output_positives = OUTPUT_DIR / "positives_SAFE.csv"
    df_positives.to_csv(output_positives, index=False, encoding='utf-8')
    print(f"✓ Positivas salvas: {output_positives}")
    print(f"  Total: {len(df_positives):,} artigos")
    
    # Com empresas
    df_with_companies = df_positives[df_positives['companies'].str.len() > 0].copy()
    output_companies = OUTPUT_DIR / "positives_with_companies_SAFE.csv"
    df_with_companies.to_csv(output_companies, index=False, encoding='utf-8')
    print(f"✓ Com empresas salvas: {output_companies}")
    print(f"  Total: {len(df_with_companies):,} artigos")
    
    # Estatísticas
    print(f"\n{'='*70}")
    print("ESTATÍSTICAS FINAIS (DADOS SEGUROS)")
    print(f"{'='*70}")
    print(f"Total: {len(df_negatives) + len(df_positives):,}")
    print(f"Negativas (label=0): {len(df_negatives):,}")
    print(f"Positivas (label=1): {len(df_positives):,}")
    print(f"  └─ Com empresas: {len(df_with_companies):,}")
    print()
    
    print("Distribuição NEGATIVAS:")
    print(df_negatives['source_portal'].value_counts())
    print()
    
    print("Distribuição POSITIVAS:")
    print(df_positives['source_portal'].value_counts())
    print()
    
    print("✅ Extração SEGURA concluída!")
    print("   Todos os dados foram processados pelo LLM")
    print("   Sem risco de contaminação com fraudes não detectadas")


if __name__ == "__main__":
    main()
