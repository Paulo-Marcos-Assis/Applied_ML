#!/usr/bin/env python3

import json
import os
from pathlib import Path
from datetime import datetime

def check_classification_progress():
    """Verifica o progresso da classificação de notícias."""
    
    base_dir = Path("/home/paulo/CascadeProjects/Applied_ML")
    results_file = base_dir / "dataset" / "classification_results" / "classification_results.json"
    
    if not results_file.exists():
        print("Nenhum arquivo de resultados encontrado.")
        return
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Erro ao ler arquivo de resultados: {e}")
        return
    
    print("="*70)
    print("RELATORIO DE PROGRESSO DA CLASSIFICACAO")
    print("="*70)
    
    # Informações gerais
    total_processed = data.get('total_processed', 0)
    category_counts = data.get('category_counts', {})
    error_counts = data.get('error_counts', {})
    
    print(f"\nRESUMO GERAL:")
    print(f"  Total processado: {total_processed}")
    print(f"  Data/hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Distribuição por categoria
    print(f"\nDISTRIBUICAO POR CATEGORIA:")
    fraud_emp_count = category_counts.get('fraud_scrap_in_may', 0)
    fraud_gen_count = category_counts.get('fraudes_gerais_may', 0)
    hard_count = category_counts.get('hard_negatives_may', 0)
    pure_count = category_counts.get('pure_negatives_may', 0)
    
    print(f"  Fraudes Empresariais: {fraud_emp_count}")
    print(f"  Fraudes Gerais: {fraud_gen_count}")
    print(f"  Hard negatives: {hard_count}")
    print(f"  Pure negatives: {pure_count}")
    print(f"  Total classificado: {fraud_emp_count + fraud_gen_count + hard_count + pure_count}")
    
    total_frauds = fraud_emp_count + fraud_gen_count
    if total_frauds > 0:
        print(f"\nRESUMO DE FRAUDES:")
        print(f"  Total de fraudes: {total_frauds}")
        print(f"  Empresariais: {fraud_emp_count} ({fraud_emp_count/total_frauds*100:.1f}%)")
        print(f"  Gerais: {fraud_gen_count} ({fraud_gen_count/total_frauds*100:.1f}%)")
    
    # Erros
    print(f"\nERROS:")
    file_errors = error_counts.get('file_errors', 0)
    classification_errors = error_counts.get('classification_errors', 0)
    timeouts = error_counts.get('timeouts', 0)
    total_errors = len(data.get('error_news', []))
    
    print(f"  Erros de arquivo: {file_errors}")
    print(f"  Erros de classificação: {classification_errors}")
    print(f"  Timeouts: {timeouts}")
    print(f"  Total de erros registrados: {total_errors}")
    
    # Últimas notícias processadas
    processed_news = data.get('processed_news', [])
    if processed_news:
        print(f"\nULTIMAS NOTICIAS PROCESSADAS (últimas 5):")
        for news in processed_news[-5:]:
            category = news.get('category', 'N/A')
            title = news.get('title', 'Sem título')[:60]
            if len(title) == 60:
                title += "..."
            print(f"  [{category:15s}] {title}")
    
    # Erros recentes
    error_news = data.get('error_news', [])
    if error_news:
        print(f"\nERROS RECENTES (últimos 5):")
        for error in error_news[-5:]:
            error_type = error.get('error_type', 'N/A')
            file_name = error.get('file', 'N/A')[:40]
            error_msg = error.get('error_message', 'N/A')[:50]
            if len(error_msg) == 50:
                error_msg += "..."
            print(f"  [{error_type:20s}] {file_name}")
            print(f"                       → {error_msg}")
    
    # Verificar se processo está rodando
    pid_file = base_dir / "classification.pid"
    if pid_file.exists():
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Verificar se processo ainda existe
            if os.path.exists(f"/proc/{pid}"):
                print(f"\nSTATUS: Processo ativo (PID: {pid})")
            else:
                print(f"\nSTATUS: Processo finalizado (arquivo PID órfão)")
                print(f"         Limpe o arquivo: rm {pid_file}")
        except:
            print(f"\nSTATUS: Não foi possível verificar PID")
    else:
        print(f"\nSTATUS: Nenhum processo rodando")
    
    print("="*70)

if __name__ == "__main__":
    check_classification_progress()
