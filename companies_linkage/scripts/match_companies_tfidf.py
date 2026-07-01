#!/usr/bin/env python3
"""
Script para vincular empresas mencionadas em notícias com o dataset de estabelecimentos.

Utiliza duas abordagens:
1. TF-IDF + Cosine Similarity para encontrar matches semânticos
2. Fuzzy String Matching para similaridade textual

Dataset de entrada:
- test/dataset/TEST_CLEAN_NO_DUPLICATES_WITH_MATCHES.csv (coluna 'empresa(s)')
- dataset/raw_companies_hmg/estabelecimento.csv (coluna 'nome_fantasia')
"""

import pandas as pd
import numpy as np
import re
import unicodedata
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from pathlib import Path
import json
from datetime import datetime


class CompanyNameNormalizer:
    """Normaliza nomes de empresas para melhorar matching."""
    
    LEGAL_SUFFIXES = [
        r'\bltda\.?$',
        r'\bs\.?a\.?$',
        r'\bs/a$',
        r'\bme$',
        r'\bepp$',
        r'\beireli$',
        r'\bcia\.?$',
        r'\blimitada$',
        r'\bsociedade anonima$',
        r'\bempresa individual de responsabilidade limitada$'
    ]
    
    @staticmethod
    def normalize(name: str) -> str:
        """
        Normaliza nome da empresa:
        - Lowercase
        - Remove acentos
        - Remove pontuação (exceto espaços)
        - Remove sufixos legais
        - Normaliza espaços
        """
        if pd.isna(name) or not isinstance(name, str):
            return ""
        
        # Lowercase
        normalized = name.lower().strip()
        
        # Remove acentos
        normalized = unicodedata.normalize('NFKD', normalized)
        normalized = normalized.encode('ASCII', 'ignore').decode('ASCII')
        
        # Remove pontuação (mantém espaços e letras)
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Remove sufixos legais
        for suffix in CompanyNameNormalizer.LEGAL_SUFFIXES:
            normalized = re.sub(suffix, '', normalized, flags=re.IGNORECASE)
        
        # Normaliza espaços múltiplos
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    @staticmethod
    def extract_core_name(name: str) -> str:
        """
        Extrai o nome 'core' da empresa (primeiras palavras significativas).
        Útil para matching de siglas ou nomes parciais.
        """
        normalized = CompanyNameNormalizer.normalize(name)
        words = normalized.split()
        
        # Remove palavras muito comuns
        stop_words = {'de', 'da', 'do', 'e', 'para', 'com', 'em'}
        core_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Retorna primeiras 3 palavras significativas
        return ' '.join(core_words[:3])


class CompanyMatcher:
    """
    Realiza matching entre empresas mencionadas e dataset de estabelecimentos.
    Combina TF-IDF + Cosine Similarity e Fuzzy Matching.
    """
    
    def __init__(
        self, 
        df_estabelecimentos: pd.DataFrame,
        tfidf_top_n: int = 10,
        fuzzy_threshold: int = 70,
        cosine_threshold: float = 0.3
    ):
        """
        Args:
            df_estabelecimentos: DataFrame com coluna 'nome_fantasia'
            tfidf_top_n: Número de candidatos a retornar via TF-IDF
            fuzzy_threshold: Score mínimo para fuzzy matching (0-100)
            cosine_threshold: Similaridade mínima para cosine (0-1)
        """
        self.df = df_estabelecimentos.copy()
        self.tfidf_top_n = tfidf_top_n
        self.fuzzy_threshold = fuzzy_threshold
        self.cosine_threshold = cosine_threshold
        
        # Pré-processa dataset
        print("Normalizando nomes de empresas...")
        self.df['nome_normalizado'] = self.df['nome_fantasia'].apply(
            CompanyNameNormalizer.normalize
        )
        
        # Remove empresas sem nome
        self.df = self.df[self.df['nome_normalizado'].str.len() > 0].copy()
        
        # Cria vetorizador TF-IDF
        print("Criando vetorizador TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb',  # Character n-grams (melhor para nomes)
            ngram_range=(2, 4),  # Bi, tri e 4-gramas
            min_df=1,
            max_features=10000
        )
        
        # Vetoriza todos os nomes normalizados
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df['nome_normalizado']
        )
        
        print(f"✓ {len(self.df)} empresas indexadas")
        print(f"✓ TF-IDF matrix shape: {self.tfidf_matrix.shape}")
    
    def find_matches(
        self, 
        company_name: str,
        method: str = 'hybrid'
    ) -> List[Dict]:
        """
        Encontra matches para uma empresa.
        
        Args:
            company_name: Nome da empresa a buscar
            method: 'tfidf', 'fuzzy', ou 'hybrid' (padrão)
        
        Returns:
            Lista de dicts com candidatos ordenados por score
        """
        if not company_name or pd.isna(company_name):
            return []
        
        normalized_query = CompanyNameNormalizer.normalize(company_name)
        
        if not normalized_query:
            return []
        
        if method == 'tfidf':
            return self._match_tfidf(normalized_query, company_name)
        elif method == 'fuzzy':
            return self._match_fuzzy(normalized_query, company_name)
        else:  # hybrid
            return self._match_hybrid(normalized_query, company_name)
    
    def _match_tfidf(
        self, 
        normalized_query: str, 
        original_query: str
    ) -> List[Dict]:
        """Matching via TF-IDF + Cosine Similarity."""
        
        # Vetoriza query
        query_vec = self.vectorizer.transform([normalized_query])
        
        # Calcula similaridade cosine
        cosine_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Pega top N candidatos
        top_indices = np.argsort(cosine_scores)[::-1][:self.tfidf_top_n]
        
        # Filtra por threshold
        candidates = []
        for idx in top_indices:
            score = cosine_scores[idx]
            
            if score < self.cosine_threshold:
                break
            
            row = self.df.iloc[idx]
            candidates.append({
                'id_estabelecimento': int(row['id_estabelecimento']),
                'nome_fantasia': str(row['nome_fantasia']),
                'nome_normalizado': str(row['nome_normalizado']),
                'cnpj_completo': str(row.get('cnpj_completo', '')),
                'id_municipio': int(row.get('id_municipio', 0)) if pd.notna(row.get('id_municipio')) else None,
                'cosine_score': float(score),
                'method': 'tfidf'
            })
        
        return candidates
    
    def _match_fuzzy(
        self, 
        normalized_query: str, 
        original_query: str
    ) -> List[Dict]:
        """Matching via Fuzzy String Similarity."""
        
        # Calcula fuzzy score para todos
        self.df['fuzzy_score'] = self.df['nome_normalizado'].apply(
            lambda x: fuzz.token_sort_ratio(normalized_query, x)
        )
        
        # Filtra e ordena
        candidates_df = self.df[
            self.df['fuzzy_score'] >= self.fuzzy_threshold
        ].sort_values('fuzzy_score', ascending=False).head(self.tfidf_top_n)
        
        candidates = []
        for _, row in candidates_df.iterrows():
            candidates.append({
                'id_estabelecimento': int(row['id_estabelecimento']),
                'nome_fantasia': str(row['nome_fantasia']),
                'nome_normalizado': str(row['nome_normalizado']),
                'cnpj_completo': str(row.get('cnpj_completo', '')),
                'id_municipio': int(row.get('id_municipio', 0)) if pd.notna(row.get('id_municipio')) else None,
                'fuzzy_score': int(row['fuzzy_score']),
                'method': 'fuzzy'
            })
        
        return candidates
    
    def _match_hybrid(
        self, 
        normalized_query: str, 
        original_query: str
    ) -> List[Dict]:
        """
        Matching híbrido: combina TF-IDF e Fuzzy.
        Retorna união dos candidatos, ordenados por score combinado.
        """
        
        # Busca via TF-IDF
        tfidf_candidates = self._match_tfidf(normalized_query, original_query)
        
        # Busca via Fuzzy
        fuzzy_candidates = self._match_fuzzy(normalized_query, original_query)
        
        # Combina resultados (união por id_estabelecimento)
        combined = {}
        
        for candidate in tfidf_candidates:
            id_est = candidate['id_estabelecimento']
            combined[id_est] = candidate
            combined[id_est]['fuzzy_score'] = 0  # Inicializa
        
        for candidate in fuzzy_candidates:
            id_est = candidate['id_estabelecimento']
            if id_est in combined:
                # Já existe, adiciona fuzzy score
                combined[id_est]['fuzzy_score'] = candidate['fuzzy_score']
            else:
                # Novo candidato
                combined[id_est] = candidate
                combined[id_est]['cosine_score'] = 0.0  # Inicializa
        
        # Calcula score combinado (média ponderada)
        for id_est in combined:
            cosine = combined[id_est].get('cosine_score', 0.0)
            fuzzy = combined[id_est].get('fuzzy_score', 0) / 100.0  # Normaliza 0-1
            
            # Média ponderada: 60% TF-IDF, 40% Fuzzy
            combined[id_est]['combined_score'] = (0.6 * cosine) + (0.4 * fuzzy)
            combined[id_est]['method'] = 'hybrid'
        
        # Ordena por score combinado
        sorted_candidates = sorted(
            combined.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        return sorted_candidates[:self.tfidf_top_n]


class CompanyLinkingPipeline:
    """Pipeline completo de vinculação de empresas."""
    
    def __init__(
        self,
        test_csv_path: str,
        estabelecimentos_csv_path: str,
        output_dir: str = "dataset/linked_companies"
    ):
        self.test_csv_path = test_csv_path
        self.estabelecimentos_csv_path = estabelecimentos_csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Carrega datasets
        print(f"\nCarregando dataset de teste: {test_csv_path}")
        self.df_test = pd.read_csv(test_csv_path)
        print(f"✓ {len(self.df_test)} notícias carregadas")
        
        print(f"\nCarregando dataset de estabelecimentos: {estabelecimentos_csv_path}")
        self.df_estabelecimentos = pd.read_csv(
            estabelecimentos_csv_path,
            low_memory=False
        )
        print(f"✓ {len(self.df_estabelecimentos)} estabelecimentos carregados")
        
        # Inicializa matcher
        self.matcher = CompanyMatcher(self.df_estabelecimentos)
        
        # Resultados
        self.results = []
        self.stats = {
            'total_news': 0,
            'news_with_companies': 0,
            'total_companies_mentioned': 0,
            'companies_matched': 0,
            'companies_not_matched': 0,
            'multiple_matches': 0,
            'exact_matches': 0
        }
    
    def process_all(self, method: str = 'hybrid', top_n: int = 5):
        """
        Processa todas as notícias do dataset de teste.
        
        Args:
            method: 'tfidf', 'fuzzy', ou 'hybrid'
            top_n: Número máximo de candidatos a retornar por empresa
        """
        print(f"\n{'='*70}")
        print(f"INICIANDO VINCULAÇÃO DE EMPRESAS")
        print(f"Método: {method.upper()}")
        print(f"{'='*70}\n")
        
        self.stats['total_news'] = len(self.df_test)
        
        # Filtra apenas notícias com empresas mencionadas
        df_with_companies = self.df_test[
            self.df_test['empresa(s)'].notna() & 
            (self.df_test['empresa(s)'].str.len() > 0)
        ].copy()
        
        self.stats['news_with_companies'] = len(df_with_companies)
        
        print(f"Notícias com empresas: {len(df_with_companies)}/{len(self.df_test)}")
        print(f"\nProcessando...\n")
        
        for idx, row in df_with_companies.iterrows():
            title = row['title']
            empresas_str = row['empresa(s)']
            fraude = row.get('fraude', '')
            
            # Separa múltiplas empresas (separadas por ';')
            empresas_list = [e.strip() for e in str(empresas_str).split(';') if e.strip()]
            
            self.stats['total_companies_mentioned'] += len(empresas_list)
            
            news_result = {
                'news_index': int(idx),
                'title': title,
                'fraude': fraude,
                'empresas_mencionadas': empresas_list,
                'matches': []
            }
            
            # Processa cada empresa mencionada
            for empresa in empresas_list:
                print(f"[{idx}] Buscando: '{empresa}'")
                
                candidates = self.matcher.find_matches(empresa, method=method)
                
                if candidates:
                    # Pega top N
                    top_candidates = candidates[:top_n]
                    
                    match_result = {
                        'empresa_mencionada': empresa,
                        'candidatos': top_candidates,
                        'num_candidatos': len(top_candidates),
                        'melhor_score': top_candidates[0].get('combined_score', 
                                                             top_candidates[0].get('cosine_score', 
                                                                                  top_candidates[0].get('fuzzy_score', 0)/100))
                    }
                    
                    news_result['matches'].append(match_result)
                    
                    self.stats['companies_matched'] += 1
                    
                    if len(top_candidates) > 1:
                        self.stats['multiple_matches'] += 1
                    
                    # Considera "exact match" se score > 0.9
                    if match_result['melhor_score'] > 0.9:
                        self.stats['exact_matches'] += 1
                    
                    print(f"  ✓ {len(top_candidates)} candidato(s) | Melhor: {top_candidates[0]['nome_fantasia']} (score: {match_result['melhor_score']:.3f})")
                else:
                    match_result = {
                        'empresa_mencionada': empresa,
                        'candidatos': [],
                        'num_candidatos': 0,
                        'melhor_score': 0.0
                    }
                    news_result['matches'].append(match_result)
                    self.stats['companies_not_matched'] += 1
                    print(f"  ✗ Nenhum candidato encontrado")
            
            self.results.append(news_result)
        
        print(f"\n{'='*70}")
        print("PROCESSAMENTO CONCLUÍDO")
        print(f"{'='*70}\n")
        
        self._print_stats()
        self._save_results(method)
    
    def _print_stats(self):
        """Imprime estatísticas do processamento."""
        print("\n" + "="*70)
        print("ESTATÍSTICAS DE VINCULAÇÃO")
        print("="*70)
        
        print(f"\nNotícias:")
        print(f"  Total processadas: {self.stats['total_news']}")
        print(f"  Com empresas mencionadas: {self.stats['news_with_companies']}")
        
        print(f"\nEmpresas:")
        print(f"  Total mencionadas: {self.stats['total_companies_mentioned']}")
        print(f"  Vinculadas (com match): {self.stats['companies_matched']}")
        print(f"  Não vinculadas: {self.stats['companies_not_matched']}")
        
        if self.stats['total_companies_mentioned'] > 0:
            match_rate = (self.stats['companies_matched'] / self.stats['total_companies_mentioned']) * 100
            print(f"\n  Taxa de sucesso: {match_rate:.1f}%")
        
        print(f"\nQualidade dos Matches:")
        print(f"  Matches exatos (score > 0.9): {self.stats['exact_matches']}")
        print(f"  Múltiplos candidatos: {self.stats['multiple_matches']}")
        
        print("="*70 + "\n")
    
    def _save_results(self, method: str):
        """Salva resultados em arquivos."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Salva resultados completos (JSON)
        results_file = self.output_dir / f"company_matches_{method}_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'timestamp': timestamp,
                    'method': method,
                    'test_dataset': self.test_csv_path,
                    'estabelecimentos_dataset': self.estabelecimentos_csv_path
                },
                'statistics': self.stats,
                'results': self.results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Resultados salvos: {results_file}")
        
        # 2. Salva CSV resumido para análise
        csv_rows = []
        for result in self.results:
            for match in result['matches']:
                empresa_mencionada = match['empresa_mencionada']
                
                if match['candidatos']:
                    for rank, candidato in enumerate(match['candidatos'], 1):
                        csv_rows.append({
                            'news_index': result['news_index'],
                            'title': result['title'][:100],
                            'empresa_mencionada': empresa_mencionada,
                            'rank': rank,
                            'id_estabelecimento': candidato['id_estabelecimento'],
                            'nome_fantasia': candidato['nome_fantasia'],
                            'cnpj': candidato.get('cnpj_completo', ''),
                            'score': candidato.get('combined_score', 
                                                  candidato.get('cosine_score',
                                                               candidato.get('fuzzy_score', 0)/100)),
                            'method': candidato['method']
                        })
                else:
                    csv_rows.append({
                        'news_index': result['news_index'],
                        'title': result['title'][:100],
                        'empresa_mencionada': empresa_mencionada,
                        'rank': 0,
                        'id_estabelecimento': None,
                        'nome_fantasia': 'NÃO ENCONTRADO',
                        'cnpj': '',
                        'score': 0.0,
                        'method': method
                    })
        
        csv_file = self.output_dir / f"company_matches_{method}_{timestamp}.csv"
        pd.DataFrame(csv_rows).to_csv(csv_file, index=False, encoding='utf-8')
        print(f"✓ CSV resumido salvo: {csv_file}")
        
        # 3. Salva estatísticas
        stats_file = self.output_dir / f"stats_{method}_{timestamp}.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RELATÓRIO DE VINCULAÇÃO DE EMPRESAS\n")
            f.write("="*70 + "\n\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Método: {method.upper()}\n\n")
            
            for key, value in self.stats.items():
                f.write(f"{key}: {value}\n")
            
            if self.stats['total_companies_mentioned'] > 0:
                match_rate = (self.stats['companies_matched'] / self.stats['total_companies_mentioned']) * 100
                f.write(f"\nTaxa de sucesso: {match_rate:.1f}%\n")
        
        print(f"✓ Estatísticas salvas: {stats_file}\n")


def main():
    """Função principal."""
    
    # Paths
    TEST_CSV = "test/dataset/TEST_CLEAN_NO_DUPLICATES_WITH_MATCHES.csv"
    ESTABELECIMENTOS_CSV = "dataset/raw_companies_hmg/estabelecimento.csv"
    OUTPUT_DIR = "dataset/linked_companies"
    
    # Inicializa pipeline
    pipeline = CompanyLinkingPipeline(
        test_csv_path=TEST_CSV,
        estabelecimentos_csv_path=ESTABELECIMENTOS_CSV,
        output_dir=OUTPUT_DIR
    )
    
    # Testa os 3 métodos
    for method in ['tfidf', 'fuzzy', 'hybrid']:
        print(f"\n{'#'*70}")
        print(f"# TESTANDO MÉTODO: {method.upper()}")
        print(f"{'#'*70}\n")
        
        pipeline.process_all(method=method, top_n=5)
        
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
