#!/usr/bin/env python3
"""
EXTRATOR OFICIAL DE ATRIBUTOS DE NOTÍCIAS DE FRAUDE EMPRESARIAL

Este módulo é responsável por extrair informações estruturadas de notícias
classificadas como fraudes empresariais pelo modelo oficial (TF-IDF + SVM).

Atributos extraídos:
- Empresas envolvidas (pessoas jurídicas)
- Pessoas envolvidas (pessoas físicas com seus papéis)
- Tipos de fraude identificados
- Nível de confiança da análise

Author: Paulo Marcos Assis
Date: Jun 2026
Version: 1.0 (Oficial)
"""

import os
import json
import csv
import time
import signal
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
SELECTED_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
LLM_TEMPERATURE = 0
TIMEOUT_SECONDS = 180
MAX_CONSECUTIVE_ERRORS = 5


class TimeoutError(Exception):
    """Exceção lançada quando o processamento excede o timeout"""
    pass


class OllamaConnectionError(Exception):
    """Exceção lançada quando há erro de conexão com Ollama"""
    pass


def timeout_handler(signum, frame):
    """Handler para timeout de processamento"""
    raise TimeoutError("Processamento excedeu o tempo limite")


class FraudFeatureExtractor:
    """
    Extrator de atributos estruturados de notícias de fraude empresarial.
    
    Este extrator utiliza LLM (via Ollama) para identificar e extrair:
    - Nomes de empresas envolvidas
    - Nomes de pessoas envolvidas (com seus papéis/funções)
    - Tipos de fraude cometidos
    - Nível de confiança da extração
    """
    
    def __init__(self, ollama_host: str = OLLAMA_HOST, model: str = SELECTED_MODEL):
        """
        Inicializa o extrator de features.
        
        Args:
            ollama_host: URL do servidor Ollama
            model: Nome do modelo LLM a ser utilizado
        """
        self.ollama_host = ollama_host
        self.model = model
        self.llm = None
        self._llm_initialized = False
        
        print(f"Feature Extractor configurado:")
        print(f"  Ollama Host: {self.ollama_host}")
        print(f"  Modelo: {self.model}")
        print(f"  Temperatura: {LLM_TEMPERATURE}")
    
    def _ensure_llm(self):
        """Garante que a conexão com LLM está estabelecida"""
        if not self._llm_initialized:
            print(f"\nConectando ao Ollama em {self.ollama_host}...")
            try:
                self.llm = ChatOllama(
                    model=self.model,
                    base_url=self.ollama_host,
                    temperature=LLM_TEMPERATURE,
                    timeout=120
                )
                self._llm_initialized = True
                print("Conexão com Ollama estabelecida com sucesso!")
            except Exception as e:
                print(f"ERRO ao conectar ao Ollama: {e}")
                self.llm = None
                self._llm_initialized = True
                raise OllamaConnectionError(f"Falha ao conectar ao Ollama: {e}")
    
    def extract_features(self, text: str, title: str = "", timeout: int = TIMEOUT_SECONDS) -> Dict:
        """
        Extrai atributos estruturados de uma notícia de fraude empresarial.
        
        Args:
            text: Texto completo da notícia
            title: Título da notícia (opcional)
            timeout: Tempo máximo de processamento em segundos
        
        Returns:
            Dicionário com os atributos extraídos:
            {
                "companies_involved": List[str],      # Empresas identificadas
                "people_involved": List[str],         # Pessoas com papéis
                "fraud_types": List[str],             # Tipos de fraude
                "confidence": str,                    # "alta", "média", "baixa"
                "extraction_successful": bool,        # Se extração foi bem-sucedida
                "execution_time_seconds": float,      # Tempo de processamento
                "error_message": str                  # Mensagem de erro (se houver)
            }
        """
        default_return = {
            "companies_involved": [],
            "people_involved": [],
            "fraud_types": [],
            "confidence": "baixa",
            "extraction_successful": False,
            "execution_time_seconds": 0.0,
            "error_message": ""
        }
        
        self._ensure_llm()
        
        if not self.llm:
            default_return["error_message"] = "LLM não inicializado"
            return default_return
        
        if not text or not isinstance(text, str):
            default_return["error_message"] = "Texto inválido ou vazio"
            return default_return
        
        full_text = f"{title}\n\n{text}" if title else text
        
        prompt_content = self._build_extraction_prompt(full_text)
        
        start_time = time.time()
        
        # Configurar timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt_content)])
            signal.alarm(0)  # Cancelar timeout
            
            result = response.content.strip()
            parsed_result = self._parse_json_response(result, default_return)
            
            execution_time = time.time() - start_time
            parsed_result["execution_time_seconds"] = round(execution_time, 2)
            parsed_result["extraction_successful"] = True
            
            return parsed_result
            
        except TimeoutError:
            signal.alarm(0)
            execution_time = time.time() - start_time
            default_return["execution_time_seconds"] = round(execution_time, 2)
            default_return["error_message"] = f"Timeout após {timeout}s"
            return default_return
            
        except Exception as e:
            signal.alarm(0)
            execution_time = time.time() - start_time
            default_return["execution_time_seconds"] = round(execution_time, 2)
            default_return["error_message"] = str(e)
            return default_return
    
    def _build_extraction_prompt(self, full_text: str) -> str:
        """
        Constrói o prompt otimizado para extração de features.
        
        Args:
            full_text: Texto completo (título + corpo da notícia)
        
        Returns:
            Prompt formatado para o LLM
        """
        return f"""
Você é um especialista em análise de notícias sobre fraudes empresariais e crimes contra a administração pública.

Sua tarefa é extrair informações estruturadas da notícia fornecida, que JÁ FOI CLASSIFICADA como fraude empresarial.

----------------------------------------------------------------------
INSTRUÇÕES DE EXTRAÇÃO:

1. Leia COMPLETAMENTE todo o texto da notícia, do início ao fim.

2. Identifique e liste os TIPOS DE FRAUDE mencionados:
   Exemplos:
   - Fraude em licitação
   - Cartel entre empresas
   - Superfaturamento
   - Corrupção
   - Lavagem de dinheiro
   - Formação de organização criminosa
   - Contratos fraudulentos
   - Pagamento de propina
   - Desvio de recursos públicos
   - Falsificação de documentos
   - Qualquer outro tipo de fraude identificado

3. Identifique e liste as EMPRESAS envolvidas:
   - APENAS PESSOAS JURÍDICAS com NOMES ESPECÍFICOS
   - Indicadores: Ltda., S.A., ME, EPP, EIRELI, palavras como "Construtora", "Serviços", "Comércio", "Engenharia", etc.
   - NÃO inclua termos genéricos como "empresas de fachada", "empresas", "esquema"
   - NÃO inclua nomes de pessoas físicas
   - Exemplos CORRETOS: "Construtora ABC Ltda.", "Serviços Gerais S.A.", "Comércio de Peças ME"
   - Se não encontrar nome específico de empresa, deixe a lista vazia

4. Identifique e liste as PESSOAS ENVOLVIDAS com seus PAPÉIS/FUNÇÕES:
   - APENAS PESSOAS FÍSICAS (não empresas)
   - FORMATO OBRIGATÓRIO: "Nome Completo (função/papel)"
   - Analise o contexto para identificar o papel da pessoa
   - Exemplos CORRETOS:
     * "João Silva (empresário)"
     * "Pedro Costa (prefeito)"
     * "Maria Santos (sócia da empresa XYZ)"
     * "Carlos Oliveira (servidor público)"
     * "Ana Lima (ex-prefeita)"
   - Exemplos INCORRETOS:
     * "João Silva" (falta o papel)
     * "Construtora Silva Ltda." (é empresa, não pessoa)

5. Determine o NÍVEL DE CONFIANÇA da extração:
   - "alta": Informações claras e explícitas no texto
   - "média": Informações presentes mas com alguns detalhes implícitos
   - "baixa": Informações vagas ou incertas

IMPORTANTE - DIFERENCIAÇÃO EMPRESA vs PESSOA:
- Se aparecer "Ltda.", "S.A.", "ME", "EPP", "EIRELI" → é EMPRESA
- Se for apenas nome e sobrenome → é PESSOA
- Se o texto menciona "o empresário [nome]", "o prefeito [nome]" → é PESSOA (adicione o papel)
- Se o texto menciona "a empresa [nome]", "a construtora [nome]" → é EMPRESA
- Em caso de dúvida, analise o contexto ao redor do nome

----------------------------------------------------------------------
FORMATO DE RESPOSTA:

Retorne APENAS um JSON válido no formato:

{{
  "tipos_de_fraude": ["tipo1", "tipo2", ...],
  "empresas_envolvidas": ["empresa1", "empresa2", ...],
  "pessoas_envolvidas": ["pessoa1 (papel1)", "pessoa2 (papel2)", ...],
  "confianca": "alta" ou "média" ou "baixa"
}}

REGRAS:
- Seja preciso e extraia apenas informações EXPLÍCITAS no texto
- Não invente ou infira informações que não estão no texto
- Se não encontrar alguma categoria, retorne lista vazia []
- Remova duplicatas

----------------------------------------------------------------------
Texto da notícia:
\"\"\"{full_text}\"\"\"

Responda APENAS com o JSON válido, sem texto adicional.
"""
    
    def _parse_json_response(self, result_str: str, default_return: Dict) -> Dict:
        """
        Faz o parsing da resposta JSON do LLM.
        
        Args:
            result_str: String de resposta do LLM
            default_return: Dicionário padrão em caso de erro
        
        Returns:
            Dicionário com os dados parseados
        """
        # Remover marcadores de código markdown se presentes
        if result_str.startswith("```json"):
            result_str = result_str[7:]
        if result_str.startswith("```"):
            result_str = result_str[3:]
        if result_str.endswith("```"):
            result_str = result_str[:-3]
        
        result_str = result_str.strip()
        
        try:
            data = json.loads(result_str)
            
            out = {
                "companies_involved": [],
                "people_involved": [],
                "fraud_types": [],
                "confidence": data.get("confianca", "baixa"),
                "extraction_successful": False,
                "execution_time_seconds": 0.0,
                "error_message": ""
            }
            
            # Mapear chaves do português para inglês
            key_mapping = {
                "tipos_de_fraude": "fraud_types",
                "empresas_envolvidas": "companies_involved",
                "pessoas_envolvidas": "people_involved"
            }
            
            for pt_key, en_key in key_mapping.items():
                value = data.get(pt_key, [])
                
                # Converter string para lista se necessário
                if isinstance(value, str):
                    value = [value] if value.strip() else []
                elif not isinstance(value, list):
                    value = []
                
                # Limpar e remover duplicatas
                clean_list = []
                seen = set()
                for item in value:
                    s = str(item).strip().strip('"').strip("'").strip()
                    if s and s not in seen:
                        clean_list.append(s)
                        seen.add(s)
                
                out[en_key] = clean_list
            
            return out
            
        except json.JSONDecodeError as e:
            print(f"Falha ao decodificar JSON: {e}")
            print(f"  Início da resposta: {result_str[:100]}...")
            default_return["error_message"] = f"Erro ao decodificar JSON: {e}"
            return default_return


def extract_from_file(json_file_path: str, extractor: FraudFeatureExtractor) -> Optional[Dict]:
    """
    Extrai features de um arquivo JSON de notícia.
    
    Args:
        json_file_path: Caminho para o arquivo JSON da notícia
        extractor: Instância do FraudFeatureExtractor
    
    Returns:
        Dicionário com os dados extraídos ou None em caso de erro
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            news_data = json.load(f)
        
        title = news_data.get("title", "")
        text = news_data.get("text", "")
        url = news_data.get("url", "")
        
        if not text:
            print(f"Arquivo {json_file_path} não contém texto")
            return None
        
        result = extractor.extract_features(text, title)
        
        # Adicionar metadados do arquivo
        result["source_file"] = Path(json_file_path).name
        result["title"] = title
        result["url"] = url
        
        return result
        
    except Exception as e:
        print(f"Erro ao processar arquivo {json_file_path}: {e}")
        return None


def extract_from_directory(
    input_dir: str,
    output_csv: str,
    output_json: str,
    resume: bool = True,
    max_files: Optional[int] = None
) -> Dict:
    """
    Extrai features de todos os arquivos JSON em um diretório.
    
    Args:
        input_dir: Diretório contendo arquivos JSON de notícias
        output_csv: Caminho para salvar CSV com resultados
        output_json: Caminho para salvar JSON com resultados detalhados
        resume: Se True, continua de onde parou
        max_files: Número máximo de arquivos a processar (None = todos)
    
    Returns:
        Dicionário com estatísticas do processamento
    """
    extractor = FraudFeatureExtractor()
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {input_dir}")
    
    json_files = sorted(list(input_path.glob("*.json")))
    
    if max_files:
        json_files = json_files[:max_files]
    
    total_files = len(json_files)
    
    print(f"\n{'='*70}")
    print(f"EXTRAÇÃO DE FEATURES DE NOTÍCIAS DE FRAUDE")
    print(f"{'='*70}")
    print(f"Diretório de entrada: {input_dir}")
    print(f"Total de arquivos: {total_files}")
    print(f"Timeout por notícia: {TIMEOUT_SECONDS}s")
    print(f"{'='*70}\n")
    
    results = []
    stats = {
        "total_processed": 0,
        "successful_extractions": 0,
        "failed_extractions": 0,
        "timeouts": 0,
        "total_companies_found": 0,
        "total_people_found": 0,
        "total_fraud_types_found": 0
    }
    
    for idx, json_file in enumerate(json_files, start=1):
        print(f"[{idx}/{total_files}] Processando: {json_file.name}")
        
        result = extract_from_file(str(json_file), extractor)
        
        if result:
            stats["total_processed"] += 1
            
            if result["extraction_successful"]:
                stats["successful_extractions"] += 1
                stats["total_companies_found"] += len(result["companies_involved"])
                stats["total_people_found"] += len(result["people_involved"])
                stats["total_fraud_types_found"] += len(result["fraud_types"])
                
                print(f"  Extração bem-sucedida (confiança: {result['confidence']})")
                if result["companies_involved"]:
                    print(f"    Empresas: {', '.join(result['companies_involved'])}")
                if result["people_involved"]:
                    print(f"    Pessoas: {', '.join(result['people_involved'])}")
                if result["fraud_types"]:
                    print(f"    Tipos: {', '.join(result['fraud_types'])}")
            else:
                stats["failed_extractions"] += 1
                print(f"  Falha na extração: {result.get('error_message', 'Erro desconhecido')}")
                
                if "Timeout" in result.get("error_message", ""):
                    stats["timeouts"] += 1
            
            results.append(result)
        
        print()
    
    # Salvar resultados
    _save_results(results, output_csv, output_json, stats)
    
    return stats


def _save_results(results: List[Dict], output_csv: str, output_json: str, stats: Dict):
    """Salva os resultados em CSV e JSON"""
    
    # Salvar JSON detalhado
    json_output = {
        "timestamp": datetime.now().isoformat(),
        "statistics": stats,
        "results": results
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    
    print(f"JSON salvo: {output_json}")
    
    # Salvar CSV simplificado
    if results:
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            fieldnames = [
                'source_file', 'title', 'url',
                'companies', 'people', 'fraud_types',
                'confidence', 'extraction_successful', 'execution_time_seconds'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                writer.writerow({
                    'source_file': result.get('source_file', ''),
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'companies': '; '.join(result.get('companies_involved', [])),
                    'people': '; '.join(result.get('people_involved', [])),
                    'fraud_types': '; '.join(result.get('fraud_types', [])),
                    'confidence': result.get('confidence', ''),
                    'extraction_successful': result.get('extraction_successful', False),
                    'execution_time_seconds': result.get('execution_time_seconds', 0)
                })
        
        print(f"CSV salvo: {output_csv}")
    
    # Imprimir estatísticas
    print(f"\n{'='*70}")
    print(f"ESTATÍSTICAS FINAIS")
    print(f"{'='*70}")
    print(f"Total processado: {stats['total_processed']}")
    print(f"Extrações bem-sucedidas: {stats['successful_extractions']}")
    print(f"Extrações falhadas: {stats['failed_extractions']}")
    print(f"Timeouts: {stats['timeouts']}")
    print(f"\nTotal de empresas encontradas: {stats['total_companies_found']}")
    print(f"Total de pessoas encontradas: {stats['total_people_found']}")
    print(f"Total de tipos de fraude: {stats['total_fraud_types_found']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import sys
    
    # Exemplo de uso
    INPUT_DIR = "/home/paulo/CascadeProjects/Applied_ML/dataset/cleaned_data_no_bias/fraud_scrap_in_may"
    OUTPUT_CSV = "/home/paulo/CascadeProjects/Applied_ML/dataset/extracted_features.csv"
    OUTPUT_JSON = "/home/paulo/CascadeProjects/Applied_ML/dataset/extracted_features.json"
    
    # Modo teste: processar apenas 5 arquivos
    test_mode = "--test" in sys.argv
    max_files = 5 if test_mode else None
    
    if test_mode:
        print("\nMODO TESTE: Processando apenas 5 arquivos\n")
    
    try:
        stats = extract_from_directory(
            input_dir=INPUT_DIR,
            output_csv=OUTPUT_CSV,
            output_json=OUTPUT_JSON,
            resume=True,
            max_files=max_files
        )
        
        print("\nProcessamento concluído com sucesso!")
        
    except Exception as e:
        print(f"\nERRO durante processamento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
