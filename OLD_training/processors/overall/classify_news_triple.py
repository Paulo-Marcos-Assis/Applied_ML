import os
import json
import csv
import time
import signal
import shutil
import sys
from datetime import datetime
from typing import Dict, List
from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434") 
SELECTED_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b") #"gpt-oss:20b" 
LLM_TEMPERATURE = 0
TIMEOUT_SECONDS = 360
START_FROM = 0
MAX_CONSECUTIVE_403_ERRORS = 3
MAX_FILES_FOR_TEST = 3  # Limitar para teste

class TimeoutError(Exception):
    pass

class OllamaError403(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Processamento excedeu o tempo limite")

class NewsClassifier:
    def __init__(self):
        print(f"News Classifier configurado para usar Ollama em {OLLAMA_HOST} (modelo: {SELECTED_MODEL})")
        self.llm = None
        self._llm_initialized = False
    
    def _ensure_llm(self):
        if not self._llm_initialized:
            print(f"Conectando ao Ollama em {OLLAMA_HOST}...")
            try:
                self.llm = ChatOllama(
                    model=SELECTED_MODEL,
                    base_url=OLLAMA_HOST,
                    temperature=LLM_TEMPERATURE,
                    timeout=120
                )
                self._llm_initialized = True
                print("Conexão com Ollama estabelecida com sucesso!")
            except Exception as e:
                print(f"ERRO ao conectar ao Ollama: {e}")
                self.llm = None
                self._llm_initialized = True

    def analyze_news(self, text: str, title: str, timeout: int = TIMEOUT_SECONDS) -> Dict:
        """
        Analisa a notícia e classifica em uma das três categorias:
        1. Fraude confirmada
        2. Hard negative
        3. Pure negative
        
        Retorna:
        {
            "is_fraud_related": bool,
            "fraud_types": List[str],
            "companies_involved": List[str],
            "people_involved": List[str],
            "hard_negative_candidate": bool,
            "hard_negative_reason": str,
            "execution_time_seconds": float
        }
        """
        default_return = {
            "is_fraud_related": False,
            "fraud_types": [],
            "companies_involved": [],
            "people_involved": [],
            "hard_negative_candidate": False,
            "hard_negative_reason": "",
            "execution_time_seconds": 0.0
        }
        
        self._ensure_llm()
        
        if not self.llm:
            print("Erro: LLM não inicializado.")
            return default_return
        
        if not text or not isinstance(text, str):
            return default_return

        full_text = f"{title}\n\n{text}" if title else text
        
        prompt_content = f"""
Você é um especialista em análise de notícias sobre fraudes, fraudes empresariais e crimes contra a administração pública.

Sua tarefa é analisar a notícia fornecida e classificá-la em uma das quatro categorias:
1. FRAUDE EMPRESARIAL CONFIRMADA (com empresas envolvidas)
2. FRAUDES GERAIS (outros tipos de fraude sem empresas)
3. HARD NEGATIVE (contexto relevante mas sem fraude)
4. PURE NEGATIVE (notícia aleatória sem contexto relevante)

Saiba que essas notícias serão utilizadas para treinar um modelo de classificação textual sobre fraudes e fraudes empresariais.
----------------------------------------------------------------------
INSTRUÇÕES:

1. Leia atentamente e COMPLETAMENTE todo o texto da notícia, do início ao fim.

2. Identifique se a notícia trata de FRAUDES ou FRAUDES EMPRESARIAIS:
   Por exemplo:
   - Em licitações
   - Cartel entre empresas
   - Superfaturamento
   - Corrupção envolvendo empresas
   - Lavagem de dinheiro
   - Contratos fraudulentos
   - Pagamento de propina 
   - Desvio de recursos públicos 
   **ENTRE QUALQUER OUTRO TIPO DE FRAUDE ou FRAUDE QUE ENVOLVA EMPRESAS**

3. Se for FRAUDE mas não tiver NOMES DE EMPRESAS IDENTIFICÁVEIS, classifique como FRAUDE GERAL:
   - Apenas menciona "empresas", "empresas de fachada", "esquema" de forma genérica
   - Fraudes envolvendo apenas pessoas físicas
   - Crimes financeiros sem empresas nomeadas
   - Outros tipos de fraude sem empresas específicas identificadas

   **IMPORTANTE**: Para ser FRAUDE EMPRESARIAL, é OBRIGATÓRIO identificar NOMES de empresas (Ltda., S.A., ME, EPP, etc.)

4. Se NÃO for fraude, verifique se é HARD NEGATIVE:
   Por exemplo:
   - Notícias sobre licitações legítimas (sem fraude) - se houver;
   - Notícias sobre empresas em contextos **não-fraudulentos** - se houver;
   - Notícias sobre processos licitatórios normais - se houver;
   - E QUALQUER OUTRA NOTÍCIA que possa ser relevante PARA COMPOR DADOS DE TREINO NEGATIVOS, NA CATEGORIA HARD NEGATIVE

5. Liste os TIPOS DE FRAUDE (apenas se relacionado_a_fraude: true)

6. Liste as EMPRESAS mencionadas:
   - APENAS PESSOAS JURÍDICAS com NOMES ESPECÍFICOS (Ltda., S.A., ME, EPP, EIRELI, etc.)
   - Exemplos: "Construtora ABC Ltda.", "Serviços Gerais S.A.", "Comércio de Peças ME"
   - **NÃO inclua termos genéricos como "empresas de fachada", "empresas", "esquema"**
   - **NÃO inclua nomes de pessoas físicas**
   - Se não encontrar NOME de empresa específico, deixe a lista vazia

7. Liste as PESSOAS ENVOLVIDAS COM SEUS PAPÉIS:
   - APENAS PESSOAS FÍSICAS com formato: "Nome Completo (função/papel)"
   - Exemplos: "João Silva (empresário)", "Pedro Costa (prefeito)"

----------------------------------------------------------------------
RESPONDA EM PORTUGUÊS BRASILEIRO;
FORMATO DE RESPOSTA:

Retorne APENAS um JSON válido no formato:

{{
  "relacionado_a_fraude": true ou false,
  "fraude_empresarial": true ou false,
  "fraude_geral": true ou false,
  "tipos_de_fraude": ["tipo1", "tipo2", ...],
  "empresas_envolvidas": ["empresa1", "empresa2", ...],
  "pessoas_envolvidas": ["pessoa1", "pessoa2", ...],
  "candidato_a_hard_negative": true ou false,
  "motivo_hard_negative": "explicação breve (apenas se candidato_a_hard_negative: true)"
}}

IMPORTANTE:
- Se relacionado_a_fraude: true → candidato_a_hard_negative deve ser false
- Se relacionado_a_fraude: false → verifique se é candidato_a_hard_negative
- Se não for fraude nem hard negative → candidato_a_hard_negative: false
- motivo_hard_negative apenas se candidato_a_hard_negative: true
- Se for fraude COM NOMES DE EMPRESAS ESPECÍFICAS → fraud_empresarial: true, fraud_geral: false
- Se for fraude SEM NOMES DE EMPRESAS ESPECÍFICAS → fraud_empresarial: false, fraud_geral: true
- Seja preciso e extraia apenas informações explícitas no texto

----------------------------------------------------------------------
Texto da notícia:
\"\"\"{full_text}\"\"\"

Responda APENAS com o JSON válido, sem texto adicional.
"""

        start_time = time.time()
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt_content)])
            signal.alarm(0)
            result = response.content.strip()
            parsed_result = self._parse_json_response(result, default_return)
            execution_time = time.time() - start_time
            parsed_result["execution_time_seconds"] = round(execution_time, 2)
            return parsed_result
        except TimeoutError:
            signal.alarm(0)
            print(f"[TIMEOUT] Processamento excedeu {timeout}s - pulando notícia")
            execution_time = time.time() - start_time
            default_return["execution_time_seconds"] = round(execution_time, 2)
            default_return["timeout_error"] = True  # Marcar como erro de timeout
            return default_return
        except Exception as e:
            signal.alarm(0)
            error_msg = str(e)
            if "403" in error_msg or "Forbidden" in error_msg:
                print(f"[ERRO 403] Ollama retornou erro de permissão: {error_msg}")
                raise OllamaError403(f"Erro 403 do Ollama: {error_msg}")
            print(f"[Erro na Análise] {e}")
            execution_time = time.time() - start_time
            default_return["execution_time_seconds"] = round(execution_time, 2)
            default_return["analysis_error"] = True
            default_return["error_message"] = error_msg
            return default_return

    def _parse_json_response(self, result_str: str, default_return: Dict) -> Dict:
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
                "is_fraud_related": bool(data.get("relacionado_a_fraude", False)),
                "fraud_empresarial": bool(data.get("fraude_empresarial", False)),
                "fraude_geral": bool(data.get("fraude_geral", False)),
                "fraud_types": [],
                "companies_involved": [],
                "people_involved": [],
                "hard_negative_candidate": bool(data.get("candidato_a_hard_negative", False)),
                "hard_negative_reason": data.get("motivo_hard_negative", ""),
                "execution_time_seconds": 0.0
            }
            
            # Mapear chaves do português para o formato interno
            key_mapping = {
                "tipos_de_fraude": "fraud_types",
                "empresas_envolvidas": "companies_involved", 
                "pessoas_envolvidas": "people_involved"
            }
            
            for pt_key, en_key in key_mapping.items():
                value = data.get(pt_key, [])
                if isinstance(value, str):
                    value = [value] if value.strip() else []
                elif not isinstance(value, list):
                    value = []
                
                clean_list = []
                seen = set()
                for item in value:
                    s = str(item).strip().strip('"').strip("'").strip()
                    if s and s not in seen:
                        clean_list.append(s)
                        seen.add(s)
                
                out[en_key] = clean_list
            
            return out
        except json.JSONDecodeError:
            print(f"Falha ao decodificar JSON. Início da resposta: {result_str[:50]}...")
            return default_return


def get_already_processed_files(output_file: str) -> set:
    output_path = Path(output_file)
    if not output_path.exists():
        return set()
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        processed_news = data.get('processed_news', [])
        return {entry['file'] for entry in processed_news}
    except:
        return set()


def get_error_files(output_file: str) -> set:
    output_path = Path(output_file)
    if not output_path.exists():
        return set()
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        error_news = data.get('error_news', [])
        return {entry['file'] for entry in error_news}
    except:
        return set()


def classify_and_move_news(json_file: Path, analysis: Dict, dataset_base: Path, news_data: Dict) -> str:
    """
    Classifica e move o arquivo JSON para a pasta correspondente.
    
    Retorna a categoria de destino.
    """
    category = ""
    
    if analysis["is_fraud_related"]:
        if analysis.get("fraud_empresarial", False):
            category = "fraud_scrap_in_may"  # Fraudes empresariais
        else:
            category = "fraudes_gerais_may"  # Fraudes gerais (sem empresas)
        target_dir = dataset_base / category
    elif analysis.get("hard_negative_candidate", False):
        category = "hard_negatives_may"
        target_dir = dataset_base / category
    else:
        category = "pure_negatives_may"
        target_dir = dataset_base / category
    
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / json_file.name
    
    try:
        shutil.copy2(json_file, target_file)
        
        # Adicionar metadados de classificação ao JSON copiado
        with open(target_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data['_classification'] = {
            'category': category,
            'is_fraud_related': analysis['is_fraud_related'],
            'hard_negative_candidate': analysis.get('hard_negative_candidate', False),
            'hard_negative_reason': analysis.get('hard_negative_reason', ''),
            'classified_at': datetime.now().isoformat(),
            'execution_time_seconds': analysis.get('execution_time_seconds', 0)
        }
        
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"  ERRO ao copiar arquivo para {category}: {e}")
        return ""
    
    return category


def process_all_news(input_dir: str, dataset_base: str, output_file: str, resume: bool = True, reprocess_errors: bool = False):
    """
    Processa todas as notícias e classifica em três categorias.
    
    Args:
        reprocess_errors: Se True, reprocessa apenas arquivos com erro anterior
    """
    classifier = NewsClassifier()
    input_path = Path(input_dir)
    dataset_path = Path(dataset_base)
    
    if not input_path.exists():
        print(f"ERRO: Diretório {input_dir} não encontrado!")
        return
    
    json_files = sorted(list(input_path.rglob("*.json")))
    
    # Limitar para teste
    if "--test" in sys.argv:
        json_files = json_files[:MAX_FILES_FOR_TEST]
        print(f"MODO TESTE: Processando apenas {len(json_files)} arquivos")
    
    total_files = len(json_files)
    
    already_processed = get_already_processed_files(output_file) if resume else set()
    error_files = get_error_files(output_file) if resume else set()
    
    processed_news = []
    error_news = []
    
    if resume and already_processed:
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            processed_news = data.get('processed_news', [])
            error_news = data.get('error_news', [])
            print(f"\nRETOMANDO PROCESSAMENTO")
            print(f"   Já processadas: {len(already_processed)} notícias")
            if error_files:
                print(f"   Com erros anteriores: {len(error_files)} notícias")
        except:
            already_processed = set()
            error_files = set()
    
    print(f"\n{'='*70}")
    print(f"CLASSIFICACAO DE NOTICIAS EM TRES CATEGORIAS")
    print(f"{'='*70}")
    print(f"Processando {total_files} notícias...")
    print(f"Timeout: {TIMEOUT_SECONDS}s por notícia")
    if START_FROM > 0:
        print(f"Começando da notícia {START_FROM}")
    if already_processed:
        print(f"Modo retomada: pulando {len(already_processed)} já processadas")
    print(f"{'='*70}\n")
    
    processed = len(already_processed)
    skipped = 0
    timeouts = 0
    consecutive_403_errors = 0
    file_errors = 0
    classification_errors = 0
    SAVE_INTERVAL = 25
    
    # Contadores por categoria
    category_counts = {
        "fraud_scrap_in_may": 0,
        "fraudes_gerais_may": 0,
        "hard_negatives_may": 0,
        "pure_negatives_may": 0
    }
    
    for index, json_file in enumerate(json_files, start=1):
        news_number = index
        
        if START_FROM > 0 and news_number < START_FROM:
            skipped += 1
            continue
        
        # Se modo reprocess_errors, processar apenas arquivos com erro
        if reprocess_errors:
            if json_file.name not in error_files:
                skipped += 1
                continue
        else:
            # Modo normal: pular já processados
            if json_file.name in already_processed:
                skipped += 1
                continue
        
        processed += 1
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                news_data = json.load(f)
            
            title = news_data.get("title", "")
            text = news_data.get("text", "")
            url = news_data.get("url", "")
            
            print(f"[{news_number}/{total_files}] Processando: {json_file.name}...")
            
            try:
                result = classifier.analyze_news(text, title, timeout=TIMEOUT_SECONDS)
                consecutive_403_errors = 0
            except OllamaError403 as e403:
                consecutive_403_errors += 1
                print(f"Erro 403 consecutivo #{consecutive_403_errors}/{MAX_CONSECUTIVE_403_ERRORS}")
                
                if consecutive_403_errors >= MAX_CONSECUTIVE_403_ERRORS:
                    print(f"\n{'='*70}")
                    print(f"INTERROMPENDO PROCESSAMENTO")
                    print(f"Motivo: {MAX_CONSECUTIVE_403_ERRORS} erros 403 consecutivos")
                    print(f"Ultima notícia: {json_file.name}")
                    print(f"{'='*70}\n")
                    
                    output_path = Path(output_file)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            "total_processed": processed,
                            "category_counts": category_counts,
                            "stopped_reason": f"Múltiplos erros 403 consecutivos ({MAX_CONSECUTIVE_403_ERRORS})",
                            "last_file": json_file.name,
                            "processed_news": processed_news,
                            "error_news": error_news
                        }, f, ensure_ascii=False, indent=2)
                    return
                
                continue
            
            # Verificar se houve erro na análise da LLM
            if result.get("analysis_error", False):
                print(f"  -> ERRO DE ANALISE LLM: {result.get('error_message', 'Erro desconhecido')}")
                
                # Registrar erro para reprocessamento
                error_entry = {
                    "file": json_file.name,
                    "title": title,
                    "url": url,
                    "error_type": "llm_analysis_error",
                    "error_message": result.get('error_message', 'Erro na análise da LLM'),
                    "timestamp": datetime.now().isoformat()
                }
                error_news.append(error_entry)
                continue
            
            # Verificar se houve timeout ou erro de análise
            if result.get('timeout_error', False) or result.get('analysis_error', False):
                print(f"  -> ERRO DETECTADO - não classificando (será registrado para reprocessamento)")
                if result.get('timeout_error', False):
                    timeouts += 1
                
                # Registrar como erro para reprocessamento
                error_entry = {
                    "file": json_file.name,
                    "title": title,
                    "url": url,
                    "error_type": "timeout_error" if result.get('timeout_error', False) else "llm_analysis_error",
                    "error_message": result.get('error_message', 'Timeout ou erro na análise'),
                    "timestamp": datetime.now().isoformat()
                }
                error_news.append(error_entry)
                continue
            
            if result.get('execution_time_seconds', 0) >= TIMEOUT_SECONDS - 1:
                timeouts += 1
            
            category = classify_and_move_news(json_file, result, dataset_path, news_data)
            
            if category:
                category_counts[category] += 1
                
                processed_entry = {
                    "file": json_file.name,
                    "title": title,
                    "url": url,
                    "category": category,
                    "analysis": result
                }
                processed_news.append(processed_entry)
                
                if result["is_fraud_related"]:
                    if result.get("fraud_empresarial", False):
                        print(f"  -> FRAUDE EMPRESARIAL DETECTADA")
                        print(f"     Tipos: {', '.join(result['fraud_types'])}")
                        if result['companies_involved']:
                            print(f"     Empresas: {', '.join(result['companies_involved'])}")
                        if result['people_involved']:
                            print(f"     Pessoas: {', '.join(result['people_involved'])}")
                    else:
                        print(f"  -> FRAUDE GERAL DETECTADA")
                        print(f"     Tipos: {', '.join(result['fraud_types'])}")
                        if result['people_involved']:
                            print(f"     Pessoas: {', '.join(result['people_involved'])}")
                elif result.get("hard_negative_candidate", False):
                    print(f"  -> HARD NEGATIVE")
                    print(f"     Motivo: {result.get('hard_negative_reason', 'N/A')}")
                else:
                    print(f"  -> PURE NEGATIVE")
            else:
                print(f"  -> ERRO na classificação")
                classification_errors += 1
                
                # Registrar erro para reprocessamento futuro
                error_entry = {
                    "file": json_file.name,
                    "title": title,
                    "url": url,
                    "error_type": "classification_error",
                    "error_message": "Falha ao classificar e mover arquivo",
                    "timestamp": datetime.now().isoformat()
                }
                error_news.append(error_entry)
        
        except Exception as e:
            print(f"  ERRO ao processar {json_file.name}: {e}")
            file_errors += 1
            
            # Registrar erro para reprocessamento futuro
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    news_data = json.load(f)
                
                error_entry = {
                    "file": json_file.name,
                    "title": news_data.get("title", ""),
                    "url": news_data.get("url", ""),
                    "error_type": "file_processing_error",
                    "error_message": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                error_news.append(error_entry)
            except:
                # Se nem conseguir ler o arquivo
                error_entry = {
                    "file": json_file.name,
                    "title": "",
                    "url": "",
                    "error_type": "file_read_error",
                    "error_message": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                error_news.append(error_entry)
            
            continue
        
        if processed % SAVE_INTERVAL == 0:
            print(f"\n{'='*70}")
            print(f"SALVAMENTO AUTOMATICO - {processed}/{total_files} notícias")
            print(f"{'='*70}")
            
            output_path = Path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "total_processed": processed,
                    "category_counts": category_counts,
                    "processed_news": processed_news,
                    "error_news": error_news,
                    "error_counts": {
                        "file_errors": file_errors,
                        "classification_errors": classification_errors,
                        "timeouts": timeouts
                    }
                }, f, ensure_ascii=False, indent=2)
            
            print(f"Arquivo de progresso salvo")
            print(f"Fraudes Empresariais: {category_counts['fraud_scrap_in_may']}")
            print(f"Fraudes Gerais: {category_counts['fraudes_gerais_may']}")
            print(f"Hard Negatives: {category_counts['hard_negatives_may']}")
            print(f"Pure Negatives: {category_counts['pure_negatives_may']}")
            if file_errors > 0 or classification_errors > 0:
                print(f"Erros: {file_errors + classification_errors} (arquivo: {file_errors}, classificação: {classification_errors})")
            print(f"{'='*70}\n")
        
        print()
    
    print(f"\n{'='*70}")
    print(f"PROCESSAMENTO CONCLUIDO")
    print(f"{'='*70}")
    print(f"Total de notícias processadas: {processed}")
    if skipped > 0:
        print(f"Notícias puladas: {skipped}")
    if timeouts > 0:
        print(f"Notícias com timeout: {timeouts}")
    
    print(f"\nDistribuição por categoria:")
    print(f"  Fraudes Empresariais: {category_counts['fraud_scrap_in_may']}")
    print(f"  Fraudes Gerais: {category_counts['fraudes_gerais_may']}")
    print(f"  Hard Negatives: {category_counts['hard_negatives_may']}")
    print(f"  Pure Negatives: {category_counts['pure_negatives_may']}")
    
    print(f"\nResumo de erros:")
    print(f"  Erros de arquivo: {file_errors}")
    print(f"  Erros de classificação: {classification_errors}")
    print(f"  Timeouts: {timeouts}")
    print(f"  Total de erros: {len(error_news)}")
    
    if category_counts['hard_negatives_may'] == 0:
        print(f"\nATENÇÃO: Nenhuma hard negative foi encontrada.")
    
    total_frauds = category_counts['fraud_scrap_in_may'] + category_counts['fraudes_gerais_may']
    if total_frauds > 0:
        print(f"\nRESUMO DE FRAUDES:")
        print(f"  Total de fraudes: {total_frauds}")
        print(f"  Empresariais: {category_counts['fraud_scrap_in_may']} ({category_counts['fraud_scrap_in_may']/total_frauds*100:.1f}%)")
        print(f"  Gerais: {category_counts['fraudes_gerais_may']} ({category_counts['fraudes_gerais_may']/total_frauds*100:.1f}%)")
    
    if error_news:
        print(f"\nATENÇÃO: {len(error_news)} arquivos com erros foram registrados para reprocessamento.")
        print(f"Para reprocessar apenas os erros, execute:")
        print(f"python processors/classify_news_triple.py --reprocess-errors")
    
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total_processed": processed,
            "category_counts": category_counts,
            "processed_news": processed_news,
            "error_news": error_news,
            "error_counts": {
                "file_errors": file_errors,
                "classification_errors": classification_errors,
                "timeouts": timeouts
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nResultados salvos em: {output_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    import sys
    import os
    
    # Forçar output imediato (sem buffer)
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    # Verificar se está rodando sem nohup/tmux/screen
    if os.getppid() == os.getsid(0):  # Terminal direto sem nohup
        print("⚠️  ATENÇÃO: Processo rodando sem nohup!")
        print("   Se fechar o terminal, o processo será interrompido.")
        print("   Para executar em background com logs visíveis:")
        print("   ./rodar_com_logs.sh")
        print("   Ou use: nohup python3 processors/classify_news_triple.py > logs/classification_$(date +%Y%m%d_%H%M%S).log 2>&1 &")
        print("   Continuando em 5 segundos...")
        import time
        time.sleep(5)
    
    INPUT_DIR = "/home/paulo/CascadeProjects/Applied_ML/collector_noticias/scraped"
    DATASET_BASE = "/home/paulo/CascadeProjects/Applied_ML/dataset"
    OUTPUT_FILE = "/home/paulo/CascadeProjects/Applied_ML/dataset/classification_results/classification_results.json"
    
    # Verificar modo de execução
    reprocess_errors = "--reprocess-errors" in sys.argv
    test_mode = "--test" in sys.argv
    
    print("\n" + "="*70)
    print("CLASSIFICADOR DE NOTICIAS - TRES CATEGORIAS")
    print("="*70)
    print(f"Diretório de entrada: {INPUT_DIR}")
    print(f"Diretório base de saída: {DATASET_BASE}")
    print(f"Arquivo de resultados: {OUTPUT_FILE}")
    print(f"Modelo: {SELECTED_MODEL}")
    if test_mode:
        print(f"Modo: TESTE (apenas {MAX_FILES_FOR_TEST} arquivos)")
    elif reprocess_errors:
        print(f"Modo: REPROCESSAR APENAS ERROS")
    else:
        print(f"Modo: PROCESSAMENTO COMPLETO")
    print("="*70 + "\n")
    
    process_all_news(INPUT_DIR, DATASET_BASE, OUTPUT_FILE, resume=True, reprocess_errors=reprocess_errors)
