#!/usr/bin/env python3
"""
Comparacao de desempenho: TF-IDF + SVM (ML) vs LLM

Avalia ambos os modelos no mesmo conjunto de teste reservado (353 exemplos)
e gera um relatorio comparativo com metricas, matrizes de confusao e CSV de predicoes.

Uso:
    python test_llm_vs_ml.py                # execucao completa (inicia do zero; apaga checkpoint existente)
    python test_llm_vs_ml.py --resume       # retomar de checkpoint salvo
    python test_llm_vs_ml.py --report-only  # gerar relatorio sem chamar a LLM
                                            # (requer checkpoint existente)

Variaveis de ambiente:
    OLLAMA_HOST   URL do servidor Ollama   (padrao: http://localhost:11434)
    OLLAMA_MODEL  Modelo a usar            (padrao: qwen3:8b)
"""

import os
import re
import sys
import json
import time
import signal
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


# ──────────────────────────────── CONFIG ─────────────────────────────────────

BASE_DIR  = Path("/home/paulo/CascadeProjects/Applied_ML")
TEST_DIR  = BASE_DIR / "test"
OUT_DIR   = TEST_DIR / "test_with_llm"

# Dataset (mesmo conjunto usado para avaliar o modelo ML)
DATASET_PATH = TEST_DIR / "dataset" / "TEST_PREPROCESSED.csv"
# Nota: o texto ja foi pre-processado para vetorizacao esparsa (stopwords removidas, etc.)
# Isso pode penalizar ligeiramente a LLM vs texto bruto, mas garante a mesma base de comparacao.

# Artefatos do modelo ML
VECTORIZER_PATH = TEST_DIR / "model" / "tfidf_ngram12_mindf2_maxdf09_maxfeat10k_20260518_220950_vectorizer.pkl"
SVM_MODEL_PATH  = TEST_DIR / "model" / "svm_tf_idf_20260519_174152.pkl"

# Saidas
CHECKPOINT_FILE  = OUT_DIR / "llm_predictions_checkpoint.json"
RESULTS_CSV      = OUT_DIR / "llm_predictions.csv"
EXTRACTIONS_CSV  = OUT_DIR / "llm_fraud_extractions.csv"   # atualizado a cada fraude detectada
REPORT_MD        = OUT_DIR / "comparison_report.md"
COMPARISON_PNG   = OUT_DIR / "comparison_confusion_matrices.png"

# LLM
OLLAMA_HOST      = os.getenv("OLLAMA_HOST", "http://localhost:11434") #'https://ollama.ceos.ufsc.br')
SELECTED_MODEL   = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
LLM_TEMPERATURE  = 0
TIMEOUT_SECONDS  = 180 
CHECKPOINT_EVERY = 25   # salvar checkpoint a cada N exemplos

# ─────────────────────────────────────────────────────────────────────────────


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Timeout excedido")


# ─────────────────────── LLM CLASSIFIER ──────────────────────────────────────

class LLMBinaryClassifier:
    """
    Classificador binario via LLM (Ollama).
    Retorna label=1 (fraude envolvendo empresas) ou label=0 (nao-fraude).
    Adaptado de processors/classify_news_triple.py.
    """

    def __init__(self):
        self.llm = None
        self._initialized = False
        print(f"LLM: {SELECTED_MODEL} @ {OLLAMA_HOST}")

    def _ensure_llm(self):
        if not self._initialized:
            print(f"Conectando ao Ollama em {OLLAMA_HOST}...")
            try:
                self.llm = ChatOllama(
                    model=SELECTED_MODEL,
                    base_url=OLLAMA_HOST,
                    temperature=LLM_TEMPERATURE,
                    timeout=TIMEOUT_SECONDS,
                )
                self._initialized = True
                print("Conexao estabelecida!")
            except Exception as e:
                print(f"ERRO ao conectar: {e}")
                self.llm = None
                self._initialized = True

    def classify(self, title: str, text: str) -> Dict:
        """
        Classifica uma noticia.

        Retorna dict com:
            label                  : int  (1=fraude, 0=nao-fraude, -1=erro)
            execution_time_seconds : float
            error                  : bool
            error_type             : str  (apenas se error=True)
        """
        default = {
            "label": -1,
            "empresas_envolvidas": [],
            "tipos_de_fraude": [],
            "execution_time_seconds": 0.0,
            "error": True,
            "error_type": "unknown",
        }

        self._ensure_llm()
        if not self.llm:
            default["error_type"] = "llm_not_initialized"
            return default

        combined = f"{title}\n\n{text}".strip() if title else text.strip()
        if not combined:
            default["error_type"] = "empty_text"
            return default

        prompt = f"""Voce e um especialista em deteccao de fraudes empresariais em noticias em portugues.

Sua tarefa e:
1. Classificar a noticia como fraude (label=1) ou nao-fraude (label=0).
   - label=1: a noticia trata de FRAUDE envolvendo empresas (fraude em licitacao, corrupcao,
     cartel, superfaturamento, lavagem de dinheiro, desvio de recursos publicos, etc.)
   - label=0: a noticia NAO trata de fraude envolvendo empresas

2. Se label=1, extraia tambem:
   - "empresas_envolvidas": lista com os NOMES das empresas citadas (Ltda., S.A., ME, EPP,
     EIRELI, Construtora, Engenharia, etc.). Apenas nomes especificos, nao termos genericos.
   - "tipos_de_fraude": lista com os tipos de fraude identificados no texto.

   Se label=0, retorne listas vazias para esses campos.

Leia o texto completo antes de decidir.

----------------------------------------------------------------------
Texto da noticia:
\"\"\"{combined}\"\"\"
----------------------------------------------------------------------

Retorne APENAS um JSON valido, sem nenhum texto adicional antes ou depois.

Exemplo para fraude:
{{"label": 1, "empresas_envolvidas": ["Construtora ABC Ltda.", "XYZ Engenharia S.A."], "tipos_de_fraude": ["fraude em licitacao", "superfaturamento"]}}

Exemplo para nao-fraude:
{{"label": 0, "empresas_envolvidas": [], "tipos_de_fraude": []}}
"""

        start = time.time()
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(TIMEOUT_SECONDS)

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            signal.alarm(0)
            raw = response.content.strip()
            parsed = self._parse_response(raw)
            parsed["execution_time_seconds"] = round(time.time() - start, 2)
            parsed["error"] = parsed["label"] == -1
            if parsed["error"]:
                parsed["error_type"] = "parse_error"
            return parsed

        except TimeoutError:
            signal.alarm(0)
            default["execution_time_seconds"] = round(time.time() - start, 2)
            default["error_type"] = "timeout"
            return default

        except Exception as e:
            signal.alarm(0)
            default["execution_time_seconds"] = round(time.time() - start, 2)
            default["error_type"] = str(e)[:120]
            return default

    @staticmethod
    def _clean_list(value) -> List[str]:
        """Normaliza um campo de lista vindo do JSON da LLM."""
        if isinstance(value, str):
            value = [value] if value.strip() else []
        elif not isinstance(value, list):
            value = []
        seen, out = set(), []
        for item in value:
            s = str(item).strip().strip('"').strip("'").strip()
            if s and s not in seen:
                out.append(s)
                seen.add(s)
        return out

    def _parse_response(self, raw: str) -> Dict:
        # Remover bloco de raciocinio (ex: <think>...</think> do qwen3)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Remover delimitadores de bloco de codigo
        raw = re.sub(r"^```(?:json)?", "", raw.strip()).strip()
        raw = re.sub(r"```$", "", raw.strip()).strip()

        try:
            data = json.loads(raw)
            label = int(data.get("label", -1))
            if label not in (0, 1):
                label = -1
            return {
                "label":               label,
                "empresas_envolvidas": self._clean_list(data.get("empresas_envolvidas", [])),
                "tipos_de_fraude":     self._clean_list(data.get("tipos_de_fraude",     [])),
            }
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: busca no texto bruto
        if re.search(r'"label"\s*:\s*1', raw):
            return {"label": 1, "empresas_envolvidas": [], "tipos_de_fraude": []}
        if re.search(r'"label"\s*:\s*0', raw):
            return {"label": 0, "empresas_envolvidas": [], "tipos_de_fraude": []}

        return {"label": -1, "empresas_envolvidas": [], "tipos_de_fraude": []}


# ─────────────────────── DATASET / MODEL ─────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    print(f"\nCarregando dataset: {DATASET_PATH.name}")
    df = pd.read_csv(DATASET_PATH)
    print(f"  {len(df)} exemplos | colunas: {list(df.columns)}")
    print(f"  Positivos (label=1): {int(df['label'].sum())} | Negativos (label=0): {int((df['label'] == 0).sum())}")
    return df.reset_index(drop=True)


def load_ml_artifacts():
    print(f"\nCarregando vetorizador TF-IDF: {VECTORIZER_PATH.name}")
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    print(f"  Features: {len(vectorizer.get_feature_names_out())} | N-gram: {vectorizer.ngram_range}")

    print(f"Carregando modelo SVM: {SVM_MODEL_PATH.name}")
    with open(SVM_MODEL_PATH, "rb") as f:
        svm_model = pickle.load(f)
    print(f"  Tipo: {type(svm_model).__name__}")

    return vectorizer, svm_model


def get_ml_predictions(df: pd.DataFrame, vectorizer, svm_model) -> Dict:
    print("\nObtendo predicoes do modelo ML (TF-IDF + SVM)...")
    combined = (
        df["title_clean"].fillna("") + " " + df["text_clean"].fillna("")
    ).tolist()
    X = vectorizer.transform(combined)
    y_pred  = svm_model.predict(X)
    y_proba = svm_model.predict_proba(X)[:, 1] if hasattr(svm_model, "predict_proba") else None
    print(f"  {len(y_pred)} predicoes obtidas")
    return {"labels": y_pred, "probas": y_proba}


# ─────────────────────── CHECKPOINT ──────────────────────────────────────────

def load_checkpoint() -> List[Dict]:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Checkpoint encontrado: {len(data)} predicoes")
        return data
    return []


def save_checkpoint(predictions: List[Dict]):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)


def write_extractions_csv(df: pd.DataFrame, extractions: List[Dict]):
    """
    Escreve (sobrescreve) o CSV de extracoes com todas as noticias classificadas
    pela LLM como fraude. Chamado apos cada nova deteccao para manter o arquivo
    sempre atualizado.
    """
    if not extractions:
        return

    rows = []
    for entry in extractions:
        idx   = entry["index"]
        title = ""
        if df is not None and idx < len(df):
            title = str(df.iloc[idx].get("title_clean", "") or "")[:120]

        rows.append({
            "news_index":          idx,
            "title_clean":         title,
            "true_label":          entry.get("true_label", ""),
            "llm_label":           entry.get("llm_label", 1),
            "empresas_envolvidas": "; ".join(entry.get("empresas_envolvidas", [])),
            "tipos_de_fraude":     "; ".join(entry.get("tipos_de_fraude",     [])),
            "execution_time_s":    entry.get("execution_time_seconds", ""),
        })

    pd.DataFrame(rows).to_csv(EXTRACTIONS_CSV, index=False, encoding="utf-8")


# ─────────────────────── LLM LOOP ────────────────────────────────────────────

def run_llm_classification(df: pd.DataFrame) -> List[Dict]:
    classifier = LLMBinaryClassifier()

    llm_predictions = load_checkpoint()
    already_done    = {p["index"] for p in llm_predictions}

    # Reconstruir lista de extracoes a partir do checkpoint (para retomada)
    extractions: List[Dict] = [
        p for p in llm_predictions
        if p.get("llm_label") == 1 and not p.get("error", False)
    ]
    if extractions:
        write_extractions_csv(df, extractions)
        print(f"  CSV de extracoes reconstruido: {len(extractions)} fraudes do checkpoint")

    total  = len(df)
    errors = 0

    print(f"\n{'='*70}")
    print(f"CLASSIFICACAO LLM — {total} exemplos")
    print(f"Modelo:  {SELECTED_MODEL}")
    print(f"Timeout: {TIMEOUT_SECONDS}s por exemplo")
    if already_done:
        print(f"Retomando: {len(already_done)}/{total} ja classificados")
    print(f"{'='*70}\n")

    for idx, row in df.iterrows():
        if idx in already_done:
            continue

        title      = str(row.get("title_clean", "") or "")
        text       = str(row.get("text_clean",  "") or "")
        true_label = int(row["label"])

        print(f"[{idx + 1:>4}/{total}] ", end="", flush=True)

        result = classifier.classify(title, text)

        if result["error"]:
            errors += 1
            print(f"ERRO: {result.get('error_type', '?')}")
        else:
            pred    = result["label"]
            correct = pred == true_label
            empresas = result.get("empresas_envolvidas", [])
            fraudes  = result.get("tipos_de_fraude", [])
            extra = ""
            if pred == 1:
                extra = f" | empresas={len(empresas)} tipos={len(fraudes)}"
            print(
                f"pred={pred} real={true_label} {'OK' if correct else 'ERROU'}"
                f"  ({result['execution_time_seconds']}s){extra}"
            )

        entry = {
            "index":               int(idx),
            "true_label":          true_label,
            "llm_label":           result.get("label", -1),
            "empresas_envolvidas": result.get("empresas_envolvidas", []),
            "tipos_de_fraude":     result.get("tipos_de_fraude",     []),
            "execution_time_seconds": result.get("execution_time_seconds", 0.0),
            "error":               result["error"],
            "error_type":          result.get("error_type", ""),
        }
        llm_predictions.append(entry)

        # Atualizar CSV de extracoes imediatamente apos cada fraude detectada
        if result.get("label") == 1 and not result["error"]:
            extractions.append(entry)
            write_extractions_csv(df, extractions)

        new_total = len(llm_predictions)
        if new_total % CHECKPOINT_EVERY == 0:
            save_checkpoint(llm_predictions)
            n_fraud = sum(1 for p in llm_predictions if p.get("llm_label") == 1)
            print(f"\n  [Checkpoint: {new_total}/{total} | fraudes: {n_fraud} | erros: {errors}]\n")

    save_checkpoint(llm_predictions)

    n_fraud = sum(1 for p in llm_predictions if p.get("llm_label") == 1 and not p.get("error"))
    print(f"\n{'='*70}")
    print(f"Classificacao LLM concluida | total: {total} | fraudes: {n_fraud} | erros: {errors}")
    print(f"CSV de extracoes atualizado: {EXTRACTIONS_CSV.name} ({len(extractions)} entradas)")
    print(f"{'='*70}\n")

    return llm_predictions


# ─────────────────────── METRICAS ────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred, name: str, y_proba=None) -> Dict:
    y_pred  = np.array(y_pred)
    y_true  = np.array(y_true)

    valid_mask = y_pred != -1
    n_errors   = int((~valid_mask).sum())
    yt = y_true[valid_mask]
    yp = y_pred[valid_mask]

    if len(yt) == 0:
        return {"name": name, "n_total": len(y_true), "n_evaluated": 0, "n_errors": n_errors}

    metrics = {
        "name":        name,
        "n_total":     len(y_true),
        "n_evaluated": len(yt),
        "n_errors":    n_errors,
        "accuracy":    round(float(accuracy_score(yt, yp)), 4),
        "precision":   round(float(precision_score(yt, yp, zero_division=0)), 4),
        "recall":      round(float(recall_score(yt, yp, zero_division=0)), 4),
        "f1":          round(float(f1_score(yt, yp, zero_division=0)), 4),
        "classification_report": classification_report(
            yt, yp, target_names=["Negative", "Positive"], digits=4
        ),
        "confusion_matrix": confusion_matrix(yt, yp).tolist(),
    }

    if y_proba is not None:
        yprob = np.array(y_proba)[valid_mask]
        metrics["roc_auc"] = round(float(roc_auc_score(yt, yprob)), 4)

    return metrics


# ─────────────────────── OUTPUTS ─────────────────────────────────────────────

def plot_comparison(ml_metrics: Dict, llm_metrics: Dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    model_label = SELECTED_MODEL.replace(":", "-")
    fig.suptitle(
        f"Confusion Matrix — TF-IDF + SVM vs LLM ({model_label})",
        fontsize=14,
        fontweight="bold",
    )

    for ax, m, subtitle in zip(
        axes,
        [ml_metrics, llm_metrics],
        [
            f"TF-IDF + SVM\nF1={ml_metrics.get('f1','?')} | Acc={ml_metrics.get('accuracy','?')}",
            f"LLM ({model_label})\nF1={llm_metrics.get('f1','?')} | Acc={llm_metrics.get('accuracy','?')}",
        ],
    ):
        cm = np.array(m.get("confusion_matrix", [[0, 0], [0, 0]]))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
        )
        ax.set_title(subtitle, fontsize=12)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(COMPARISON_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figura salva: {COMPARISON_PNG.name}")


def generate_report(ml_metrics: Dict, llm_metrics: Dict, llm_predictions: List[Dict]):
    valid_times = [
        p["execution_time_seconds"]
        for p in llm_predictions
        if not p.get("error", False) and p.get("execution_time_seconds", 0) > 0
    ]
    total_time = sum(valid_times)
    avg_time   = total_time / len(valid_times) if valid_times else 0.0
    n_errors   = sum(1 for p in llm_predictions if p.get("error", False))

    model_label = SELECTED_MODEL.replace(":", "-")

    def row(metric):
        ml_val  = ml_metrics.get(metric, "N/A")
        llm_val = llm_metrics.get(metric, "N/A")
        ml_str  = f"{ml_val:.4f}"  if isinstance(ml_val,  float) else str(ml_val)
        llm_str = f"{llm_val:.4f}" if isinstance(llm_val, float) else str(llm_val)
        return f"| {metric.capitalize():10} | {ml_str:>12} | {llm_str:>12} |"

    lines = [
        "# Comparacao: TF-IDF + SVM vs LLM",
        "",
        f"**Data:** {datetime.now().strftime('%d/%m/%Y %H:%M')}  ",
        f"**Dataset:** `TEST_PREPROCESSED.csv` — {ml_metrics.get('n_total', '?')} exemplos (mesmo conjunto da avaliacao do ML)  ",
        f"**Modelo LLM:** `{SELECTED_MODEL}`  ",
        f"**Servidor Ollama:** `{OLLAMA_HOST}`  ",
        "",
        "> **Nota:** O texto enviado a LLM e o texto pre-processado para vetorizacao esparsa",
        "> (stopwords removidas). Isso pode penalizar a LLM vs uso de texto bruto.",
        "",
        "---",
        "",
        "## Metricas Comparativas",
        "",
        f"| Metrica    | TF-IDF + SVM | LLM ({model_label}) |",
        "|------------|-------------|" + "-" * (len(model_label) + 7) + "|",
        row("accuracy"),
        row("precision"),
        row("recall"),
        row("f1"),
    ]

    if "roc_auc" in ml_metrics:
        lines.append(
            f"| ROC-AUC    | {ml_metrics['roc_auc']:.4f} | N/A (sem probabilidades) |"
        )

    lines += [
        "",
        "## Detalhes da Avaliacao LLM",
        "",
        f"- Exemplos avaliados: {llm_metrics.get('n_evaluated', '?')} / {llm_metrics.get('n_total', '?')}",
        f"- Erros / timeouts: {n_errors}",
        f"- Tempo total de inferencia: {total_time:.1f}s ({total_time / 60:.1f} min)",
        f"- Tempo medio por exemplo: {avg_time:.1f}s",
        "",
        "## Classification Report — TF-IDF + SVM",
        "",
        "```",
        ml_metrics.get("classification_report", "N/A"),
        "```",
        "",
        f"## Classification Report — LLM ({model_label})",
        "",
        "```",
        llm_metrics.get("classification_report", "N/A"),
        "```",
        "",
        "## Analise",
        "",
    ]

    f1_ml  = ml_metrics.get("f1",  0.0)
    f1_llm = llm_metrics.get("f1", 0.0)
    delta  = f1_llm - f1_ml

    pct = f" ({abs(delta) / f1_ml * 100:.1f}% de diferenca)" if f1_ml != 0 else ""
    if delta > 0:
        lines.append(
            f"A LLM **superou** o modelo ML em F1-Score: **{f1_llm:.4f}** vs {f1_ml:.4f}"
            f" (delta = {delta:+.4f}{pct})."
        )
    elif delta < 0:
        lines.append(
            f"O modelo ML **superou** a LLM em F1-Score: **{f1_ml:.4f}** vs {f1_llm:.4f}"
            f" (delta = {delta:+.4f}{pct})."
        )
    else:
        lines.append("Os dois modelos alcancaram o mesmo F1-Score.")

    n_fraud_detected = sum(
        1 for p in llm_predictions
        if p.get("llm_label") == 1 and not p.get("error", False)
    )
    n_with_companies = sum(
        1 for p in llm_predictions
        if p.get("llm_label") == 1 and p.get("empresas_envolvidas")
    )
    n_with_types = sum(
        1 for p in llm_predictions
        if p.get("llm_label") == 1 and p.get("tipos_de_fraude")
    )

    lines += [
        "",
        "## Extracoes da LLM (noticias classificadas como fraude)",
        "",
        f"- Noticias classificadas como fraude pela LLM: **{n_fraud_detected}**",
        f"- Com empresas identificadas: {n_with_companies}",
        f"- Com tipos de fraude identificados: {n_with_types}",
        f"- Ver detalhes: `{EXTRACTIONS_CSV.name}`",
        "",
        "## Arquivos Gerados",
        "",
        f"- `{COMPARISON_PNG.name}` — matrizes de confusao lado a lado",
        f"- `{RESULTS_CSV.name}` — todas as predicoes (LLM + ML + ground truth + extracoes)",
        f"- `{EXTRACTIONS_CSV.name}` — apenas fraudes detectadas pela LLM com extracoes",
        f"- `{CHECKPOINT_FILE.name}` — checkpoint com predicoes brutas da LLM",
        "",
    ]

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Relatorio salvo: {REPORT_MD.name}")


def save_results_csv(df: pd.DataFrame, llm_predictions: List[Dict], ml_pred_dict: Dict):
    llm_map  = {p["index"]: p for p in llm_predictions}
    ml_preds = ml_pred_dict["labels"]

    rows = []
    for idx, row in df.iterrows():
        lp          = llm_map.get(idx, {})
        true_label  = int(row["label"])
        ml_pred     = int(ml_preds[idx])
        llm_pred    = lp.get("llm_label", -1)
        llm_valid   = llm_pred in (0, 1)

        rows.append(
            {
                "index":               idx,
                "true_label":          true_label,
                "ml_prediction":       ml_pred,
                "llm_prediction":      llm_pred if llm_valid else None,
                "ml_correct":          int(true_label == ml_pred),
                "llm_correct":         int(true_label == llm_pred) if llm_valid else None,
                "llm_empresas":        "; ".join(lp.get("empresas_envolvidas", [])),
                "llm_tipos_fraude":    "; ".join(lp.get("tipos_de_fraude",     [])),
                "llm_time_seconds":    lp.get("execution_time_seconds", None),
                "llm_error":           lp.get("error", True),
                "llm_error_type":      lp.get("error_type", ""),
            }
        )

    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
    print(f"CSV de resultados salvo: {RESULTS_CSV.name}")


# ─────────────────────── MAIN ────────────────────────────────────────────────

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report_only = "--report-only" in sys.argv
    resume      = "--resume"      in sys.argv

    # Sem --resume e sem --report-only: apagar checkpoint para iniciar do zero
    if not resume and not report_only and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("[INFO] Checkpoint anterior removido. Iniciando do zero.")
        print("       Use --resume para retomar de onde parou.\n")

    print("=" * 70)
    print("COMPARACAO: TF-IDF + SVM  vs  LLM")
    print("=" * 70)
    print(f"Dataset:    {DATASET_PATH}")
    print(f"Modelo ML:  {SVM_MODEL_PATH.name}")
    print(f"LLM:        {SELECTED_MODEL} @ {OLLAMA_HOST}")
    print(f"Saida:      {OUT_DIR}")
    print("=" * 70 + "\n")

    # 1. Dataset
    df = load_dataset()
    y_true = df["label"].values

    # 2. Predicoes LLM
    if report_only:
        llm_predictions = load_checkpoint()
        if not llm_predictions:
            print("ERRO: nenhum checkpoint encontrado. Execute sem --report-only primeiro.")
            return 1
        print(f"Modo relatorio: {len(llm_predictions)} predicoes carregadas do checkpoint.")
        # Regenerar CSV de extracoes a partir do checkpoint
        extractions = [
            p for p in llm_predictions
            if p.get("llm_label") == 1 and not p.get("error", False)
        ]
        if extractions:
            write_extractions_csv(df, extractions)
            print(f"CSV de extracoes regenerado: {len(extractions)} fraudes")
    else:
        llm_predictions = run_llm_classification(df)

    # 3. Predicoes ML
    vectorizer, svm_model = load_ml_artifacts()
    ml_pred_dict = get_ml_predictions(df, vectorizer, svm_model)

    # 4. Metricas
    # Constroi vetor alinhado com y_true (posicao i = exemplo i); usa -1 para ausentes
    llm_idx_map = {p["index"]: p.get("llm_label", -1) for p in llm_predictions}
    llm_labels  = np.array([llm_idx_map.get(i, -1) for i in range(len(df))])

    ml_metrics  = compute_metrics(y_true, ml_pred_dict["labels"], "TF-IDF + SVM",
                                  y_proba=ml_pred_dict["probas"])
    llm_metrics = compute_metrics(y_true, llm_labels, f"LLM ({SELECTED_MODEL})")

    print("\n" + "=" * 70)
    print("RESULTADOS COMPARATIVOS")
    print("=" * 70)
    header = f"  {'Metrica':<12} {'TF-IDF+SVM':>12}   {'LLM':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for metric in ("accuracy", "precision", "recall", "f1"):
        ml_v  = ml_metrics.get(metric,  "N/A")
        llm_v = llm_metrics.get(metric, "N/A")
        print(
            f"  {metric.capitalize():<12} "
            f"{(f'{ml_v:.4f}' if isinstance(ml_v, float) else str(ml_v)):>12}   "
            f"{(f'{llm_v:.4f}' if isinstance(llm_v, float) else str(llm_v)):>12}"
        )
    if "roc_auc" in ml_metrics:
        print(f"  {'ROC-AUC':<12} {ml_metrics['roc_auc']:>12.4f}   {'N/A':>12}")
    print("=" * 70 + "\n")

    # 5. Salvar
    save_results_csv(df, llm_predictions, ml_pred_dict)
    plot_comparison(ml_metrics, llm_metrics)
    generate_report(ml_metrics, llm_metrics, llm_predictions)

    print("\n" + "=" * 70)
    print("CONCLUIDO — arquivos salvos em:")
    print(f"  {OUT_DIR}/")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
