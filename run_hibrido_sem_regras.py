#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, json, math, argparse, time, hashlib, joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sentence_transformers import SentenceTransformer
import faiss

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ==========================
# OpenAI (GPT como refinador)
# ==========================
from openai import OpenAI

def make_client():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Defina OPENAI_API_KEY no ambiente.")
    return OpenAI(api_key=key)

SYSTEM_PT = (
    "Você é um refinador de classificação. "
    "Tarefa: DADO um rótulo provisório do modelo base e exemplos rotulados recuperados, "
    "CONFIRME ou CORRIJA o rótulo final para 0 (VERDADEIRO) ou 1 (FALSO). "
    "Responda estritamente no formato:\nLABEL: 0\nJUSTIFICATIVA: ...\n"
)

PROMPT_REFINO = """Você receberá:
1) A previsão PROVISÓRIA do modelo base e sua confiança.
2) EXEMPLOS rotulados (RAG) do conjunto de treino.
3) O TEXTO alvo a classificar.

Seu trabalho:
- Se o provisório parece coerente com os exemplos e o texto, CONFIRME.
- Caso contrário, CORRIJA para o rótulo adequado.
- Seja sucinto na justificativa (1–2 frases), referenciando padrões dos exemplos quando possível.

=== PROVISÓRIO DO MODELO BASE ===
label_provisorio: {label_base}
confianca_base: {conf_base:.4f}  # prob do rótulo escolhido

=== EXEMPLOS (label conhecido) ===
{exemplos}

=== TEXTO ALVO ===
\"\"\"{texto}\"\"\"

Responda no formato:
LABEL: 0
JUSTIFICATIVA: <1-2 frases>
"""

def call_gpt(client, model, prompt, max_retries=5):
    for i in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":SYSTEM_PT},
                          {"role":"user","content":prompt}],
                temperature=0.0
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            time.sleep(2**i)
    return None

def parse_label(txt):
    if not txt:
        return None, ""
    m = re.search(r"LABEL\s*:\s*([01])", txt, flags=re.I)
    lab = int(m.group(1)) if m else None
    just = ""
    m2 = re.search(r"JUSTIFICATIVA\s*:\s*(.*)", txt, flags=re.I|re.S)
    if m2:
        just = re.sub(r"\s+"," ", m2.group(1).strip())
    return lab, just

# ==========================
# Cache para respostas do GPT
# ==========================
class JsonlCache:
    def __init__(self, path):
        self.path = path
        self.map = {}
        if os.path.exists(path):
            with open(path,"r",encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        self.map[obj["key"]] = obj["value"]
                    except:
                        pass

    def _key(self, **kwargs):
        payload = json.dumps(kwargs, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, **kwargs):
        k = self._key(**kwargs)
        return self.map.get(k), k

    def set(self, key, value):
        self.map[key] = value
        with open(self.path,"a",encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

# ==========================
# RAG (só sobre o TREINO)
# ==========================
class Retriever:
    def __init__(self, texts, labels, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.texts = np.array(texts)
        self.labels = np.array(labels)
        emb = self.encoder.encode(list(texts), batch_size=256, show_progress_bar=True, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb.astype("float32"))

    def topk(self, query_text, k=5, max_chars_each=220):
        q = self.encoder.encode([query_text], normalize_embeddings=True).astype("float32")
        D, I = self.index.search(q, k)
        pares = []
        for idx in I[0]:
            if idx < 0: 
                continue
            t = re.sub(r"\s+"," ", self.texts[idx]).strip()
            if len(t) > max_chars_each:
                t = t[:max_chars_each] + " ..."
            pares.append((t, int(self.labels[idx])))
        return pares

def exemplos_block(pares):
    return "\n".join([f"- [label={lab}] {txt}" for txt, lab in pares])

# ==========================
# Incerteza do modelo base
# ==========================
def prob_to_entropy(p1):
    p = np.clip([1-p1, p1], 1e-8, 1-1e-8)
    return -np.sum(p*np.log2(p))

def margin(p1):
    return abs(p1-0.5)

# ==========================
# Treino do modelo base (seu SVM + TF-IDF) com calibração
# ==========================
def treinar_base_svm_tfidf(train_texts, train_labels):
    base = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            min_df=5, max_df=0.95,
            sublinear_tf=True,
            lowercase=True,
            analyzer="word"
        )),
        ("svm", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE))
    ])
    # Calibra para obter predict_proba (usa decision_function internamente)
    base_cal = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    base_cal.fit(train_texts, train_labels)
    return base_cal

# ==========================
# Avaliação híbrida (sem regras)
# ==========================
def avalia_hibrido(part_df, label_col, text_col, base_model, retriever, args, split_name, outdir):
    client = make_client()
    cache  = JsonlCache(os.path.join(outdir, f"{args.prefixo}.cache.jsonl"))

    registros, y_true, y_pred = [], [], []

    for _, row in tqdm(part_df.reset_index(drop=True).iterrows(), total=len(part_df), desc=f"{split_name}"):
        texto = str(row[text_col])
        true_lab = int(row[label_col])

        # 1) Modelo base (com probas)
        proba = base_model.predict_proba([texto])[0]  # [p0, p1]
        p1 = float(proba[1])              # prob de FALSO (label=1)
        lab_base = int(p1 >= 0.5)
        ent = prob_to_entropy(p1)
        mg  = margin(p1)
        conf_ok = (mg >= args.uncert_th) and (ent <= args.entropy_th)

        if conf_ok:
            final_lab = lab_base
            origem = "ml_base_confiante"
            just = f"p1={p1:.3f}; mg={mg:.3f}; ent={ent:.3f}"
        else:
            # 2) Refino GPT + RAG
            exs = retriever.topk(texto, k=args.k)
            prompt = PROMPT_REFINO.format(
                label_base=lab_base,
                conf_base=(p1 if lab_base==1 else 1-p1),
                exemplos=exemplos_block(exs),
                texto=texto if len(texto) <= args.max_chars else (texto[:args.max_chars] + " ...")
            )
            cached, key = cache.get(model=args.model, prompt=prompt, split=split_name)
            if cached is None:
                resp = call_gpt(client, args.model, prompt)
                if resp is None:
                    lab_ref, just_ref = lab_base, "falha_API"
                    raw = ""
                else:
                    lab_ref, just_ref = parse_label(resp)
                    if lab_ref is None:
                        lab_ref, just_ref = lab_base, "parse_falhou"
                    raw = resp
                cache.set(key, {"raw": raw, "pred": int(lab_ref), "just": just_ref})
            else:
                raw = cached.get("raw","")
                lab_ref = int(cached.get("pred", lab_base))
                just_ref= cached.get("just","")

            final_lab = lab_ref
            origem = "gpt_refiner"
            just = f"base(p1={p1:.3f}; mg={mg:.3f}; ent={ent:.3f}) | {just_ref}"

        y_true.append(true_lab)
        y_pred.append(final_lab)
        registros.append({
            "split": split_name,
            "texto": texto,
            "true_label": true_lab,
            "pred_label": final_lab,
            "origem": origem,
            "justificativa": just
        })

    # métricas
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm  = confusion_matrix(y_true, y_pred, labels=[0,1])

    # salvar artefatos
    casos_path = os.path.join(outdir, f"{args.prefixo}.casos_{split_name}.jsonl")
    with open(casos_path, "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cm_path = os.path.join(outdir, f"{args.prefixo}.cm_{split_name}.csv")
    pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv(cm_path, index=True, encoding="utf-8")

    met = {
        "split": split_name,
        "acc": acc, "macroF1": f1m,
        "F1_0": rep.get("0",{}).get("f1-score",0.0),
        "F1_1": rep.get("1",{}).get("f1-score",0.0),
        "recall_0": rep.get("0",{}).get("recall",0.0),
        "recall_1": rep.get("1",{}).get("recall",0.0),
        "precision_0": rep.get("0",{}).get("precision",0.0),
        "precision_1": rep.get("1",{}).get("precision",0.0)
    }
    # breakdown da origem
    origem_counts = pd.Series([r["origem"] for r in registros]).value_counts().to_dict()
    for k,v in origem_counts.items():
        met[f"n_{k}"] = v
    return met

# ==========================
# MAIN
# ==========================
def main(args):
    os.makedirs(args.out, exist_ok=True)

    # carregar dados
    df = pd.read_csv(args.data, sep=";", encoding="utf-8-sig")
    assert args.text_col in df.columns and args.label_col in df.columns
    keep_cols = [args.text_col, args.label_col]
    if "fonte" in df.columns:
        keep_cols.append("fonte")
    df = df[keep_cols].dropna(subset=[args.text_col, args.label_col]).copy()
    df[args.label_col] = df[args.label_col].astype(int)

    # splits 80/10/10
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df[args.label_col], random_state=RANDOM_STATE)
    val_df,   test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df[args.label_col], random_state=RANDOM_STATE)

    # limitar (opcional)
    if args.max_per_split:
        def cap(df_, n):
            return df_.groupby(args.label_col, group_keys=False).apply(lambda g: g.sample(min(len(g), n), random_state=RANDOM_STATE))
        train_df = cap(train_df, args.max_per_split)
        val_df   = cap(val_df,   max(1, args.max_per_split//4))
        test_df  = cap(test_df,  max(1, args.max_per_split//4))

    # 1) treinar base (SVM+TFIDF calibrado) e salvar
    base_model = treinar_base_svm_tfidf(train_df[args.text_col], train_df[args.label_col])
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "svm_tfidf_calibrado.joblib")
    joblib.dump(base_model, model_path)
    print("Modelo base salvo em:", model_path)

    # 2) retriever com o TREINO
    retriever = Retriever(
        texts=train_df[args.text_col].tolist(),
        labels=train_df[args.label_col].tolist(),
        model_name=args.emb_model
    )

    # 3) avaliação híbrida (val e test)
    resultados = []
    for nome, part in [("val", val_df), ("test", test_df)]:
        met = avalia_hibrido(part, args.label_col, args.text_col, base_model, retriever, args, nome, args.out)
        resultados.append(met)

    met_df = pd.DataFrame(resultados)
    met_csv = os.path.join(args.out, f"{args.prefixo}.metricas.csv")
    met_df.to_csv(met_csv, index=False, encoding="utf-8")

    # Excel consolidado
    xlsx = os.path.join(args.out, f"{args.prefixo}.xlsx")
    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as w:
        met_df.to_excel(w, index=False, sheet_name="metricas")
        for nome in ["val","test"]:
            cm_path = os.path.join(args.out, f"{args.prefixo}.cm_{nome}.csv")
            if os.path.exists(cm_path):
                pd.read_csv(cm_path).to_excel(w, index=False, sheet_name=f"cm_{nome}")

        # Se tiver coluna 'fonte', gera resumo por fonte no TEST (usando predição do modelo base)
        if "fonte" in df.columns:
            test = test_df.copy()
            proba = base_model.predict_proba(test[args.text_col])
            yhat  = (proba[:,1] >= 0.5).astype(int)
            linhas = []
            for fonte, sub in test.groupby("fonte"):
                y_true_f = sub[args.label_col].to_numpy()
                y_pred_f = yhat[sub.index]
                rep = classification_report(y_true_f, y_pred_f, output_dict=True, zero_division=0)
                linhas.append([fonte, "VERÍDICO", rep["0"]["precision"], rep["0"]["recall"], rep["0"]["f1-score"]])
                linhas.append([fonte, "FALSO",    rep["1"]["precision"], rep["1"]["recall"], rep["1"]["f1-score"]])
                linhas.append([fonte, "Average",
                               (rep["0"]["precision"]+rep["1"]["precision"])/2,
                               (rep["0"]["recall"]+rep["1"]["recall"])/2,
                               (rep["0"]["f1-score"]+rep["1"]["f1-score"])/2])
            if linhas:
                df_fontes = pd.DataFrame(linhas, columns=["fonte","class","precision","recall","F1"])
                df_fontes.to_excel(w, index=False, sheet_name="por_fonte_test")

    print("[OK] métricas ->", met_csv)
    print("[OK] planilha ->", xlsx)
    print("[OK] casos    ->", os.path.join(args.out, f"{args.prefixo}.casos_val.jsonl"),
          " / ", os.path.join(args.out, f"{args.prefixo}.casos_test.jsonl"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-col", default="texto")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--out", default="resultados_hibrido_v3")
    ap.add_argument("--prefixo", default="resultados_hibrido_v3")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--emb-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--uncert-th", type=float, default=0.08)
    ap.add_argument("--entropy-th", type=float, default=0.70)
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--max-per-split", type=int, default=None)
    args = ap.parse_args()
    main(args)
