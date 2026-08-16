# -*- coding: utf-8 -*-
# (a) Compare retrieval backends (char-TFIDF, word-TFIDF, BM25[, embeddings]) on dev via the full
#     pipeline; (b) dump real misclassified dev examples per segment type as a LaTeX fragment.
import io, re, sys, os
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from rapidfuzz import fuzz
import iepipe as ie

# Expects a local clone of the shared-task repo (corpora + data + scorer):
#   git clone https://github.com/Watheq9/IslamicEval2026.git
# then run from a directory containing it, or set ISLAMICEVAL_REPO=/path/to/IslamicEval2026
REPO = Path(os.environ.get("ISLAMICEVAL_REPO", "IslamicEval2026"))
out = io.open("compare_out.txt", "w", encoding="utf-8"); P = lambda *a: (print(*a), print(*a, file=out), out.flush())

P("loading corpora...")
QURAN = ie.load_quran(REPO/"Corpora/quranic_verses.json")
HADITH = ie.load_hadith(REPO/"Corpora/six_hadith_books.json", keep_full=True)
dev, _ = ie.load_segments(REPO/"dev_set/dev.jsonl")
train, _ = ie.load_segments(REPO/"train_set/train.jsonl")
keep = set(list(dict.fromkeys(s["resp_id"] for s in train))[:600])
tune = [s for s in train if s["resp_id"] in keep]
P("dev", len(dev), "tune", len(tune))

# ---------- backends: each exposes score_spans(spans, topn) -> [(best_sim, rec, [(sc,rec)...])] ----------
def _rerank(qn, cand_idx, records, topn):
    scored = []
    for j in cand_idx:
        r = records[j]
        sc = max(fuzz.token_set_ratio(qn, r["norm"]), fuzz.partial_ratio(qn, r["norm"]))/100.0
        scored.append((sc, r))
    scored.sort(key=lambda x: -x[0])
    return (scored[0][0], scored[0][1], scored[:topn]) if scored else (0.0, None, [])

class CharTFIDF:
    name = "char-TFIDF (ours)"
    def __init__(s, recs):
        s.recs = recs; s.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1)
        s.mat = s.vec.fit_transform([r["norm"] for r in recs])
    def score_spans(s, spans, k=15, topn=1, chunk=256):
        qn = [ie.normalize(x) for x in spans]; res = [(0.0,None,[]) for _ in spans]
        idx = [i for i,q in enumerate(qn) if q]
        if not idx: return res
        Q = s.vec.transform([qn[i] for i in idx])
        for st in range(0, len(idx), chunk):
            sub = idx[st:st+chunk]; sims = linear_kernel(Q[st:st+chunk], s.mat)
            for row, i in enumerate(sub):
                kk = min(k, sims.shape[1]); top = np.argpartition(sims[row], -kk)[-kk:]
                res[i] = _rerank(qn[i], top, s.recs, topn)
        return res

class WordTFIDF(CharTFIDF):
    name = "word-TFIDF"
    def __init__(s, recs):
        s.recs = recs; s.vec = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=1)
        s.mat = s.vec.fit_transform([r["norm"] for r in recs])

class BM25:
    name = "BM25"
    def __init__(s, recs):
        from rank_bm25 import BM25Okapi
        s.recs = recs; s.toks = [r["norm"].split() for r in recs]; s.bm = BM25Okapi(s.toks)
    def score_spans(s, spans, k=15, topn=1, chunk=None):
        res = []
        for x in spans:
            qn = ie.normalize(x)
            if not qn: res.append((0.0,None,[])); continue
            sc = s.bm.get_scores(qn.split()); top = np.argpartition(sc, -k)[-k:]
            res.append(_rerank(qn, top, s.recs, topn))
        return res

# ---------- verifiers (identical across backends) ----------
SURAH = {ie.normalize(v["surah_name"]): v["surah_id"] for v in QURAN if v.get("surah_name") and v.get("surah_id") is not None}
AR2EN = str.maketrans(''.join(chr(0x660+i) for i in range(10)), '0123456789')
def find_number(t):
    m = re.search(r'\d+', str(t).translate(AR2EN)); return int(m.group()) if m else None
def _w(*c): return ie.normalize(''.join(chr(x) for x in c))
BOOKS = [_w(0x627,0x644,0x628,0x62E,0x627,0x631,0x64A),_w(0x645,0x633,0x644,0x645),_w(0x627,0x644,0x62A,0x631,0x645,0x630,0x64A),
 _w(0x627,0x644,0x646,0x633,0x627,0x626,0x64A),_w(0x627,0x628,0x646,0x20,0x645,0x627,0x62C,0x647),_w(0x627,0x62D,0x645,0x62F),_w(0x645,0x627,0x644,0x643)]
def verify_cs(span, pk, pr):
    c = ie.normalize(span)
    if pr is None or not c: return "correct"
    if pk == "Ayah":
        sid = next((v for n,v in SURAH.items() if n and len(n)>2 and n in c), None)
        if sid is None: return "correct"
        if str(sid) != str(pr.get("surah_id")): return "incorrect"
        n = find_number(span)
        if n is not None and pr.get("ayah_id") is not None: return "correct" if str(n)==str(pr.get("ayah_id")) else "incorrect"
        return "correct"
    cb = next((b for b in BOOKS if b in c), None); tb = ie.normalize(str(pr.get("book") or ""))
    if cb is None or not tb: return "correct"
    return "correct" if (cb in tb or tb in cb) else "incorrect"
TAU_I = 0.85
def precompute(segs, QB, HB):
    rows = [dict(s) for s in segs]
    by = {t: [i for i,s in enumerate(segs) if (s["seg_type"] or "").strip()==t] for t in ie.SEG_TYPES}
    parent = {}
    for pos,(sc,rec,_) in zip(by["Ayah"], QB.score_spans([segs[i]["span_text"] for i in by["Ayah"]])):
        rows[pos].update(_score=sc,_rec=rec); parent[(segs[pos]["resp_id"],segs[pos]["ann_id"])]=("Ayah",rec,[rec])
    for pos,(sc,rec,top3) in zip(by["matn"], HB.score_spans([segs[i]["span_text"] for i in by["matn"]],topn=3)):
        rows[pos].update(_score=sc,_rec=rec); parent[(segs[pos]["resp_id"],segs[pos]["ann_id"])]=("matn",rec,[r for _,r in top3])
    for pos in by["claimed_source"]:
        pk,pr,_=parent.get((segs[pos]["resp_id"],segs[pos]["ann_id"]),(None,None,[])); rows[pos].update(_cs=verify_cs(segs[pos]["span_text"],pk,pr),_rec=pr)
    for pos in by["isnad"]:
        pk,pr,tops=parent.get((segs[pos]["resp_id"],segs[pos]["ann_id"]),(None,None,[])); q=ie.normalize(segs[pos]["span_text"]); fs=0.0; best=None
        if q and pk=="matn":
            for r in tops:
                if r:
                    v=max(fuzz.token_set_ratio(q,r.get("full_norm","")),fuzz.partial_ratio(q,r.get("full_norm","")))/100.0
                    if v>fs: fs,best=v,r
        rows[pos].update(_isnad=fs,_rec=best)
    for r in rows: r.setdefault("_score",0.0); r.setdefault("_cs","incorrect"); r.setdefault("_isnad",0.0); r.setdefault("_rec",None)
    return rows
def apply_(rows, ta, tm, ti=TAU_I):
    o=[]
    for r in rows:
        st=r["seg_type"]
        if st=="Ayah": p="correct" if r["_score"]>=ta else "incorrect"
        elif st=="matn": p="correct" if r["_score"]>=tm else "incorrect"
        elif st=="claimed_source": p=r["_cs"]
        elif st=="isnad": p="correct" if r["_isnad"]>=ti else "incorrect"
        else: p="incorrect"
        o.append({**r,"pred":p})
    import pandas as pd; return pd.DataFrame(o)
def run_backend(QB, HB, label):
    tr=precompute(tune,QB,HB); dr=precompute(dev,QB,HB)
    best=-1; bc=(0.9,0.82)
    for ta in [round(x,2) for x in np.arange(0.80,0.99,0.02)]:
        for tm in [round(x,2) for x in np.arange(0.70,0.95,0.02)]:
            m=ie.macro_accuracy(apply_(tr,ta,tm))["MACRO"]
            if m>best: best,bc=m,(ta,tm)
    ta,tm=bc; m=ie.macro_accuracy(apply_(dr,ta,tm))
    P(f"[{label}] taus=({ta},{tm}) DEV "+str({k:round(v,3) for k,v in m.items()}))
    return dr, m

P("=== backend comparison (dev, full pipeline) ===")
QC,HC=CharTFIDF(QURAN),CharTFIDF(HADITH)
dr_char,_=run_backend(QC,HC,"char-TFIDF (ours)")
QW,HW=WordTFIDF(QURAN),WordTFIDF(HADITH)
run_backend(QW,HW,"word-TFIDF")
try:
    QB,HB=BM25(QURAN),BM25(HADITH)
    run_backend(QB,HB,"BM25")
except Exception as e:
    P("BM25 skipped:",repr(e))

# ---------- misclassified examples (char backend / our system) ----------
def esc(t):
    t=str(t).replace("\n"," ").replace("\\","")
    for a,b in [("&","\\&"),("%","\\%"),("_","\\_"),("#","\\#"),("$","\\$"),("{","\\{"),("}","\\}"),("~"," "),("^"," ")]:
        t=t.replace(a,b)
    return t.strip()
def trunc(t,n=55):
    t=str(t).strip(); return t[:n]+("\\ldots" if len(t)>n else "")
# use tuned char taus (re-tune quickly for dev application already applied in dr_char via run_backend? we need a df)
# rebuild dev predictions at submitted thresholds:
dr=precompute(dev,QC,HC); pred=apply_(dr,0.98,0.94)
frag=io.open("examples_gen.tex","w",encoding="utf-8")
frag.write("\\begin{table}[h]\n\\centering\\small\n\\setlength{\\tabcolsep}{4pt}\n")
frag.write("\\begin{tabular}{@{}llp{3.1cm}p{3.1cm}@{}}\n\\toprule\n")
frag.write("\\textbf{Type} & \\textbf{gold/pred} & \\textbf{quoted span} & \\textbf{nearest source} \\\\\n\\midrule\n")
import pandas as pd
for st in ["Ayah","matn","isnad","claimed_source"]:
    sub=pred[(pred["seg_type"]==st) & (pred["gold"].isin(["correct","incorrect"])) & (pred["pred"]!=pred["gold"])]
    picks=sub[sub["span_text"].str.len()>8].head(1)
    if len(picks)==0: picks=sub.head(1)
    for _,r in picks.iterrows():
        rec=r.get("_rec") or {}; srctxt=rec.get("text","") if isinstance(rec,dict) else ""
        lbl=("claimed src" if st=="claimed_source" else st)
        frag.write(f"{lbl} & {r['gold']}/{r['pred']} & \\ar{{{esc(trunc(r['span_text']))}}} & \\ar{{{esc(trunc(srctxt))}}} \\\\\n\\addlinespace[2pt]\n")
frag.write("\\bottomrule\n\\end{tabular}\n")
frag.write("\\caption{Representative development misclassifications, one per segment type: the quoted span, the gold and predicted labels, and the nearest canonical source retrieved.}\n\\label{tab:errors}\n\\end{table}\n")
frag.close()
P("wrote examples_gen.tex")
out.close()
