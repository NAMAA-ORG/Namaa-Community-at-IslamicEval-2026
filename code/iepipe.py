# -*- coding: utf-8 -*-
"""Robust IslamicEval Subtask-2 pipeline. Arabic ranges built from codepoints (ASCII source)."""
import re, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from rapidfuzz import fuzz

_TASH_RANGES=[(0x610,0x61A),(0x64B,0x65F),(0x670,0x670),(0x6D6,0x6DC),(0x6DF,0x6E8),(0x6EA,0x6ED)]
_TASHKEEL=re.compile('['+''.join(chr(a)+'-'+chr(b) for a,b in _TASH_RANGES)+']')
_TATWEEL=chr(0x640)
_NON_AR=re.compile('[^'+chr(0x621)+'-'+chr(0x64A)+'\\s]')
_SPACES=re.compile(r'\s+')
_ALEF=re.compile('['+''.join(chr(c) for c in (0x622,0x623,0x625,0x627,0x671,0x621))+']')
def strip_diacritics(t):
    if not t: return ''
    return _SPACES.sub(' ', _TASHKEEL.sub('', str(t)).replace(_TATWEEL,'')).strip()
def normalize(text, letters=True):
    t=strip_diacritics(text)
    if letters:
        t=_ALEF.sub(chr(0x627), t)
        t=(t.replace(chr(0x649),chr(0x64A)).replace(chr(0x624),chr(0x648))
            .replace(chr(0x626),chr(0x64A)).replace(chr(0x629),chr(0x647)))
        t=_SPACES.sub(' ', _NON_AR.sub(' ', t)).strip()
    return t

def read_json_any(path):
    txt=Path(path).read_text(encoding="utf-8").strip()
    try: return json.loads(txt)
    except json.JSONDecodeError:
        return [json.loads(l) for l in txt.splitlines() if l.strip()]
def first_key(d, keys):
    for k in keys:
        if k in d and d[k] not in (None, ""): return d[k]
    return None
def load_quran(path):
    out=[]
    for d in read_json_any(path):
        t=first_key(d,["ayah_text","text","full_text"])
        if not t: continue
        out.append({"text":str(t),"norm":normalize(t),"surah_id":first_key(d,["surah_id","surah"]),
                    "surah_name":first_key(d,["surah_name","surahName"]),"ayah_id":first_key(d,["ayah_id","ayahId"])})
    return out
def load_hadith(path, keep_full=False):
    out=[]
    for d in read_json_any(path):
        m=first_key(d,["Matn","matn","hadith_text","text"])
        if not m: continue
        rec={"text":str(m),"norm":normalize(m),"book":first_key(d,["title","book","BookName"]),"book_id":first_key(d,["BookID","book_id"])}
        if keep_full:
            full=first_key(d,["hadithTxt","hadith_text","full_text"]) or ""
            nf=normalize(full); nm=rec["norm"]; rec["full_norm"]=nf
            rec["chain_norm"]=nf.replace(nm," ").strip() if nm and nm in nf else nf
        out.append(rec)
    return out
def load_segments(path):
    data=read_json_any(path); segs=[]
    for rec in data:
        rid=first_key(rec,["id","Response_ID"]); ans=first_key(rec,["generated_answer","response","answer","text"]) or ""
        for ann in (rec.get("annotations") or []):
            aid=first_key(ann,["annotation_id","id"])
            for s in (ann.get("segments") or []):
                st=first_key(s,["type","segment_type","Segment_Type"])
                a=first_key(s,["span_start","start","char_start"]); b=first_key(s,["span_end","end","char_end"])
                span=first_key(s,["span_text","text"])
                if span is None and a is not None and b is not None and int(b)>int(a): span=ans[int(a):int(b)]
                segs.append({"resp_id":rid,"ann_id":aid,"seg_type":st,"span_text":span or "","gold":first_key(s,["label","Label","gold"])})
    return segs,data
class Retriever:
    def __init__(self, records, ngram=(3,5)):
        self.records=records
        self.vec=TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, min_df=1)
        self.mat=self.vec.fit_transform([r["norm"] for r in records]) if records else None
    def score_spans(self, spans, k=15, chunk=256, topn=1):
        qn=[normalize(s) for s in spans]; res=[(0.0,None,[]) for _ in spans]
        idxs=[i for i,q in enumerate(qn) if q]
        if not idxs or self.mat is None: return res
        Q=self.vec.transform([qn[i] for i in idxs])
        for st in range(0,len(idxs),chunk):
            sub=idxs[st:st+chunk]; sims=linear_kernel(Q[st:st+chunk],self.mat)
            for row,i in enumerate(sub):
                kk=min(k,sims.shape[1]); top=np.argpartition(sims[row],-kk)[-kk:]; q=qn[i]; scored=[]
                for j in top:
                    rec=self.records[j]
                    sc=max(fuzz.token_set_ratio(q,rec["norm"]),fuzz.partial_ratio(q,rec["norm"]))/100.0
                    scored.append((sc,rec))
                scored.sort(key=lambda x:-x[0]); res[i]=(scored[0][0],scored[0][1],scored[:topn])
        return res
SEG_TYPES=["Ayah","matn","isnad","claimed_source"]
def macro_accuracy(df):
    per={}
    for st in SEG_TYPES:
        sub=df[(df["seg_type"]==st)&(df["gold"].isin(["correct","incorrect"]))]
        per[st]=float((sub["pred"]==sub["gold"]).mean()) if len(sub) else float("nan")
    valid=[v for v in per.values() if v==v]; per["MACRO"]=sum(valid)/len(valid) if valid else float("nan")
    return per
