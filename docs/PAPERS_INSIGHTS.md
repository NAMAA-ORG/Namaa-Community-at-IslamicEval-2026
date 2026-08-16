# IslamicEval 2025 → 2026: what the winning papers did, and how we beat them

Distilled from the 9 papers in `papers/` (the 2025 overview + all system papers). Focus: techniques
that are **reusable on a CPU-only, ≤13B budget** and map onto the **2026** subtasks we're submitting.

---

## 1. The 2025 leaderboard (verbatim) — what "90+" actually means

**Task 1 (2025)** — 1A Identification (char-level macro-F1), 1B Validation (accuracy), 1C Correction (accuracy):

| Team | 1A F1 | 1B Acc | 1C Acc | Approach |
|---|---|---|---|---|
| **Burhan AI** | **90.06** 🥇 | 88.60 | 66.56 | agentic LLM (o4-mini/gpt-4.1-mini + code-interpreter) for 1A; hierarchical index cascade + LLM repair for 1B/1C |
| **HUMAIN** | 87.20 | 86.14 | **68.18** 🥇 | LLM span tagging (TANL) + Needleman–Wunsch offset alignment; classical index + LCS + bge-reranker |
| **TCE** | 86.11 | **89.82** 🥇 | – | few-shot Qwen-235B/GPT-4o + rapidfuzz; RAG verifier (Quran-strict / Hadith-lenient) |
| **Isnad AI** (ours, 2025) | 66.97 | – | – | AraBERTv2 token classifier trained on **rule-based synthetic data** |
| mucAI | 44.88 | – | – | – |
| **majority baseline** | 36.17 | 70.00 | 67.52 | – |

**Only ONE score cleared 90 in the entire 2025 task: Burhan AI's 90.06 on span detection**, and it required an **agentic LLM with a code-interpreter tool** (to count character offsets reliably). Pure
rule-based detection tops out ~35% (our 2025 Database-Lookup ablation) to ~67% (our AraBERTv2 model).
1C correction is essentially unsolved (best 68.18 barely beats the 67.52 "all-uncorrectable" baseline).

---

## 2. Mapping 2025 → 2026 (important: the tasks were renumbered)

| 2026 subtask | 2025 equivalent | 2025 best | Our current dev |
|---|---|---|---|
| **Task 1** — span detection (Ayah/matn/isnad/claimed_source) | Subtask 1A (Ayah/Hadith only) | 90.06 F1 (Burhan, agentic LLM) | **~0.48** char-F1 |
| **Task 2** — verification correct/incorrect | Subtask 1B (validation) | **89.82 acc** (TCE) | **0.841** macro |
| **Task 3** — correction | Subtask 1C | 68.18 acc (HUMAIN) | not built |
| **Task 4** — answer relevance | (new; loosely Task 2 QA) | – | **0.618** F1 |

Caveats: 2026 Task 1 adds **isnad + claimed_source** (2025 was Ayah/Hadith only) → harder. 2026 Task 2
is **macro over 4 types** (isnad/claimed_source each 25%), whereas 2025 1B accuracy was Ayah/Hadith
only → our 0.841 is measured on a harder metric than TCE's 0.898.

---

## 3. The reusable, CPU-friendly recipe (converged across all three top teams)

### 3a. Arabic normalization (universal)
Strip diacritics/tashkeel + tatweel (`ً-ْـ` and the full annotation range); unify alef
variants (إ/أ/آ/ٱ→ا), ya/waw-hamza, ta-marbuta; strip honorifics; collapse whitespace.
**Nuance:** strip diacritics for *indexing/matching*, but TCE showed keeping diacritics for an *LLM
verifier* is +2–3 pts. (We already do the matching normalization.)

### 3b. The matching cascade (BurhanAI's index + HUMAIN + TCE), cheap → expensive, early-stop
1. **exact hash** (raw) →
2. **normalized hash** (diacritic-free) →
3. **strict substring containment** →
4. **char n-gram fuzzy** (3-grams; rapidfuzz) →
5. **LCS ratio** (Quran ≥0.85–0.90, Hadith ≥0.75) →
6. **semantic** (small Arabic sentence-transformer / bge-reranker-v2-m3 ~568M — optional, CPU-OK) →
7. **token-overlap / Jaccard-on-trigrams** (last resort).
Prefilter with **length bucketing**. Clean quotes resolve at step 1–3 instantly and *exactly*.

### 3c. The single highest-value verification rule (TCE **and** HUMAIN, independently)
> **Qur'an = strict word-for-word substring** (diacritics/spacing ignored). **Hadith = paraphrase-
> tolerant** (the *matn* legitimately varies across the six books — don't demand exact match).
This is exactly the Ayah-strict / matn-fuzzy split we can bake into Task 2.

### 3d. Span detection (Task 1) without an expensive model
- **Trigger-word / citation-pattern prompting** (TCE lists the templates): قال الله تعالى، قوله تعالى،
  قال رسول الله ﷺ، رواه البخاري … These bracket the citation and boost recall.
- **Offsets deterministically**: never trust model-emitted indices — locate the matched substring in
  the source and compute start/end (replaces Burhan's code-interpreter; HUMAIN's Needleman–Wunsch is
  the alignment alternative; TCE's rapidfuzz sliding-window @90% is the fuzzy alternative).
- **Chunking is essential**: TCE sentence-aware 800-char chunks (+6.5 pts).
- **Isnad AI (our 2025) Database-Lookup cascade** (the reusable rule system): normalize → build
  **overlapping 5–15-word segments (step 3)** of original+normalized → sort KB by length desc →
  substring-match **longest-first** → **char-boolean overlap guard** → `No_Spans` fallback. Weakness:
  Hadith recall and a flood of short false positives.

---

## 4. Concrete plan per 2026 task (prioritized by expected gain / effort, CPU-only)

### Task 2 — verification (now 0.841; 2025 analogue hit ~0.90)
1. **Add exact + normalized-substring cascade before fuzzy** for Ayah (strict) and keep fuzzy for matn
   (paraphrase-tolerant). *(experiment running now.)*
2. Keep the **isnad grounding** (already +0.045) and tune `topn`.
3. Optional: a small local LLM verifier (Gemma-2-9B / ALLaM-7B, ≤13B) as a tie-breaker on ambiguous
   matches, Quran-strict / Hadith-lenient two-prompt design + early-exit (TCE's 0.898 recipe).
   Target: **0.86–0.90**.

### Task 1 — span detection (now ~0.48; rule ceiling ~0.67, LLM ceiling ~0.90)
1. Replace clause-splitting with the **Isnad-AI Database-Lookup cascade** (overlapping-segment index,
   longest-match, overlap guard) for Ayah/matn → should reach **~0.6**.
2. Better isnad/claimed_source via the **trigger-word templates** + numeric/collection regex.
3. To truly chase 90: **fine-tune a ≤13B model** (our 2025 AraBERTv2 stack, or guided-JSON decoding on
   Command-R7B/Jais-13B) — needs GPU for training. This is the only path to 90 and matches your 2025 work.

### Task 4 — relevance (now 0.618 all-relevant)
No 2025 analogue helps directly (those were retrieval-QA). Beating all-1 needs a **semantic relevance
model** (question↔span with an Arabic embedding). Lexical signal alone is too weak. Target modest.

### Task 3 — correction (not built)
If wanted: HUMAIN's cascade — exact substring → LCS (Quran 0.85 / Hadith 0.75) → bge-reranker-v2-m3
(α=0.7 blend); output canonical text **with diacritics restored**; mark uncorrectable as `خطأ`.
Note: correction is the hardest task (2025 best only 68%).

---

## 5. Honest ceiling statement
On a **CPU-only, no-LLM** budget, realistic targets are: Task 2 ~0.86–0.90, Task 1 ~0.6–0.67, Task 4
~0.62. **Reaching 90 on span detection specifically requires an LLM detector** (fine-tuned ≤13B or
agentic) — that's the one place the 2025 winner needed real model horsepower, and it's the natural
extension of your own 2025 Isnad AI AraBERT system.
