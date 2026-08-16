# IslamicEval 2026 — Subtask 2: Methodology (what the submission notebook actually does)

This documents, step by step, the pipeline in
`notebooks/IslamicEval2026_Subtask2_Submission.ipynb` — the notebook that produced the dev
submission uploaded to Hugging Face. It is written so you can defend every design choice in a paper.

---

## 0. What problem we solved

Subtask 2 gives us, for each LLM response, a set of **already-located citation segments** and asks us
to label each one **`correct`** or **`incorrect`**:

| Segment type | What it is | Share of score |
|---|---|---|
| `Ayah` | a quoted Qur'anic verse | 25% |
| `matn` | the text of a hadith | 25% |
| `isnad` | the hadith's chain of narration | 25% |
| `claimed_source` | the stated attribution (surah name / verse no., or collection like Bukhari) | 25% |

Metric: **accuracy per type, macro-averaged, gold-`N/A` rows excluded**. Because it is *macro*, a
type with 30 instances (isnad) counts as much as one with 698 (Ayah).

**We did NOT train a model.** We built a *retrieval + string-matching classifier*: for a quoted span,
find the nearest authentic text in the canonical corpus and decide `correct` if the quote is close
enough. This is what "RAG" means in this project — **no LLM, no generation** — which is why the ≤13B
parameter limit is met trivially.

---

## 1. Data sources — and what we did / did NOT use

The notebook **clones** `github.com/Watheq9/IslamicEval2026` and uses, directly:

- `Corpora/quranic_verses.json` — **6,236** canonical verses `{surah_id, surah_name, ayah_id, ayah_text}`.
- `Corpora/six_hadith_books.json` — **34,994** hadith `{hadithID, BookID, title, hadithTxt, Matn}`.
- `dev_set/dev.jsonl` — **484** responses to label.
- `train_set/train.jsonl` — **4,706** responses, used **only to tune thresholds**.
- `Scoring_scripts/task2_scoring.py` — the official scorer, run locally.

**Not used:** the *preprocessing notebook's* outputs (the segmented `quran_segmented.csv`, the
diacritic-augmented `*_augmented.csv`, and the overlapping-segment knowledge base `kb_*`). Those live
in your Google Drive and would break a self-contained "clone-and-run" notebook. Instead we load the
**raw** corpora and normalize them **in memory** using the *same* normalization the preprocessing
notebook applies — so the result is equivalent for matching, without the Drive dependency.
> To use the preprocessed corpora instead: set `USE_DRIVE = True` in Cell 1 and point the paths at
> your `processed/` folder. The retrieval logic is unchanged.

---

## 2. Arabic normalization (the single most important step)

**Why:** an LLM quote is usually un-vocalized ("بسم الله الرحمن الرحيم") while the canonical verse is
fully vocalized Uthmani script ("بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ"). Compared literally they look
different; we must erase that difference so equal meaning ⇒ equal string.

**What we do**, applied **identically to the corpus and to every quoted span**:

1. **Strip تشكيل + tatweel** — remove the full Qur'anic diacritic/annotation range
   (`U+0610–061A, 064B–065F, 0670, 06D6–06ED`) and the elongation character `ـ`.
2. **Unify letters** — `أ إ آ ٱ → ا`, `ى → ي`, `ؤ/ئ → و/ي`, `ة → ه`.
3. **Cleanup** — drop non-Arabic characters, collapse whitespace.

This is levels L1–L3 of the scheme in your *Morphological Analysis* sheet. We stop at L3 (no
lemma/root) because L4/L5 can over-merge distinct verses and hurt precision.

---

## 3. Recovering what to classify

`dev.jsonl` gives each segment as **character offsets** into `generated_answer`
(`span_start`, `span_end`). We slice the answer text to recover the exact quoted string, and keep the
gold `label` (present in dev) so we can tune and self-score. Result: a flat list of **2,728 dev
segments**, each `{resp_id, ann_id, seg_type, span_text, gold}`.

---

## 4. Retrieval index (finding candidate sources fast)

For each corpus we build a **character n-gram TF-IDF index** (`char_wb`, n = 3–5) with scikit-learn.

**Why char n-grams (not words):** Arabic is morphologically rich (prefixes/suffixes glue onto words).
Sub-word character sequences stay stable under that inflection, so a slightly different word form still
retrieves the right verse. TF-IDF then ranks corpus texts by shared n-grams with the query.

This gives a fast **shortlist** (top-15 candidates) per span. TF-IDF alone is recall-oriented and
approximate — hence step 5.

---

## 5. Precise similarity (RapidFuzz re-ranking)

For each shortlisted candidate we compute the **best of two** RapidFuzz measures against the span and
keep the maximum as `best_score ∈ [0,1]`:

- `token_set_ratio` — order-insensitive; forgiving of extra/missing words (good for whole quotes).
- `partial_ratio` — best alignment of a short span inside a longer verse (good for fragments).

**Performance note:** retrieval is done **once per span, batched** (one vectorized TF-IDF matmul per
chunk of 256 spans, then fuzzy re-rank). This is what makes threshold tuning essentially free — see
step 7 — and turns a ~40-minute run into a couple of minutes on CPU.

---

## 6. The decision rule per segment type

| Type | Rule |
|---|---|
| **Ayah** | `best_score` vs Qur'an ≥ `tau_ayah` → `correct`, else `incorrect` |
| **matn** | `best_score` vs Hadith ≥ `tau_matn` → `correct`, else `incorrect` |
| **claimed_source** | verified against the **parent** Ayah/matn's matched source record (below) |
| **isnad** | **grounded**: similarity of the quoted chain to the parent hadith's full narration ≥ `tau_isnad` (below) |

**claimed_source — the important fix.** A `claimed_source` span is a *citation label* (e.g. "البقرة:
31" or "رواه البخاري"), **not** verse text. Matching that label against the corpus (as the original
RAG notebook did) is meaningless and scored only 0.35. Instead we:

1. Remember, for each annotation, the source record its **Ayah/matn** matched (the "parent").
2. Parse the claim: for Qur'an, detect the surah name (and verse number) named in the span; for
   Hadith, detect the collection (Bukhari, Muslim, …).
3. Compare to the parent's true reference. **Prior-anchored:** predict the majority label (`correct`)
   *unless* we positively detect a mismatch (named surah ≠ matched surah, or wrong verse number, or
   wrong collection). This lifted claimed_source from **0.35 → 0.81**.

**isnad — now grounded (v2).** Earlier we used the majority prior (0.53 on dev). We now **ground** it:
the isnad segment lives in an annotation whose matn matched a specific hadith record, and that record
carries the *full narration* (`hadithTxt` = chain + matn). We compare the quoted isnad to that full
narration (max over the parent matn's top-3 matches) and threshold. This separates cleanly — gold-
correct chains score ~0.84 similarity, gold-incorrect ~0.75 — and lifts isnad **0.53 → 0.70**, macro
**0.796 → 0.841**. `tau_isnad` is tuned on train (≈0.85). Still improvable with a dedicated narrator
DB (see §9).

---

## 7. Threshold tuning — on TRAIN, not dev

`tau_ayah` and `tau_matn` are swept over a grid and the pair maximizing **macro accuracy** is chosen.
Crucially we tune on a **train sample (1,200 responses), never on dev**, then apply the frozen
thresholds to dev. This keeps the reported dev number an honest estimate of blind-test performance
rather than an over-fit. Because retrieval is precomputed (step 5), the whole grid is re-scored
instantly.

Chosen values on this data: `tau_ayah ≈ 0.96`, `tau_matn ≈ 0.92`.

---

## 8. Output, self-scoring, packaging

- **Write** `submission_dev.tsv`: `Response_ID  Annotation_ID  Segment_Type  Label`, tab-separated,
  with header; one row per segment, `correct`/`incorrect` only (never `N/A`); duplicate keys dropped.
  The scorer matches by `(Response_ID, Annotation_ID, Segment_Type)` and ignores gold-`N/A` rows, so
  emitting a label for every segment is safe and guarantees no "missing prediction" penalties.
- **Score** with the official `task2_scoring.py` — identical to what CodaBench runs.
- **Zip** → upload to CodaBench competition 17483.

### Verified dev result (official scorer)

| metric | v1 | **v2 (current)** |
|---|---|---|
| **accuracy (macro)** | 0.796 | **0.841** |
| accuracy_Ayah | 0.944 | 0.944 |
| accuracy_matn | 0.901 | 0.913 |
| accuracy_claimed_source | 0.807 | 0.807 |
| accuracy_isnad | 0.533 | **0.700** |
| missing_predictions | 0 | 0 |

---

## 8b. A correctness bug we caught (important)

An early version wrote the diacritic-stripping regex using **literal Arabic combining marks** inside a
range, e.g. `[ؐ-ً...]`. Combining marks reorder around the range dashes when a file is saved, which
turned the intended range `0610–061A` into `0610–064B` — silently swallowing the **base Arabic letters**
(0621–064A). The normalizer then deleted *all* text, so every span matched nothing and was labelled
`incorrect`. The fix: **build every Arabic range from integer codepoints via `chr()`** (pure-ASCII
source), which cannot reorder. Lesson for the paper's reproducibility section: never embed literal
Arabic combining marks in a regex character class.

---

## 9. Where the score is lost, and how to improve it

1. **isnad (now 0.70)** — a dedicated narrator DB / exact-chain corpus would push it further; also tune
   `topn` and the `token_set`/`partial` mix.
2. **claimed_source (0.81)** — extend the collection list, handle numeric `surah:ayah` references and
   kunya spellings; gate the flip on parent-match confidence.
3. **matn / Ayah (0.91 / 0.94)** — near the ceiling for lexical matching; a multilingual-embedding
   retrieval pass or `nine_hadith_books.csv` adds recall for paraphrases.
4. **Ensembling** — averaging retrieval variants mostly helps Ayah/matn (already high); the remaining
   error is dominated by isnad and claimed_source, so spend effort there.
5. **Supervised verifier** — if lexical matching plateaus, fine-tune AraBERT on `(span,
   retrieved_source) → correct/incorrect` from `train.jsonl`.

---

## Appendix — how this relates to your other two notebooks

- **`IslamicEval2026_Subtask2_RAG.ipynb`** — the origin of the method. The submission notebook is a
  corrected, self-contained descendant: fixed the record schema (`type`/`span_start`/`span_end`,
  `annotation_id`), fixed `claimed_source` to use the parent record, and batched retrieval for speed.
- **`IslamicEval_Preprocessing_Artifacts.ipynb`** — builds cleaned/segmented/augmented corpora and the
  overlapping-segment KB on your Drive, plus paper tables. The submission notebook **does not depend on
  it** (it normalizes raw corpora in memory instead), but its outputs are drop-in compatible via
  `USE_DRIVE = True` if you want the exact same corpus across all notebooks.
```
