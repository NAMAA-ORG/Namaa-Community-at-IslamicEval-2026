# Namaa Community at IslamicEval 2026 — Subtask 2 (Hallucination Identification)

System-description paper, code, notebooks and experiments for the **Namaa Community** submission to
**Subtask 2 of [IslamicEval 2026](https://github.com/Watheq9/IslamicEval2026)** (CodaBench competition
**17483**): deciding, for each citation segment in an Arabic LLM response — a quoted verse (**Ayah**),
hadith body (**matn**), chain of narration (**isnad**), or stated attribution (**claimed source**) —
whether it faithfully matches an authentic source.

The system is **retrieval-grounded verification**: each segment is normalised, matched against the
canonical Qur'an and the six major hadith collections with a character *n*-gram index refined by
edit-distance re-ranking, and adjudicated by a verifier chosen by its type. It uses no trained model
and runs on CPU.

## Results (official scorer)

| Split | Ayah | matn | claimed source | isnad | **Macro** |
|---|---|---|---|---|---|
| Development (submitted) | 0.961 | 0.913 | 0.811 | 0.700 | **0.846** |
| Official blind test | 0.818 | 0.622 | 0.340 | 0.895 | **0.668** |

**Ablation (development, cumulative)**

| Configuration | Ayah | matn | claimed source | isnad | Macro |
|---|---|---|---|---|---|
| Attribution matched as text (initial) | 0.961 | 0.913 | 0.492 | 0.533 | 0.725 |
| + parent-linked attribution | 0.961 | 0.913 | 0.811 | 0.533 | 0.805 |
| + grounded isnad (submitted) | 0.961 | 0.913 | 0.811 | 0.700 | **0.846** |

**Retrieval-backend comparison (development)**

| Backend | Ayah | matn | claimed source | isnad | Macro |
|---|---|---|---|---|---|
| character *n*-gram TF-IDF (ours) | 0.961 | 0.913 | 0.811 | 0.700 | **0.846** |
| word-level TF-IDF | 0.961 | 0.912 | 0.823 | 0.667 | 0.841 |
| Okapi BM25 | 0.963 | 0.927 | 0.823 | 0.667 | 0.845 |

Raw numbers and per-type misclassified examples are in [`experiments/`](experiments).

## Method

1. **Preprocessing** — length filtering; content-aware segmentation of over-length verses; diacritic
   augmentation (keep the vocalised original, add a diacritic-free copy); overlapping-window expansion
   for partial quotations; one normaliser for corpus and query alike, with its Arabic ranges built
   from Unicode code points rather than literal combining marks.
2. **Retrieval** — character *n*-gram TF-IDF shortlist → RapidFuzz re-rank → similarity σ ∈ [0,1].
3. **Typed verifiers** — Ayah/matn thresholded (Qur'an near-exact τₐ=0.98, matn tolerant τₘ=0.94);
   claimed source checked against the record its *parent* Ayah/matn matched; isnad **grounded** in the
   parent hadith's complete narration (τᵢ=0.85).

Full write-up: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) and the paper in [`paper/`](paper).

## Repository structure

```
paper/         system paper (PDF + LaTeX sources: .tex, .bib, acl.sty, acl_natbib.bst, build.ps1)
notebooks/     self-contained Colab notebooks (clone the task repo → run → official score → zip/push)
code/          iepipe.py (core pipeline) + compare_and_examples.py (experiment driver)
experiments/   results.json, ablation.tsv, backend_comparison.tsv, misclassified_examples.tsv, examples_table.tex
submissions/   dev prediction TSV (+ zip)
docs/          METHODOLOGY.md, PAPERS_INSIGHTS.md
```

| Notebook | Purpose |
|---|---|
| `notebooks/IslamicEval2026_Subtask2_Submission.ipynb` | End-to-end: clone → predict → official score → zip |
| `notebooks/IslamicEval2026_Task2_Experiments_Colab.ipynb` | Full pipeline + ablation + backend comparison + misclassified examples |
| `notebooks/IslamicEval2026_Task2_Verifier_GPU.ipynb` | Optional AraBERTv2 pair-verifier for Ayah/matn (GPU) |
| `notebooks/IslamicEval2026_Subtask2_RAG.ipynb` | Earlier retrieval-matching experiment |
| `notebooks/IslamicEval_Preprocessing_Artifacts.ipynb` | Corpus preprocessing (segmentation, diacritic augmentation, KB) |

## Reproduce

Every notebook is self-contained: it `git clone`s the official task repo (corpora + data + scorer),
runs, and reports the official score. `code/iepipe.py` is the importable pipeline module used by the
experiment driver; `code/compare_and_examples.py` regenerates the ablation, backend comparison and
misclassified examples in `experiments/`.

## Submission format (Subtask 2)

Tab-separated, with header, columns `Response_ID  Annotation_ID  Segment_Type  Label`
(`Label` ∈ {`correct`, `incorrect`}; never `N/A`), matched to gold by
`(Response_ID, Annotation_ID, Segment_Type)`.

## Citation

```bibtex
@inproceedings{alharbi-etal-2026-islamiceval,
  title     = {IslamicEval 2026: The Second Shared Task of Capturing LLMs Hallucination in Islamic Content},
  author    = {Alharbi, Rahaf and Alturki, Abdulelah and Mansour, Watheq and Malhas, Rana and Mubarak, Hamdy and Darwish, Kareem and Elsayed, Tamer and Magdy, Walid},
  booktitle = {Proceedings of the Fourth Arabic Natural Language Processing Conference (ArabicNLP 2026)},
  year      = {2026}
}
```

## Team & license

**Namaa Community** — Fatimah Emad Eldin, Israa, Omer Nacar, Khloud Al Jallad.
Code released under the **MIT** license. The Qur'an and hadith corpora and the task data are provided
by the IslamicEval 2026 organisers under their own terms; this repository contains only our code,
predictions, and derived analysis.
