# rag-hallu-bench

A comparison of hallucination checkers on medical RAG answers. Each checker reads a question, the retrieved documents and the answer, and gives a probability that the answer says something the documents don't support. All of them are scored on the same rows of three public datasets. The checkers include TypeSafe's Jev, two open models of the same kind, open-weight verifiers, LLMs used as judges with and without reasoning, the grounding checks sold by AWS and Google, and two trivial references.

I started it while adding claim verification to [ebm-lens](https://github.com/mauroforlin/ebm-lens), which now uses Jev, but nothing here imports from that project. The write-up is at https://mauroforlin.github.io/rag-hallu-bench/.

## This branch: MedDecider

This branch adds [MedDecider](https://huggingface.co/collections/lion-ai/meddecider-6acd32515928b98a6671de7b) (thelion.ai) 4B, 9B and 27B to the benchmark. It is not part of the published article. Each model is loaded with its authors' `meddecider_infer.py` from a pinned revision and gets the same question as `jev-choice`; the score is 1 − P(faithful), averaged over both option orders (`hallubench/detectors/generative.py`). The 4B and 9B ran in bf16, the 27B quantized to int8 (bitsandbytes) to fit on a 40 GB A100. All three scored every row.

| Checker | Macro AUROC | RAGBench PubMedQA |
|---|---|---|
| Qwen3-235B, yes/no probability | 0.925 | 0.781 |
| MedDecider 27B (int8) | 0.920 | 0.802 |
| Jev Choice | 0.917 | 0.815 |
| MedDecider 9B | 0.903 | 0.771 |
| MedDecider 4B | 0.900 | 0.790 |
| BioDecision v2 4B | 0.881 | 0.703 |

To rerun, open `notebooks/colab_gpu.ipynb` on an A100 with `ONLY_MEDDECIDER = True`. The MedDecider adapters are CC-BY-NC-4.0. The registry also lists `meddecider-31b`, which has no results here.

## What counts as a hallucination

An answer is counted as hallucinated when it states something the retrieved documents don't, even if it is medically correct. The test is about faithfulness to the documents and says nothing about whether the documents are right. MedHallu partly mixes in correctness, as noted below.

## Data

`hallubench build-data` writes `data/examples.jsonl`, 5,246 rows from three datasets pinned to exact Hub revisions and sampled with a fixed seed, so every machine builds the same file. The command prints its sha256 (`df950765b366f89c`) for checking.

| Dataset | Rows | Answers | Labels | Caveats |
|---|---|---|---|---|
| [RAGBench](https://huggingface.co/datasets/galileo-ai/ragbench) PubMedQA + CovidQA-RAG | 1,000 + 246 | GPT-3.5 and Claude 3 Haiku answering over retrieved passages, the closest thing here to production RAG output | GPT-4-turbo / GPT-4o, sentence by sentence | Labels come from a model, which may favour judges that behave like GPT-4. PubMedQA answers each question twice, once per generator, and both answers share a split and a bootstrap group. RAGBench also ships precomputed RAGAS and TruLens scores |
| [HaluBench](https://huggingface.co/datasets/PatronusAI/HaluBench) PubMedQA + CovidQA | 1,000 + 1,000 | Gold answers plus answers rewritten to be plausible but unfaithful | Derived from the source QA datasets | Easy for the strongest checkers, which score 0.95-0.99. Lynx was trained on these sources. In CovidQA about half the answers share their question with another answer |
| [MedHallu](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu) pqa_labeled | 2,000 (1,000 questions, gold + hallucinated) | PubMedQA gold answers and generated hallucinations, graded easy, medium or hard | Generation pipeline with verification | The gold answer is the abstract's conclusion and the passages are the rest of the abstract, so a gold answer can say more than its passages. Wrong answers are also shorter than gold ones (length alone gives AUROC 0.745). BioDecision was trained on MedHallu's generated split |

Each dataset is split 70/30 into test and dev by a hash of the question, so every answer to the same question lands on the same side: MedHallu's gold/hallucinated pairs, RAGBench's two answers to one question, and HaluBench-CovidQA's faithful and rewritten answers, which have no question ID and are grouped by the question's text. A stratified core slice of about 1,000 rows (19% of each dataset, split and label) is used for the calibration numbers and the prompt ablation.

### ExpertQA, tried and left out

The Healthcare / Medicine part of [ExpertQA](https://github.com/chaitanyamalaviya/ExpertQA) (702 claims, each rated by a medical professional) would be the only human-labelled set here. Its loader is in `data.py` but switched off (`INCLUDE_EXPERTQA = False`). Every checker scored close to chance on it (AUROC about 0.50), much as every model does on the ExpertQA column of [LLM-AggreFact](https://llm-aggrefact.github.io/). The stored evidence is an 80-190 word fragment of each page. For 7 of 16 claims that the experts marked as supported and Jev did not, I recovered the full 2023 page from the Wayback Machine, and Jev still found all 7 unsupported. I checked one of them by hand and the page doesn't say what the claim says. The sample is small, but it suggests some ratings follow medical correctness more than support by the cited page.

## Checkers

Every checker outputs one number per row, P(hallucinated) between 0 and 1, and each one's docstring explains how its native output is mapped to it. They are grouped by where the number comes from.

Computed by a model built to return probabilities:

| Name | What it is | Runs on |
|---|---|---|
| `jev-choice` | TypeSafe Jev (`jev-1.13.0`), one Choice over the whole answer (faithful / unsupported / contradicted), score 1 - P(faithful) | API |
| `jev-noul` | Jev, one Noul asking whether the answer contains anything the documents don't back | API |
| `jev-sentences` | Jev, one Choice per answer sentence (supports / contradicts / says nothing) in a single request, answer scored by its worst sentence | API |
| `laya` | Open decision model of the same kind (Convai, 0.4B encoder, Apache-2.0), asked Jev's exact questions | CPU / GPU |
| `biodecision-v2-4b` | Open medical decision model (Qwen3.5-4B, Together's Tev1 recipe, 1.75M biomedical decisions), asked Jev's exact questions through its authors' server code. Trained on MedHallu/HaluBench-related data, so in-distribution there and clean on RAGBench | GPU |

Computed from token probabilities or a classifier head, by models not trained to calibrate them:

| Name | What it is | Runs on |
|---|---|---|
| `judge-logprob-qwen3-235b` | Qwen3-235B-A22B-Instruct-2507 answering yes or no, score P(no) from its token probabilities. Pinned to one provider (Google Vertex) so every row is judged by the same deployment | API |
| `hhem` | Vectara HHEM-2.1-Open, T5 consistency classifier | CPU / GPU |
| `minicheck-ft5`, `minicheck-deberta` | MiniCheck (EMNLP 2024) sentence-level checkers, reimplemented from the reference code | CPU / GPU |
| `alignscore-large` | AlignScore (ACL 2023), the RoBERTa-large checkpoint NVIDIA NeMo Guardrails uses, reimplemented from the reference code in its documented `nli_sp` mode | CPU / GPU |
| `lettucedetect-large` | ModernBERT token classifier trained on RAGTruth, marks the unsupported words | CPU / GPU |
| `bespoke-minicheck-7b` | The strongest MiniCheck model on LLM-AggreFact, read as P(Yes) per sentence | GPU ≥ 24 GB |
| `granite-guardian-8b` | IBM Granite Guardian 3.3, groundedness risk read as P(yes) at the verdict token. It gives "yes" and "no" the same weight, to the precision it runs at, on 26 rows, which therefore sit at 0.5 | GPU ≥ 24 GB |
| `lynx-8b` | Patronus Lynx v1.1, which writes its reasoning before PASS/FAIL, read as P(FAIL) at the verdict token | GPU ≥ 24 GB |

Written by the model as text:

| Name | What it is | Runs on |
|---|---|---|
| `judge-qwen3-235b` | The same Qwen3-235B on the same provider, asked for a JSON verdict with a self-reported confidence like the judges below, so the two Qwen rows differ only in how the number is read | API |
| `judge-gpt-6-luna` | OpenAI GPT-6 Luna as judge, JSON verdict with a self-reported confidence | API |
| `judge-gpt-5.6-luna` | The same prompt on the previous Luna generation | API |
| `judge-gemini-2.5-flash` | The same prompt on Gemini 2.5 Flash with reasoning off | API |
| `judge-claude-haiku-5.5-thinking` | The same prompt on Claude Haiku 5.5 with reasoning on. Anthropic's content filter declines 9 rows (virology, infections), which stay unscored, so its coverage is 99.8% | API |
| `judge-gemini-3.8-flash-thinking` | The same prompt on Gemini 3.8 Flash with reasoning on, RAGBench rows only for cost | API |
| `ragas-faithfulness`, `trulens-groundedness` | LLM-as-judge pipelines precomputed by the RAGBench authors, RAGBench rows only. The score is a share of supported statements and not a probability | free |

Sold by cloud providers:

| Name | What it is | Runs on |
|---|---|---|
| `bedrock-contextual-grounding` | AWS Bedrock Guardrails' contextual grounding check through the ApplyGuardrail API, given the documents, the question and the answer, score 1 - grounding. Billed per 1,000 characters | AWS |
| `vertex-check-grounding` | Google's Check Grounding API (Vertex AI Search), given the documents (cut into facts of up to 10,000 characters) and the answer, which is all it accepts, score 1 - support. When it finds no claim worth checking it returns no score and flags nothing, and those 110 rows are graded as not flagged (macro AUROC 0.746; 0.761 leaving them out, 0.757 scoring them 0.5) | Google Cloud |

Trivial references, graded like the checkers but kept out of the comparisons with Jev:

| Name | What it is |
|---|---|
| `baseline-length` | The answer's length in words |
| `baseline-novel-words` | The share of the answer's content words that never appear in the documents |

A surface feature can point either way (MedHallu's wrong answers are shorter, RAGBench's longer), so each reference takes its direction from the dev rows of each dataset, the way the checkers take their thresholds.

The LLM judges and the Jev-style models get the same three-way rubric. Qwen3-235B is run both ways, once read through its token probabilities and once writing its confidence, so for that model the effect of the method can be separated from the effect of the model. For the other judges the two still go together.

## Metrics

- AUROC, the main number: the chance that a randomly picked hallucinated answer scores higher than a randomly picked faithful one, from 0.5 (chance) to 1. It doesn't depend on a threshold or on how the score is scaled, so very different checkers can be compared. It is reported per dataset with a 95% bootstrap interval, and checkers are ranked by macro AUROC, the mean over the five datasets, because pooling the datasets would reward a checker for scoring whole datasets differently.
- AUPRC, which also reflects how rare hallucinations are. On RAGBench only 14-30% of answers are hallucinated.
- Precision, recall and F1 at a threshold chosen on dev and applied to test, both per dataset and with one global threshold for all of them.
- Calibration error (ECE) and the precision of confident flags, meaning how often a "90% or more" is right.
- Precision and NPV at realistic hallucination rates. The datasets are about 45% hallucinated, so `acting` in `metrics.json` recomputes both at 5, 10 and 20% from each checker's hit rates with Bayes' rule, with intervals. These pool all five datasets; `docs/data.json` also has them on RAGBench alone (`acting_ragbench`).
- Sentence-level AUROC on RAGBench, for the checkers that work sentence by sentence.
- MedHallu by difficulty, and how often each checker flags MedHallu's gold answers.
- Paired difference in AUROC against Jev on the rows both scored, with a paired cluster bootstrap (20,000 resamples of whole questions) and Holm-corrected p-values over every checker and dataset (106 comparisons).
- Macro AUROC without MedHallu, without HaluBench, and on RAGBench alone.
- A prompt ablation, with Jev and the Qwen judge asked in two other ways on the core slice (family `ablation`, kept out of the main table).
- Run-to-run stability of Jev and Qwen from a second full run (`results/rerun1/`, `hallubench compare-runs`).
- Cost per 1,000 answers from the APIs' own usage reports (AWS's billed units at list price, Google's list price), and latency measured one request at a time (`run --workers 1` into `results/latency/`): over the network from Italy for the API checkers, on a Colab A100 for the open ones. The two cloud checks were timed only inside the main parallel run.

## Rebuilding the numbers

The scores every checker gave every row are attached to the [v1.0 release](https://github.com/mauroforlin/rag-hallu-bench/releases/tag/v1.0) as `scores.zip`. With them, every number in the article can be rebuilt without calling any model:

```bash
pip install -e .
hallubench reproduce          # download scores.zip, rebuild the rows, grade, compare with docs/data.json
```

It ends by saying whether every number in `docs/data.json` came out the same. NumPy doesn't promise the same random streams across versions, so for an exact match install the analysis libraries at the versions in `requirements-lock.txt` (`pip install numpy==2.5.4 scipy==1.18.1 scikit-learn==1.9.1`), as [`notebooks/reproduce.ipynb`](notebooks/reproduce.ipynb) does on a free Colab runtime.

The article is `docs/article.src.html` plus `docs/data.json`. `hallubench article` regenerates `data.json` from `results/report/`, and `python docs/build.py` builds `index.html` for GitHub Pages.

## Rerunning the checkers

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -e ".[jev,local,cloud]" --extra-index-url https://pypi.typesafe.ai/simple/
cp .env.example .env                                  # keys, see below

hallubench build-data                                 # check the printed sha256
hallubench list
hallubench run jev-choice --limit 20                  # smoke test
hallubench run --where api                            # Jev, LLM judges, cloud checks, precomputed
hallubench evaluate                                   # table + results/report/metrics.json
hallubench article                                    # docs/data.json
```

`.env` takes `OPENROUTER_API_KEY` (LLM judges) and `TYPESAFE_KEY` (Jev); for the cloud checks, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_REGION` (an IAM user with Bedrock access; the guardrail is created on first use), and `GCP_PROJECT` with a service-account key in `gcp-key.json` (role Discovery Engine Editor, Discovery Engine API enabled). The AWS check cost about $2.84 on all rows and Google's well under a cent; the API checkers cost from $0.07 (Jev) to $2.07 (Gemini 3.8 Flash with reasoning) per 1,000 answers.

Runs can be resumed and append to `results/raw/<detector>.jsonl`, one line per row. Useful flags are `--core` (core slice only), `--datasets ...`, `--workers N` (concurrency for API checkers, batch size for local ones) and `--only-errors` (retry failed rows without extending a run). Setting `HALLUBENCH_RAW` to another folder keeps smoke tests apart from real results. The API models and both cloud checks can change behind the same name, so a rerun may not match the published scores exactly.

The open models run in [`notebooks/colab_gpu.ipynb`](notebooks/colab_gpu.ipynb) on an A100. *Run all* skips every checker already complete on Drive, tries each missing one on 20 rows and stops if those look wrong, then runs it in full; copy the `.jsonl` files it lists into `results/raw/` and run `hallubench evaluate` again, since runs merge by row ID.

## Licences

The code is MIT. The datasets keep their own licences: RAGBench is CC-BY-4.0, HaluBench CC-BY-NC-2.0 and ExpertQA MIT, while MedHallu declares none on its Hub card. No data is committed, `build-data` downloads it from the sources. The released scores hold ids, scores and the checkers' own outputs, and the generative judges' outputs can quote passages from the datasets, so the datasets' licences apply to those too. Among the models, HHEM-2.1-Open, Granite Guardian and Laya are Apache-2.0, AlignScore and LettuceDetect are MIT, Lynx and Bespoke-MiniCheck-7B are CC-BY-NC-4.0 and BioDecision v2 is research-only. They are used here for research.
