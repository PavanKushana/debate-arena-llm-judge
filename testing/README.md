# Judge v2 — Turn-Level Metrics Testing

This folder contains the testing of the three **turn-level metrics** of judge v2:

- **Stance Support**: does the turn argue for the speaker's side?
- **Logical Cogency**: are the premises acceptable, relevant and sufficient for the claim?
- **Consistency**: does the speaker contradict their own earlier turns?

**Test data:** 5 annotated debates × 6 turns = **30 turns** (15 PRO, 15 CON).
**Results:** see [`turn_level_metrics_summary.md`](turn_level_metrics_summary.md).
**Judge code tested:** [`presentations/final_debate_judge/judge_service_v2`](../../final_debate_judge/judge_service_v2) (commit `dd5232c`, 27 Sep 2026).

---

## 1. How the testing was done

### Step 1: Choose the test debates
Five debates that were already annotated by five different team members:

| Topic | Annotator |
|---|---|
| Governments should give citizens free money to end poverty | Rosen |
| Growing up as an only child does more harm than good | Immanuvel |
| Holding a grudge is healthier than always forgiving and forgetting | Sneha |
| Space exploration is worth the cost even when there are major problems on Earth | Manoj |
| Working from home is better than working in an office | Pavan |

Each debate has two speakers (**a** = PRO, **b** = CON) and three rounds each.

### Step 2: Prepare the inputs for each metric
Each metric was given exactly the inputs it expects, taken from the annotated files:

| Metric | API endpoint | Inputs |
|---|---|---|
| Stance Support | `POST /score/stance-support` | topic, turn text, stance, claim, side hypothesis |
| Logical Cogency | `POST /score/logical-cogency` | claim, premises |
| Consistency | `POST /score/consistency` | claim, all earlier claims of the same speaker |

- **Claims and premises** are the human-extracted ones from the annotation files.
- **Side hypotheses:** stance support needs one sentence per side. Four files had none, so I added them (the WFH file already had them).
  - PRO = the motion, e.g. *"Working from home is better than working in an office."*
  - CON = the **denial** of the motion, e.g. *"Working from home is **not** better than working in an office."* CON only has to reject the motion, not prove the opposite.

### Step 3: Run the metrics
- Every turn was scored by all three metrics through the judge v2 API.
- Logical cogency uses an LLM, so I checked with **two models**: `qwen2.5:7b` and `gemma4:26b`.
- Stance support and consistency use the DeBERTa-v3 NLI model.
- For every call, the **input and the raw output** were saved. No judge code, prompt or setting was changed.

### Step 4: Review every output
Each output was compared with **the metric's own definition** (for cogency, with the judge's scoring rubric in `judge_service_v2/prompts/`) and marked:

> ✅ Correct · ⚠️ Partly correct · ❌ Wrong

Examples of the rules used:
- **Stance Support:** a turn that clearly argues for its side should score HIGH or MEDIUM; a turn that leaves its side should score LOW.
- **Logical Cogency:** each score (1–3) and its written reason should fit the rubric; missing reasons count as a fault.
- **Consistency:** a turn that keeps the same position should not be flagged; a real change of position should be.

The counts and examples are in the [summary](turn_level_metrics_summary.md).

---

## 2. What is in this folder

```
last_week_pavan/
├── README.md                          ← this file
├── turn_level_metrics_summary.md      ← results, examples and suggested fixes
├── input_topics/                      ← the 5 debates used as test input
├── outputs_qwen2.5-7b/                ← outputs of all 3 metrics (cogency with qwen2.5:7b)
├── outputs_gemma4-26b_cogency/        ← outputs of logical cogency with gemma4:26b
└── run_turn_metrics.py                ← automation script (see section 4)
```

Each folder has **one file per topic**, named after the topic, so the same topic can be found in all three folders.

---

## 3. How to read the files

### `input_topics/`: the test input
The annotated debate files, plus the side hypotheses.

| Field | Meaning |
|---|---|
| `topic` | the motion of the debate |
| `speakers` | `a` and `b`, each with `stance` (pro / con) and `hypothesis` (the side sentence used by stance support) |
| `turns` | the 6 turns; each has `id`, `round`, `speaker`, `stance`, `text` |
| `claim`, `premises` | the human-extracted argument of the turn (in some files inside `arguments`) |
| other fields (`winner`, scores, …) | the annotator's own scores; they are kept as they were and were **not** sent to the judge |

**Turn IDs:** `a-r2` = speaker **a** (PRO), round **2**. `b-r3` = speaker **b** (CON), round **3**.

### `outputs_qwen2.5-7b/`: results for all 3 metrics
One file per topic, e.g. `working-from-home-...results.json`:

```
{
  "topic": "...",
  "source_file": "<file in input_topics/>",
  "model": "qwen2.5:7b",
  "hypotheses": { "a": {...}, "b": {...} },
  "turns": [
    {
      "turn_id": "a-r2", "speaker": "a", "stance": "PRO",
      "stance_support":  { "input": {...}, "output": {...} },
      "logical_cogency": { "input": {...}, "output": {...} },
      "consistency":     { "input": {...}, "output": {...} }
    },
    ...
  ]
}
```

Every metric shows **what was sent** (`input`) and **what the judge answered** (`output`), unchanged.

**What to look at in each output:**

| Metric | Key fields | How to read them |
|---|---|---|
| Stance Support | `stance_support_score`, `label`, `probs` | score = entailment probability; `label` is HIGH (≥ 0.60), MEDIUM (≥ 0.35) or LOW; `probs` shows entailment / neutral / contradiction |
| Logical Cogency | `local_acceptability`, `local_relevance`, `local_sufficiency` + `…_rationale` | each 1–3, with the model's written reason |
| Consistency | `consistency`, `most_contradicted` | 1.0 = consistent, below 0.5 = contradicting; `most_contradicted` is the earlier claim that clashes most; `null` for a speaker's first turn |

**Example:** WFH, turn a-r2 (PRO), stance support:
```
"stance_support_score": 0.0523, "label": "LOW",
"probs": {"entailment": 0.0523, "neutral": 0.9448, "contradiction": 0.0029}
```
The turn clearly argues for working from home, but NLI reads it as *neutral*, so it gets LOW. This is ❌ in the summary.

### `outputs_gemma4-26b_cogency/`: cogency with a second model
Same layout, but only `logical_cogency` is included, with `"model": "gemma4:26b"`. Compare it turn by turn with the same file in `outputs_qwen2.5-7b/` to see how the two models differ.

---

## 4. Automation script: `run_turn_metrics.py`

A small script was also created that repeats the whole test in one command, so it can be run again (for example after the judge changes) with exactly the same inputs.

**What it does:**
1. Reads every topic file in `input_topics/`.
2. Builds the inputs for each metric, as in step 2 above. For consistency, it collects the earlier claims of the same speaker.
3. Sends each input to the running judge API.
4. Saves the input and the raw output per topic in the output folder. The file is saved after every turn, so nothing is lost if a call fails.

**How to run it:**

1. Start the judge (from `presentations/final_debate_judge`):
   ```
   uvicorn judge_service_v2.api:app --port 8500
   ```
2. Run the test (from this folder, in a second terminal):
   ```
   python run_turn_metrics.py
   ```

**Options:**

| Command | What it does |
|---|---|
| `python run_turn_metrics.py` | all topics, all 3 metrics, qwen2.5:7b → `outputs_qwen2.5-7b/` |
| `python run_turn_metrics.py --only grudge` | only topics whose file name contains "grudge" |
| `python run_turn_metrics.py --metrics logical_cogency --model gemma4:26b --out-dir outputs_gemma4-26b_cogency` | cogency with gemma |
| `python run_turn_metrics.py --dry-run --out-dir <folder>` | only builds and saves the inputs, no API calls |

**Note:** running the script again **overwrites** the files in the output folder. Use `--out-dir` with a new folder to keep the original results.

**Requirements:** Python 3.9+, `requests`, the judge API running, and access to the Ollama server set in `judge_service_v2/utils/judge_config.py`.
