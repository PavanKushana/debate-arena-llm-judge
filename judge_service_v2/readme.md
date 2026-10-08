# Judge Service v2

A FastAPI-based debate evaluation service. It scores a debate using several independent metrics and, for a two-speaker debate, determines a winner using a **deterministic two-level decision rule**.

The project is designed so that each metric can be tested separately, while `/score/debate` runs the complete evaluation in one request.

---

## 1. Quick Start

Follow these steps in order before testing the project.

### Step 1 — Start Ollama

The project uses Ollama for the LLM-based parts of the evaluation. First, make sure Ollama is running with the required model (locally: install from https://ollama.com and run `ollama pull qwen2.5:7b`; the default address is `http://localhost:11434`).

Before starting the Python API, verify that the Ollama service is available from the machine where you will run `judge_service_v2`.

You can check it later through:

```text
GET /health/ollama
```

### Step 2 — Start the Python API

Run this command from the directory that **contains `judge_service_v2/`**:

```bash
uvicorn judge_service_v2.api:app --host 0.0.0.0 --port 8500 --reload
```

The API will then be available at:

```text
http://localhost:8500
```

### Step 3 — Test the APIs using the HTML tester

You can test the available APIs directly through the project's `tester.html` file.

You can either open the tester through the running API:

```text
http://localhost:8500/
```

or open the `tester.html` file directly in a browser if that is more convenient.

The tester provides the currently used metric/API screens, including:

- Whole Debate
- Stance Support
- Logical Cogency
- Consistency
- Rebuttal
- Information Density
- Topic Hypotheses
- Argument Mining

### Step 4 — Change the port or Ollama address if needed

If you need to use a **different API port** or change the **address of the Ollama server** (`OLLAMA_HOST`), update the project's configuration file:

```text
judge_service_v2/utils/judge_config.py
```

Change the relevant configuration values there instead of changing them in individual metric files. After changing the configuration, restart the Python API so that the new settings are loaded.

For example, the API port is also reflected in the `uvicorn` command:

```bash
uvicorn judge_service_v2.api:app --host 0.0.0.0 --port <YOUR_PORT> --reload
```

If the tester is opened directly as an HTML file, make sure its API base URL also points to the correct API host/port.

### Health checks

Use these endpoints when troubleshooting:

```text
GET /health
GET /health/ollama
GET /health/all
```

- `/health` checks whether the FastAPI service is running.
- `/health/ollama` checks whether Ollama is reachable and shows the available models.
- `/health/all` checks both services together.

### Quick startup summary

```text
1. Start Ollama (locally or on a server)
        ↓
2. Check the Ollama configuration in judge_config.py
        ↓
3. Start the Python FastAPI service
        ↓
4. Open tester.html
        ↓
5. Select a metric and send the request
```

## 2. Project Structure

Only the main files are listed here. Helper and cache files are intentionally omitted to keep the structure easy to understand.

```text
judge_service_v2/
│
├── api.py
├── orchestrator.py
├── tester.html
│
├── turn_level_scoring/
│   ├── stance_support_score.py
│   ├── logical_cogency_score.py
│   ├── consistency_score.py
│   └── delta_votes_score.py
│
├── debate_level_scoring/
│   ├── rebuttal_effectiveness_dl.py
│   ├── information_density_dl.py
│   ├── debate_winner.py
│   └── argument_survival_dl.py          # optional / future use
│
└── utils/
    ├── argument_miner.py
    ├── topic_hypotheses.py
    ├── judge_config.py
    ├── telemetry.py
    └── logging_config.py
```

### Main file responsibilities

| File | Purpose |
|---|---|
| `api.py` | Defines the FastAPI endpoints and request/response models. It connects HTTP requests to the scoring functions. |
| `orchestrator.py` | Runs the complete debate evaluation. It prepares claims/premises, runs the metrics, creates per-speaker results, and calls the deterministic winner engine. |
| `tester.html` | Browser-based interface for testing the API. It currently exposes Whole Debate, Stance Support, Logical Cogency, Consistency, Rebuttal, Information Density, Topic Hypotheses, and Argument Mining. |
| `turn_level_scoring/stance_support_score.py` | Measures whether a speaker's turn supports the side they are arguing for. |
| `turn_level_scoring/logical_cogency_score.py` | Scores acceptability, relevance, and sufficiency separately. |
| `turn_level_scoring/consistency_score.py` | Checks whether a new claim is consistent with the speaker's previous claims. |
| `turn_level_scoring/delta_votes_score.py` | Processes Delta/Votes information when external retrieval results are supplied. It is used by the whole-debate pipeline but is not a separate tester screen at present. |
| `debate_level_scoring/rebuttal_effectiveness_dl.py` | Finds rebuttal links between turns and scores their effectiveness. |
| `debate_level_scoring/information_density_dl.py` | Measures how much new, distinct argument information a speaker contributes. |
| `debate_level_scoring/debate_winner.py` | Deterministic winner logic. It compares the selected metrics without a weighted total score. |
| `utils/argument_miner.py` | Extracts a claim and premises from a turn when they are not already supplied. |
| `utils/topic_hypotheses.py` | Creates the PRO and CON hypotheses used by stance support. |

---

## 3. How a Whole Debate Is Processed

The normal workflow is:

```text
Debate topic + turns
        │
        ▼
Resolve speaker stances
        │
        ▼
Generate PRO/CON topic hypotheses
        │
        ▼
Mine claim + premises for each turn
        │
        ├──► Stance Support
        ├──► Logical Cogency
        ├──► Consistency
        └──► Delta / Votes (when retrieval data is supplied)
        │
        ▼
Rebuttal analysis
        │
        ▼
Information Density
        │
        ▼
Per-speaker metric profile
        │
        ▼
Deterministic Winner Engine
        │
        ├── Level 1: Cogency + Rebuttal
        └── Level 2: Stance Support + Consistency + Information Density
```

Claims and premises are mined once per turn and then reused by the relevant metrics. Topic hypotheses are generated once per topic and reused for stance support.

---

# 4. Metrics

## 4.1 Stance Support

**Question:** Does the speaker's turn support the position they are supposed to defend?

**Scale:** `0–1`

A higher value means the turn provides stronger support for the assigned stance.

### Example request

```http
POST /score/stance-support
Content-Type: application/json
```

```json
{
  "topic": "Schools should teach social media literacy.",
  "turn": "Students need to understand misinformation and online privacy.",
  "stance": "PRO"
}
```

### Example result

```json
{
  "stance_support_score": 0.82
}
```

The exact response can contain additional fields depending on the scoring implementation.

### Batch version

```text
POST /score/stance-support-batch
```

This scores multiple turns and also returns a per-speaker aggregation.

---

## 4.2 Logical Cogency

**Question:** How logically strong is the relationship between the claim and its premises?

Logical cogency is kept as **three separate dimensions**:

- **Acceptability** — are the premises acceptable/reasonable?
- **Relevance** — are the premises relevant to the claim?
- **Sufficiency** — do the premises provide enough support for the claim?

**Scale:** `1–3` for each dimension.

These values are **not averaged into one cogency score**.

### Example request

```text
POST /score/logical-cogency
```

```json
{
  "claim": "Schools should teach social media literacy.",
  "premises": [
    "Students regularly encounter misleading information online.",
    "Understanding how to evaluate online information can reduce misinformation."]
}
```

### Example result

```json
{
  "local_acceptability": 3,
  "local_relevance": 3,
  "local_sufficiency": 2
}
```

For the whole debate, the three dimensions are averaged **separately per speaker**. There is no single combined cogency number.

---

## 4.3 Consistency

**Question:** Does the speaker's current claim contradict their earlier claims?

**Scale:** `0–1`.

A higher value indicates stronger consistency with the speaker's previous claims.

### Example request

```text
POST /score/consistency
```

```json
{
  "claim": "Social media literacy should be taught in schools.",
  "prior_claims": [
    {"claim": "Schools should help students understand online information."}
  ]
}
```

### Example result

```json
{
  "consistency": 0.91
}
```

The whole-debate pipeline keeps previous claims separately for each speaker.

---

## 4.4 Rebuttal Effectiveness

**Question:** Does a speaker effectively respond to another speaker's argument?

The rebuttal module works at sentence/argument level. It identifies links between a rebutting sentence and a target from another turn.

Current attack types include:

- `undermining`
- `rebutting`
- `undercutting`

**Effectiveness scale:** `1–3`

- `1` = low
- `2` = moderate
- `3` = high

NLI is used as a candidate/screening signal. The NLI value itself is **not** the final rebuttal effectiveness score.

### Example request

```text
POST /score/rebuttal-debate
```

```json
{
  "turns": [
    {
      "speaker": "Alice",
      "turn_no": 1,
      "text": "Social media literacy helps students identify misinformation."
    },
    {
      "speaker": "Bob",
      "turn_no": 2,
      "text": "That does not solve the problem because students may still trust unreliable sources."
    }
  ],
  "use_llm": true
}
```

### Example result

```json
{
  "rebuttals": [
    {
      "from": {"turn": 2, "speaker": "Bob"},
      "target": {"turn": 1, "speaker": "Alice"},
      "attack_type": "undercutting",
      "effectiveness": 2
    }
  ],
  "by_speaker": {
    "Alice": {"mean_effectiveness": 0.0},
    "Bob": {"mean_effectiveness": 2.0}
  }
}
```

The exact fields can contain additional information such as the sentence, target text, rationale, and NLI screening value.

---

## 4.5 Information Density

**Question:** How much distinct, on-topic argument information does a speaker contribute instead of repeating the same points?

The metric focuses on **distinct points relative to the speaker's total statements**. Similar or repeated points can be grouped together.

**Main score:** `0–1`.

### Example

Suppose a speaker makes 5 statements and 4 of them represent distinct on-topic points:

```text
4 distinct points / 5 statements = 0.80
```

### API

```text
POST /score/information-density
```

```json
{
  "topic": "Schools should teach social media literacy.",
  "turns": [
    "Students need to identify misinformation.",
    "They should learn how to check sources.",
    "Source checking reduces misinformation.",
    "Students also need to understand online privacy."
  ],
  "speaker": "Alice"
}
```

### Example result

```json
{
  "information_density": 0.75
}
```

The exact response may also contain supporting counts or clustering information.

---

## 4.6 Delta / Votes

Delta/Votes is available inside the whole-debate pipeline when Elasticsearch/retrieval results are supplied for turns.

It is **not currently a separate tester screen** and is not part of the current winner hierarchy.

API:

```text
POST /score/delta-votes
```

Input is a list of retrieval results:

```json
{
  "es_results": []
}
```

This metric is kept mainly for the broader evaluation pipeline and can be extended later.

---

# 5. Winner Determination

The whole-debate endpoint returns a deterministic `winner_determination` object for a debate with exactly two speakers.

The winner logic does **not** calculate a weighted total score and does **not** average different metrics together.

## Level 1 — Core Debate Performance

The first comparison uses:

1. Acceptability
2. Relevance
3. Sufficiency
4. Rebuttal Effectiveness

The first three are the separate dimensions of Logical Cogency.

A speaker dominates Level 1 when they:

- are not meaningfully worse on any Level-1 metric, and
- are better on at least one Level-1 metric.

If one speaker dominates, that speaker is selected and Level 2 is not needed.

## Level 2 — Supporting Argument Quality

Level 2 is used only when Level 1 does not distinguish the speakers.

It compares:

1. Stance Support
2. Consistency
3. Information Density

For these `0–1` metrics, the current meaningful-difference threshold is:

```text
0.10
```

These thresholds are initial calibration parameters and can be validated against human-judged debates later.

### Example

```text
Alice
Acceptability: 3
Relevance:     3
Sufficiency:   2
Rebuttal:      2

Bob
Acceptability: 3
Relevance:     3
Sufficiency:   3
Rebuttal:      3
```

Bob is equal on acceptability and relevance and better on sufficiency and rebuttal. Therefore Bob dominates Level 1.

If Level 1 is tied/unresolved, the system moves to Level 2.

If neither speaker dominates Level 2, the result is:

```json
{
  "winner": null,
  "level": "Tie"
}
```

### Winner response example

```json
{
  "winner": "Bob",
  "level": "Level 1",
  "reason": "Bob dominates Alice at Level 1: equal or better on every core metric and better on sufficiency, rebuttal_effectiveness.",
  "level_1_comparison": {
    "equal_metrics": ["acceptability", "relevance"],
    "first_speaker_better_metrics": [],
    "second_speaker_better_metrics": ["sufficiency", "rebuttal_effectiveness"],
    "missing_metrics": [],
    "first_speaker_dominates": false,
    "second_speaker_dominates": true
  }
}
```

Missing required values do not automatically produce a winner. The comparison remains unresolved when the necessary metric data is unavailable.

---

# 6. Main API Endpoints

## Health

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check whether the API is running. |
| GET | `/health/ollama` | Check Ollama connection and available models. |
| GET | `/health/all` | Combined API + Ollama health check. |
| GET | `/` | Open the tester UI. |

## Current tester / main scoring APIs

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/score/debate` | Run the complete debate pipeline and return per-speaker metrics plus winner determination. |
| POST | `/score/stance-support` | Score one turn for stance support. |
| POST | `/score/stance-support-batch` | Score multiple turns and aggregate by speaker. |
| POST | `/score/logical-cogency` | Score acceptability, relevance, and sufficiency. |
| POST | `/score/consistency` | Check consistency against previous claims. |
| POST | `/score/rebuttal-debate` | Find and score rebuttal links. |
| POST | `/score/information-density` | Score distinct information contributed by a speaker. |
| POST | `/util/topic-hypotheses` | Generate PRO and CON hypotheses for a topic. |
| POST | `/util/mine-argument` | Extract a claim and premises from an argument. |

## Available but not part of the current tester flow

| Method | Endpoint | Status / use |
|---|---|---|
| POST | `/score/delta-votes` | Used by the broader whole-debate pipeline when retrieval results are provided. |
| POST | `/score/argument-survival` | Optional/future analysis. The current tester does not expose it. |
| POST | `/score/debate/llm-winner` | Separate LLM-as-a-judge endpoint. Not used by the deterministic winner flow. |

---

# 7. Whole Debate API

The main endpoint is:

```text
POST /score/debate
```

### Basic request

```json
{
  "topic": "Schools should be required to teach social media literacy.",
  "turns": [
    {
      "speaker": "Alice",
      "turn_no": 1,
      "text": "Students need to learn how to identify misinformation.",
      "stance": "PRO"
    },
    {
      "speaker": "Bob",
      "turn_no": 2,
      "text": "Teaching this subject may take time away from core subjects.",
      "stance": "CON"
    }
  ],
  "stance_by_speaker": {
    "Alice": "PRO",
    "Bob": "CON"
  },
  "model": "<your-ollama-model>"
}
```

The response contains the main sections:

```text
by_speaker
per_turn
winner_determination
debate_level
```

The `by_speaker` section contains the metrics used to build the winner profile.

The `winner_determination` section contains the final deterministic decision and the comparison details used to reach it.

---

# 8. Utility APIs

### Topic hypotheses

```text
POST /util/topic-hypotheses
```

Example:

```json
{
  "topic": "Schools should teach social media literacy.",
  "use_cache": true
}
```

Example result:

```json
{
  "pro_hypothesis": "Schools should teach social media literacy.",
  "con_hypothesis": "Schools should not be required to teach social media literacy."
}
```

### Argument mining

```text
POST /util/mine-argument
```

Example:

```json
{
  "argument": "Schools should teach media literacy because students regularly encounter misinformation online."
}
```

Example result:

```json
{
  "claim": "Schools should teach media literacy.",
  "premises": [
    "Students regularly encounter misinformation online."
  ]
}
```

These utilities are used internally by the whole-debate pipeline when the required information is not already supplied.

---

# 9. Important Design Notes

### No weighted winner score

The winner is not calculated as:

```text
0.30 × metric A + 0.20 × metric B + ...
```

Instead, the system uses the priority levels and dominance comparison described above.

### No combined logical-cogency score

Acceptability, relevance, and sufficiency remain separate because they answer different questions.

### Winner determination is deterministic

The winner engine uses the already-computed metric values. It does not make another LLM call.

### Metric scales are different by design

- Stance Support: `0–1`
- Consistency: `0–1`
- Information Density: `0–1`
- Logical Cogency dimensions: `1–3`
- Rebuttal Effectiveness: `1–3`

These scales should not be directly averaged together.

### Two speakers are required for the current winner engine

If the debate contains more or fewer than two speakers, the metrics can still be evaluated, but the current winner determination returns an unresolved result with an explanation.

---

# 10. Future Improvements

Possible future work includes:

- Validate the Level-2 `0.10` thresholds against human-judged debates.
- Improve calibration of individual metrics using human annotations.
- Add a dedicated tester view for Delta/Votes if needed.
- Reintroduce Argument Survival as an optional analysis view if it becomes useful.
- Improve error handling and missing-data reporting for incomplete debates.
- Add automated tests for API endpoints and winner-decision edge cases.

These are extensions; they are not required for the current core workflow.

---

## 11. Current Recommended Workflow

For normal use, start the API and open the tester:

```bash
uvicorn judge_service_v2.api:app --host 0.0.0.0 --port 8500 --reload
```

Then open:

```text
http://localhost:8500/
```

For an automated application, use:

```text
POST /score/debate
```

For debugging or testing one metric at a time, use the individual `/score/...` endpoints listed above.
