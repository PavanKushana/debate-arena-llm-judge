# Debate Judge — Metric Tester

A small web tool for exercising every scoring metric in `judge_service_v2`
individually. Each metric is exposed as its own HTTP endpoint; the tester
(`tester.html`) lets you edit the request body, pick the LLM model, send the
call, and read the response — with live health indicators for both the API and
Ollama.

Two files provide it:

- **`api.py`** — a FastAPI app that wraps each metric as a `POST` endpoint and
  serves the tester UI at `/`.
- **`tester.html`** — the browser UI (self-contained vanilla HTML/JS, no build
  step). Served by `api.py`, so it is same-origin and needs no configuration.

> `main.py` in this package is an older service file and does **not** run
> against the current modules. Use `api.py` for testing.

---

## 1. Prerequisites

- **Python 3.11+** (tested on 3.13).
- **Ollama** reachable for the LLM-backed metrics (optional — non-LLM metrics
  work without it). Default host is set in `utils/judge_config.py` and is
  overridable with the `OLLAMA_HOST` environment variable.
- The NLI-backed metrics load a DeBERTa model on first use via
  `sentence-transformers` / `torch` — the **first such call is slow**; the app
  itself starts without them.

---

## 2. Install

Run from the folder that **contains** `judge_service_v2/`.

```powershell
# from inside judge_service_v2/
python -m pip install -r requirements.txt
```

Minimum set just to bring the tester up (add `sentence-transformers` before you
call any NLI metric):

```powershell
python -m pip install fastapi "uvicorn[standard]" pydantic requests numpy
```

If `torch` fails to install on a very new Python, install a CPU build first:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

---

## 3. Run

Run from the directory **one level above** `judge_service_v2/` (the folder that
contains the package):

```powershell
uvicorn judge_service_v2.api:app --host 0.0.0.0 --port 8500 --reload
```

The `uvicorn` target reads as `package.module:variable`:
`judge_service_v2` (folder) · `api` (`api.py`) · `app` (the FastAPI object).
It is imported relative to your current working directory, so that directory
must contain the `judge_service_v2` folder — not be inside it.

To run without changing directory:

```powershell
uvicorn judge_service_v2.api:app --app-dir .. --host 0.0.0.0 --port 8500 --reload
```

Then open:

```
http://localhost:8500/
```

---

## 4. Using the tester

- **Pick a metric** in the left sidebar (grouped Turn level / Debate level /
  Utilities). The request panel pre-fills with a working example.
- **Edit the JSON** request body as needed.
- **Send request** (button, or `Ctrl/Cmd + Enter`). The response panel shows the
  JSON with the HTTP status and round-trip time.
- **LLM dropdown** (top bar): choose the model used for every LLM call. The
  selection is injected as `"model"` into LLM requests and persists across
  reloads.
- **Health dots** (top bar): `api` = judge service reachable; `ollama` = Ollama
  reachable, with the count of models it has loaded. If the model you selected
  is not among Ollama's loaded models, a warning appears under the request panel.
- **API field** (top bar): the base URL the UI calls (defaults to the page
  origin). Change it to test a remote instance.

---

## 5. Endpoints

Base: `http://localhost:8500`

### Health

| Method | Path             | Purpose                                             |
|--------|------------------|-----------------------------------------------------|
| GET    | `/health`        | Judge-API liveness.                                 |
| GET    | `/health/ollama` | Ollama reachability + list of models loaded.        |
| GET    | `/health/all`    | Both services in one call.                           |
| GET    | `/`              | Serves the tester UI (`tester.html`).               |

### Turn-level metrics

| Method | Path                          | Key fields |
|--------|-------------------------------|------------|
| POST   | `/score/stance-support`       | `topic`, `turn`, `stance` (PRO/CON), `interpret` |
| POST   | `/score/stance-support-dual`  | `topic`, `turn`, `stance`, `pro_hypothesis`, `con_hypothesis` |
| POST   | `/score/logical-cogency`      | `claim`, `premises[]`, `use_llm_sufficiency`, `prior_claims`, `model` |
| POST   | `/score/consistency`          | `claim`, `prior_claims[]` |
| POST   | `/score/reasoning-defects`    | `claim`, `reasons`, `model` — **LLM** |
| POST   | `/score/delta-votes`          | `es_results[]` (Elasticsearch granular-similarity results) |

### Debate-level metrics

| Method | Path                             | Key fields |
|--------|----------------------------------|------------|
| POST   | `/score/rebuttal-effectiveness`  | `opponent_claim`, `opponent_premises[]`, `rebuttal_text`, `rebuttal_claim`, `rebuttal_premises[]`, `use_llm_attack_type`, `model` — scores ONE (rebuttal, target) pair |
| POST   | `/score/rebuttal-debate`         | `turns[]` (each: `speaker`, `turn_no`, `claim`, `premises[]`, `text`, `cogency`, `replies_to`), `exclude_opener_targets`, `lookback`, `use_llm_attack_type`, `model` — non-adjacent rebuttal pass over a whole debate |
| POST   | `/score/information-density`     | `turns[]`, `topic`, `speaker`, `method` (`both`/`nli`/`llm`, default `both`), `model` |
| POST   | `/score/argument-survival`       | `turns[]` (each: `speaker`, `claim`, `premises[]`, `text`, `cogency`, `replies_to`), `use_llm_attack_type`, `model` |

### LLM utilities

| Method | Path                      | Key fields |
|--------|---------------------------|------------|
| POST   | `/util/topic-hypotheses`  | `topic`, `use_cache`, `model` — **LLM** |
| POST   | `/util/mine-argument`     | `argument`, `model` — **LLM** |

Endpoints marked **LLM**, and any metric where you set `use_llm_sufficiency`,
`use_llm_attack_type`, or `method: "llm"`, call Ollama. Interactive docs are
also available at `http://localhost:8500/docs`.

---

## 6. Quick checks with curl

Health:

```bash
curl http://localhost:8500/health
curl http://localhost:8500/health/ollama
```

A deterministic metric (no LLM, but loads the NLI model on first call):

```bash
curl -X POST http://localhost:8500/score/stance-support \
  -H "Content-Type: application/json" \
  -d '{"topic":"Money can buy happiness","turn":"Financial security reduces anxiety.","stance":"PRO"}'
```

An LLM metric with an explicit model:

```bash
curl -X POST http://localhost:8500/util/topic-hypotheses \
  -H "Content-Type: application/json" \
  -d '{"topic":"Money can buy happiness","model":"gemma4:26b"}'
```

---

## 7. Configuration (environment variables)

Set before launching `uvicorn` (or via a `.env` in the package folder, loaded by
`utils/judge_config.py`).

| Variable        | Default (see `judge_config.py`)                      | Effect |
|-----------------|------------------------------------------------------|--------|
| `OLLAMA_HOST`   | `http://localhost:11434`                              | Ollama base URL for all LLM calls and `/health/ollama`. |
| `OLLAMA_MODEL`  | `qwen2.5:7b`                                          | Default model when a request omits `model`. The tester dropdown overrides this per call. |
| `ES_API`        | (see config)                                         | Elasticsearch base URL used by `/score/delta-votes`. |

The models offered in the tester dropdown are defined near the top of the
`<script>` block in `tester.html` (the `MODELS` array). Edit that list to match
exactly the tags your Ollama has (`ollama list`), since the dropdown value is
sent verbatim.

---

## 8. Troubleshooting

- **`ModuleNotFoundError: No module named 'judge_service_v2'`** — you launched
  from inside the package folder. Run from the parent directory, or use
  `--app-dir ..`.
- **`No module named 'judge_api'`** — you used the old target. Use
  `judge_service_v2.api:app`.
- **A metric returns HTTP 400** — the scorer raised (e.g. an invalid `stance`,
  or Ollama unreachable for an LLM metric). The response `detail` names the
  cause; check the `ollama` health dot.
- **Validation error (HTTP 422)** — a required field is missing or mistyped in
  the request body.
- **First NLI call is very slow / seems to hang** — the DeBERTa model is
  downloading/loading. Subsequent calls are fast.
- **LLM warning "model not in Ollama's loaded models"** — the dropdown value
  does not match a tag Ollama has. Compare against the models shown on the
  `ollama` health chip and `ollama pull` it, or edit the `MODELS` array.
