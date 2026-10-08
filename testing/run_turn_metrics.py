"""
run_turn_metrics.py - run the turn-level metrics of judge_service_v2 over every
topic in input_topics/ and save each metric's input and output.

Metrics (each called through the running judge API):
    stance support   POST /score/stance-support    topic, turn, stance, claim, hypothesis, model
    logical cogency  POST /score/logical-cogency   claim, premises, model
    consistency      POST /score/consistency       claim, prior_claims (same speaker, earlier turns)

One output file per topic: outputs_qwen2.5-7b/<topic-file-name>.results.json
The file is rewritten after every turn, so a crash keeps what was done so far.

Usage (API must be running, see judge_service_v2/readme.md):
    python run_turn_metrics.py
    python run_turn_metrics.py --only grudge          # one topic
    python run_turn_metrics.py --dry-run              # build inputs only, no API calls
    python run_turn_metrics.py --metrics logical_cogency --model gemma4:26b --out-dir outputs_gemma4-26b_cogency
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
TIMEOUT = {"stance_support": 120, "logical_cogency": 330, "consistency": 120}
ENDPOINT = {"stance_support": "/score/stance-support",
            "logical_cogency": "/score/logical-cogency",
            "consistency": "/score/consistency"}


# -- reading the topic files (two annotation layouts) ---------------------------

def _premise_text(p) -> str:
    return (p.get("text", "") if isinstance(p, dict) else str(p)).strip()


def _arguments(turn: dict) -> list:
    """Claim + premises of a turn. Pavan's layout keeps them on the turn; the
    others keep a list under `arguments`."""
    if "arguments" in turn:
        args = turn["arguments"]
    else:
        args = [{"claim": turn.get("claim"), "premises": turn.get("premises")}]
    out = []
    for i, a in enumerate(args, 1):
        out.append({
            "argument_id": a.get("id") or (turn["id"] if len(args) == 1 else f"{turn['id']}-{i:02d}"),
            "claim": (a.get("claim") or "").strip(),
            "premises": [t for t in map(_premise_text, a.get("premises") or []) if t],
        })
    return out


def build_inputs(data: dict, model: str) -> list:
    """Every metric request for one topic, in turn order."""
    topic = data["topic"]
    hyp = {sp: info.get("hypothesis") for sp, info in data["speakers"].items()}
    prior: dict = {}
    rows = []
    for turn in data["turns"]:
        sp = turn["speaker"]
        for arg in _arguments(turn):
            rows.append({
                "turn_id": turn["id"],
                "argument_id": arg["argument_id"],
                "round": turn.get("round"),
                "speaker": sp,
                "stance": turn["stance"].upper(),
                "stance_support": {"input": {
                    "topic": topic, "turn": turn["text"],
                    "stance": turn["stance"].upper(), "claim": arg["claim"],
                    "hypothesis": hyp.get(sp), "model": model}},
                "logical_cogency": {"input": {
                    "claim": arg["claim"], "premises": arg["premises"],
                    "model": model}},
                "consistency": {"input": {
                    "claim": arg["claim"],
                    "prior_claims": list(prior.get(sp, []))}},
            })
            prior.setdefault(sp, []).append(
                {"claim": arg["claim"], "turn_no": turn["id"]})
    return rows


# -- calling the API --------------------------------------------------------------

def call(api: str, metric: str, payload: dict) -> tuple:
    """Returns (response body, console status). Only the body is saved; on a
    failed call the body is the API's error detail."""
    t0 = time.time()
    try:
        r = requests.post(api + ENDPOINT[metric], json=payload,
                          timeout=TIMEOUT[metric])
        body = r.json() if r.headers.get("content-type", "").startswith(
            "application/json") else r.text
        status = "ok" if r.status_code == 200 else f"error {r.status_code}"
    except requests.RequestException as e:
        body, status = {"detail": str(e)}, "error"
    return body, f"{status}({round(time.time() - t0, 2)}s)"


def check_health(api: str) -> None:
    try:
        h = requests.get(api + "/health/all", timeout=10).json()
    except requests.RequestException as e:
        sys.exit(f"Judge API not reachable at {api}: {e}")
    ollama = h.get("ollama", {})
    if not ollama.get("reachable"):
        sys.exit(f"Ollama not reachable ({ollama.get('host')}): {ollama.get('error')}")
    print(f"API ok | Ollama ok at {ollama.get('host')} | models: {ollama.get('models')}")


# -- main ---------------------------------------------------------------------------

def run_topic(path: Path, out_dir: Path, api: str, model: str, dry: bool,
              metrics: list) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = build_inputs(data, model)
    for row in rows:                      # keep only the selected metrics
        for m in ENDPOINT:
            if m not in metrics:
                del row[m]
    out = out_dir / (path.name.removesuffix(".json") + ".results.json")
    result = {
        "topic": data["topic"],
        "source_file": path.name,
        "api": api,
        "model": model,
        "hypotheses": {sp: {"stance": i.get("stance"), "hypothesis": i.get("hypothesis")}
                       for sp, i in data["speakers"].items()},
        "turns": rows,
    }
    print(f"\n== {data['topic']}  ({len(rows)} turns)")
    for row in rows:
        line = [f"  {row['argument_id']:<10}"]
        for metric in metrics:
            if dry:
                continue
            row[metric]["output"], status = call(api, metric, row[metric]["input"])
            line.append(f"{metric}={status}")
        print(" ".join(line))
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  saved -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://localhost:8500")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--topics-dir", default=str(HERE / "input_topics"))
    ap.add_argument("--out-dir", default=str(HERE / "outputs_qwen2.5-7b"))
    ap.add_argument("--only", help="run only topic files whose name contains this text")
    ap.add_argument("--metrics", nargs="+", choices=list(ENDPOINT), default=list(ENDPOINT),
                    help="which metrics to run (default: all three)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and save the inputs without calling the API")
    args = ap.parse_args()

    api = args.api.rstrip("/")
    files = sorted(Path(args.topics_dir).glob("*.json"))
    if args.only:
        files = [f for f in files if args.only.lower() in f.name.lower()]
    if not files:
        sys.exit("No topic files found.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        check_health(api)
    for f in files:
        run_topic(f, out_dir, api, args.model, args.dry_run, args.metrics)


if __name__ == "__main__":
    main()
