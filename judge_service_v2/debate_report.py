"""
debate_report.py — write one human-readable Markdown report per debate run.

Called by orchestrator.score_debate after all metrics are computed. It reformats
the result the orchestrator already produced (no re-computation), so it is pure
presentation. One timestamped file per call under REPORT_DIR (default logs/reports/).

The report is FULL-DETAIL: a per-speaker metric table, the rebuttal map (which
sentence rebutted which point), argument-survival outcomes, key flags, AND every
turn's full per-metric response. It is separate from the two logs:
  * judge_api.log            — narrative
  * judge_api_telemetry.jsonl — structured per-call
  * logs/reports/debate_*.md — THIS: one readable summary per debate.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

REPORT_DIR = os.getenv("REPORT_DIR", os.path.join(os.getenv("LOG_DIR", "logs"), "reports"))


def _fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _pct(v):
    return "—" if v is None else f"{round(v*100)}%"


def _speaker_table(by_speaker: dict) -> str:
    metrics = [
        ("Stance support", "stance_support"),
        ("Logical cogency", "logical_cogency"),
        ("Consistency", "consistency"),
        ("Votes", "votes"),
    ]
    speakers = list(by_speaker.keys())
    lines = ["| Metric | " + " | ".join(speakers) + " |",
             "|---|" + "|".join(["---"] * len(speakers)) + "|"]
    for label, key in metrics:
        row = [label] + [_fmt(by_speaker[s].get(key)) for s in speakers]
        lines.append("| " + " | ".join(row) + " |")
    # rebuttal effectiveness + counts (nested dict)
    reb = [(by_speaker[s].get("rebuttal_effectiveness") or {}) for s in speakers]
    lines.append("| Rebuttal effectiveness | " +
                 " | ".join(_fmt(r.get("rebuttal_effectiveness")) for r in reb) + " |")
    lines.append("| Rebuttals made | " +
                 " | ".join(_fmt(r.get("rebuttals_made") or r.get("rebuttals_made", 0), 0) for r in reb) + " |")
    # argument survival
    surv = [(by_speaker[s].get("argument_survival") or {}) for s in speakers]
    lines.append("| Argument survival (mean) | " +
                 " | ".join(_fmt(x.get("argument_survival_score")) for x in surv) + " |")
    lines.append("| Arguments still standing | " +
                 " | ".join(f"{x.get('still_standing','—')}/{x.get('arguments_introduced','—')}" for x in surv) + " |")
    # information density (may be {"nli":..,"llm":..} or flat)
    def id_val(s):
        d = by_speaker[s].get("information_density")
        if not d:
            return "—"
        if "nli" in d or "llm" in d:
            n = (d.get("nli") or {}).get("information_density")
            l = (d.get("llm") or {}).get("information_density")
            return f"nli {_fmt(n)} / llm {_fmt(l)}"
        return _fmt(d.get("information_density"))
    lines.append("| Information density | " + " | ".join(id_val(s) for s in speakers) + " |")
    return "\n".join(lines)


def _rebuttal_map(rebuttal: dict) -> str:
    if not isinstance(rebuttal, dict) or "error" in rebuttal:
        return f"_Rebuttal unavailable: {rebuttal.get('error') if isinstance(rebuttal,dict) else rebuttal}_"
    rbs = rebuttal.get("rebuttals")
    if rbs is None:
        # turn-level shape (score_rebuttal_debate): list of turns with targets
        rows = []
        for t in rebuttal.get("turns", []):
            for tg in t.get("targets", []):
                rows.append(f"- Turn {t['turn_no']} ({t['speaker']}) → Turn {tg['target_turn_no']} "
                            f"[{tg.get('attack_target','?')}] · eff {_fmt(tg.get('rebuttal_effectiveness'))}"
                            + ("  *(primary)*" if tg.get("primary") else ""))
        return "\n".join(rows) if rows else "_No rebuttals detected._"
    # sentence-level shape
    if not rbs:
        return "_No rebuttals detected._"
    rows = []
    for r in rbs:
        rows.append(
            f"- **Turn {r['from']['turn']} ({r['from']['speaker']})** "
            f"— _{r['attack_type']}_ → **Turn {r['target']['turn']} ({r['target']['speaker']}) "
            f"{r['target']['kind']}** · eff {_fmt(r.get('effectiveness'))}\n"
            f"    - sentence: “{r['from']['sentence']}”\n"
            f"    - target: “{r['target']['text']}”"
            + (f"\n    - why: {r['reason']}" if r.get("reason") else ""))
    return "\n".join(rows)


def _survival_section(survival: dict) -> str:
    if not isinstance(survival, dict) or "error" in survival:
        return f"_Survival unavailable._"
    lines = []
    for tid, thr in (survival.get("threads") or {}).items():
        hist = " → ".join(f"{h['event']}:{_fmt(h['survival'],2)}" for h in thr.get("history", []))
        status = "alive" if thr.get("alive") else "**dead**"
        lines.append(f"- **{thr.get('owner')}**, Turn {thr.get('birth_turn')} "
                     f"(“{(thr.get('claim') or '')[:70]}”) — final {_fmt(thr.get('survival'),2)} [{status}]\n"
                     f"    - {hist}")
    return "\n".join(lines) if lines else "_No argument threads._"


def _per_turn_full(per_turn: list) -> str:
    if not per_turn:
        return "_Per-turn detail not included._"
    out = []
    for t in per_turn:
        out.append(f"### Turn {t['turn_no']} · {t['speaker']} ({t.get('stance','?')})")
        out.append(f"**Claim:** {t.get('claim','')}")
        if t.get("premises"):
            out.append("**Premises:** " + "; ".join(t["premises"]))
        # full per-metric responses as fenced JSON
        for label, key in [("Stance support", "stance_support"),
                           ("Logical cogency", "logical_cogency"),
                           ("Consistency", "consistency"),
                           ("Delta / Votes", "delta_votes")]:
            out.append(f"<details><summary>{label}</summary>\n\n```json\n"
                       + json.dumps(t.get(key, {}), indent=2, ensure_ascii=False)
                       + "\n```\n</details>")
        out.append("")
    return "\n".join(out)


def _reasoning_section(rs: dict) -> str:
    """Reasoning-model summary + any flagged score↔content disagreements."""
    if rs.get("error"):
        return (f"\n## Reasoning summary\n\n_Not produced: {rs['error']}_")
    out = ["\n## Reasoning summary",
           f"_by {rs.get('model','reasoning model')}_\n",
           (rs.get("narrative") or "").strip() or "_(no narrative)_"]
    disc = rs.get("discrepancies") or []
    out.append("\n### ⚠ Score ↔ reasoning discrepancies")
    if disc:
        out.append("_The reasoning model flagged these scores as contradicting "
                   "what the turns show — inspect them (a flag is a pointer for "
                   "review, not a verdict that the score is wrong)._\n")
        for d in disc:
            out.append(f"- **{d.get('metric','?')}** — {d.get('reason','')}")
    else:
        out.append("_None — the reasoning model concurred with all computed scores._")
    return "\n".join(out)


def build_debate_report_md(result: dict) -> str:
    topic = result.get("topic", "(no topic)")
    stances = result.get("stances", {})
    by_speaker = result.get("by_speaker", {})
    dl = result.get("debate_level", {})
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = []
    md.append(f"# Debate Judge Report")
    md.append(f"**Topic:** {topic}  ")
    md.append(f"**Generated:** {ts}  ")
    md.append(f"**Sides:** " + ", ".join(f"{s} = {st}" for s, st in stances.items()) + "  ")
    md.append(f"**Rebuttal mode:** {dl.get('rebuttal_mode','?')}")
    if result.get("note"):
        md.append(f"\n> ⚠ {result['note']}")

    # ── reasoning summary + score↔reasoning discrepancies (top of report) ────
    rs = result.get("reasoning_summary") or {}
    if rs:
        md.append(_reasoning_section(rs))

    md.append("\n## 1. Per-speaker scores\n")
    md.append(_speaker_table(by_speaker))

    md.append("\n## 2. Rebuttal map (which sentence rebutted what)\n")
    md.append(_rebuttal_map(dl.get("rebuttal", {})))

    md.append("\n## 3. Argument survival\n")
    md.append(_survival_section(dl.get("argument_survival", {})))

    # key flags
    md.append("\n## 4. Key flags\n")
    flags = []
    surv = dl.get("argument_survival", {})
    if isinstance(surv, dict):
        for tid, thr in (surv.get("threads") or {}).items():
            if not thr.get("alive"):
                flags.append(f"- Argument from Turn {thr.get('birth_turn')} ({thr.get('owner')}) was **defeated** (survival {_fmt(thr.get('survival'),2)}).")
    for sp, d in by_speaker.items():
        cons = d.get("consistency")
        cval = (cons.get("mean_consistency") if isinstance(cons, dict) else cons)
        if cval is not None and cval < 0.5:
            flags.append(f"- {sp} shows an **internal inconsistency** (consistency {_fmt(cval)}).")
    md.append("\n".join(flags) if flags else "_No notable flags._")

    md.append("\n## 5. Full per-turn detail\n")
    md.append(_per_turn_full(result.get("per_turn", [])))

    return "\n".join(md)


def write_debate_report(result: dict) -> str:
    """Write the Markdown report to REPORT_DIR and return its absolute path."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = Path(REPORT_DIR) / f"debate_{stamp}.md"
    path.write_text(build_debate_report_md(result), encoding="utf-8")
    return str(path.resolve())
