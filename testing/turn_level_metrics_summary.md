# Judge v2 — Turn-Level Metrics: Testing Summary

**Metrics tested:** Stance Support · Logical Cogency · Consistency

**Data:** 5 debate topics × 6 turns = **30 turns** (15 PRO, 15 CON)

---

## 1. What I Tested

| Topic | Annotator |
|---|---|
| Governments should give citizens free money to end poverty | Rosen |
| Growing up as an only child does more harm than good | Immanuvel |
| Holding a grudge is healthier than always forgiving and forgetting | Sneha |
| Space exploration is worth the cost even when there are major problems on Earth | Manoj |
| Working from home is better than working in an office | Pavan |

**Models used by the judge**

| Metric | How it scores | Model |
|---|---|---|
| Stance Support | NLI | DeBERTa-v3 (MNLI) |
| Logical Cogency | LLM | **qwen2.5:7b** and **gemma4:26b** (run with both) |
| Consistency | NLI | DeBERTa-v3 (MNLI) |

> ✅ Correct · ⚠️ Partly correct · ❌ Wrong

---

## 2. Stance Support

### How it works
The turn's **claim** is compared with a one-sentence **hypothesis of the speaker's side** using NLI.
The model gives three probabilities — *entailment / neutral / contradiction* — and the **score is the entailment probability only** (HIGH ≥ 0.60, MEDIUM ≥ 0.35, LOW below).

CON hypotheses are written as a **denial of the motion** (e.g. *"Working from home is **not** better than working in an office"*), since CON only has to reject the motion, not prove the opposite.

### How v2 responded

| Topic | ✅ | ⚠️ | ❌ |
|---|---|---|---|
| Govt free money | 0 | 0 | 6 |
| Only child | 1 | 1 | 4 |
| Grudge | 3 | 1 | 2 |
| Space | 2 | 1 | 3 |
| WFH | 2* | 2 | 2 |
| **Total** | **8** | **5** | **17 / 30** |

\* one of them correct only by chance (see below)

- 29 of 30 turns clearly argue their assigned side, but the judge labelled **18 of 30 as LOW**.
- **PRO:** 4 correct · 1 partly · 10 wrong. **CON:** 4 correct · 4 partly · 7 wrong.
- **Govt topic: 0 of 6 turns scored correctly.**

### Where it failed

| Failure | Count | What happens |
|---|---|---|
| Reason / evidence read as **neutral** | 12 | The claim supports the side with a reason, but does not restate it |
| Rebuttal read as **neutral** | 3 | The claim attacks the opponent instead of stating its own side |
| Support read as **contradiction** | 2 | Negation / concession words flip the result |

### Highlighted examples

> **❌ WFH, turn a-r2 (PRO) — clear support scored LOW**
> **Claim:** "The flexibility and personal benefits of working from home often outweigh the concerns about the office."
> **Side:** "Working from home is better than working in an office."
> **Result:** entailment 0.05 · **neutral 0.95** → **0.05 LOW**
> **Why:** "Outweigh" is a comparison in PRO's favour, but it is indirect — the claim never says "home is better than the office", so NLI reads it as neutral.

> **❌ Govt, turn a-r2 (PRO) — own side read as contradiction**
> **Claim:** "I acknowledge the concern about dependency, but a well-structured UBI program can include features to encourage work, such as a gradual reduction in benefits…"
> **Side:** "Governments should give citizens free money to end poverty."
> **Result:** **contradiction 0.97** → **0.00 LOW**
> **Why:** The PRO speaker is defending UBI, but "acknowledge the concern" and "reduction in benefits" push the model to *contradiction*.

> **✅ Space, turn a-r1 (PRO) — the kind of claim that passes**
> **Claim:** "Space exploration is invaluable, even when Earth faces significant challenges."
> **Result:** entailment **0.98** → **HIGH**
> **Why:** The claim covers the *whole* side sentence (value + "even when Earth has problems").

> **WFH, turn a-r3 (PRO) — correct only by chance**
> The speaker drifts to "best of both worlds" (a real stance failure). It scored LOW (0.008) — correct — but the speaker's clear PRO turns a-r1 and a-r2 are also LOW (0.05), so the metric **cannot tell a real drift from normal turns**.

### Verdict
**❌ Not working as intended.** The inputs and code are correct; the problem is the method. **NLI entailment asks "does the claim say the same as the side?", but stance support needs "does the claim argue for the side?"** Debaters argue with reasons, comparisons and rebuttals, which entailment does not reward.

---

## 3. Logical Cogency

### How it works
**One LLM call** reads the claim + premises and gives three scores, **1–3**, each with a written reason:
- **Acceptability** — are the premises believable?
- **Relevance** — are the premises about the claim?
- **Sufficiency** — *if* the premises are true, are they enough for the claim?

Run with **qwen2.5:7b** (the same model that generated the debates) and with **gemma4:26b**.

### How v2 responded

| Model | ✅ | ⚠️ | ❌ |
|---|---|---|---|
| **qwen2.5:7b** | 12 | 14 | 4 |
| **gemma4:26b** | **19** | 10 | **1** |

| | Acceptability | Relevance | Sufficiency | Empty reasons |
|---|---|---|---|---|
| **qwen** | 3 ×22, 2 ×8 | **3 on all 30** | **2 on all 30** | **7 turns** |
| **gemma** | 3 ×11, 2 ×19 | 3 ×27, 2 ×1, 1 ×2 | 3 ×7, 2 ×20, 1 ×3 | 0 |

The two models give the same three scores on **only 12 of 30** turns.

### Where it failed

**qwen2.5:7b**

| Failure | Count |
|---|---|
| Relevance always 3, sufficiency always 2 — cannot separate weak from strong arguments | 30 / 30 |
| **No reasons at all** | 7 turns |
| **Reason describes a 2, score is 3** ("plausible but lacks evidence" → 3) | 7 turns |
| Copy-paste reasons (same premise quoted for all three) | 3 turns |
| Misses premises that are not about the claim | 2 turns |

**gemma4:26b**

| Failure | Count |
|---|---|
| Sufficiency 3 when **part** of the claim is not supported (a comparison, condition or extra element) | 3 turns |
| Misreads concessions ("While X, Y" — asks for proof of X, which the speaker concedes) | 3 turns (score mostly unaffected) |
| Other small reason/score slips | 3 turns |

### Highlighted examples

> **Govt, turn a-r3 (PRO) — gemma ✅, qwen ❌**
> **Claim:** "…studies show that UBI recipients often use the income to improve their lives and invest in education or job training…"
> **Premises:** economic stimulus from UBI; designing UBI programs thoughtfully.
> **qwen:** 2 / 3 / 2 — **with no reasons at all**
> **gemma:** 2 / **1 / 1** — *"The premises regarding 'economic stimulus' and 'designing UBI programs' do not address the specific conclusion about 'UBI recipients' using income for 'education or job training'."*
> **Why:** The premises are about something else. gemma sees it; qwen gives its default scores.

> **❌ WFH, turn a-r3 (PRO) — both models miss**
> **Claim:** "The key is to leverage the best of both worlds, combining… home with… office."
> **Premise:** "The social interaction benefits of the office can be effectively replicated through virtual meetups…"
> **Both:** sufficiency **2** — should be **1**
> **Why:** The premise says the office is *not needed*, which argues **against** combining both. Neither model catches premises that work against the claim.

### Verdict
- **Code:** ✅ works; valid scores; but it **accepts empty reasons** (`default=""` in the schema).
- **qwen2.5:7b:** ❌ **not usable** for this metric — relevance and sufficiency are constant, reasons missing or contradicting the score.
- **gemma4:26b:** ✅ **mostly working** — varied, explained scores that catch real gaps; weak on multi-part claims.
- **Not tested by this data:** acceptability 1 was never given, so the rule "judge sufficiency as if premises are true" was never exercised; repeatability was not tested.

---

## 4. Consistency

### How it works
The current claim is compared by NLI with **each earlier claim of the same speaker**.
**Score = 1 − the highest contradiction probability** (1.0 = consistent; below 0.5 = contradicting). Opening turns get `null`.

### How v2 responded

| | Turns |
|---|---|
| Opening turns (`null`, correct by design) | 10 |
| Scored turns | 20 |
| ✅ Correct (all ~0.9–1.0) | **18** |
| ❌ False alarm | **1** |
| ❌ Missed a real shift | **1** |

### Highlighted examples

> **❌ WFH, turn a-r3 (PRO) — real shift missed**
> **Now:** "The key is to leverage the best of both worlds, combining… home with… office."
> **Before (a-r2):** "The flexibility and personal benefits of working from home often outweigh the concerns about the office."
> **Result:** neutral 0.97 · contradiction 0.00 → **consistency 0.997**
> **Why:** The speaker moved from *home is better* to *a mix of both*. NLI only detects a hard "A vs not-A" clash — a change of position reads as neutral.

> **❌ Govt, turn a-r2 (PRO) — false alarm**
> **Now:** "…a well-structured UBI program can include features to encourage work, such as a gradual reduction in benefits…"
> **Before (a-r1):** "Providing citizens with free money directly addresses the root causes of poverty…"
> **Result:** contradiction 0.69 → **consistency 0.31 (flagged)**
> **Why:** Both claims defend UBI. The only flag raised in all 5 topics is wrong — same negation/concession trigger as in stance support.

### Where it failed
| Failure | Example |
|---|---|
| Shifts / softening / "both sides" not detected | WFH a-r3 |
| Negation words cause false contradiction | Govt a-r2 |
| Only claims are compared — drift inside premises is invisible | Only-child a-r3, Grudge a-r3 |
| Noisy middle values without a real reason | Only-child b-r2 → 0.76 |

### Verdict
**⚠️ Works for what it measures, but not proven for its purpose.** 18/20 looks good, but these debates contain almost no real inconsistency — it caught **0 of 1** real shifts and raised **1 false alarm**. A score of ~1.0 means "no hard contradiction found", not "the speaker held the same line".

---

## 5. How to Fix It

### Stance Support
| Option | Idea | Effort |
|---|---|---|
| 1 | Score the claim against **both** sides and use which side it is closer to, instead of entailment of one side | Low |
| 2 | Score the **whole turn** instead of only the claim (more context, e.g. the missing half of a comparison) | Very low |
| 3 | Replace NLI with an **LLM stance judgement** (0–3 with a reason) — the same move cogency already made for relevance | Medium |

### Logical Cogency
| Option | Idea | Effort |
|---|---|---|
| 1 | **Use gemma4:26b, not qwen2.5:7b** (and not the model that generated the debates) | Very low |
| 2 | Make reasons **required** so empty reasons fail loudly | Very low (code) |
| 3 | Prompt: "check **every part** of the conclusion (comparisons, conditions)"; "do not require support for a clause the speaker **concedes**" | Low |
| 4 | Add **control arguments** (false premise, off-topic premises, premises against the claim) and **repeat runs** | Low |

### Consistency
| Option | Idea | Effort |
|---|---|---|
| 1 | Add **planted test cases**: direct negation, "both sides" shift, unrelated claim | Low |
| 2 | Compare **premises** too, and run NLI in **both directions** | Low |
| 3 | Use an **LLM judgement** with the rule *"Can the speaker reasonably hold both claims at the same time?"* | Medium |

---

## 6. Key Takeaways

1. **Stance Support fails on 17 of 30 turns (Govt: 0 of 6)** — NLI entailment only rewards claims that restate the side, not claims that argue for it.
2. **Logical Cogency depends on the model** — qwen 7B gives near-constant scores (and 7 empty reasons); gemma 26B gives varied, explained scores and was right on 19 of 30.
3. **Consistency looks good (18/20) but is untested for its real job** — it missed the only real position shift and its only flag was a false alarm.
4. **Same weakness in both NLI metrics:** negation/concession wording flips results (Govt a-r2 fails both stance and consistency), and NLI cannot recognise *arguing for* or *drifting from* a position.
5. **One turn tells the whole story — WFH a-r3** ("best of both worlds"): a real stance drift that stance support flagged only by chance, cogency scored as moderately sufficient, and consistency scored 0.997.

---

*Test inputs: `input_topics/` · Raw outputs: `outputs_qwen2.5-7b/` (all three metrics; cogency with qwen) and `outputs_gemma4-26b_cogency/` (cogency with gemma)*
