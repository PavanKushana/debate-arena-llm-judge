# Manual Annotation Rules for LLM-Generated Debate

## Purpose

Use these rules to manually evaluate each debate turn in a simple and
consistent way.

## General Scoring Scale

For numeric metrics:

- **0 = Absent / Very poor**
- **1 = Weak**
- **2 = Moderate / Clear**
- **3 = Strong**

Judge the **meaning and reasoning**, not just the wording or writing
quality.

------------------------------------------------------------------------

## 1. Stance Support

**Question:** Does the turn support the speaker’s assigned side?

- **0:** Does not support the speaker’s side or argues against it.
- **1:** Slight or unclear support.
- **2:** Clearly supports the side.
- **3:** Strong and direct support.

------------------------------------------------------------------------

## 2. Logical Cogency

Evaluate three separate dimensions.

### 2.1 Local Acceptability

**Question:** Are the premises/reasons believable or reasonable?

- **0:** Clearly false, implausible, or unsupported.
- **1:** Weakly believable or questionable.
- **2:** Generally reasonable/believable.
- **3:** Clearly credible and well-supported.

**Remember:** Do not ask whether the premises prove the claim. That is
**sufficiency**.

### 2.2 Local Relevance

**Question:** Do the premises/reasons actually relate to the claim?

- **0:** Unrelated.
- **1:** Only weakly related.
- **2:** Clearly related.
- **3:** Directly and strongly relevant.

**Remember:** A premise can be relevant without being sufficient.

### 2.3 Local Sufficiency

**Question:** If the premises are true, are they enough to justify the
claim?

- **0:** Do not justify the claim.
- **1:** Provide only weak/partial support.
- **2:** Provide reasonably strong support.
- **3:** Provide strong and sufficient support.

**Simple distinction:**

> Acceptability = Are the reasons believable?  
> Relevance = Are the reasons about the claim?  
> Sufficiency = Are the reasons enough to support the claim?

------------------------------------------------------------------------

## 3. Consistency

**Question:** Is the current turn consistent with the speaker’s previous
claims?

- **0:** Clear contradiction.
- **1:** Significant tension or inconsistency.
- **2:** Mostly consistent, with a minor change or qualification.
- **3:** Fully consistent.

**Remember:** A clarification, qualification, or change of emphasis is
not automatically a contradiction.

Ask:

> “Can the speaker reasonably maintain both claims at the same time?”

------------------------------------------------------------------------

## 4. Rebuttal Effectiveness

First decide whether the turn actually responds to an opponent’s
argument.

### Step 1: Is there a rebuttal?

- **No:** Mark **No Rebuttal**.
- **Yes:** Identify what is being attacked:
  - Claim
  - Premise
  - Evidence
  - Inference/reasoning

### Step 2: Rate the rebuttal

- **0:** No meaningful attack or only disagreement.
- **1:** Weak attack with little reasoning/evidence.
- **2:** Clear and relevant challenge with reasonable support.
- **3:** Strong attack that substantially undermines the opponent’s
  argument.

**Remember:** “I disagree” by itself is not a strong rebuttal.

------------------------------------------------------------------------

## 5. Information Density

**Question:** Does the turn add new, meaningful, topic-relevant
information?

Focus on **semantic novelty**, not new wording.

- **0:** Almost entirely repetition or irrelevant content.
- **1:** Small amount of new information.
- **2:** Several meaningful new points/details.
- **3:** Strong amount of relevant new information.

**Remember:** Rephrasing is not new information. A new reason, example,
evidence, or implication is.

------------------------------------------------------------------------

## 6. Argument Survival

Track each major argument thread across the debate.

### Unchallenged

The argument has not received a meaningful attack.

### Defended

The argument was attacked, and its owner later responds to the attack.

### Weakened

The argument was meaningfully attacked but was not successfully
defended.

### Refuted

The argument is decisively defeated.

Use **Refuted** only when:

- The opponent gives a **strong (3/3) rebuttal**, and
- The argument owner does **not successfully defend it afterward**.

**Remember:** An argument is not refuted simply because someone
disagrees with it.

------------------------------------------------------------------------

# Recommended Annotation Workflow

For each turn:

1.  Identify the **main claim**.
2.  Identify the **premises/reasons/evidence**.
3.  Score **Stance Support**.
4.  Score **Logical Cogency**:
    - Acceptability
    - Relevance
    - Sufficiency
5.  Compare with previous same-speaker claims and score **Consistency**.
6.  Check whether the turn contains a **rebuttal**.
7.  If yes, identify the **target** and score Rebuttal Effectiveness.
8.  Score **Information Density**.
9.  Update the relevant **Argument Survival** status.
10. Add a **short justification** for the important scores.

------------------------------------------------------------------------

# Annotation Template

| Field                        | Annotation                                   |
|------------------------------|----------------------------------------------|
| Turn ID                      |                                              |
| Speaker                      |                                              |
| Main Claim                   |                                              |
| Premises / Reasons           |                                              |
| Stance Support (0–3)         |                                              |
| Acceptability (0–3)          |                                              |
| Relevance (0–3)              |                                              |
| Sufficiency (0–3)            |                                              |
| Consistency (0–3)            |                                              |
| Rebuttal? (Yes/No)           |                                              |
| Rebuttal Target              | Claim / Premise / Evidence / Inference       |
| Rebuttal Effectiveness (0–3) |                                              |
| Information Density (0–3)    |                                              |
| Argument Thread              |                                              |
| Argument Survival            | Unchallenged / Defended / Weakened / Refuted |
| Short Justification          |                                              |

------------------------------------------------------------------------

# Quick Rules to Remember

- **Believable ≠ sufficient.**
- **Relevant ≠ sufficient.**
- **Disagreement ≠ rebuttal.**
- **Rephrasing ≠ new information.**
- **Qualification ≠ contradiction.**
- **An attack ≠ refutation.**
- **Refutation requires a decisive attack and no successful defense.**
- Judge the **argument**, not the writing style.
- Use the **debate context**, not the turn in isolation.
