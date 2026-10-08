# Why We Built V2

## Overview

We built **V2 (`judge_service_v2`) because V1 (`judge_api`) tried to answer the final question — *“Who won?”* — before establishing whether its underlying measurements, constructs, weighting scheme, and ground truth were sufficiently valid to support that decision.

V1 was therefore an important **prototype and feasibility study**. It demonstrated that an automated debate-judging pipeline could combine multiple signals and produce an overall speaker score. However, the development and evaluation of V1 revealed a deeper research problem: **before building a reliable automated judge, we first need to establish what can actually be measured reliably.**

V2 is the methodological evolution of that work. It moves from a **winner-oriented scoring system** toward a **multidimensional argumentative analysis framework** in which individual constructs can be measured and validated before they are combined into an overall judgment.

---

## V1 — `judge_api`

V1 was designed as an end-to-end automated judging pipeline. It combined several signals to produce a composite speaker score and ultimately determine a winner.

Its main signals included:

* NLI
* Argument quality
* Evidence / votes
* Engagement
* Rebuttal effectiveness
* Information density
* Composure

The overall process was:

```text
Individual metrics
        ↓
Manual weighting
        ↓
Composite speaker score
        ↓
0–100 score
        ↓
Winner
```
---

## What V1 Revealed

### 1. The Argument-Quality Signal Does Not Automatically Generalize to Debate

V1 relied heavily on **IBM ArgQ-Rank-30k**, a well-established argument-quality benchmark containing more than 30,000 crowd-sourced arguments across 71 controversial topics, with quality judgments aggregated from 10 annotators.

The dataset is valuable for its intended purpose, but its characteristics do not directly match our debate setting:

* **Short arguments:** The dataset contains short, standalone arguments, unlike the multi-sentence turns produced in our debates.
* **Limited topic coverage:** Its 71 topics create a potential generalization problem for dynamically generated and previously unseen debate topics.
* **Similarity threshold:** A fixed cosine-similarity threshold such as `0.8` is not scientifically justified for comparing substantially different-length debate turns with short reference arguments.

The important conclusion is **not that IBM ArgQ is unreliable**. Rather, it is a strong benchmark for its intended task, but its **data distribution and annotation objective do not directly correspond to our multi-sentence, dynamically generated debate setting**.

This creates a generalization and construct-validity question that V1 could not resolve.

---

### 2. Manual Weighting Hid the Uncertainty

V1 combined heterogeneous metrics using manually selected weights. These weights were not supported by sufficient empirical ground truth to establish that one construct should contribute more than another.

This creates two problems:

1. We cannot confidently justify the relative importance assigned to each metric.
2. A single composite score can hide meaningful differences between speakers.

For example, one speaker may provide stronger stance support while another may demonstrate stronger rebuttal effectiveness. Reducing both profiles to one number can remove precisely the information we want to understand.

---

### 3. The Ground Truth Was Not Established

A composite score was implicitly treated as sufficient to rank speakers, but there was no established gold standard demonstrating that the resulting ranking corresponds to human judgments of debate performance.

This raised several fundamental questions:

* What exactly does each metric measure?
* Are different metrics measuring the same underlying construct?
* Do the metrics correspond to human judgments?
* What should the relative weights be?
* Does a metric trained on one domain generalize to debate?
* Can overall debate quality legitimately be represented by one score?

These are **measurement questions**, not merely implementation problems.

---

## Why We Did Not Simply Improve V1

V1 and V2 have different research objectives.

V1 starts with:

> **“Given a debate, who is the winner?”**

That objective requires a trustworthy aggregation framework, justified weights, valid individual measurements, and an appropriate ground truth.

During V1 development, we found that these assumptions had not yet been established.

Therefore, simply improving V1's implementation would not solve the underlying problem. Better code could make the pipeline more robust, but it would not answer whether the **measurement framework itself is valid**.

V2 consequently treats the problem as a **research framework problem rather than only a software engineering problem**.

---

# V2 — `judge_service_v2`

V2 changes the central question from:

> **“Who won?”**

to:

> **“What can we measure reliably, and where is one speaker stronger than the other?”**


| Aspect            | V1 — `judge_api`                       | V2 — `judge_service_v2`                                          |
| ----------------- | -------------------------------------- | ---------------------------------------------------------------- |
| Main unit         | Argument → speaker → winner            | Turn → metric → speaker comparison                               |
| Primary objective | Produce a winner                       | Measure and validate argumentative constructs                    |
| Ground truth      | Implicitly assumes a composite ranking | Explicitly acknowledges the lack of an established gold standard |
| Output            | Composite 0–100 score                  | Multidimensional speaker profile                                 |
| Validation        | Limited construct validation           | Human labels, agreement ceiling, discriminant validity           |

---

## What V2 Measures

V2 separates several constructs that were previously combined into a final score.

### Stance Support

Measures whether a turn supports the speaker's position.

```text
Topic hypothesis
      ↓
Debate turn
      ↓
NLI comparison
      ↓
Stance-support signal
```

The purpose is to measure **support for the speaker's position**, rather than treating NLI as a general-purpose indicator of debate quality.

### Logical Cogency

Measures how well the reasons in a turn support the local claim.

It considers:

* **Acceptability**
* **Relevance**
* **Sufficiency**

These dimensions are analyzed as aspects of local reasoning rather than being conflated with stance support.

### Dialectical Behaviour

V2 examines how arguments function within the interaction itself, including:

* **Rebuttal effectiveness**
* **Argument survival**

This moves beyond evaluating an argument in isolation and considers its behaviour across the debate.

### Additional Signals

V2 also retains useful signals where appropriate, including:

* **Consistency**
* **Information density**

These signals remain separate rather than being automatically converted into a single overall score.

---

# Validation Before Aggregation

A central principle of V2 is:

> **A metric should be evaluated before it is trusted as evidence for an overall judgment.**

V2 therefore introduces validation at the construct level using:

* **Human labels** — comparison with human judgments of the relevant construct.
* **Agreement ceiling** — recognition that automated performance should be interpreted relative to the level of agreement achievable between humans.
* **Discriminant validity** — testing whether supposedly different metrics actually capture distinguishable constructs rather than measuring the same underlying property.

This allows us to determine which measurements are meaningful, which overlap, and which require further refinement.

Only after the individual constructs are sufficiently understood does it become scientifically reasonable to consider whether they should be combined into an overall judgment.

---

# From V1 to V2

The evolution can be summarized as:

```text
V1 — Automated Winner
        ↓
Individual metrics
        ↓
Manual weights
        ↓
Composite score
        ↓
Winner
        ↓
Research review
        ├── Are the weights justified?
        ├── What does each metric actually measure?
        ├── Are the metrics overlapping?
        ├── Where is the ground truth?
        └── Does IBM ArgQ generalize to our setting?
        ↓
V2 — Validated Multidimensional Analysis
        ↓
Stance Support
Logical Cogency
Dialectical Behaviour
        ├── Rebuttal
        └── Argument Survival
        ↓
Additional signals
        ├── Consistency
        └── Information Density
        ↓
Human validation
        ├── Human labels
        ├── Agreement ceiling
        └── Discriminant validity
        ↓
Multidimensional speaker profile
        ↓
Speaker comparison
        ├── Where is Speaker A stronger?
        └── Where is Speaker B stronger?
```

---

# Why V2 Is a Research Evolution

The progression is therefore:

### V1 — Prototype

V1 demonstrated that an automated debate-judging pipeline could be built and used to generate a winner.

### Research Problem

V1 exposed limitations involving:

1. **Construct separation**
2. **IBM ArgQ-Rank-30k's topic and annotation dependence**
3. **Manual weighting**
4. **Lack of an established gold standard**
5. **Information loss from a single composite score**

### V2 — Methodological Refinement

V2 addresses these limitations by:

* separating argumentative constructs,
* reporting their signals independently,
* treating unavailable measurements explicitly rather than inventing fallback scores,
* evaluating measurements against human annotations,

The purpose of V2 is therefore **not simply to produce a better version of the V1 winner score**.

It is to establish a more scientifically defensible framework for automated debate analysis—one in which we first determine **what can be measured reliably**, validate those measurements, and then use them to understand **where and why one speaker is stronger than another**.

> **V1 asked: “Who won?”**
> **V2 asks: “What can we measure reliably, and where is one speaker stronger than the other?”**

That shift—from **winner prediction** to **validated multidimensional measurement**—is the fundamental reason we built V2.
