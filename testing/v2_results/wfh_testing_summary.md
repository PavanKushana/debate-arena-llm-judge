# Testing Summary — Working from Home vs Office

## Topic
**Working from home is better than working in an office.**

## Status
These are **actual values recorded from the HTML/API tester** during testing.

## Metric Implementation Used in Testing
All four tested metrics were calculated using the **NLI-based implementation only**.

- **Stance Support:** NLI
- **Stance Support · Dual:** NLI
- **Logical Cogency:** NLI
- **Consistency:** NLI

For Logical Cogency, `use_llm_sufficiency` was kept as `false` in every test. Therefore, the `model` field in the request payload was **not used to calculate the metric**. Sufficiency was calculated through the NLI fallback path.

## Metrics Tested
- Stance Support
- Stance Support · Dual
- Logical Cogency
- Consistency

---

## Turn 1 — PRO

### Stance Support
- **Stance support score:** 0.9952
- **Net stance:** 0.9941
- **Label:** HIGH
- **Off-topic:** false
- **Entailment:** 0.9952
- **Neutral:** 0.0037
- **Contradiction:** 0.0011

**Observation:**  
The turn strongly supports the PRO position. The result is correct and very stable.

### Stance Support · Dual
- **PRO score:** 0.9945
- **CON score:** 0.0017
- **Net stance:** 0.9928
- **Label:** HIGH
- **PRO entailment:** 0.9945
- **CON contradiction:** 0.9967

**Observation:**  
The dual metric also identifies the turn as clearly PRO.

### Logical Cogency
- **Cogency score:** 0.2062
- **Label:** LOW
- **Acceptability:** 0.9998
- **Relevance:** 0.0080
- **Sufficiency:** 0.0077
- **Entailment:** 0.0077
- **Neutral:** 0.9920
- **Contradiction:** 0.0002
- **Sufficiency method:** nli_fallback
- **LLM sufficiency used:** false

**Observation:**  
This result is unexpectedly low. The premises clearly provide reasons for the claim, but the NLI fallback treats them as almost neutral. This suggests that Logical Cogency can be very sensitive to how a broad evaluative claim such as *“WFH is a better choice”* is phrased.

---

## Turn 2 — CON

The generated turn ended with an incomplete sentence:

> “Overall, while WFH has its merits, it is not always the optimal choice for”

For this reason, the stance metrics were tested twice.

### Stance Support — Original Incomplete Text
- **Stance support score:** 0.1398
- **Net stance:** -0.6787
- **Label:** LOW
- **Off-topic:** false
- **Entailment:** 0.8185
- **Neutral:** 0.0417
- **Contradiction:** 0.1398

**Observation:**  
The result is incorrect for the intended CON stance. The incomplete ending caused the NLI classifier to assign a very high entailment score to the PRO thesis.

### Stance Support — Incomplete Ending Removed
- **Stance support score:** 0.8348
- **Net stance:** 0.8178
- **Label:** HIGH
- **Off-topic:** false
- **Entailment:** 0.0170
- **Neutral:** 0.1481
- **Contradiction:** 0.8348

**Observation:**  
Removing the incomplete ending changes the result from LOW to HIGH and makes contradiction dominant. This is a strong indication that truncated text can distort NLI-based stance classification.

### Stance Support · Dual — Original Incomplete Text
- **PRO score:** 0.8185
- **CON score:** 0.9988
- **Net stance:** 0.1802
- **Label:** HIGH

**Observation:**  
The CON hypothesis is strongly supported, but the PRO hypothesis is also incorrectly high. The low net stance reveals the ambiguity.

### Stance Support · Dual — Incomplete Ending Removed
- **PRO score:** 0.0170
- **CON score:** 0.9939
- **Net stance:** 0.9768
- **Label:** HIGH

**Observation:**  
After removing the incomplete ending, the dual result becomes much cleaner and strongly favors the CON side.

### Logical Cogency
- **Cogency score:** 0.9968
- **Label:** HIGH
- **Acceptability:** 0.9997
- **Relevance:** 0.9963
- **Sufficiency:** 0.9959
- **Entailment:** 0.9959
- **Neutral:** 0.0037
- **Contradiction:** 0.0003
- **Sufficiency method:** nli_fallback
- **LLM sufficiency used:** false

**Observation:**  
The premises directly support the claim that WFH is not universally superior. The metric behaves as expected.

---

## Turn 3 — PRO

### Stance Support
- **Stance support score:** 0.9267
- **Net stance:** 0.8777
- **Label:** HIGH
- **Off-topic:** false
- **Entailment:** 0.9267
- **Neutral:** 0.0243
- **Contradiction:** 0.0490

**Observation:**  
The turn is correctly recognized as PRO.

### Stance Support · Dual
- **PRO score:** 0.9267
- **CON score:** 0.0076
- **Net stance:** 0.9191
- **Label:** HIGH
- **CON contradiction:** 0.9855

**Observation:**  
The dual result is clean and strongly favors the PRO hypothesis.

### Logical Cogency
- **Cogency score:** 0.9940
- **Label:** HIGH
- **Acceptability:** 0.9992
- **Relevance:** 0.9932
- **Sufficiency:** 0.9924
- **Entailment:** 0.9924
- **Neutral:** 0.0068
- **Contradiction:** 0.0008
- **Sufficiency method:** nli_fallback
- **LLM sufficiency used:** false

**Observation:**  
The metric correctly identifies the premises as relevant and sufficient. This also shows that having several premises does not itself cause low cogency.

---

## Turn 4 — CON

The generated turn also ended incompletely, so Stance Support and Dual were checked both with and without the final incomplete part.

### Stance Support — Original Text
- **Stance support score:** 0.1087
- **Net stance:** 0.0975
- **Label:** LOW
- **Off-topic:** true
- **Entailment:** 0.0112
- **Neutral:** 0.8801
- **Contradiction:** 0.1087

### Stance Support — Incomplete Ending Removed
- **Stance support score:** 0.0952
- **Net stance:** 0.0704
- **Label:** LOW
- **Off-topic:** true
- **Entailment:** 0.0248
- **Neutral:** 0.8800
- **Contradiction:** 0.0952

**Observation:**  
Unlike Turn 2, removing the incomplete ending does not fix the result. The metric still treats the turn as mostly neutral, even though it gives several clear reasons against WFH being better.

### Stance Support · Dual — Original Text
- **PRO score:** 0.0112
- **CON score:** 0.8838
- **Net stance:** 0.8726
- **Label:** HIGH
- **Off-topic:** false

### Stance Support · Dual — Incomplete Ending Removed
- **PRO score:** 0.0248
- **CON score:** 0.9307
- **Net stance:** 0.9059
- **Label:** HIGH
- **Off-topic:** false

**Observation:**  
Dual mode performs much better than affirmative mode for this indirect CON argument. The explicit CON hypothesis allows the metric to recognize the intended side correctly.

### Logical Cogency
- **Cogency score:** 0.9795
- **Label:** HIGH
- **Acceptability:** 0.9913
- **Relevance:** 0.9814
- **Sufficiency:** 0.9727
- **Entailment:** 0.9727
- **Neutral:** 0.0186
- **Contradiction:** 0.0087
- **Sufficiency method:** nli_fallback
- **LLM sufficiency used:** false

**Observation:**  
The premises strongly support the claim. The result is consistent with the argument structure.

---

## Consistency

### PRO Claims
- **Consistency score:** 0.9996
- **Maximum contradiction:** 0.0004
- **Inconsistent:** false
- **Number of prior claims:** 1

**Observation:**  
The second PRO claim develops the first PRO position without contradicting it.

### CON Claims
- **Consistency score:** 0.9994
- **Maximum contradiction:** 0.0006
- **Inconsistent:** false
- **Number of prior claims:** 1

**Observation:**  
The second CON claim remains fully consistent with the earlier CON position.

---

# Overall Findings

1. **All four metrics in this test were NLI-based; no LLM was used for metric calculation.**
3. **Clear PRO turns were detected reliably.**
3. **Incomplete generated text can strongly affect stance classification**, as demonstrated by Turn 2.
4. Truncation does **not always** cause an error; Turn 3 remained stable despite being incomplete.
5. **Affirmative Stance Support can struggle with indirect CON arguments.**
6. **Dual Stance Support was more reliable for CON arguments** because it directly evaluates an explicit CON hypothesis.
7. The **net stance** in dual mode is useful for detecting cases where both hypotheses receive high support.
8. **Logical Cogency worked well for Turns 2–4**, but Turn 1 exposed sensitivity to broad claim wording under the NLI fallback.
9. **Consistency performed very strongly** for both PRO and CON claims.
10. The main technical weaknesses observed were **NLI sensitivity to truncation, claim formulation, and indirect contradiction**.

## Short Presentation Conclusion

The WFH test shows that the scoring pipeline generally behaves well, but NLI-based metrics are sensitive to wording and incomplete generation. Dual stance evaluation is especially useful for CON arguments, while Logical Cogency may require additional attention when premises support a broad evaluative conclusion indirectly. **No LLM was used to calculate these four metric scores.**
