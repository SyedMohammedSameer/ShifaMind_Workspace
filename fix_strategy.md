# Fix Strategy - Evidence-Based Approach

## Current Status: NO FIXES YET

I haven't applied any fixes yet. The `shifamind302_phase1_FIXED.py` makes assumptions that need validation.

## Step 1: Run Rigorous Analysis FIRST

```bash
python rigorous_analysis.py
```

This will PROVE (not guess):
- Is max_length actually different? (load both configs)
- Is AMP actually enabled in 302? (load config)
- Which concepts are predictive vs noise? (correlation analysis)
- What's optimal sparsity? (current: 78% → 62%)
- Do Top-50 codes match? (compare with counts)

## Step 2: Identify PRIMARY Issue (After Evidence)

Based on evidence, we'll identify the ONE main culprit:

**Hypothesis A: UMLS over-extraction is the primary issue**
- Evidence needed:
  - 302 has 99.58 concepts/sample vs 301's 24.5
  - >50% of concepts have weak correlation with diagnoses (<0.1)
  - Performance drops correlate with concept density
- Fix: Stricter UMLS filtering
- Expected: Concepts drop to 15-25/sample, F1 recovers

**Hypothesis B: Config changes (max_length/AMP) are primary**
- Evidence needed:
  - Config shows max_length 512→384
  - Config shows AMP enabled in 302 but not 301
  - No major difference in concept distribution
- Fix: Restore config to 301 values
- Expected: F1 recovers without concept changes

**Hypothesis C: Top-50 codes are wrong**
- Evidence needed:
  - Codes differ between 301 and 302
  - OR counts differ significantly (>20%)
- Fix: Use 301's Top-50 codes
- Expected: F1 recovers with correct targets

**Hypothesis D: Multiple compounding issues**
- Evidence needed:
  - All of the above show problems
  - No single issue dominates
- Fix: Prioritize by impact, fix one at a time

## Step 3: ONE Controlled Experiment

**NOT doing**: Change 6 things at once (my mistake)

**DOING**: Change ONE variable

### Experiment Design:
1. **Baseline**: ShifaMind301 (Macro F1: 0.2801)
2. **Failed**: ShifaMind302 (Macro F1: 0.2524)
3. **Experiment**: 302 + ONE fix
4. **Measure**: Did F1 recover?

### If Primary Issue = UMLS Over-Extraction:
```
Experiment: ShifaMind302 with stricter UMLS filtering
- Keep: max_length=384 (don't change multiple things)
- Keep: 263 concepts vocabulary
- Change: ONLY add UMLS confidence threshold >0.7
- Expected concepts/sample: 30-40 (not 99.58, not 15-25 yet)
- Expected F1: 0.26-0.27 (partial recovery)
- Runtime: 6-8 hours

If this works → gradually tune threshold
If doesn't work → UMLS not the issue, try Hypothesis B
```

### If Primary Issue = Config Changes:
```
Experiment: ShifaMind302 with restored config
- Change: max_length back to 512
- Change: Disable AMP
- Keep: UMLS extraction unchanged (still 99.58 concepts)
- Expected F1: 0.26-0.27 (partial recovery)
- Runtime: 6-8 hours

If this works → config was the issue
If doesn't work → config not the issue, try Hypothesis A
```

## Step 4: Iterate Based on Results

### Scenario 1: Fix A works partially (F1: 0.25 → 0.27)
- Apply Fix B on top of Fix A
- Measure again
- Continue until F1 ≥ 0.28

### Scenario 2: Fix A doesn't work (F1 stays ~0.25)
- Revert Fix A
- Try Fix B
- Measure

### Scenario 3: Multiple fixes needed
- Fix them sequentially
- Measure impact of each
- Document what actually matters

## Why This Approach?

**Problem with my previous approach:**
- Changed 6 things at once
- If it works, don't know which fix mattered
- If it fails, don't know which change broke it
- No learning, just guessing

**Better approach:**
- Change one variable
- Measure impact
- Learn what actually affects performance
- Build understanding, not just solutions

## Questions to Answer With Evidence

1. **Is UMLS the problem?**
   - If we reduce concepts to 30/sample, does F1 recover?
   - Which concepts are actually predictive?

2. **Is max_length the problem?**
   - Are clinical notes being truncated at 384?
   - What % of notes are >384 tokens?

3. **Is AMP the problem?**
   - Does FP16 precision hurt concept prediction?
   - Can we measure precision loss?

4. **Are Top-50 codes correct?**
   - Do they match 301 exactly?
   - Are we targeting the right diagnoses?

5. **Is concept vocabulary the problem?**
   - Are 263 concepts too many?
   - Which of the 150 new concepts are useful?

## Timeline

1. **Rigorous Analysis**: 10-15 minutes (run script)
2. **Review Evidence**: 15-30 minutes (read output, make decision)
3. **ONE Experiment**: 6-8 hours (re-run Phase 1)
4. **Evaluate Results**: 15 minutes (did it work?)
5. **Iterate**: Repeat if needed

Total: 1-2 days to properly diagnose and fix

## Bottom Line

**I was wrong to make 6 changes without evidence.**

Let's do this properly:
1. Run `rigorous_analysis.py`
2. Review evidence together
3. Agree on PRIMARY issue
4. Fix ONLY that issue
5. Measure
6. Iterate

**No more guessing.**
