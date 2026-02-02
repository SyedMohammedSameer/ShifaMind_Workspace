# ShifaMind302 Phase 1 - Evidence-Based Fix

**Problem**: Phase 1 failed (-10% Macro F1: 0.28 → 0.25)

**Status**: Gathering evidence BEFORE making changes

---

## Step 1: Run Rigorous Analysis (DO THIS FIRST)

```bash
python rigorous_analysis.py
```

**Time**: 10-15 minutes

**Outputs**:
- Complete config comparison (proves if max_length, AMP differ)
- Deep concept analysis (proves if UMLS over-extracted)
- Concept-diagnosis correlation (proves which concepts are noise)
- Top-50 code verification (proves if codes match)
- Performance breakdown (proves where F1 dropped)

**Evidence saved**: `/content/drive/MyDrive/ShifaMind/11_ShifaMind_v302/analysis/rigorous_evidence.json`

---

## Step 2: Review Evidence & Decide

Read `rigorous_evidence.json` and identify PRIMARY issue:

- **Hypothesis A**: UMLS over-extraction (99.58 concepts/sample)
- **Hypothesis B**: Config changes (max_length, AMP)
- **Hypothesis C**: Top-50 codes wrong
- **Hypothesis D**: Multiple issues

See `fix_strategy.md` for decision tree.

---

## Step 3: ONE Controlled Experiment

Fix ONLY the primary issue, re-run Phase 1, measure.

**NOT doing**: Changing 6 things at once (can't isolate cause)

**DOING**: Change one variable, measure impact, iterate

---

## Current Files

- `rigorous_analysis.py` - Evidence gathering (run first)
- `fix_strategy.md` - Decision framework (read after analysis)
- `shifamind302_phase1_FIXED.py` - Proposed fix (DON'T use until evidence reviewed)

---

## Why This Approach?

Previous approach was rushed:
- Made 6 changes without evidence
- If it works, don't know which change mattered
- If it fails, don't know which change broke it

New approach:
- Gather evidence first
- Identify primary issue
- Fix one thing
- Measure impact
- Learn and iterate

**No more guessing. Evidence-based decisions only.**
