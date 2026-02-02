# ShifaMind302 Phase 1 Fix

**Problem**: Phase 1 failed with -10% Macro F1 drop (0.28 → 0.25)

**Root causes**:
- GPU never used (you confirmed)
- AMP added without request
- max_length truncated 512→384
- UMLS over-extracted 99.58 concepts/sample
- Too many concepts (263 vs 113)

---

## Quick Start

### 1. Run Analysis (10 min)
```bash
python analyze_and_fix_phase1.py
```
Outputs: `/content/drive/MyDrive/ShifaMind/11_ShifaMind_v302/analysis/analysis_results.json`

### 2. Run Fixed Version (6-8 hrs)
Upload `shifamind302_phase1_FIXED.py` to Colab
- Ensure GPU enabled (Runtime → GPU)
- Run all cells
- Expected: Macro F1 ≥ 0.28

---

## Fixes Applied

| Issue | Fix |
|-------|-----|
| max_length 384 | → 512 |
| AMP enabled | → Disabled |
| 263 concepts | → 113 |
| Batch 16/32 | → 8/16 |
| UMLS no filter | → Strict (conf>0.7, semantic types) |
| No GPU check | → Verification added |

Expected: 15-25 concepts/sample (not 99.58), Macro F1 ≥ 0.28
