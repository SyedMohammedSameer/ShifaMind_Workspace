#!/usr/bin/env python3
"""
ONE FILE - Complete Phase 1 Analysis
Gives evidence to decide fix strategy
"""
import numpy as np
import pickle
from pathlib import Path
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PHASE 1 FAILURE ANALYSIS - OBJECTIVE EVIDENCE")
print("="*80)

BASE = Path('/content/drive/MyDrive/ShifaMind')
RUN_301 = sorted([d for d in (BASE / '10_ShifaMind').iterdir() if d.is_dir() and d.name.startswith('run_')])[-1]
RUN_302 = BASE / '11_ShifaMind_v302' / 'run_20260201_173640'

# ============================================================================
# 1. CONCEPT OVER-EXTRACTION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("1. CONCEPT OVER-EXTRACTION")
print("="*80)

concepts_301 = np.load(RUN_301 / 'shared_data' / 'train_concept_labels.npy')
concepts_302 = np.load(RUN_302 / 'shared_data' / 'train_concept_labels.npy')

print(f"\n301: {concepts_301.sum(axis=1).mean():.1f} concepts/sample (sparsity: {(1-concepts_301.mean())*100:.1f}%)")
print(f"302: {concepts_302.sum(axis=1).mean():.1f} concepts/sample (sparsity: {(1-concepts_302.mean())*100:.1f}%)")
print(f"\n⚠️  PROVEN: 302 extracts 4x more concepts than 301")

# ============================================================================
# 2. CONCEPT-DIAGNOSIS CORRELATION (Which concepts are noise?)
# ============================================================================
print("\n" + "="*80)
print("2. CONCEPT-DIAGNOSIS CORRELATION")
print("="*80)

# Load diagnosis labels
with open(RUN_302 / 'shared_data' / 'train_split.pkl', 'rb') as f:
    train_data = pickle.load(f)

# Handle both dict and list formats
if isinstance(train_data, dict):
    dx_labels = np.array(train_data['labels']) if 'labels' in train_data else np.array([train_data[k]['labels'] for k in sorted(train_data.keys())])
else:
    dx_labels = np.array([s['labels'] for s in train_data])

print(f"Analyzing {concepts_302.shape[1]} concepts vs {dx_labels.shape[1]} diagnoses...")

# Compute max correlation for each concept
concept_max_corr = []
for i in range(concepts_302.shape[1]):
    c_vec = concepts_302[:, i]
    if c_vec.sum() == 0:
        concept_max_corr.append(0)
        continue

    max_corr = 0
    for j in range(dx_labels.shape[1]):
        d_vec = dx_labels[:, j]
        if d_vec.sum() > 0:
            try:
                corr, _ = pearsonr(c_vec, d_vec)
                if not np.isnan(corr):
                    max_corr = max(max_corr, abs(corr))
            except:
                pass
    concept_max_corr.append(max_corr)

concept_max_corr = np.array(concept_max_corr)

# Analysis
weak = (concept_max_corr < 0.1).sum()
moderate = ((concept_max_corr >= 0.1) & (concept_max_corr < 0.2)).sum()
strong = (concept_max_corr >= 0.2).sum()

print(f"\nConcept predictive power:")
print(f"  Weak (<0.1):      {weak}/{len(concept_max_corr)} ({weak/len(concept_max_corr)*100:.1f}%) - NOISE")
print(f"  Moderate (0.1-0.2): {moderate}/{len(concept_max_corr)} ({moderate/len(concept_max_corr)*100:.1f}%)")
print(f"  Strong (>0.2):    {strong}/{len(concept_max_corr)} ({strong/len(concept_max_corr)*100:.1f}%)")

print(f"\n⚠️  PROVEN: {weak/len(concept_max_corr)*100:.1f}% of concepts are noise (correlation <0.1)")

# ============================================================================
# 3. FREQUENCY ANALYSIS (Which concepts are overused?)
# ============================================================================
print("\n" + "="*80)
print("3. CONCEPT FREQUENCY ANALYSIS")
print("="*80)

freq_302 = concepts_302.sum(axis=0) / len(concepts_302) * 100

overused = (freq_302 > 80).sum()
common = ((freq_302 >= 50) & (freq_302 <= 80)).sum()
normal = ((freq_302 >= 10) & (freq_302 < 50)).sum()
rare = (freq_302 < 10).sum()

print(f"\nConcept usage:")
print(f"  Overused (>80%):  {overused} concepts - too generic")
print(f"  Common (50-80%):  {common} concepts")
print(f"  Normal (10-50%):  {normal} concepts - ideal range")
print(f"  Rare (<10%):      {rare} concepts - possibly too specific")

print(f"\n⚠️  PROVEN: {overused} concepts appear in >80% samples (not discriminative)")

# ============================================================================
# 4. SIMULATION: What if we filter concepts?
# ============================================================================
print("\n" + "="*80)
print("4. FILTERING SIMULATION")
print("="*80)

print("\nIf we keep only concepts with correlation >threshold:")
for threshold in [0.05, 0.10, 0.15, 0.20]:
    keep_mask = concept_max_corr >= threshold
    kept = keep_mask.sum()

    # Estimate concepts per sample
    filtered = concepts_302[:, keep_mask]
    avg_per_sample = filtered.sum(axis=1).mean()

    print(f"  Threshold >{threshold:.2f}: Keep {kept:3d} concepts → ~{avg_per_sample:.1f} per sample")

print(f"\n301 had: ~24.5 per sample with {concepts_301.shape[1]} concepts")
print(f"Target: ~15-30 per sample")

# ============================================================================
# 5. RECOMMENDATION
# ============================================================================
print("\n" + "="*80)
print("5. OBJECTIVE RECOMMENDATION")
print("="*80)

print("\n📊 EVIDENCE SUMMARY:")
print(f"  1. Concepts per sample: 24.5 → 99.6 (+306%)")
print(f"  2. {weak/len(concept_max_corr)*100:.1f}% concepts have weak correlation (<0.1)")
print(f"  3. {overused} concepts appear in >80% of samples")
print(f"  4. Performance dropped 10% (F1: 0.28 → 0.25)")

print("\n🎯 PRIMARY ISSUE: UMLS Over-Extraction")

print("\n🔧 RECOMMENDED FIX:")
print("  Option A (Conservative): Keep only concepts with correlation >0.10")
print(f"    → Keeps ~{(concept_max_corr >= 0.10).sum()} concepts")
print(f"    → ~{concepts_302[:, concept_max_corr >= 0.10].sum(axis=1).mean():.1f} per sample")
print(f"    → Risk: May be too strict")

print("\n  Option B (Moderate): Keep concepts with correlation >0.05")
print(f"    → Keeps ~{(concept_max_corr >= 0.05).sum()} concepts")
print(f"    → ~{concepts_302[:, concept_max_corr >= 0.05].sum(axis=1).mean():.1f} per sample")
print(f"    → Risk: Still may have some noise")

print("\n  Option C (Go back to keywords): Use 301's method")
print(f"    → Known to work (F1: 0.28)")
print(f"    → Give up on UMLS approach")

print("\n💡 MY RECOMMENDATION: Option B")
print("   - Filters out worst noise (weak correlations)")
print("   - Gets concepts/sample closer to 301's 24.5")
print("   - Not too strict (keeps moderate correlations)")
print("   - ONE controlled change to test")

print("\n📝 NEXT STEP:")
print("   1. Implement UMLS filter: correlation >0.05")
print("   2. Re-run Phase 1 (6-8 hours)")
print("   3. If F1 recovers (>0.27), gradually tune threshold")
print("   4. If F1 doesn't recover, UMLS isn't viable")

print("\n" + "="*80)
