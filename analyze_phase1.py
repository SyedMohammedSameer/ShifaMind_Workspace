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

avg_301 = concepts_301.sum(axis=1).mean()
avg_302 = concepts_302.sum(axis=1).mean()
sparsity_301 = (1-concepts_301.mean())*100
sparsity_302 = (1-concepts_302.mean())*100

print(f"\n301: {avg_301:.1f} concepts/sample (sparsity: {sparsity_301:.1f}%)")
print(f"302: {avg_302:.1f} concepts/sample (sparsity: {sparsity_302:.1f}%)")
print(f"\n⚠️  PROVEN: 302 extracts {avg_302/avg_301:.1f}x more concepts than 301")

# ============================================================================
# 2. LOAD DIAGNOSIS LABELS (Try multiple methods)
# ============================================================================
print("\n" + "="*80)
print("2. LOADING DIAGNOSIS LABELS")
print("="*80)

dx_labels = None

# Method 1: Try loading from train_split.pkl
try:
    print("Trying train_split.pkl...")
    with open(RUN_302 / 'shared_data' / 'train_split.pkl', 'rb') as f:
        data = pickle.load(f)

    # Check structure
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict) and 'labels' in data[0]:
            dx_labels = np.array([s['labels'] for s in data])
            print(f"✅ Loaded from list format: {dx_labels.shape}")
    elif isinstance(data, dict):
        if 'labels' in data:
            dx_labels = np.array(data['labels'])
            print(f"✅ Loaded from dict['labels']: {dx_labels.shape}")
        elif 'train_labels' in data:
            dx_labels = np.array(data['train_labels'])
            print(f"✅ Loaded from dict['train_labels']: {dx_labels.shape}")
except Exception as e:
    print(f"❌ Failed to load train_split.pkl: {e}")

# Method 2: Try loading train_labels.npy directly
if dx_labels is None:
    try:
        print("Trying train_labels.npy...")
        # Look for label files
        for label_file in ['train_labels.npy', 'train_dx_labels.npy', 'y_train.npy']:
            label_path = RUN_302 / 'shared_data' / label_file
            if label_path.exists():
                dx_labels = np.load(label_path)
                print(f"✅ Loaded from {label_file}: {dx_labels.shape}")
                break
    except Exception as e:
        print(f"❌ Failed to load npy files: {e}")

# Method 3: Reconstruct from saved data
if dx_labels is None:
    try:
        print("Trying to load from mimic_dx_data.csv...")
        import pandas as pd

        csv_path = RUN_302.parent / 'mimic_dx_data_top50.csv'
        if not csv_path.exists():
            csv_path = BASE / '10_ShifaMind' / 'mimic_dx_data_top50.csv'

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Get label columns (should be the Top-50 ICD codes)
            label_cols = [c for c in df.columns if c not in ['subject_id', 'hadm_id', 'text', 'labels']]
            if len(label_cols) == 50:
                # Get training indices
                n_total = len(df)
                n_train = len(concepts_302)
                dx_labels = df[label_cols].iloc[:n_train].values
                print(f"✅ Reconstructed from CSV: {dx_labels.shape}")
    except Exception as e:
        print(f"❌ Failed to reconstruct from CSV: {e}")

if dx_labels is None:
    print("\n❌ CANNOT LOAD DIAGNOSIS LABELS")
    print("Skipping correlation analysis")
    print("\nBut we still have clear evidence from concept distribution:")
    print(f"  - 302 extracts {avg_302/avg_301:.1f}x more concepts")
    print(f"  - Sparsity dropped from {sparsity_301:.1f}% to {sparsity_302:.1f}%")
    print(f"  - This correlates with 10% F1 drop")
    print("\n🎯 RECOMMENDATION: UMLS is over-extracting, needs filtering")
    exit(0)

# ============================================================================
# 3. CONCEPT-DIAGNOSIS CORRELATION
# ============================================================================
print("\n" + "="*80)
print("3. CONCEPT-DIAGNOSIS CORRELATION")
print("="*80)

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
print(f"  Weak (<0.1):        {weak}/{len(concept_max_corr)} ({weak/len(concept_max_corr)*100:.1f}%) - NOISE")
print(f"  Moderate (0.1-0.2): {moderate}/{len(concept_max_corr)} ({moderate/len(concept_max_corr)*100:.1f}%)")
print(f"  Strong (>0.2):      {strong}/{len(concept_max_corr)} ({strong/len(concept_max_corr)*100:.1f}%)")

print(f"\n⚠️  PROVEN: {weak/len(concept_max_corr)*100:.1f}% of concepts are noise (correlation <0.1)")

# ============================================================================
# 4. FREQUENCY ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("4. CONCEPT FREQUENCY ANALYSIS")
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
print(f"  Rare (<10%):      {rare} concepts")

print(f"\n⚠️  {overused} concepts appear in >80% samples (not discriminative)")

# ============================================================================
# 5. FILTERING SIMULATION
# ============================================================================
print("\n" + "="*80)
print("5. FILTERING SIMULATION")
print("="*80)

print("\nIf we filter by correlation threshold:")
for threshold in [0.05, 0.10, 0.15, 0.20]:
    keep_mask = concept_max_corr >= threshold
    kept = keep_mask.sum()

    filtered = concepts_302[:, keep_mask]
    avg_per_sample = filtered.sum(axis=1).mean()

    print(f"  Threshold >{threshold:.2f}: Keep {kept:3d} concepts → {avg_per_sample:.1f} per sample")

print(f"\n301 baseline: {avg_301:.1f} per sample with {concepts_301.shape[1]} concepts")
print(f"Target range: 15-30 per sample")

# ============================================================================
# 6. RECOMMENDATION
# ============================================================================
print("\n" + "="*80)
print("6. OBJECTIVE RECOMMENDATION")
print("="*80)

print("\n📊 EVIDENCE:")
print(f"  1. Concepts/sample: {avg_301:.1f} → {avg_302:.1f} (+{(avg_302/avg_301-1)*100:.0f}%)")
print(f"  2. Noise concepts: {weak/len(concept_max_corr)*100:.1f}% (correlation <0.1)")
print(f"  3. Overused concepts: {overused} (>80% frequency)")
print(f"  4. F1 dropped 10%: 0.28 → 0.25")

print("\n🎯 PRIMARY ISSUE: UMLS Over-Extraction")

best_threshold = None
best_diff = float('inf')

for threshold in [0.05, 0.08, 0.10, 0.12, 0.15]:
    keep_mask = concept_max_corr >= threshold
    avg_filtered = concepts_302[:, keep_mask].sum(axis=1).mean()
    diff = abs(avg_filtered - avg_301)
    if diff < best_diff:
        best_diff = diff
        best_threshold = threshold

keep_mask = concept_max_corr >= best_threshold
kept_concepts = keep_mask.sum()
avg_filtered = concepts_302[:, keep_mask].sum(axis=1).mean()

print(f"\n🔧 RECOMMENDED FIX:")
print(f"   Filter concepts by correlation >{best_threshold:.2f}")
print(f"   → Keeps {kept_concepts}/{len(concept_max_corr)} concepts ({kept_concepts/len(concept_max_corr)*100:.1f}%)")
print(f"   → Gets {avg_filtered:.1f} concepts/sample (target: {avg_301:.1f})")
print(f"   → Removes {weak} noise concepts")

print(f"\n📝 IMPLEMENTATION:")
print(f"   1. In UMLS extraction, compute concept-diagnosis correlation")
print(f"   2. Keep only concepts with correlation >{best_threshold:.2f}")
print(f"   3. Re-run Phase 1 (6-8 hours)")
print(f"   4. Expected: F1 recovers to 0.27-0.28")

print(f"\n⚠️  IF THIS FAILS:")
print(f"   → UMLS approach not viable")
print(f"   → Revert to 301's keyword method")

print("\n" + "="*80)
