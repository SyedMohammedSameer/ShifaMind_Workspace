#!/usr/bin/env python3
import numpy as np
import pickle
from pathlib import Path
from scipy.stats import pearsonr

print("="*80)
print("PHASE 1 FAILURE ANALYSIS")
print("="*80)

BASE = Path('/content/drive/MyDrive/ShifaMind')
RUN_301 = BASE / '10_ShifaMind' / 'run_20260102_053548'
RUN_302 = BASE / '11_ShifaMind_v302' / 'run_20260201_173640'

# Load concepts
concepts_301 = np.load(RUN_301 / 'shared_data' / 'train_concept_labels.npy')
concepts_302 = np.load(RUN_302 / 'shared_data' / 'train_concept_labels.npy')

avg_301 = concepts_301.sum(axis=1).mean()
avg_302 = concepts_302.sum(axis=1).mean()

print(f"\n301: {avg_301:.1f} concepts/sample")
print(f"302: {avg_302:.1f} concepts/sample")
print(f"⚠️  302 extracts {avg_302/avg_301:.1f}x more concepts")

# Load diagnosis labels - just check what format it is
with open(RUN_302 / 'shared_data' / 'train_split.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"\nPickle data type: {type(data)}")
print(f"Pickle data keys/len: {data.keys() if isinstance(data, dict) else len(data) if hasattr(data, '__len__') else 'unknown'}")

if isinstance(data, dict):
    print(f"Dict keys: {list(data.keys())[:5]}")
    first_key = list(data.keys())[0]
    print(f"First item type: {type(data[first_key])}")
    print(f"First item: {data[first_key]}")
elif isinstance(data, list) and len(data) > 0:
    print(f"List first item type: {type(data[0])}")
    print(f"List first item: {data[0]}")
else:
    print(f"Data structure: {data}")

# Try to get labels
dx_labels = None

try:
    if isinstance(data, dict) and 'labels' in data:
        dx_labels = np.array(data['labels'])
    elif isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], dict) and 'labels' in data[0]:
            dx_labels = np.array([s['labels'] for s in data])
        elif hasattr(data[0], 'labels'):
            dx_labels = np.array([s.labels for s in data])

    if dx_labels is not None:
        print(f"✅ Loaded labels: {dx_labels.shape}")
    else:
        print("❌ Could not extract labels - showing structure")
        exit(0)
except Exception as e:
    print(f"❌ Error loading labels: {e}")
    exit(0)

# Correlation analysis
print(f"\nAnalyzing {concepts_302.shape[1]} concepts vs {dx_labels.shape[1]} diagnoses...")

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

weak = (concept_max_corr < 0.1).sum()
moderate = ((concept_max_corr >= 0.1) & (concept_max_corr < 0.2)).sum()
strong = (concept_max_corr >= 0.2).sum()

print(f"\nConcept predictive power:")
print(f"  Weak (<0.1):        {weak}/{len(concept_max_corr)} ({weak/len(concept_max_corr)*100:.1f}%)")
print(f"  Moderate (0.1-0.2): {moderate}/{len(concept_max_corr)} ({moderate/len(concept_max_corr)*100:.1f}%)")
print(f"  Strong (>0.2):      {strong}/{len(concept_max_corr)} ({strong/len(concept_max_corr)*100:.1f}%)")

# Simulation
print("\n" + "="*80)
print("FILTERING SIMULATION")
print("="*80)

for threshold in [0.05, 0.10, 0.15, 0.20]:
    keep_mask = concept_max_corr >= threshold
    kept = keep_mask.sum()
    avg_filtered = concepts_302[:, keep_mask].sum(axis=1).mean()
    print(f"Threshold >{threshold:.2f}: {kept:3d} concepts → {avg_filtered:.1f} per sample")

print(f"\n301 baseline: {avg_301:.1f} per sample")

# Find best threshold
best_threshold = None
best_diff = float('inf')
for threshold in [0.05, 0.08, 0.10, 0.12, 0.15]:
    keep_mask = concept_max_corr >= threshold
    avg_filtered = concepts_302[:, keep_mask].sum(axis=1).mean()
    diff = abs(avg_filtered - avg_301)
    if diff < best_diff:
        best_diff = diff
        best_threshold = threshold

kept_concepts = (concept_max_corr >= best_threshold).sum()
avg_filtered = concepts_302[:, concept_max_corr >= best_threshold].sum(axis=1).mean()

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print(f"\nFilter concepts by correlation >{best_threshold:.2f}")
print(f"→ Keeps {kept_concepts}/{len(concept_max_corr)} concepts ({kept_concepts/len(concept_max_corr)*100:.1f}%)")
print(f"→ Gets {avg_filtered:.1f} concepts/sample (target: {avg_301:.1f})")
print(f"→ Removes {weak} noise concepts")
print(f"\nExpected F1 recovery: 0.27-0.28")
print("="*80)
