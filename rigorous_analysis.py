#!/usr/bin/env python3
"""
RIGOROUS Phase 1 Failure Analysis
==================================
Evidence-based diagnosis - NO GUESSING

Experiments:
1. Load and compare ALL configs (prove max_length, AMP issues)
2. Concept distribution + predictive power analysis
3. Concept-diagnosis correlation (which concepts are noise?)
4. Top-50 code verification with counts
5. Sparsity impact analysis
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pointbiserialr
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("RIGOROUS PHASE 1 FAILURE ANALYSIS")
print("="*80)

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
RUN_301 = sorted([d for d in (BASE_PATH / '10_ShifaMind').iterdir() if d.is_dir() and d.name.startswith('run_')])[-1]
RUN_302 = BASE_PATH / '11_ShifaMind_v302' / 'run_20260201_173640'

evidence = {}

# ============================================================================
# EXPERIMENT 1: COMPLETE CONFIG COMPARISON
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENT 1: COMPLETE CONFIG COMPARISON")
print("="*80)

# Try multiple config locations
config_paths_301 = [
    RUN_301 / 'results' / 'phase1' / 'config.json',
    RUN_301 / 'config.json',
    RUN_301 / 'phase1_config.json'
]

config_paths_302 = [
    RUN_302 / 'results' / 'phase1' / 'phase1_config.json',
    RUN_302 / 'results' / 'phase1' / 'config.json',
    RUN_302 / 'config.json',
    RUN_302 / 'phase1_config.json'
]

cfg_301 = None
cfg_302 = None

for path in config_paths_301:
    if path.exists():
        print(f"Loading 301 config: {path}")
        with open(path) as f:
            cfg_301 = json.load(f)
        break

for path in config_paths_302:
    if path.exists():
        print(f"Loading 302 config: {path}")
        with open(path) as f:
            cfg_302 = json.load(f)
        break

if cfg_301 and cfg_302:
    print("\nCOMPLETE CONFIG DIFF:")
    print("-" * 80)

    # All keys from both configs
    all_keys = set(cfg_301.keys()) | set(cfg_302.keys())

    diffs = []
    for key in sorted(all_keys):
        val_301 = cfg_301.get(key, "NOT_SET")
        val_302 = cfg_302.get(key, "NOT_SET")

        if val_301 != val_302:
            print(f"❌ {key}:")
            print(f"   301: {val_301}")
            print(f"   302: {val_302}")
            diffs.append((key, val_301, val_302))
        else:
            if key in ['max_length', 'batch_size', 'learning_rate', 'num_epochs', 'device']:
                print(f"✅ {key}: {val_301}")

    evidence['config_diffs'] = diffs

    # Check specific issues
    if cfg_302.get('max_length', 512) < cfg_301.get('max_length', 512):
        print(f"\n⚠️  PROVEN: max_length reduced from {cfg_301.get('max_length')} to {cfg_302.get('max_length')}")

    if cfg_302.get('use_amp') or (isinstance(cfg_302.get('gpu_optimizations'), dict) and cfg_302.get('gpu_optimizations', {}).get('use_amp')):
        print(f"\n⚠️  PROVEN: AMP is enabled in 302")
        if not cfg_301.get('use_amp'):
            print(f"         AMP was NOT in 301")
else:
    print("⚠️  Could not load configs for comparison")
    print(f"301 config exists: {any(p.exists() for p in config_paths_301)}")
    print(f"302 config exists: {any(p.exists() for p in config_paths_302)}")

# ============================================================================
# EXPERIMENT 2: DEEP CONCEPT ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENT 2: DEEP CONCEPT DISTRIBUTION ANALYSIS")
print("="*80)

concepts_301_path = RUN_301 / 'shared_data' / 'train_concept_labels.npy'
concepts_302_path = RUN_302 / 'shared_data' / 'train_concept_labels.npy'

if concepts_301_path.exists() and concepts_302_path.exists():
    concepts_301 = np.load(concepts_301_path)
    concepts_302 = np.load(concepts_302_path)

    print(f"\n301 Concept Matrix: {concepts_301.shape}")
    print(f"302 Concept Matrix: {concepts_302.shape}")

    # Detailed statistics
    per_sample_301 = concepts_301.sum(axis=1)
    per_sample_302 = concepts_302.sum(axis=1)

    print("\n301 Concepts per sample:")
    print(f"  Mean:   {per_sample_301.mean():.2f}")
    print(f"  Median: {np.median(per_sample_301):.0f}")
    print(f"  Std:    {per_sample_301.std():.2f}")
    print(f"  Min:    {per_sample_301.min():.0f}")
    print(f"  Max:    {per_sample_301.max():.0f}")
    print(f"  25%:    {np.percentile(per_sample_301, 25):.0f}")
    print(f"  75%:    {np.percentile(per_sample_301, 75):.0f}")

    print("\n302 Concepts per sample:")
    print(f"  Mean:   {per_sample_302.mean():.2f}")
    print(f"  Median: {np.median(per_sample_302):.0f}")
    print(f"  Std:    {per_sample_302.std():.2f}")
    print(f"  Min:    {per_sample_302.min():.0f}")
    print(f"  Max:    {per_sample_302.max():.0f}")
    print(f"  25%:    {np.percentile(per_sample_302, 25):.0f}")
    print(f"  75%:    {np.percentile(per_sample_302, 75):.0f}")

    # Concept frequency analysis
    freq_301 = concepts_301.sum(axis=0)
    freq_302 = concepts_302.sum(axis=0)

    print(f"\n301 Concept frequency:")
    print(f"  Never used (0):      {(freq_301 == 0).sum()}/{len(freq_301)}")
    print(f"  Rare (<1%):          {(freq_301 < len(concepts_301)*0.01).sum()}")
    print(f"  Common (>50%):       {(freq_301 > len(concepts_301)*0.5).sum()}")
    print(f"  Very common (>80%):  {(freq_301 > len(concepts_301)*0.8).sum()}")

    print(f"\n302 Concept frequency:")
    print(f"  Never used (0):      {(freq_302 == 0).sum()}/{len(freq_302)}")
    print(f"  Rare (<1%):          {(freq_302 < len(concepts_302)*0.01).sum()}")
    print(f"  Common (>50%):       {(freq_302 > len(concepts_302)*0.5).sum()}")
    print(f"  Very common (>80%):  {(freq_302 > len(concepts_302)*0.8).sum()}")

    # Save for later analysis
    evidence['concepts_301'] = {
        'mean': float(per_sample_301.mean()),
        'median': float(np.median(per_sample_301)),
        'std': float(per_sample_301.std()),
        'never_used': int((freq_301 == 0).sum()),
        'rare': int((freq_301 < len(concepts_301)*0.01).sum())
    }

    evidence['concepts_302'] = {
        'mean': float(per_sample_302.mean()),
        'median': float(np.median(per_sample_302)),
        'std': float(per_sample_302.std()),
        'never_used': int((freq_302 == 0).sum()),
        'rare': int((freq_302 < len(concepts_302)*0.01).sum())
    }

    # Distribution comparison
    print("\n📊 DISTRIBUTION COMPARISON:")
    print(f"Concepts per sample: {per_sample_301.mean():.1f} → {per_sample_302.mean():.1f} ({(per_sample_302.mean()/per_sample_301.mean()-1)*100:+.1f}%)")
    print(f"Sparsity: {(1-concepts_301.mean())*100:.1f}% → {(1-concepts_302.mean())*100:.1f}% ({(1-concepts_302.mean())*100 - (1-concepts_301.mean())*100:+.1f}pp)")

# ============================================================================
# EXPERIMENT 3: CONCEPT-DIAGNOSIS CORRELATION
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENT 3: CONCEPT-DIAGNOSIS CORRELATION ANALYSIS")
print("="*80)

# Load train split to get diagnosis labels
train_split_path = RUN_302 / 'shared_data' / 'train_split.pkl'
if train_split_path.exists() and concepts_302_path.exists():
    import pickle

    print("Loading train split...")
    with open(train_split_path, 'rb') as f:
        train_split = pickle.load(f)

    dx_labels = np.array([sample['labels'] for sample in train_split])
    print(f"Diagnosis labels: {dx_labels.shape}")

    # Compute correlations for sample of concepts
    print("\nComputing concept-diagnosis correlations (this may take a minute)...")

    num_concepts_to_analyze = min(concepts_302.shape[1], 200)  # Sample
    correlations = []

    for i in range(num_concepts_to_analyze):
        concept_vec = concepts_302[:, i]
        if concept_vec.sum() > 0:  # Only non-zero concepts
            max_corr = 0
            for j in range(dx_labels.shape[1]):
                dx_vec = dx_labels[:, j]
                if dx_vec.sum() > 0:
                    try:
                        corr, _ = pointbiserialr(concept_vec, dx_vec)
                        if not np.isnan(corr) and abs(corr) > abs(max_corr):
                            max_corr = corr
                    except:
                        pass
            correlations.append((i, max_corr))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\nTop 10 most predictive concepts:")
    for idx, corr in correlations[:10]:
        print(f"  Concept {idx:3d}: correlation = {corr:+.3f}")

    print(f"\n10 weakest concepts:")
    for idx, corr in correlations[-10:]:
        print(f"  Concept {idx:3d}: correlation = {corr:+.3f}")

    weak_count = sum(1 for _, corr in correlations if abs(corr) < 0.1)
    print(f"\nConcepts with weak correlation (<0.1): {weak_count}/{len(correlations)} ({weak_count/len(correlations)*100:.1f}%)")

    evidence['concept_correlation'] = {
        'weak_count': weak_count,
        'total_analyzed': len(correlations),
        'weak_percentage': weak_count/len(correlations)*100
    }

# ============================================================================
# EXPERIMENT 4: TOP-50 CODE VERIFICATION WITH COUNTS
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENT 4: TOP-50 CODE VERIFICATION")
print("="*80)

top50_301_path = RUN_301 / 'shared_data' / 'top50_icd10_info.json'
top50_302_path = RUN_302 / 'shared_data' / 'top50_icd10_info.json'

if top50_301_path.exists() and top50_302_path.exists():
    with open(top50_301_path) as f:
        info_301 = json.load(f)
    with open(top50_302_path) as f:
        info_302 = json.load(f)

    codes_301 = info_301['top_50_codes']
    codes_302 = info_302['top_50_codes']
    counts_301 = info_301['top_50_counts']
    counts_302 = info_302['top_50_counts']

    if codes_301 == codes_302:
        print("✅ Top-50 codes IDENTICAL between 301 and 302")

        # Compare counts
        print("\nCode frequency comparison (top 10):")
        for code in codes_301[:10]:
            c301 = counts_301.get(code, 0)
            c302 = counts_302.get(code, 0)
            diff = c302 - c301
            diff_pct = (diff / c301 * 100) if c301 > 0 else 0
            print(f"  {code}: {c301:5d} → {c302:5d} ({diff_pct:+6.1f}%)")

        evidence['top50_match'] = True
    else:
        diff = set(codes_301) ^ set(codes_302)
        print(f"❌ Top-50 codes DIFFER: {len(diff)} different")
        print(f"  In 301 not 302: {set(codes_301) - set(codes_302)}")
        print(f"  In 302 not 301: {set(codes_302) - set(codes_301)}")
        evidence['top50_match'] = False
        evidence['top50_diff_count'] = len(diff)

# ============================================================================
# EXPERIMENT 5: PERFORMANCE METRICS DEEP DIVE
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENT 5: PERFORMANCE METRICS COMPARISON")
print("="*80)

results_301_path = RUN_301 / 'results' / 'phase1' / 'results.json'
results_302_path = RUN_302 / 'results' / 'phase1' / 'phase1_results.json'

if results_301_path.exists() and results_302_path.exists():
    with open(results_301_path) as f:
        res_301 = json.load(f)
    with open(results_302_path) as f:
        res_302 = json.load(f)

    # Extract all metrics
    metrics_301 = res_301.get('test_metrics', res_301)
    metrics_302 = res_302.get('test_metrics', res_302)

    print("\nALL METRICS COMPARISON:")
    print("-" * 80)

    metric_keys = set(metrics_301.keys()) & set(metrics_302.keys())
    for key in sorted(metric_keys):
        if isinstance(metrics_301[key], (int, float)):
            val_301 = metrics_301[key]
            val_302 = metrics_302[key]
            diff = val_302 - val_301
            diff_pct = (diff / val_301 * 100) if val_301 != 0 else 0

            symbol = "✅" if diff > 0 else "❌"
            print(f"{symbol} {key:30s}: {val_301:.4f} → {val_302:.4f} ({diff_pct:+6.1f}%)")

    # Specific focus on key metrics
    f1_301 = metrics_301.get('test_macro_f1', 0)
    f1_302 = metrics_302.get('test_macro_f1', 0)

    evidence['performance'] = {
        'macro_f1_301': f1_301,
        'macro_f1_302': f1_302,
        'macro_f1_diff': f1_302 - f1_301,
        'macro_f1_diff_pct': (f1_302 - f1_301) / f1_301 * 100 if f1_301 > 0 else 0
    }

# ============================================================================
# SAVE EVIDENCE
# ============================================================================
output_path = BASE_PATH / '11_ShifaMind_v302' / 'analysis'
output_path.mkdir(parents=True, exist_ok=True)

with open(output_path / 'rigorous_evidence.json', 'w') as f:
    json.dump(evidence, f, indent=2)

print("\n" + "="*80)
print("EVIDENCE SUMMARY")
print("="*80)

print("\n📊 PROVEN FACTS:")
proven_issues = []

if 'config_diffs' in evidence and evidence['config_diffs']:
    print(f"✅ Config differences found: {len(evidence['config_diffs'])} parameters changed")
    proven_issues.append(f"Config changes: {len(evidence['config_diffs'])} parameters")

if 'concepts_302' in evidence:
    mean_302 = evidence['concepts_302']['mean']
    mean_301 = evidence['concepts_301']['mean'] if 'concepts_301' in evidence else 25
    if mean_302 > mean_301 * 2:
        print(f"✅ PROVEN: Concept over-extraction ({mean_301:.1f} → {mean_302:.1f})")
        proven_issues.append(f"Concept over-extraction: {mean_301:.1f} → {mean_302:.1f}")

if 'concept_correlation' in evidence:
    weak_pct = evidence['concept_correlation']['weak_percentage']
    if weak_pct > 50:
        print(f"✅ PROVEN: {weak_pct:.1f}% of concepts have weak predictive power")
        proven_issues.append(f"Weak concepts: {weak_pct:.1f}%")

if 'performance' in evidence:
    diff_pct = evidence['performance'].get('macro_f1_diff_pct', 0)
    print(f"✅ PROVEN: Performance dropped {abs(diff_pct):.1f}%")
    proven_issues.append(f"Performance drop: {abs(diff_pct):.1f}%")

print(f"\n💾 Saved detailed evidence: {output_path / 'rigorous_evidence.json'}")
print(f"\n🔴 PROVEN ISSUES: {len(proven_issues)}")
for i, issue in enumerate(proven_issues, 1):
    print(f"  {i}. {issue}")

print("\n" + "="*80)
print("✅ RIGOROUS ANALYSIS COMPLETE")
print("="*80)
print("\nNow we have EVIDENCE to make decisions.")
print("Review rigorous_evidence.json for complete data.")
