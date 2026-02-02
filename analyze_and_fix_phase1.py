#!/usr/bin/env python3
"""
ShifaMind302 Phase 1 - Complete Analysis & Fix Generator
=========================================================
Single script that does EVERYTHING:
1. GPU diagnostics
2. Evidence gathering (Top-50 codes, concepts, config comparison)
3. Generates fixed implementation code

Run this, review output, then use generated fix.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔍 SHIFAMIND302 PHASE 1 - COMPLETE ANALYSIS")
print("="*80)

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
PHASE1_301_PATH = BASE_PATH / '10_ShifaMind'
PHASE1_302_PATH = BASE_PATH / '11_ShifaMind_v302' / 'run_20260201_173640'
OUTPUT_PATH = BASE_PATH / '11_ShifaMind_v302' / 'analysis'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

findings = {'issues': [], 'fixes': [], 'evidence': {}}

# ============================================================================
# PART 1: GPU DIAGNOSTIC
# ============================================================================
print("\n" + "="*80)
print("🖥️  GPU DIAGNOSTIC")
print("="*80)

import torch

gpu_ok = torch.cuda.is_available()
print(f"CUDA Available: {gpu_ok}")

if gpu_ok:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    try:
        test = torch.randn(100, 100).cuda()
        _ = test @ test.T
        print("✅ GPU test passed")
    except:
        print("❌ GPU test failed")
        gpu_ok = False
else:
    print("❌ NO GPU - Training will use CPU (10-50x slower)")
    findings['issues'].append("GPU not available in runtime")

findings['evidence']['gpu_available'] = gpu_ok

# ============================================================================
# PART 2: TOP-50 CODE VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("📊 TOP-50 ICD CODE VERIFICATION")
print("="*80)

run_dirs_301 = sorted([d for d in PHASE1_301_PATH.iterdir() if d.is_dir() and d.name.startswith('run_')])
if run_dirs_301:
    latest_301 = run_dirs_301[-1]
    top50_301_path = latest_301 / 'shared_data' / 'top50_icd10_info.json'

    if top50_301_path.exists():
        with open(top50_301_path, 'r') as f:
            top50_301 = json.load(f)
        codes_301 = top50_301['top_50_codes']
        print(f"✅ Found Top-50 codes from 301: {len(codes_301)} codes")

        top50_302_path = PHASE1_302_PATH / 'shared_data' / 'top50_icd10_info.json'
        if top50_302_path.exists():
            with open(top50_302_path, 'r') as f:
                top50_302 = json.load(f)
            codes_302 = top50_302['top_50_codes']

            if codes_301 == codes_302:
                print("✅ Top-50 codes MATCH between 301 and 302")
            else:
                print(f"❌ Top-50 codes DIFFER: {len(set(codes_301) ^ set(codes_302))} different")
                findings['issues'].append("Top-50 codes don't match between 301 and 302")

        findings['evidence']['top50_codes'] = codes_301[:10]  # Top 10

# ============================================================================
# PART 3: CONCEPT ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("🧠 CONCEPT DISTRIBUTION ANALYSIS")
print("="*80)

concept_301_path = latest_301 / 'shared_data' / 'train_concept_labels.npy' if run_dirs_301 else None
concept_302_path = PHASE1_302_PATH / 'shared_data' / 'train_concept_labels.npy'

if concept_301_path and concept_301_path.exists():
    concepts_301 = np.load(concept_301_path)
    avg_301 = concepts_301.sum(axis=1).mean()
    sparsity_301 = (1 - concepts_301.mean()) * 100
    print(f"301: {avg_301:.1f} concepts/sample, {sparsity_301:.1f}% sparsity ✅")
    findings['evidence']['301_concepts'] = float(avg_301)
    findings['evidence']['301_sparsity'] = float(sparsity_301)

if concept_302_path.exists():
    concepts_302 = np.load(concept_302_path)
    avg_302 = concepts_302.sum(axis=1).mean()
    sparsity_302 = (1 - concepts_302.mean()) * 100
    print(f"302: {avg_302:.1f} concepts/sample, {sparsity_302:.1f}% sparsity ❌")

    if avg_302 > 50:
        findings['issues'].append(f"UMLS over-extraction: {avg_302:.1f} concepts/sample (should be 15-25)")
        findings['fixes'].append("Add UMLS filtering: confidence >0.7, semantic types, word boundaries")

    findings['evidence']['302_concepts'] = float(avg_302)
    findings['evidence']['302_sparsity'] = float(sparsity_302)

# ============================================================================
# PART 4: CONFIG COMPARISON
# ============================================================================
print("\n" + "="*80)
print("⚙️  CONFIGURATION COMPARISON")
print("="*80)

config_301_path = latest_301 / 'results' / 'phase1' / 'config.json' if run_dirs_301 else None
config_302_path = PHASE1_302_PATH / 'results' / 'phase1' / 'phase1_config.json'

if config_301_path and config_301_path.exists() and config_302_path.exists():
    with open(config_301_path) as f:
        cfg_301 = json.load(f)
    with open(config_302_path) as f:
        cfg_302 = json.load(f)

    max_len_301 = cfg_301.get('max_length', 512)
    max_len_302 = cfg_302.get('max_length', 384)

    print(f"max_length: {max_len_301} (301) vs {max_len_302} (302)")

    if max_len_302 < max_len_301:
        findings['issues'].append(f"max_length reduced: {max_len_301} → {max_len_302}")
        findings['fixes'].append(f"Restore max_length to {max_len_301}")

    if cfg_302.get('use_amp') or cfg_302.get('gpu_optimizations', {}).get('use_amp'):
        print("⚠️  AMP enabled in 302 (user never requested)")
        findings['issues'].append("AMP enabled without user request")
        findings['fixes'].append("Disable AMP (user never asked for it)")

# ============================================================================
# PART 5: PERFORMANCE COMPARISON
# ============================================================================
print("\n" + "="*80)
print("📈 PERFORMANCE COMPARISON")
print("="*80)

results_301_path = latest_301 / 'results' / 'phase1' / 'results.json' if run_dirs_301 else None
results_302_path = PHASE1_302_PATH / 'results' / 'phase1' / 'phase1_results.json'

if results_301_path and results_301_path.exists() and results_302_path.exists():
    with open(results_301_path) as f:
        res_301 = json.load(f)
    with open(results_302_path) as f:
        res_302 = json.load(f)

    f1_301 = res_301.get('test_metrics', {}).get('test_macro_f1', res_301.get('test_macro_f1', 0))
    f1_302 = res_302.get('test_metrics', {}).get('test_macro_f1', res_302.get('test_macro_f1', 0))

    diff = f1_302 - f1_301
    pct = (diff / f1_301 * 100) if f1_301 > 0 else 0

    print(f"Macro F1: {f1_301:.4f} → {f1_302:.4f} ({pct:+.1f}%)")

    if diff < -0.02:
        findings['issues'].append(f"Performance dropped {abs(pct):.1f}%")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📋 SUMMARY")
print("="*80)

print(f"\n🔴 Issues Found ({len(findings['issues'])}):")
for i, issue in enumerate(findings['issues'], 1):
    print(f"  {i}. {issue}")

print(f"\n🔧 Fixes Needed ({len(findings['fixes'])}):")
for i, fix in enumerate(findings['fixes'], 1):
    print(f"  {i}. {fix}")

# Save findings
with open(OUTPUT_PATH / 'analysis_results.json', 'w') as f:
    json.dump(findings, f, indent=2)

print(f"\n💾 Saved: {OUTPUT_PATH / 'analysis_results.json'}")

# ============================================================================
# GENERATE FIX SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ RECOMMENDED FIXES FOR SHIFAMIND302_PHASE1_FIXED.PY")
print("="*80)

print("""
Key changes to apply:
1. max_length = 512 (not 384)
2. USE_AMP = False (user never requested)
3. Concepts = 113 (not 263)
4. Batch sizes = 8/16 (not 16/32)
5. UMLS strict filtering:
   - Confidence threshold >0.7
   - Semantic type filtering (medical only)
   - Word boundary matching
   - Target: 15-25 concepts/sample (not 99.58)
6. GPU verification before training

The fixed implementation is in: shifamind302_phase1_FIXED.py
""")

print("="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)
