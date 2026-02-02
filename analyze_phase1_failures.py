#!/usr/bin/env python3
"""
AUTOMATED EVIDENCE GATHERING: Phase 1 Failure Analysis
=======================================================

This script automatically investigates why ShifaMind302 Phase 1 performed worse
than ShifaMind301 and provides evidence-based recommendations.

CRITICAL ISSUES TO INVESTIGATE:
1. GPU was never used during training (most critical!)
2. AMP added without request
3. max_length truncated from 512→384
4. UMLS over-extraction (99.58 concepts/sample, 38% sparsity)
5. Top-50 ICD codes verification

NO USER INTERVENTION REQUIRED - Fully automated analysis
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
PHASE1_301_PATH = BASE_PATH / '10_ShifaMind'  # ShifaMind301 runs
PHASE1_302_PATH = BASE_PATH / '11_ShifaMind_v302' / 'run_20260201_173640'  # Failed run
ANALYSIS_OUTPUT = BASE_PATH / '11_ShifaMind_v302' / 'analysis_evidence'
ANALYSIS_OUTPUT.mkdir(parents=True, exist_ok=True)

print("="*80)
print("🔍 AUTOMATED PHASE 1 FAILURE ANALYSIS")
print("="*80)
print(f"📁 Output: {ANALYSIS_OUTPUT}")
print()

# Store all findings
findings = {
    'critical_issues': [],
    'recommendations': [],
    'evidence': {}
}

# ============================================================================
# INVESTIGATION 1: GPU USAGE CHECK (MOST CRITICAL)
# ============================================================================
print("\n" + "="*80)
print("🖥️  INVESTIGATION 1: GPU USAGE VERIFICATION (CRITICAL)")
print("="*80)

gpu_evidence = {
    'gpu_detected': False,
    'gpu_used_in_training': False,
    'evidence_files': []
}

# Check training logs for GPU usage patterns
log_files = list(PHASE1_302_PATH.glob('logs/**/*.log'))
if not log_files:
    log_files = list(PHASE1_302_PATH.glob('**/*.log'))

print(f"📄 Found {len(log_files)} log files")

gpu_patterns = [
    'cuda',
    'GPU',
    'gpu',
    'Device:',
    'device:',
    'CUDA',
]

training_speed_indicators = []

for log_file in log_files:
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

            # Check for GPU detection
            if any(pattern in content for pattern in gpu_patterns):
                gpu_evidence['evidence_files'].append(str(log_file))

                # Look for device info
                for line in content.split('\n'):
                    if 'Device:' in line or 'device:' in line:
                        print(f"   Found: {line.strip()}")
                        if 'cuda' in line.lower():
                            gpu_evidence['gpu_detected'] = True
                        else:
                            findings['critical_issues'].append(
                                "❌ CRITICAL: Device was set to 'cpu' not 'cuda'"
                            )

                    # Check training speed (GPU trains much faster)
                    if 'epoch' in line.lower() and 'time' in line.lower():
                        training_speed_indicators.append(line)
    except Exception as e:
        print(f"   ⚠️  Could not read {log_file}: {e}")

# Check config files for GPU settings
config_files = list(PHASE1_302_PATH.glob('**/config.json'))
config_files.extend(list(PHASE1_302_PATH.glob('**/phase1_config.json')))

for config_file in config_files:
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            if 'device' in config:
                print(f"\n📋 Config device: {config['device']}")
                gpu_evidence['device_in_config'] = config['device']
            if 'gpu_optimizations' in config:
                print(f"   GPU optimizations: {config['gpu_optimizations']}")
    except Exception as e:
        print(f"   ⚠️  Could not read {config_file}: {e}")

# Analyze training speed
if training_speed_indicators:
    print(f"\n⏱️  Training speed samples:")
    for indicator in training_speed_indicators[:3]:
        print(f"   {indicator.strip()}")
else:
    findings['critical_issues'].append(
        "❌ CRITICAL: No training time logs found - cannot verify GPU usage"
    )

if not gpu_evidence['gpu_detected']:
    findings['critical_issues'].append(
        "❌ CRITICAL: GPU was NOT detected during training! Code shows 'cuda' but runtime had no GPU."
    )
    findings['recommendations'].append(
        "🔧 FIX: Ensure Google Colab runtime is set to GPU (Runtime → Change runtime type → T4/L4 GPU)"
    )
else:
    print("✅ GPU was detected in logs")

findings['evidence']['gpu_investigation'] = gpu_evidence

# ============================================================================
# INVESTIGATION 2: TOP-50 ICD CODE VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("📊 INVESTIGATION 2: TOP-50 ICD CODE VERIFICATION")
print("="*80)

# Find most recent 301 run
run_dirs_301 = sorted([d for d in PHASE1_301_PATH.iterdir() if d.is_dir() and d.name.startswith('run_')])
if run_dirs_301:
    latest_301_run = run_dirs_301[-1]
    print(f"📁 Latest 301 run: {latest_301_run.name}")

    # Load Top-50 codes from 301
    top50_301_path = latest_301_run / 'shared_data' / 'top50_icd10_info.json'
    if top50_301_path.exists():
        with open(top50_301_path, 'r') as f:
            top50_301 = json.load(f)
        codes_301 = top50_301['top_50_codes']
        counts_301 = top50_301['top_50_counts']

        print(f"✅ Found Top-50 codes from ShifaMind301")
        print(f"   Total codes: {len(codes_301)}")
        print(f"\n   Top 10 by frequency:")
        sorted_codes = sorted(counts_301.items(), key=lambda x: x[1], reverse=True)[:10]
        for code, count in sorted_codes:
            print(f"      {code}: {count:,} admissions")

        # Load Top-50 codes from 302
        top50_302_path = PHASE1_302_PATH / 'shared_data' / 'top50_icd10_info.json'
        if top50_302_path.exists():
            with open(top50_302_path, 'r') as f:
                top50_302 = json.load(f)
            codes_302 = top50_302['top_50_codes']

            # Compare
            if codes_301 == codes_302:
                print(f"\n✅ Top-50 codes MATCH between 301 and 302")
            else:
                diff_codes = set(codes_301) ^ set(codes_302)
                print(f"\n❌ Top-50 codes DIFFER!")
                print(f"   Different codes: {len(diff_codes)}")
                print(f"   Codes in 301 not in 302: {set(codes_301) - set(codes_302)}")
                print(f"   Codes in 302 not in 301: {set(codes_302) - set(codes_301)}")
                findings['critical_issues'].append(
                    f"❌ Top-50 ICD codes differ between 301 and 302! {len(diff_codes)} codes different."
                )
        else:
            print(f"\n⚠️  Could not find Top-50 codes in 302 run")

        findings['evidence']['top50_codes_301'] = codes_301
        findings['evidence']['top50_counts_301'] = counts_301
    else:
        print(f"⚠️  Top-50 info not found at: {top50_301_path}")
        findings['critical_issues'].append(
            "⚠️  Cannot verify Top-50 codes - file not found in 301"
        )
else:
    print("⚠️  No ShifaMind301 runs found")

# ============================================================================
# INVESTIGATION 3: CONCEPT DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("🧠 INVESTIGATION 3: CONCEPT DISTRIBUTION ANALYSIS")
print("="*80)

# Load concept labels from both versions
concept_labels_301_path = latest_301_run / 'shared_data' / 'train_concept_labels.npy' if run_dirs_301 else None
concept_labels_302_path = PHASE1_302_PATH / 'shared_data' / 'train_concept_labels.npy'

concept_analysis = {}

if concept_labels_301_path and concept_labels_301_path.exists():
    concepts_301 = np.load(concept_labels_301_path)
    print(f"📊 ShifaMind301 concepts:")
    print(f"   Shape: {concepts_301.shape}")
    print(f"   Concepts per sample: {concepts_301.sum(axis=1).mean():.2f} ± {concepts_301.sum(axis=1).std():.2f}")
    print(f"   Sparsity: {(1 - concepts_301.mean()) * 100:.1f}%")
    print(f"   Non-zero concepts: {(concepts_301.sum(axis=0) > 0).sum()} / {concepts_301.shape[1]}")

    # Concept frequency
    concept_freq_301 = concepts_301.sum(axis=0)
    print(f"\n   Concept usage distribution:")
    print(f"      Min: {concept_freq_301.min():.0f}")
    print(f"      25%: {np.percentile(concept_freq_301, 25):.0f}")
    print(f"      50%: {np.percentile(concept_freq_301, 50):.0f}")
    print(f"      75%: {np.percentile(concept_freq_301, 75):.0f}")
    print(f"      Max: {concept_freq_301.max():.0f}")

    concept_analysis['301'] = {
        'shape': concepts_301.shape,
        'avg_per_sample': float(concepts_301.sum(axis=1).mean()),
        'std_per_sample': float(concepts_301.sum(axis=1).std()),
        'sparsity': float((1 - concepts_301.mean()) * 100),
        'concept_frequencies': concept_freq_301.tolist()
    }
else:
    print("⚠️  Could not load ShifaMind301 concept labels")

if concept_labels_302_path.exists():
    concepts_302 = np.load(concept_labels_302_path)
    print(f"\n📊 ShifaMind302 concepts:")
    print(f"   Shape: {concepts_302.shape}")
    print(f"   Concepts per sample: {concepts_302.sum(axis=1).mean():.2f} ± {concepts_302.sum(axis=1).std():.2f}")
    print(f"   Sparsity: {(1 - concepts_302.mean()) * 100:.1f}%")
    print(f"   Non-zero concepts: {(concepts_302.sum(axis=0) > 0).sum()} / {concepts_302.shape[1]}")

    # Concept frequency
    concept_freq_302 = concepts_302.sum(axis=0)
    print(f"\n   Concept usage distribution:")
    print(f"      Min: {concept_freq_302.min():.0f}")
    print(f"      25%: {np.percentile(concept_freq_302, 25):.0f}")
    print(f"      50%: {np.percentile(concept_freq_302, 50):.0f}")
    print(f"      75%: {np.percentile(concept_freq_302, 75):.0f}")
    print(f"      Max: {concept_freq_302.max():.0f}")

    # Identify rarely used concepts (noise candidates)
    rare_threshold = concepts_302.shape[0] * 0.01  # <1% of samples
    rare_concepts = (concept_freq_302 < rare_threshold).sum()
    print(f"\n   🚨 Rarely used concepts (<1% samples): {rare_concepts} / {concepts_302.shape[1]}")

    # Identify overused concepts (potential noise)
    overused_threshold = concepts_302.shape[0] * 0.8  # >80% of samples
    overused_concepts = (concept_freq_302 > overused_threshold).sum()
    print(f"   🚨 Overused concepts (>80% samples): {overused_concepts} / {concepts_302.shape[1]}")

    concept_analysis['302'] = {
        'shape': concepts_302.shape,
        'avg_per_sample': float(concepts_302.sum(axis=1).mean()),
        'std_per_sample': float(concepts_302.sum(axis=1).std()),
        'sparsity': float((1 - concepts_302.mean()) * 100),
        'concept_frequencies': concept_freq_302.tolist(),
        'rare_concepts': int(rare_concepts),
        'overused_concepts': int(overused_concepts)
    }

    # Compare sparsity
    if concept_labels_301_path and concept_labels_301_path.exists():
        sparsity_diff = concept_analysis['302']['sparsity'] - concept_analysis['301']['sparsity']
        concepts_diff = concept_analysis['302']['avg_per_sample'] - concept_analysis['301']['avg_per_sample']

        print(f"\n📈 Comparison:")
        print(f"   Sparsity change: {sparsity_diff:+.1f}% ({concept_analysis['301']['sparsity']:.1f}% → {concept_analysis['302']['sparsity']:.1f}%)")
        print(f"   Concepts per sample: {concepts_diff:+.1f} ({concept_analysis['301']['avg_per_sample']:.1f} → {concept_analysis['302']['avg_per_sample']:.1f})")

        if sparsity_diff < -10:
            findings['critical_issues'].append(
                f"❌ Sparsity decreased by {abs(sparsity_diff):.1f}% - concepts too dense!"
            )
            findings['recommendations'].append(
                "🔧 FIX: Implement stricter concept filtering (confidence threshold, semantic type filtering)"
            )

        if concepts_diff > 50:
            findings['critical_issues'].append(
                f"❌ Concepts per sample increased by {concepts_diff:.1f} - UMLS over-extraction!"
            )
            findings['recommendations'].append(
                "🔧 FIX: Add UMLS confidence threshold (>0.8) and semantic type filtering"
            )
else:
    print("⚠️  Could not load ShifaMind302 concept labels")

findings['evidence']['concept_analysis'] = concept_analysis

# ============================================================================
# INVESTIGATION 4: CONCEPT-DIAGNOSIS CORRELATION
# ============================================================================
print("\n" + "="*80)
print("🔗 INVESTIGATION 4: CONCEPT-DIAGNOSIS CORRELATION")
print("="*80)

if concept_labels_302_path.exists():
    # Load diagnosis labels
    train_split_path = PHASE1_302_PATH / 'shared_data' / 'train_split.pkl'
    if train_split_path.exists():
        import pickle
        with open(train_split_path, 'rb') as f:
            train_split = pickle.load(f)

        dx_labels = np.array([sample['labels'] for sample in train_split])

        print(f"📊 Computing concept-diagnosis correlations...")
        print(f"   Concepts: {concepts_302.shape[1]}")
        print(f"   Diagnoses: {dx_labels.shape[1]}")

        # Compute correlation matrix
        correlations = []
        for i in range(min(concepts_302.shape[1], 100)):  # Sample first 100 concepts
            concept_vec = concepts_302[:, i]
            if concept_vec.sum() > 0:  # Only non-zero concepts
                for j in range(dx_labels.shape[1]):
                    dx_vec = dx_labels[:, j]
                    if dx_vec.sum() > 0:  # Only non-zero diagnoses
                        try:
                            corr, _ = pearsonr(concept_vec, dx_vec)
                            if not np.isnan(corr):
                                correlations.append(abs(corr))
                        except:
                            pass

        if correlations:
            print(f"\n   Correlation statistics (|r|):")
            print(f"      Mean: {np.mean(correlations):.4f}")
            print(f"      Median: {np.median(correlations):.4f}")
            print(f"      75th percentile: {np.percentile(correlations, 75):.4f}")
            print(f"      95th percentile: {np.percentile(correlations, 95):.4f}")

            weak_corr = (np.array(correlations) < 0.1).sum() / len(correlations) * 100
            print(f"\n   🚨 Weak correlations (<0.1): {weak_corr:.1f}%")

            if weak_corr > 50:
                findings['critical_issues'].append(
                    f"❌ {weak_corr:.1f}% of concept-diagnosis correlations are weak (<0.1) - many concepts are noise!"
                )
                findings['recommendations'].append(
                    "🔧 FIX: Filter concepts by minimum correlation with at least one diagnosis (>0.15)"
                )

            findings['evidence']['correlation_analysis'] = {
                'mean_correlation': float(np.mean(correlations)),
                'median_correlation': float(np.median(correlations)),
                'weak_correlation_pct': float(weak_corr)
            }
    else:
        print("⚠️  Could not load training split for correlation analysis")
else:
    print("⚠️  Skipping correlation analysis - concepts not loaded")

# ============================================================================
# INVESTIGATION 5: MODEL CONFIGURATION COMPARISON
# ============================================================================
print("\n" + "="*80)
print("⚙️  INVESTIGATION 5: MODEL CONFIGURATION COMPARISON")
print("="*80)

config_301_path = latest_301_run / 'results' / 'phase1' / 'config.json' if run_dirs_301 else None
config_302_path = PHASE1_302_PATH / 'results' / 'phase1' / 'phase1_config.json'

config_diffs = []

if config_301_path and config_301_path.exists() and config_302_path.exists():
    with open(config_301_path, 'r') as f:
        config_301 = json.load(f)
    with open(config_302_path, 'r') as f:
        config_302 = json.load(f)

    # Check critical parameters
    critical_params = ['max_length', 'batch_size', 'learning_rate', 'num_epochs']

    print("📋 Configuration comparison:")
    for param in critical_params:
        val_301 = config_301.get(param, 'N/A')
        val_302 = config_302.get(param, 'N/A')

        if val_301 != val_302:
            print(f"   ⚠️  {param}: {val_301} (301) → {val_302} (302)")
            config_diffs.append({
                'parameter': param,
                'value_301': val_301,
                'value_302': val_302
            })

            if param == 'max_length' and val_301 > val_302:
                findings['critical_issues'].append(
                    f"❌ CRITICAL: max_length reduced from {val_301} to {val_302} - truncating clinical notes!"
                )
                findings['recommendations'].append(
                    f"🔧 FIX: Restore max_length to {val_301}"
                )
        else:
            print(f"   ✅ {param}: {val_301}")

    # Check for AMP
    if config_302.get('use_amp', False) or config_302.get('gpu_optimizations', {}).get('use_amp', False):
        print(f"\n   ⚠️  AMP enabled in 302 (was it in 301?)")
        if not config_301.get('use_amp', False):
            findings['critical_issues'].append(
                "⚠️  AMP (Mixed Precision) was added in 302 but not present in 301"
            )
            findings['recommendations'].append(
                "🔧 CONSIDER: Disable AMP or verify it doesn't hurt performance (precision loss)"
            )

    findings['evidence']['config_differences'] = config_diffs
else:
    print("⚠️  Could not compare configurations - files not found")

# ============================================================================
# INVESTIGATION 6: PERFORMANCE METRICS COMPARISON
# ============================================================================
print("\n" + "="*80)
print("📈 INVESTIGATION 6: PERFORMANCE METRICS COMPARISON")
print("="*80)

results_301_path = latest_301_run / 'results' / 'phase1' / 'results.json' if run_dirs_301 else None
results_302_path = PHASE1_302_PATH / 'results' / 'phase1' / 'phase1_results.json'

if results_301_path and results_301_path.exists() and results_302_path.exists():
    with open(results_301_path, 'r') as f:
        results_301 = json.load(f)
    with open(results_302_path, 'r') as f:
        results_302 = json.load(f)

    metrics_to_compare = [
        ('test_macro_f1', 'Macro F1'),
        ('test_micro_f1', 'Micro F1'),
        ('concept_f1', 'Concept F1'),
    ]

    print("📊 Performance comparison:")
    for metric_key, metric_name in metrics_to_compare:
        val_301 = results_301.get('test_metrics', {}).get(metric_key, results_301.get(metric_key, 0))
        val_302 = results_302.get('test_metrics', {}).get(metric_key, results_302.get(metric_key, 0))

        diff = val_302 - val_301
        pct_change = (diff / val_301 * 100) if val_301 > 0 else 0

        symbol = "✅" if diff > 0 else "❌"
        print(f"   {symbol} {metric_name}: {val_301:.4f} → {val_302:.4f} ({pct_change:+.1f}%)")

        if diff < -0.02:  # Performance dropped >2%
            findings['critical_issues'].append(
                f"❌ {metric_name} dropped by {abs(pct_change):.1f}% ({val_301:.4f} → {val_302:.4f})"
            )

    findings['evidence']['performance_comparison'] = {
        '301': results_301.get('test_metrics', {}),
        '302': results_302.get('test_metrics', {})
    }
else:
    print("⚠️  Could not compare performance - results files not found")

# ============================================================================
# INVESTIGATION 7: UMLS LABELING TIME & EFFICIENCY
# ============================================================================
print("\n" + "="*80)
print("⏱️  INVESTIGATION 7: UMLS LABELING EFFICIENCY")
print("="*80)

# Check if concept extraction time was logged
concept_extract_time = None
num_samples = None

if config_302_path.exists():
    with open(config_302_path, 'r') as f:
        config_302 = json.load(f)

    # Try to find timing info
    concept_extract_time = config_302.get('concept_extraction_time_hours')
    num_samples = config_302.get('num_train_samples')

    if concept_extract_time:
        print(f"📊 UMLS labeling time: {concept_extract_time:.2f} hours")
        if num_samples:
            time_per_sample = (concept_extract_time * 3600) / num_samples
            print(f"   Time per sample: {time_per_sample:.2f} seconds")

            if time_per_sample > 1.0:
                findings['critical_issues'].append(
                    f"⚠️  UMLS labeling is slow: {time_per_sample:.2f} sec/sample ({concept_extract_time:.1f} hours total)"
                )
                findings['recommendations'].append(
                    "🔧 CONSIDER: Cache UMLS entity linking results or use faster NER (keyword matching)"
                )
    else:
        print("⚠️  Concept extraction time not logged")

# ============================================================================
# SAVE ALL FINDINGS
# ============================================================================
print("\n" + "="*80)
print("💾 SAVING ANALYSIS RESULTS")
print("="*80)

# Save detailed findings
findings_path = ANALYSIS_OUTPUT / 'detailed_findings.json'
with open(findings_path, 'w') as f:
    json.dump(findings, f, indent=2)
print(f"✅ Saved detailed findings: {findings_path}")

# Create summary report
summary_path = ANALYSIS_OUTPUT / 'SUMMARY_REPORT.txt'
with open(summary_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("SHIFAMIND302 PHASE 1 FAILURE ANALYSIS - SUMMARY REPORT\n")
    f.write("="*80 + "\n\n")

    f.write("CRITICAL ISSUES FOUND:\n")
    f.write("-" * 80 + "\n")
    for i, issue in enumerate(findings['critical_issues'], 1):
        f.write(f"{i}. {issue}\n")

    f.write("\n\nRECOMMENDED FIXES:\n")
    f.write("-" * 80 + "\n")
    for i, rec in enumerate(findings['recommendations'], 1):
        f.write(f"{i}. {rec}\n")

    f.write("\n\nEVIDENCE SUMMARY:\n")
    f.write("-" * 80 + "\n")

    if 'concept_analysis' in findings['evidence']:
        ca = findings['evidence']['concept_analysis']
        if '301' in ca and '302' in ca:
            f.write(f"\nConcept Distribution:\n")
            f.write(f"  301: {ca['301']['avg_per_sample']:.1f} concepts/sample, {ca['301']['sparsity']:.1f}% sparsity\n")
            f.write(f"  302: {ca['302']['avg_per_sample']:.1f} concepts/sample, {ca['302']['sparsity']:.1f}% sparsity\n")
            f.write(f"  Change: {ca['302']['avg_per_sample'] - ca['301']['avg_per_sample']:+.1f} concepts/sample\n")

    if 'performance_comparison' in findings['evidence']:
        pc = findings['evidence']['performance_comparison']
        if '301' in pc and '302' in pc:
            f.write(f"\nPerformance Metrics:\n")
            for key in ['test_macro_f1', 'test_micro_f1']:
                if key in pc['301'] and key in pc['302']:
                    val_301 = pc['301'][key]
                    val_302 = pc['302'][key]
                    diff = val_302 - val_301
                    f.write(f"  {key}: {val_301:.4f} → {val_302:.4f} ({diff:+.4f})\n")

print(f"✅ Saved summary report: {summary_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📋 FINAL SUMMARY")
print("="*80)

print(f"\n🔴 Critical Issues Found: {len(findings['critical_issues'])}")
for issue in findings['critical_issues']:
    print(f"   {issue}")

print(f"\n🔧 Recommended Fixes: {len(findings['recommendations'])}")
for rec in findings['recommendations']:
    print(f"   {rec}")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)
print(f"\n📁 All results saved to: {ANALYSIS_OUTPUT}")
print(f"   - detailed_findings.json: Complete evidence data")
print(f"   - SUMMARY_REPORT.txt: Human-readable summary")
print()
