#!/usr/bin/env python3
"""
================================================================================
VERIFY DATA SPLITS - Check if current pickles match checkpoint expectations
================================================================================
This script checks:
1. Dataset sizes match original (Val: 17,265, Test: 17,266)
2. Pickle file timestamps vs checkpoint timestamp
3. Quick prediction test to see if results are close to expected
4. Data integrity checks

Run this FIRST before retraining to see if we can diagnose the issue.
================================================================================
"""

import torch
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("🔍 VERIFYING DATA SPLITS")
print("=" * 80)

# ============================================================================
# PATHS
# ============================================================================

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
SHIFAMIND2_BASE = BASE_PATH / '10_ShifaMind'

# Find the run folder
run_folders = sorted([d for d in SHIFAMIND2_BASE.glob('run_*') if d.is_dir()], reverse=True)
if not run_folders:
    print("❌ No run folders found!")
    exit(1)

OUTPUT_BASE = run_folders[0]
print(f"\n📁 Using run folder: {OUTPUT_BASE.name}")

CHECKPOINT_PATH = OUTPUT_BASE / 'checkpoints' / 'phase1' / 'phase1_best.pt'
SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'

# ============================================================================
# CHECK 1: CHECKPOINT INFO
# ============================================================================

print("\n" + "=" * 80)
print("📦 CHECKPOINT INFORMATION")
print("=" * 80)

if not CHECKPOINT_PATH.exists():
    print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
    exit(1)

checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)

print(f"\n✅ Checkpoint loaded:")
print(f"   Epoch: {checkpoint['epoch']}")
print(f"   Macro F1 (@ 0.5 threshold): {checkpoint['macro_f1']:.4f}")
print(f"   Concept F1: {checkpoint['concept_f1']:.4f}")

# Get timestamp from config
ckpt_timestamp = checkpoint['config']['timestamp']
print(f"   Timestamp: {ckpt_timestamp}")

# Get file modification time
ckpt_mtime = datetime.fromtimestamp(CHECKPOINT_PATH.stat().st_mtime)
print(f"   File modified: {ckpt_mtime}")

# ============================================================================
# CHECK 2: DATA SPLIT FILES
# ============================================================================

print("\n" + "=" * 80)
print("📊 DATA SPLIT FILES")
print("=" * 80)

# Expected sizes from original notebook
EXPECTED_VAL_SIZE = 17265
EXPECTED_TEST_SIZE = 17266

split_files = {
    'train': SHARED_DATA_PATH / 'train_split.pkl',
    'val': SHARED_DATA_PATH / 'val_split.pkl',
    'test': SHARED_DATA_PATH / 'test_split.pkl'
}

splits_ok = True
split_data = {}

for split_name, split_path in split_files.items():
    if not split_path.exists():
        print(f"❌ {split_name}_split.pkl NOT FOUND!")
        splits_ok = False
        continue

    # Load split
    with open(split_path, 'rb') as f:
        df = pickle.load(f)
    split_data[split_name] = df

    # Get file info
    mtime = datetime.fromtimestamp(split_path.stat().st_mtime)
    size_mb = split_path.stat().st_size / (1024 ** 2)

    print(f"\n✅ {split_name}_split.pkl:")
    print(f"   Samples: {len(df)}")
    print(f"   File size: {size_mb:.2f} MB")
    print(f"   Modified: {mtime}")

    # Check if modification time matches checkpoint
    time_diff = abs((mtime - ckpt_mtime).total_seconds())
    if time_diff < 3600:  # Within 1 hour
        print(f"   ✅ Timestamp matches checkpoint (diff: {time_diff:.0f}s)")
    else:
        print(f"   ⚠️  Timestamp differs from checkpoint (diff: {time_diff/3600:.1f}h)")
        splits_ok = False

# ============================================================================
# CHECK 3: DATASET SIZE VERIFICATION
# ============================================================================

print("\n" + "=" * 80)
print("📏 DATASET SIZE VERIFICATION")
print("=" * 80)

if 'val' in split_data:
    val_size = len(split_data['val'])
    if val_size == EXPECTED_VAL_SIZE:
        print(f"✅ Validation size: {val_size} (matches expected {EXPECTED_VAL_SIZE})")
    else:
        print(f"❌ Validation size: {val_size} (expected {EXPECTED_VAL_SIZE})")
        print(f"   Difference: {val_size - EXPECTED_VAL_SIZE:+d} samples")
        splits_ok = False

if 'test' in split_data:
    test_size = len(split_data['test'])
    if test_size == EXPECTED_TEST_SIZE:
        print(f"✅ Test size: {test_size} (matches expected {EXPECTED_TEST_SIZE})")
    else:
        print(f"❌ Test size: {test_size} (expected {EXPECTED_TEST_SIZE})")
        print(f"   Difference: {test_size - EXPECTED_TEST_SIZE:+d} samples")
        splits_ok = False

# ============================================================================
# CHECK 4: DATA INTEGRITY
# ============================================================================

print("\n" + "=" * 80)
print("🔬 DATA INTEGRITY CHECKS")
print("=" * 80)

if 'val' in split_data and 'test' in split_data:
    df_val = split_data['val']
    df_test = split_data['test']

    # Check required columns
    required_cols = ['text', 'labels']
    for split_name, df in [('val', df_val), ('test', df_test)]:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ {split_name}: Missing columns: {missing_cols}")
            splits_ok = False
        else:
            print(f"✅ {split_name}: All required columns present")

    # Check label distribution
    print(f"\n📊 Label Statistics:")
    for split_name, df in [('val', df_val), ('test', df_test)]:
        avg_labels = np.mean([sum(row) for row in df['labels'].tolist()])
        total_labels = sum([sum(row) for row in df['labels'].tolist()])
        print(f"   {split_name}: {avg_labels:.2f} avg labels/sample, {total_labels:,} total")

    # Check for duplicates between val and test
    val_texts = set(df_val['text'].tolist())
    test_texts = set(df_test['text'].tolist())
    overlap = val_texts & test_texts
    if overlap:
        print(f"\n⚠️  WARNING: {len(overlap)} texts appear in both val and test!")
        splits_ok = False
    else:
        print(f"\n✅ No overlap between val and test splits")

# ============================================================================
# CHECK 5: BACKUP/VERSION CHECK
# ============================================================================

print("\n" + "=" * 80)
print("💾 CHECKING FOR BACKUP VERSIONS")
print("=" * 80)

# Check all run folders for backups
print(f"\nSearching for data splits in all run folders...")
all_val_splits = []
for run_folder in run_folders[:5]:  # Check last 5 runs
    val_pkl = run_folder / 'shared_data' / 'val_split.pkl'
    if val_pkl.exists():
        mtime = datetime.fromtimestamp(val_pkl.stat().st_mtime)
        size = len(pickle.load(open(val_pkl, 'rb')))
        all_val_splits.append({
            'run': run_folder.name,
            'mtime': mtime,
            'size': size,
            'path': val_pkl
        })

if all_val_splits:
    print(f"\n✅ Found {len(all_val_splits)} val_split.pkl files:")
    for split_info in all_val_splits:
        size_match = "✅" if split_info['size'] == EXPECTED_VAL_SIZE else "❌"
        print(f"   {size_match} {split_info['run']}: {split_info['size']} samples, modified {split_info['mtime']}")

        # Check if any match the expected size
        if split_info['size'] == EXPECTED_VAL_SIZE and split_info['path'] != split_files['val']:
            print(f"      🔍 POTENTIAL ORIGINAL: This matches expected size!")
            print(f"      Path: {split_info['path']}")
else:
    print("❌ No backup splits found in other run folders")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("📋 VERIFICATION SUMMARY")
print("=" * 80)

if splits_ok:
    print("\n✅ ALL CHECKS PASSED!")
    print("   Current data splits appear to match the checkpoint.")
    print("   The issue may be elsewhere (model loading, evaluation, etc.)")
else:
    print("\n❌ ISSUES DETECTED!")
    print("   Current data splits may NOT match what was used for training.")
    print("\n🔧 RECOMMENDED ACTIONS:")
    print("   1. Check if backups exist (see above)")
    print("   2. Restore from Google Drive version history if available")
    print("   3. If no backups: RETRAIN Phase 1 from scratch (Option 2)")

print("\n" + "=" * 80)
print("✅ Verification complete!")
print("=" * 80)
