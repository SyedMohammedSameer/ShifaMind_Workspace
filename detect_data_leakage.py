#!/usr/bin/env python3
"""
================================================================================
DATA LEAKAGE DETECTION SCRIPT
================================================================================
Check if the original Phase 1 results had data leakage, which would explain
why we can't replicate the high performance (0.4360).

Possible leakage scenarios:
1. Train/Val/Test splits overlap (samples appear in multiple sets)
2. Threshold tuned on test instead of val
3. Test labels leaked during training
4. Wrong splits used during evaluation
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
SHIFAMIND2_BASE = BASE_PATH / '10_ShifaMind'

# Find latest run folder
run_folders = sorted([d for d in SHIFAMIND2_BASE.glob('run_*') if d.is_dir()], reverse=True)
OUTPUT_BASE = run_folders[0]

SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'

print("="*80)
print("DATA LEAKAGE DETECTION")
print("="*80)

# ============================================================================
# Load all splits
# ============================================================================

print("\n📂 Loading data splits...")

with open(SHARED_DATA_PATH / 'train_split.pkl', 'rb') as f:
    df_train = pickle.load(f)

with open(SHARED_DATA_PATH / 'val_split.pkl', 'rb') as f:
    df_val = pickle.load(f)

with open(SHARED_DATA_PATH / 'test_split.pkl', 'rb') as f:
    df_test = pickle.load(f)

print(f"✅ Train: {len(df_train)} samples")
print(f"✅ Val: {len(df_val)} samples")
print(f"✅ Test: {len(df_test)} samples")
print(f"✅ Total: {len(df_train) + len(df_val) + len(df_test)} samples")

# ============================================================================
# CHECK 1: Split Overlap
# ============================================================================

print("\n" + "="*80)
print("CHECK 1: SPLIT OVERLAP (Most Critical)")
print("="*80)

# Check if there's an ID column
if 'hadm_id' in df_train.columns:
    id_col = 'hadm_id'
elif 'subject_id' in df_train.columns:
    id_col = 'subject_id'
elif 'row_id' in df_train.columns:
    id_col = 'row_id'
else:
    # Use index
    id_col = None
    print("⚠️  No ID column found - using text hashes")
    train_ids = set([hash(str(text)) for text in df_train['text'].tolist()])
    val_ids = set([hash(str(text)) for text in df_val['text'].tolist()])
    test_ids = set([hash(str(text)) for text in df_test['text'].tolist()])

if id_col:
    print(f"📋 Using ID column: {id_col}")
    train_ids = set(df_train[id_col].tolist())
    val_ids = set(df_val[id_col].tolist())
    test_ids = set(df_test[id_col].tolist())

# Check overlaps
train_val_overlap = train_ids & val_ids
train_test_overlap = train_ids & test_ids
val_test_overlap = val_ids & test_ids

print(f"\n🔍 Overlap Analysis:")
print(f"   Train ∩ Val: {len(train_val_overlap)} samples")
print(f"   Train ∩ Test: {len(train_test_overlap)} samples")
print(f"   Val ∩ Test: {len(val_test_overlap)} samples")

if len(train_val_overlap) > 0 or len(train_test_overlap) > 0 or len(val_test_overlap) > 0:
    print("\n❌ DATA LEAKAGE DETECTED! Splits overlap!")
    if len(train_test_overlap) > 0:
        print(f"   🚨 CRITICAL: {len(train_test_overlap)} samples in BOTH train and test!")
        print("   → This inflates test performance!")
    if len(val_test_overlap) > 0:
        print(f"   🚨 CRITICAL: {len(val_test_overlap)} samples in BOTH val and test!")
        print("   → This makes threshold tuning invalid!")
else:
    print("\n✅ No overlap between splits - GOOD!")

# ============================================================================
# CHECK 2: Label Distribution
# ============================================================================

print("\n" + "="*80)
print("CHECK 2: LABEL DISTRIBUTION")
print("="*80)

train_labels = np.array(df_train['labels'].tolist())
val_labels = np.array(df_val['labels'].tolist())
test_labels = np.array(df_test['labels'].tolist())

train_pos_rate = train_labels.mean()
val_pos_rate = val_labels.mean()
test_pos_rate = test_labels.mean()

print(f"\n📊 Positive label rate:")
print(f"   Train: {train_pos_rate:.4f}")
print(f"   Val:   {val_pos_rate:.4f}")
print(f"   Test:  {test_pos_rate:.4f}")

distribution_diff = abs(train_pos_rate - test_pos_rate)
if distribution_diff > 0.05:
    print(f"\n⚠️  Large distribution difference: {distribution_diff:.4f}")
    print("   → Train/test might be from different distributions")
else:
    print("\n✅ Label distributions are similar")

# ============================================================================
# CHECK 3: Class Distribution
# ============================================================================

print("\n" + "="*80)
print("CHECK 3: PER-CLASS DISTRIBUTION")
print("="*80)

train_class_counts = train_labels.sum(axis=0)
val_class_counts = val_labels.sum(axis=0)
test_class_counts = test_labels.sum(axis=0)

print(f"\n📊 Top 10 most common classes:")
top_classes = np.argsort(train_class_counts)[-10:][::-1]

print(f"{'Class':<8} {'Train':<10} {'Val':<10} {'Test':<10} {'Test/Train':<12}")
print("-" * 60)
for cls_idx in top_classes:
    ratio = test_class_counts[cls_idx] / max(train_class_counts[cls_idx], 1)
    print(f"{cls_idx:<8} {train_class_counts[cls_idx]:<10.0f} "
          f"{val_class_counts[cls_idx]:<10.0f} {test_class_counts[cls_idx]:<10.0f} "
          f"{ratio:<12.2f}")

# ============================================================================
# CHECK 4: Text Length Distribution
# ============================================================================

print("\n" + "="*80)
print("CHECK 4: TEXT LENGTH DISTRIBUTION")
print("="*80)

train_lengths = [len(str(text).split()) for text in df_train['text'].tolist()]
val_lengths = [len(str(text).split()) for text in df_val['text'].tolist()]
test_lengths = [len(str(text).split()) for text in df_test['text'].tolist()]

print(f"\n📊 Average text length (words):")
print(f"   Train: {np.mean(train_lengths):.1f} ± {np.std(train_lengths):.1f}")
print(f"   Val:   {np.mean(val_lengths):.1f} ± {np.std(val_lengths):.1f}")
print(f"   Test:  {np.mean(test_lengths):.1f} ± {np.std(test_lengths):.1f}")

# ============================================================================
# CHECK 5: Original Split Creation Code
# ============================================================================

print("\n" + "="*80)
print("CHECK 5: SPLIT METADATA")
print("="*80)

# Check if there's metadata about how splits were created
metadata_files = list(SHARED_DATA_PATH.glob('*metadata*.json'))
if metadata_files:
    print(f"\n📋 Found {len(metadata_files)} metadata files:")
    for mfile in metadata_files:
        print(f"   - {mfile.name}")
        with open(mfile, 'r') as f:
            metadata = json.load(f)
            print(f"     {metadata}")
else:
    print("\n⚠️  No metadata files found about split creation")

# ============================================================================
# FINAL VERDICT
# ============================================================================

print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

has_leakage = False
leakage_reasons = []

if len(train_test_overlap) > 0:
    has_leakage = True
    leakage_reasons.append(f"Train/Test overlap: {len(train_test_overlap)} samples")

if len(val_test_overlap) > 0:
    has_leakage = True
    leakage_reasons.append(f"Val/Test overlap: {len(val_test_overlap)} samples")

if distribution_diff > 0.05:
    leakage_reasons.append(f"Large distribution shift: {distribution_diff:.4f}")

if has_leakage:
    print("\n❌ DATA LEAKAGE DETECTED!")
    print("\nReasons:")
    for reason in leakage_reasons:
        print(f"   - {reason}")

    print("\n💡 CONCLUSION:")
    print("   The original Phase 1 results (0.4360) likely had data leakage.")
    print("   Your current results (0.3717) are CORRECT and HONEST.")
    print("   → You should report 0.3717 as the true Phase 1 performance!")

    print("\n🔧 RECOMMENDATION:")
    print("   1. Acknowledge the original results were inflated")
    print("   2. Use current results (0.3717) as the true baseline")
    print("   3. Focus on improving Phase 2 and Phase 3 on top of this honest baseline")

else:
    print("\n✅ No obvious data leakage detected")
    print("\n💡 CONCLUSION:")
    print("   The splits appear clean, but performance difference might be due to:")
    print("   1. Undertraining (checkpoint only at epoch 4)")
    print("   2. Different hyperparameters in original training")
    print("   3. Evaluation methodology differences")

    print("\n🔧 RECOMMENDATION:")
    print("   1. Train Phase 1 for more epochs (>10)")
    print("   2. Monitor validation metrics during training")
    print("   3. Save checkpoint when val micro-F1 peaks")

print("\n" + "="*80)
print("DETECTION COMPLETE")
print("="*80)
