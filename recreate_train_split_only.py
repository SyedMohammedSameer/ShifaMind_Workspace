"""
QUICK FIX: Recreate ONLY train_split.pkl

The user has everything except train_split.pkl.
We can recreate it by:
1. Loading the original CSV
2. Loading val_split.pkl and test_split.pkl
3. Removing val+test from original to get train
4. Verifying it matches train_concept_labels.npy dimensions
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')

# Original data
ORIGINAL_RUN = BASE_PATH / '10_ShifaMind' / 'run_20260102_203225'
ORIGINAL_CSV = ORIGINAL_RUN / 'mimic_dx_data_top50.csv'

# Phase 1 run shared_data
PHASE1_RUN = BASE_PATH / '10_ShifaMind' / 'run_20260215_013437'
SHARED_DATA_PATH = PHASE1_RUN / 'shared_data'

print("="*80)
print("🔧 RECREATING ONLY train_split.pkl")
print("="*80)

# ============================================================================
# VERIFY WHAT EXISTS
# ============================================================================

print("\n📋 Checking existing files...")

required_files = {
    'val_split.pkl': SHARED_DATA_PATH / 'val_split.pkl',
    'test_split.pkl': SHARED_DATA_PATH / 'test_split.pkl',
    'train_concept_labels.npy': SHARED_DATA_PATH / 'train_concept_labels.npy',
}

for name, path in required_files.items():
    if path.exists():
        print(f"   ✅ {name}")
    else:
        print(f"   ❌ {name} - MISSING!")
        raise FileNotFoundError(f"Required file missing: {path}")

# ============================================================================
# LOAD ORIGINAL CSV
# ============================================================================

print(f"\n📊 Loading original CSV...")
if not ORIGINAL_CSV.exists():
    print(f"❌ ERROR: Original CSV not found at {ORIGINAL_CSV}")
    raise FileNotFoundError(f"Missing: {ORIGINAL_CSV}")

df_full = pd.read_csv(ORIGINAL_CSV)
print(f"✅ Loaded {len(df_full):,} samples")

# ============================================================================
# LOAD EXISTING SPLITS
# ============================================================================

print(f"\n📦 Loading existing val and test splits...")

with open(SHARED_DATA_PATH / 'val_split.pkl', 'rb') as f:
    df_val = pickle.load(f)
print(f"   Val: {len(df_val):,} samples")

with open(SHARED_DATA_PATH / 'test_split.pkl', 'rb') as f:
    df_test = pickle.load(f)
print(f"   Test: {len(df_test):,} samples")

# ============================================================================
# RECREATE TRAIN SPLIT
# ============================================================================

print(f"\n🔧 Recreating train split...")

# Get indices of val and test samples
if 'subject_id' in df_val.columns and 'hadm_id' in df_val.columns:
    # Use subject_id + hadm_id as unique identifier
    val_ids = set(zip(df_val['subject_id'], df_val['hadm_id']))
    test_ids = set(zip(df_test['subject_id'], df_test['hadm_id']))
    full_ids = list(zip(df_full['subject_id'], df_full['hadm_id']))

    # Get train indices (everything not in val or test)
    train_mask = [id_pair not in val_ids and id_pair not in test_ids
                  for id_pair in full_ids]
    df_train = df_full[train_mask].reset_index(drop=True)

elif hasattr(df_val, 'index'):
    # Try using pandas index
    val_idx = set(df_val.index)
    test_idx = set(df_test.index)
    train_mask = [i not in val_idx and i not in test_idx
                  for i in df_full.index]
    df_train = df_full[train_mask].reset_index(drop=True)

else:
    # Fallback: use row position
    print("⚠️  Warning: Using row-based split (less reliable)")
    n_val = len(df_val)
    n_test = len(df_test)
    n_train = len(df_full) - n_val - n_test
    df_train = df_full.iloc[:n_train].reset_index(drop=True)

print(f"✅ Train split recreated: {len(df_train):,} samples")

# ============================================================================
# VERIFY
# ============================================================================

print(f"\n🔍 Verifying against train_concept_labels.npy...")

train_concept_labels = np.load(SHARED_DATA_PATH / 'train_concept_labels.npy')
expected_train_size = train_concept_labels.shape[0]

print(f"   Expected: {expected_train_size:,} samples")
print(f"   Got:      {len(df_train):,} samples")

if len(df_train) == expected_train_size:
    print(f"   ✅ MATCH!")
else:
    print(f"   ⚠️  SIZE MISMATCH!")
    print(f"   Difference: {abs(len(df_train) - expected_train_size):,} samples")
    print(f"   This might still work, but the split may not be exact.")

# ============================================================================
# SAVE
# ============================================================================

print(f"\n💾 Saving train_split.pkl...")

output_path = SHARED_DATA_PATH / 'train_split.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(df_train, f)

print(f"✅ Saved to: {output_path}")

# ============================================================================
# FINAL VERIFICATION
# ============================================================================

print("\n" + "="*80)
print("✅ RECONSTRUCTION COMPLETE!")
print("="*80)

print(f"\n📊 Dataset sizes:")
print(f"   Train: {len(df_train):,} ({len(df_train)/len(df_full)*100:.1f}%)")
print(f"   Val:   {len(df_val):,} ({len(df_val)/len(df_full)*100:.1f}%)")
print(f"   Test:  {len(df_test):,} ({len(df_test)/len(df_full)*100:.1f}%)")
print(f"   Total: {len(df_train)+len(df_val)+len(df_test):,}")

print(f"\n🚀 NOW RUN PHASE 3:")
print(f"!python /content/drive/MyDrive/ShifaMind/phase3_training_optimized.py")
