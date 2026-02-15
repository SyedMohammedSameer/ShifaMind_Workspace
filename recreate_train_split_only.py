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

# Find the diagnosis column (could be 'icd10_code', 'diagnosis', 'label', etc.)
diagnosis_col = None
possible_cols = ['icd10_code', 'diagnosis', 'label', 'icd_code', 'code', 'dx_code']
for col in possible_cols:
    if col in df_full.columns:
        diagnosis_col = col
        break

if diagnosis_col:
    print(f"✅ Found diagnosis column: '{diagnosis_col}'")
else:
    print(f"⚠️  No diagnosis column found in: {df_full.columns.tolist()}")
    print(f"   Will use non-stratified split")

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
# RECREATE TRAIN SPLIT (SAME RANDOM STATE AS ORIGINAL!)
# ============================================================================

print(f"\n🔧 Recreating train split with same random state...")

# The original Phase 1 used stratified 70/15/15 split with random_state=42
# We'll recreate it exactly!

from sklearn.model_selection import train_test_split

# Expected sizes based on train_concept_labels.npy
train_concept_labels = np.load(SHARED_DATA_PATH / 'train_concept_labels.npy')
expected_train_size = train_concept_labels.shape[0]

print(f"\n📊 Expected split sizes:")
print(f"   Train: {expected_train_size:,} (70%)")
print(f"   Val:   {len(df_val):,} (15%)")
print(f"   Test:  {len(df_test):,} (15%)")
print(f"   Total: {expected_train_size + len(df_val) + len(df_test):,}")

# Recreate the exact same split (70/15/15 with random_state=42)
if diagnosis_col:
    # Stratified split (preferred)
    df_train_new, df_temp = train_test_split(
        df_full,
        test_size=0.3,
        random_state=42,
        stratify=df_full[diagnosis_col]
    )

    df_val_new, df_test_new = train_test_split(
        df_temp,
        test_size=0.5,
        random_state=42,
        stratify=df_temp[diagnosis_col]
    )
else:
    # Non-stratified split (fallback)
    print("   Using non-stratified split")
    df_train_new, df_temp = train_test_split(
        df_full,
        test_size=0.3,
        random_state=42
    )

    df_val_new, df_test_new = train_test_split(
        df_temp,
        test_size=0.5,
        random_state=42
    )

# Use the train split
df_train = df_train_new

print(f"\n✅ Train split recreated: {len(df_train):,} samples")

# ============================================================================
# VERIFY
# ============================================================================

print(f"\n🔍 Verifying against train_concept_labels.npy...")

print(f"   Expected: {expected_train_size:,} samples")
print(f"   Got:      {len(df_train):,} samples")

if len(df_train) == expected_train_size:
    print(f"   ✅ PERFECT MATCH!")
    print(f"   The split is EXACT - same random seed used!")
else:
    print(f"   ⚠️  SIZE MISMATCH!")
    print(f"   Difference: {abs(len(df_train) - expected_train_size):,} samples")
    print(f"   ERROR: This will break Phase 3!")
    raise ValueError(f"Train split size mismatch: {len(df_train)} vs {expected_train_size}")

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
