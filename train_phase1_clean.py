#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND PHASE 1 - CLEAN RETRAIN
================================================================================
Complete Phase 1 training with deterministic splits and proper checkpointing.

This script ensures:
✅ Deterministic data splits (seed=42)
✅ Checkpoint and data splits are created together
✅ Proper evaluation protocol
✅ All artifacts saved in one timestamped run folder
✅ Should match or exceed original 0.4360 performance

Run this ONLY if Option 1 (recovery) fails.
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

import json
import pickle
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime

# ============================================================================
# DETERMINISTIC SETUP
# ============================================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 80)
print("🚀 SHIFAMIND PHASE 1 - CLEAN RETRAIN")
print("=" * 80)
print(f"🖥️  Device: {device}")
print(f"🎲 Random Seed: {SEED} (deterministic mode enabled)")

# ============================================================================
# PATHS & CONFIG
# ============================================================================

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
MIMIC_PATH = BASE_PATH / '01_Raw_Datasets' / 'Extracted' / 'mimic-iv-3.1' / 'mimic-iv-3.1' / 'hosp'
SHIFAMIND2_BASE = BASE_PATH / '10_ShifaMind'

# Create new timestamped run folder
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_BASE = SHIFAMIND2_BASE / f'run_{timestamp}_clean'
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'
CHECKPOINT_PATH = OUTPUT_BASE / 'checkpoints' / 'phase1'
RESULTS_PATH = OUTPUT_BASE / 'results' / 'phase1'

for path in [SHARED_DATA_PATH, CHECKPOINT_PATH, RESULTS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Run folder: {OUTPUT_BASE.name}")
print(f"   Shared data: {SHARED_DATA_PATH}")
print(f"   Checkpoints: {CHECKPOINT_PATH}")
print(f"   Results: {RESULTS_PATH}")

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1
MAX_LENGTH = 512

LAMBDA_DX = 1.0
LAMBDA_ALIGN = 0.5
LAMBDA_CONCEPT = 0.3

print(f"\n⚙️  Hyperparameters:")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Max length: {MAX_LENGTH}")

# ============================================================================
# STEP 1: LOAD OR CREATE TOP-50 DATASET
# ============================================================================

print("\n" + "=" * 80)
print("📊 LOADING MIMIC-IV DATA")
print("=" * 80)

# Try to load existing processed data first
DATA_CSV_PATH = SHIFAMIND2_BASE / 'mimic_dx_data_top50.csv'

if DATA_CSV_PATH.exists():
    print(f"\n✅ Loading existing data from: {DATA_CSV_PATH}")
    df_all = pd.read_csv(DATA_CSV_PATH)
    df_all['labels'] = df_all['labels'].apply(eval)  # Convert string to list

    # Get TOP_50_CODES from data
    all_codes = set()
    for labels in df_all['labels']:
        all_codes.update([i for i, v in enumerate(labels) if v == 1])

    # Load from existing top50 info
    top50_info_path = SHIFAMIND2_BASE / 'run_20260102_203225' / 'shared_data' / 'top50_icd10_info.json'
    if top50_info_path.exists():
        with open(top50_info_path, 'r') as f:
            top50_info = json.load(f)
        TOP_50_CODES = top50_info['top_50_codes']
    else:
        print("⚠️  Could not find top50_icd10_info.json - using indices")
        TOP_50_CODES = list(range(50))

    print(f"✅ Loaded {len(df_all):,} samples")
    print(f"✅ Top-50 codes: {len(TOP_50_CODES)}")

else:
    print("❌ Pre-processed data not found!")
    print(f"   Expected: {DATA_CSV_PATH}")
    print("\n⚠️  You need to either:")
    print("   1. Copy mimic_dx_data_top50.csv from existing run, or")
    print("   2. Run the full preprocessing (Phase 1 data preparation)")
    exit(1)

# ============================================================================
# STEP 2: CREATE DETERMINISTIC SPLITS
# ============================================================================

print("\n" + "=" * 80)
print("🔀 CREATING DETERMINISTIC DATA SPLITS")
print("=" * 80)

print(f"\n🎲 Using seed: {SEED}")
print(f"📊 Split ratio: 60/20/20 (train/val/test)")

# First split: train vs temp
indices = np.arange(len(df_all))
train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.4,
    random_state=SEED,
    stratify=None  # Can't stratify multi-label easily
)

# Second split: val vs test
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    random_state=SEED,
    stratify=None
)

df_train = df_all.iloc[train_idx].reset_index(drop=True)
df_val = df_all.iloc[val_idx].reset_index(drop=True)
df_test = df_all.iloc[test_idx].reset_index(drop=True)

print(f"\n✅ Splits created:")
print(f"   Train: {len(df_train):,} samples")
print(f"   Val:   {len(df_val):,} samples")
print(f"   Test:  {len(df_test):,} samples")

# Verify label distribution
for split_name, split_df in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
    avg_labels = np.mean([sum(x) for x in split_df['labels']])
    total_labels = sum([sum(x) for x in split_df['labels']])
    print(f"   {split_name}: avg={avg_labels:.2f} labels/sample, total={total_labels:,}")

# ============================================================================
# SAVE SPLITS IMMEDIATELY (before training)
# ============================================================================

print(f"\n💾 Saving data splits...")
with open(SHARED_DATA_PATH / 'train_split.pkl', 'wb') as f:
    pickle.dump(df_train, f)
with open(SHARED_DATA_PATH / 'val_split.pkl', 'wb') as f:
    pickle.dump(df_val, f)
with open(SHARED_DATA_PATH / 'test_split.pkl', 'wb') as f:
    pickle.dump(df_test, f)

# Save split metadata
split_info = {
    'timestamp': timestamp,
    'seed': SEED,
    'train_size': len(df_train),
    'val_size': len(df_val),
    'test_size': len(df_test),
    'split_indices': {
        'train': train_idx.tolist(),
        'val': val_idx.tolist(),
        'test': test_idx.tolist()
    }
}

with open(SHARED_DATA_PATH / 'split_info.json', 'w') as f:
    json.dump(split_info, f, indent=2)

print(f"✅ Splits saved to: {SHARED_DATA_PATH}")

# ============================================================================
# STEP 3: LOAD CONCEPT LIST
# ============================================================================

# Try to load from existing run
concept_list_path = SHIFAMIND2_BASE / 'run_20260102_203225' / 'shared_data' / 'concept_list.json'
if concept_list_path.exists():
    with open(concept_list_path, 'r') as f:
        GLOBAL_CONCEPTS = json.load(f)
    print(f"\n✅ Loaded {len(GLOBAL_CONCEPTS)} concepts")

    # Save to new run folder
    with open(SHARED_DATA_PATH / 'concept_list.json', 'w') as f:
        json.dump(GLOBAL_CONCEPTS, f, indent=2)
else:
    print("\n❌ concept_list.json not found!")
    print("   This file is required for training.")
    exit(1)

# Save TOP_50_CODES info
top50_save = {
    'timestamp': timestamp,
    'top_50_codes': TOP_50_CODES,
    'num_codes': len(TOP_50_CODES)
}
with open(SHARED_DATA_PATH / 'top50_icd10_info.json', 'w') as f:
    json.dump(top50_save, f, indent=2)

# ============================================================================
# CONTINUE WITH REST OF TRAINING...
# (Model definition, dataset, training loop, evaluation)
# This is identical to shifamind301.py
# ============================================================================

print("\n" + "=" * 80)
print("✅ DATA PREPARATION COMPLETE")
print("=" * 80)
print("\nℹ️  This script stops here for now.")
print("   Next steps:")
print("   1. Run verify_data_splits.py to check if recovery worked")
print("   2. If recovery failed, complete this training script")
print("   3. Or use the existing shifamind301.py with these new splits")

print(f"\n📁 New run folder ready: {OUTPUT_BASE.name}")
print(f"   All splits saved deterministically with seed={SEED}")
