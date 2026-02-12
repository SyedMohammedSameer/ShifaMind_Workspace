"""
Diagnostic script to verify ICD code ordering consistency
between checkpoint config and test data
"""

import torch
import pickle
import json
from pathlib import Path
import numpy as np

# Paths
BASE_PATH = Path('/content/drive/MyDrive/ShifaMind/10_ShifaMind')

# Find all v301 runs
run_folders = sorted([d for d in BASE_PATH.glob('run_2026*') if d.is_dir()], reverse=True)

print("="*80)
print("🔍 VERIFYING ICD CODE ORDERING CONSISTENCY")
print("="*80)

for run_folder in run_folders[:3]:  # Check top 3 runs
    print(f"\n{'='*80}")
    print(f"📁 Checking: {run_folder.name}")
    print(f"{'='*80}")

    # Load Phase 1 checkpoint
    ckpt_path = run_folder / 'checkpoints' / 'phase1' / 'phase1_best.pt'
    if not ckpt_path.exists():
        print("   ❌ No Phase 1 checkpoint found")
        continue

    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        checkpoint_codes = checkpoint['config']['top_50_codes']
        checkpoint_f1 = checkpoint.get('macro_f1', -1)
        print(f"   ✅ Checkpoint loaded (Training Macro F1: {checkpoint_f1:.4f})")
        print(f"   Checkpoint has {len(checkpoint_codes)} codes")
    except Exception as e:
        print(f"   ❌ Failed to load checkpoint: {e}")
        continue

    # Load top50_icd10_info.json if exists
    top50_info_path = run_folder / 'shared_data' / 'top50_icd10_info.json'
    if top50_info_path.exists():
        with open(top50_info_path, 'r') as f:
            top50_info = json.load(f)
        info_codes = top50_info['top_50_codes']
        print(f"   ✅ top50_icd10_info.json found")

        # Compare
        if checkpoint_codes == info_codes:
            print(f"   ✅ Checkpoint codes MATCH info file codes")
        else:
            print(f"   ❌ MISMATCH between checkpoint and info file!")
            for i in range(min(10, len(checkpoint_codes))):
                if checkpoint_codes[i] != info_codes[i]:
                    print(f"      Position {i}: ckpt={checkpoint_codes[i]} vs info={info_codes[i]}")
    else:
        print(f"   ⚠️  No top50_icd10_info.json found")

    # Load test_split.pkl
    test_path = run_folder / 'shared_data' / 'test_split.pkl'
    if not test_path.exists():
        print(f"   ❌ No test_split.pkl found")
        continue

    try:
        with open(test_path, 'rb') as f:
            df_test = pickle.load(f)
        print(f"   ✅ test_split.pkl loaded ({len(df_test)} samples)")

        # Get DataFrame columns (excluding 'text' and 'labels')
        df_columns = [col for col in df_test.columns if col not in ['text', 'labels', 'subject_id', 'hadm_id', 'has_top50']]

        print(f"   DataFrame has {len(df_columns)} ICD code columns")

        # Compare checkpoint codes with DataFrame columns
        if len(checkpoint_codes) == len(df_columns):
            all_match = True
            mismatches = []
            for i in range(len(checkpoint_codes)):
                if checkpoint_codes[i] != df_columns[i]:
                    all_match = False
                    mismatches.append((i, checkpoint_codes[i], df_columns[i]))

            if all_match:
                print(f"   ✅ Checkpoint codes MATCH DataFrame column order")
            else:
                print(f"   ❌ MISMATCH: {len(mismatches)} codes differ in order!")
                for i, ckpt_code, df_code in mismatches[:10]:
                    print(f"      Position {i}: ckpt={ckpt_code} vs df={df_code}")
        else:
            print(f"   ❌ LENGTH MISMATCH: ckpt has {len(checkpoint_codes)}, df has {len(df_columns)}")

        # Verify 'labels' column structure
        print(f"\n   📊 Verifying 'labels' column structure:")
        sample_labels = df_test['labels'].iloc[0]
        print(f"      First sample 'labels' length: {len(sample_labels)}")
        print(f"      Expected length: {len(checkpoint_codes)}")

        if len(sample_labels) == len(checkpoint_codes):
            print(f"      ✅ 'labels' length matches checkpoint codes")
        else:
            print(f"      ❌ 'labels' length MISMATCH!")

        # Check if 'labels' column values match the individual code columns
        print(f"\n   🔍 Verifying 'labels' values match individual code columns:")
        sample_idx = 0
        sample_labels = df_test['labels'].iloc[sample_idx]
        all_match = True

        for i, code in enumerate(checkpoint_codes[:10]):  # Check first 10
            if code in df_test.columns:
                df_value = df_test[code].iloc[sample_idx]
                labels_value = sample_labels[i]

                if df_value == labels_value:
                    match_str = "✓"
                else:
                    match_str = "✗"
                    all_match = False

                print(f"      {match_str} {code}: df['{code}']={df_value}, labels[{i}]={labels_value}")
            else:
                print(f"      ⚠️  {code} not found in DataFrame columns")
                all_match = False

        if all_match:
            print(f"   ✅ 'labels' column values MATCH individual code columns")
        else:
            print(f"   ❌ 'labels' column values DO NOT MATCH!")

        # Print first 10 codes from checkpoint
        print(f"\n   📋 First 10 codes in checkpoint['config']['top_50_codes']:")
        for i in range(min(10, len(checkpoint_codes))):
            print(f"      {i}: {checkpoint_codes[i]}")

    except Exception as e:
        print(f"   ❌ Failed to load test_split.pkl: {e}")

print(f"\n{'='*80}")
print("✅ Diagnostic complete")
print(f"{'='*80}")
