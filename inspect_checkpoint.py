#!/usr/bin/env python3
"""
Inspect what metrics are stored in the Phase 1 checkpoint
"""

import torch
from pathlib import Path

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
SHIFAMIND2_BASE = BASE_PATH / '10_ShifaMind'

# Find latest run folder
run_folders = sorted([d for d in SHIFAMIND2_BASE.glob('run_*') if d.is_dir()], reverse=True)
OUTPUT_BASE = run_folders[0]

PHASE1_CHECKPOINT_PATH = OUTPUT_BASE / 'checkpoints' / 'phase1' / 'phase1_best.pt'

print(f"Loading checkpoint: {PHASE1_CHECKPOINT_PATH}")
checkpoint = torch.load(PHASE1_CHECKPOINT_PATH, map_location='cpu', weights_only=False)

print("\n" + "="*80)
print("CHECKPOINT METRICS")
print("="*80)

print(f"\n📊 Training Metrics Saved in Checkpoint:")
print(f"   - Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"   - Macro F1: {checkpoint.get('macro_f1', 'N/A')}")
print(f"   - Concept F1: {checkpoint.get('concept_f1', 'N/A')}")

if 'val_metrics' in checkpoint:
    print(f"\n📊 Validation Metrics:")
    for key, value in checkpoint['val_metrics'].items():
        print(f"   - {key}: {value}")

if 'test_metrics' in checkpoint:
    print(f"\n📊 Test Metrics:")
    for key, value in checkpoint['test_metrics'].items():
        print(f"   - {key}: {value}")

print("\n" + "="*80)
print("COMPARISON WITH TARGET")
print("="*80)

print("\n🎯 What we NEED (from outputs.md):")
print("   - Val micro-F1: 0.5363 (at threshold 0.25)")
print("   - Test tuned: 0.4360")

print("\n📊 What we HAVE in checkpoint:")
if 'macro_f1' in checkpoint:
    print(f"   - Saved macro_f1: {checkpoint['macro_f1']:.4f}")
else:
    print("   - No validation metrics saved!")

print("\n💡 DIAGNOSIS:")
if checkpoint.get('macro_f1', 0) < 0.4:
    print("   ❌ Checkpoint is UNDERTRAINED - need to retrain Phase 1")
    print("   → The Phase 1 training didn't converge to target performance")
    print("   → Need to train longer or with better hyperparameters")
else:
    print("   ✅ Checkpoint has good training metrics")
    print("   ⚠️  But evaluation gives different results - might be wrong split")

print("\n" + "="*80)
print("ALL CHECKPOINT KEYS")
print("="*80)
for key in checkpoint.keys():
    if key not in ['model_state_dict', 'optimizer_state_dict']:
        print(f"   - {key}: {checkpoint[key]}")
