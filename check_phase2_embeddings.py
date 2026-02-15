"""
Quick script to check what's in the Phase 2 checkpoint
Run this in Colab to see if we can extract concept embeddings
"""

import torch
from pathlib import Path

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
PHASE2_RUN = BASE_PATH / '11_ShifaMind_v302' / 'run_20260215_022518'
PHASE2_CHECKPOINT = PHASE2_RUN / 'phase_2_models' / 'phase2_best.pt'

print("🔍 Loading Phase 2 checkpoint...")
checkpoint = torch.load(PHASE2_CHECKPOINT, map_location='cpu', weights_only=False)

print("\n📋 Checkpoint keys:")
for key in checkpoint.keys():
    print(f"   - {key}")

print("\n🔑 Model state_dict keys:")
for i, key in enumerate(checkpoint['model_state_dict'].keys()):
    tensor_shape = checkpoint['model_state_dict'][key].shape
    print(f"   {i+1}. {key}: {tensor_shape}")

print("\n🔍 Searching for concept-related keys:")
for key in checkpoint['model_state_dict'].keys():
    if 'concept' in key.lower() or 'embed' in key.lower():
        tensor_shape = checkpoint['model_state_dict'][key].shape
        print(f"   ✅ {key}: {tensor_shape}")
