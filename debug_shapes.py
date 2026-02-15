"""
Quick debug script to check loaded concept embeddings shape
"""

import torch
from pathlib import Path

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
PHASE2_RUN = BASE_PATH / '11_ShifaMind_v302' / 'run_20260215_022518'
PHASE2_CHECKPOINT = PHASE2_RUN / 'phase_2_models' / 'phase2_best.pt'

checkpoint = torch.load(PHASE2_CHECKPOINT, map_location='cpu', weights_only=False)

print("Concept embeddings shape:", checkpoint['concept_embeddings'].shape)
print("Expected: torch.Size([111, 768])")
print("\nFirst few values:")
print(checkpoint['concept_embeddings'][:3, :5])
