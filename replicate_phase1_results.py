#!/usr/bin/env python3
"""
================================================================================
DIAGNOSTIC SCRIPT: Replicate Phase 1 Results from Phase 5
================================================================================
Target Results:
   Best threshold: 0.25 (val micro-F1: 0.5363)
   Test: Fixed@0.5=0.2934, Tuned@0.25=0.4360, Top-5=0.3896

This script will:
1. Verify checkpoint exists and contents
2. Load data exactly as Phase 5 does
3. Load model with exact architecture from Phase 5
4. Run evaluation with exact protocol
5. Debug any mismatches
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

import json
import pickle
from pathlib import Path
import sys

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}")

# ============================================================================
# STEP 1: LOCATE CHECKPOINT
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOCATING CHECKPOINT")
print("="*80)

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
SHIFAMIND2_BASE = BASE_PATH / '10_ShifaMind'

# Find latest run folder
run_folders = sorted([d for d in SHIFAMIND2_BASE.glob('run_*') if d.is_dir()], reverse=True)
if not run_folders:
    print("❌ No run folders found!")
    sys.exit(1)

OUTPUT_BASE = run_folders[0]
print(f"✅ Found run folder: {OUTPUT_BASE.name}")

PHASE1_CHECKPOINT_PATH = OUTPUT_BASE / 'checkpoints' / 'phase1' / 'phase1_best.pt'

if not PHASE1_CHECKPOINT_PATH.exists():
    print(f"❌ Checkpoint not found at: {PHASE1_CHECKPOINT_PATH}")
    print("\nSearching for other checkpoint locations...")

    # Search for any phase1 checkpoints
    possible_paths = list(OUTPUT_BASE.rglob('phase1*.pt'))
    if possible_paths:
        print(f"✅ Found {len(possible_paths)} potential checkpoints:")
        for p in possible_paths:
            print(f"   {p}")
        PHASE1_CHECKPOINT_PATH = possible_paths[0]
        print(f"\n📌 Using: {PHASE1_CHECKPOINT_PATH}")
    else:
        print("❌ No Phase 1 checkpoints found anywhere!")
        sys.exit(1)
else:
    print(f"✅ Checkpoint found: {PHASE1_CHECKPOINT_PATH}")

# ============================================================================
# STEP 2: VERIFY CHECKPOINT CONTENTS
# ============================================================================

print("\n" + "="*80)
print("STEP 2: VERIFYING CHECKPOINT CONTENTS")
print("="*80)

checkpoint = torch.load(PHASE1_CHECKPOINT_PATH, map_location='cpu', weights_only=False)

print(f"✅ Checkpoint loaded successfully")
print(f"\n📋 Checkpoint keys:")
for key in checkpoint.keys():
    print(f"   - {key}")

if 'model_state_dict' in checkpoint:
    print(f"\n📋 Model state dict keys (first 10):")
    state_keys = list(checkpoint['model_state_dict'].keys())
    for i, key in enumerate(state_keys[:10]):
        print(f"   {i+1}. {key}")
    if len(state_keys) > 10:
        print(f"   ... and {len(state_keys) - 10} more")
    print(f"\n✅ Total parameters in checkpoint: {len(state_keys)}")

if 'config' in checkpoint:
    print(f"\n📋 Config:")
    config = checkpoint['config']
    for key, value in config.items():
        if isinstance(value, list) and len(value) > 5:
            print(f"   - {key}: {type(value).__name__} (length {len(value)})")
        else:
            print(f"   - {key}: {value}")
    TOP_50_CODES = config['top_50_codes']
else:
    print("⚠️  No config found in checkpoint!")
    TOP_50_CODES = None

# ============================================================================
# STEP 3: LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 3: LOADING DATA")
print("="*80)

SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'

# Check data files
required_files = ['val_split.pkl', 'test_split.pkl', 'concept_list.json']
for fname in required_files:
    fpath = SHARED_DATA_PATH / fname
    if fpath.exists():
        print(f"✅ {fname} found")
    else:
        print(f"❌ {fname} NOT found at {fpath}")
        sys.exit(1)

# Load data
with open(SHARED_DATA_PATH / 'val_split.pkl', 'rb') as f:
    df_val = pickle.load(f)

with open(SHARED_DATA_PATH / 'test_split.pkl', 'rb') as f:
    df_test = pickle.load(f)

with open(SHARED_DATA_PATH / 'concept_list.json', 'r') as f:
    ALL_CONCEPTS = json.load(f)

print(f"\n✅ Data loaded:")
print(f"   - Validation: {len(df_val)} samples")
print(f"   - Test: {len(df_test)} samples")
print(f"   - Concepts: {len(ALL_CONCEPTS)}")
print(f"   - Diagnoses: {len(TOP_50_CODES) if TOP_50_CODES else 'Unknown'}")

# Calculate TOP_K
avg_labels_per_sample = np.array([sum(row) for row in df_val['labels'].tolist()]).mean()
TOP_K = int(round(avg_labels_per_sample))
print(f"   - Top-K: {TOP_K}")

# ============================================================================
# STEP 4: DEFINE MODEL ARCHITECTURE (EXACT FROM PHASE 5)
# ============================================================================

print("\n" + "="*80)
print("STEP 4: DEFINING MODEL ARCHITECTURE")
print("="*80)

class ConceptBottleneckCrossAttention(nn.Module):
    """Multiplicative concept bottleneck with cross-attention"""
    def __init__(self, hidden_size, num_heads=8, dropout=0.1, layer_idx=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.layer_idx = layer_idx

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, concept_embeddings, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        num_concepts = concept_embeddings.shape[0]

        concepts_batch = concept_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(concepts_batch).view(batch_size, num_concepts, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(concepts_batch).view(batch_size, num_concepts, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        context = self.out_proj(context)

        pooled_text = hidden_states.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        pooled_context = context.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        gate_input = torch.cat([pooled_text, pooled_context], dim=-1)
        gate = self.gate_net(gate_input)

        output = gate * context
        output = self.layer_norm(output)

        return output, attn_weights.mean(dim=1), gate.mean()


class ShifaMind2Phase1(nn.Module):
    """ShifaMind2 Phase 1: Concept Bottleneck with Top-50 ICD-10"""
    def __init__(self, base_model, num_concepts, num_classes, fusion_layers=[9, 11]):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        self.num_concepts = num_concepts
        self.fusion_layers = fusion_layers

        self.concept_embeddings = nn.Parameter(
            torch.randn(num_concepts, self.hidden_size) * 0.02
        )

        self.fusion_modules = nn.ModuleDict({
            str(layer): ConceptBottleneckCrossAttention(self.hidden_size, layer_idx=layer)
            for layer in fusion_layers
        })

        self.concept_head = nn.Linear(self.hidden_size, num_concepts)
        self.diagnosis_head = nn.Linear(self.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, concept_embeddings_external, input_texts=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states
        current_hidden = outputs.last_hidden_state

        for layer_idx in self.fusion_layers:
            if str(layer_idx) in self.fusion_modules:
                layer_hidden = hidden_states[layer_idx]
                fused_hidden, attn, gate = self.fusion_modules[str(layer_idx)](
                    layer_hidden, self.concept_embeddings, attention_mask
                )
                current_hidden = fused_hidden

        cls_hidden = self.dropout(current_hidden[:, 0, :])
        concept_scores = torch.sigmoid(self.concept_head(cls_hidden))
        diagnosis_logits = self.diagnosis_head(cls_hidden)

        return {
            'logits': diagnosis_logits,
            'concept_scores': concept_scores
        }

print("✅ Model architecture defined")

# ============================================================================
# STEP 5: DATASET
# ============================================================================

class EvalDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df['text'].tolist()
        self.labels = df['labels'].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'text': str(self.texts[idx]),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

# ============================================================================
# STEP 6: EVALUATION FUNCTIONS (EXACT FROM PHASE 5)
# ============================================================================

print("\n" + "="*80)
print("STEP 5: DEFINING EVALUATION FUNCTIONS")
print("="*80)

def tune_global_threshold(probs_val, y_val):
    """Find optimal threshold on validation"""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.05, 0.61, 0.01):
        preds = (probs_val > threshold).astype(int)
        f1 = f1_score(y_val, preds, average='micro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"   Best threshold: {best_threshold:.2f} (val micro-F1: {best_f1:.4f})")
    return best_threshold

def eval_with_threshold(probs, y_true, threshold):
    preds = (probs > threshold).astype(int)
    return {
        'macro_f1': float(f1_score(y_true, preds, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y_true, preds, average='micro', zero_division=0))
    }

def eval_with_topk(probs, y_true, k):
    preds = np.zeros_like(probs)
    for i in range(len(probs)):
        top_k_indices = np.argsort(probs[i])[-k:]
        preds[i, top_k_indices] = 1
    return {
        'macro_f1': float(f1_score(y_true, preds, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y_true, preds, average='micro', zero_division=0))
    }

def get_probs_from_model(model, loader, concept_embeddings):
    """Get probabilities from model"""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Getting predictions", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            texts = batch['text']

            outputs = model(input_ids, attention_mask, concept_embeddings, input_texts=texts)
            logits = outputs['logits']

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    return np.vstack(all_probs), np.vstack(all_labels)

print("✅ Evaluation functions defined")

# ============================================================================
# STEP 7: LOAD MODEL & CHECKPOINT
# ============================================================================

print("\n" + "="*80)
print("STEP 6: LOADING MODEL WITH CHECKPOINT")
print("="*80)

tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
print("✅ Tokenizer loaded")

base_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device)
print("✅ Base model loaded")

model = ShifaMind2Phase1(base_model, len(ALL_CONCEPTS), len(TOP_50_CODES)).to(device)
print("✅ ShifaMind Phase 1 model initialized")

# Fix checkpoint keys (from Phase 5 code)
def fix_checkpoint_keys(state_dict):
    """Keep base_model.* and include concept_embeddings for Phase 1"""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[key] = value
    return new_state_dict

print("\n🔄 Loading checkpoint weights...")
fixed_state_dict = fix_checkpoint_keys(checkpoint['model_state_dict'])

# Try to load
missing_keys, unexpected_keys = model.load_state_dict(fixed_state_dict, strict=False)

if missing_keys:
    print(f"\n⚠️  Missing keys ({len(missing_keys)}):")
    for key in missing_keys[:10]:
        print(f"   - {key}")
    if len(missing_keys) > 10:
        print(f"   ... and {len(missing_keys) - 10} more")

if unexpected_keys:
    print(f"\n⚠️  Unexpected keys ({len(unexpected_keys)}):")
    for key in unexpected_keys[:10]:
        print(f"   - {key}")
    if len(unexpected_keys) > 10:
        print(f"   ... and {len(unexpected_keys) - 10} more")

if not missing_keys and not unexpected_keys:
    print("✅ Checkpoint loaded perfectly - all keys matched!")
elif len(missing_keys) < 5 and len(unexpected_keys) < 5:
    print("⚠️  Minor key mismatches - should be OK")
else:
    print("❌ Significant key mismatches - results may differ!")

# Get concept embeddings from loaded model
concept_embeddings = model.concept_embeddings.detach()
print(f"✅ Concept embeddings extracted: {concept_embeddings.shape}")

# ============================================================================
# STEP 8: CREATE DATA LOADERS
# ============================================================================

print("\n" + "="*80)
print("STEP 7: CREATING DATA LOADERS")
print("="*80)

val_dataset = EvalDataset(df_val, tokenizer)
test_dataset = EvalDataset(df_test, tokenizer)

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"✅ Validation loader: {len(val_loader)} batches")
print(f"✅ Test loader: {len(test_loader)} batches")

# ============================================================================
# STEP 9: RUN EVALUATION
# ============================================================================

print("\n" + "="*80)
print("STEP 8: RUNNING EVALUATION")
print("="*80)
print("🔍 This is the critical step - replicating Phase 5 evaluation...")

print("\n📊 Getting validation predictions...")
probs_val, y_val = get_probs_from_model(model, val_loader, concept_embeddings)
print(f"✅ Validation predictions: {probs_val.shape}")

print("\n📊 Getting test predictions...")
probs_test, y_test = get_probs_from_model(model, test_loader, concept_embeddings)
print(f"✅ Test predictions: {probs_test.shape}")

print("\n🎯 Tuning threshold on validation...")
tuned_threshold = tune_global_threshold(probs_val, y_val)

print("\n📊 Evaluating on validation set...")
val_fixed = eval_with_threshold(probs_val, y_val, 0.5)
val_tuned = eval_with_threshold(probs_val, y_val, tuned_threshold)
val_topk = eval_with_topk(probs_val, y_val, TOP_K)

print(f"   Fixed@0.5: Macro={val_fixed['macro_f1']:.4f}, Micro={val_fixed['micro_f1']:.4f}")
print(f"   Tuned@{tuned_threshold:.2f}: Macro={val_tuned['macro_f1']:.4f}, Micro={val_tuned['micro_f1']:.4f}")
print(f"   Top-{TOP_K}: Macro={val_topk['macro_f1']:.4f}, Micro={val_topk['micro_f1']:.4f}")

print("\n📊 Evaluating on test set...")
test_fixed = eval_with_threshold(probs_test, y_test, 0.5)
test_tuned = eval_with_threshold(probs_test, y_test, tuned_threshold)
test_topk = eval_with_topk(probs_test, y_test, TOP_K)

print(f"   Fixed@0.5: Macro={test_fixed['macro_f1']:.4f}, Micro={test_fixed['micro_f1']:.4f}")
print(f"   Tuned@{tuned_threshold:.2f}: Macro={test_tuned['macro_f1']:.4f}, Micro={test_tuned['micro_f1']:.4f}")
print(f"   Top-{TOP_K}: Macro={test_topk['macro_f1']:.4f}, Micro={test_topk['micro_f1']:.4f}")

# ============================================================================
# STEP 10: COMPARE WITH TARGET
# ============================================================================

print("\n" + "="*80)
print("FINAL COMPARISON WITH TARGET RESULTS")
print("="*80)

print("\n🎯 TARGET (from outputs.md):")
print("   Best threshold: 0.25 (val micro-F1: 0.5363)")
print("   Test: Fixed@0.5=0.2934, Tuned@0.25=0.4360, Top-5=0.3896")

print("\n📊 YOUR RESULTS:")
print(f"   Best threshold: {tuned_threshold:.2f} (val micro-F1: {val_tuned['micro_f1']:.4f})")
print(f"   Test: Fixed@0.5={test_fixed['macro_f1']:.4f}, Tuned@{tuned_threshold:.2f}={test_tuned['macro_f1']:.4f}, Top-{TOP_K}={test_topk['macro_f1']:.4f}")

print("\n📈 DIFFERENCES:")
target_threshold = 0.25
target_val_micro = 0.5363
target_test_fixed = 0.2934
target_test_tuned = 0.4360
target_test_topk = 0.3896

threshold_diff = abs(tuned_threshold - target_threshold)
val_micro_diff = abs(val_tuned['micro_f1'] - target_val_micro)
test_fixed_diff = abs(test_fixed['macro_f1'] - target_test_fixed)
test_tuned_diff = abs(test_tuned['macro_f1'] - target_test_tuned)
test_topk_diff = abs(test_topk['macro_f1'] - target_test_topk)

print(f"   Threshold difference: {threshold_diff:.2f}")
print(f"   Val micro-F1 difference: {val_micro_diff:.4f}")
print(f"   Test fixed@0.5 difference: {test_fixed_diff:.4f}")
print(f"   Test tuned difference: {test_tuned_diff:.4f}")
print(f"   Test top-k difference: {test_topk_diff:.4f}")

if test_tuned_diff < 0.01:
    print("\n✅ PERFECT MATCH! Results replicated successfully!")
elif test_tuned_diff < 0.05:
    print("\n✅ VERY CLOSE! Minor differences likely due to environment.")
elif test_tuned_diff < 0.1:
    print("\n⚠️  MODERATE DIFFERENCE - Check checkpoint or data integrity")
else:
    print("\n❌ SIGNIFICANT DIFFERENCE - Investigation needed")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
