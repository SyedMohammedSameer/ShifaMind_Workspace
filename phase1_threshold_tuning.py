#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND2 PHASE 1: THRESHOLD TUNING (POST-HOC OPTIMIZATION)
================================================================================
Author: Mohammed Sameer Syed
University of Arizona - MS in AI Capstone

Optimizes classification thresholds per-label to maximize F1 scores.
Finds optimal thresholds for each of the Top-50 ICD-10 codes independently.

Similar to Phase 5 threshold tuning but for Phase 1 multi-label classification.
================================================================================
"""

print("="*80)
print("🎯 SHIFAMIND2 PHASE 1 - THRESHOLD TUNING")
print("="*80)

# ============================================================================
# IMPORTS
# ============================================================================

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModel

import json
import pickle
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")

# ============================================================================
# CONFIGURATION - POINT TO YOUR RUN FOLDER
# ============================================================================

print("\n" + "="*80)
print("⚙️  CONFIGURATION")
print("="*80)

# CHANGE THIS to your run folder from the training output
RUN_FOLDER = '/content/drive/MyDrive/ShifaMind/10_ShifaMind/run_20260215_013437'

BASE_PATH = Path(RUN_FOLDER)
CHECKPOINT_PATH = BASE_PATH / 'checkpoints' / 'phase1' / 'phase1_best.pt'
RESULTS_PATH = BASE_PATH / 'results' / 'phase1'
SHARED_DATA_PATH = BASE_PATH / 'shared_data'

print(f"\n📁 Run Folder: {BASE_PATH}")
print(f"📁 Checkpoint: {CHECKPOINT_PATH}")

if not CHECKPOINT_PATH.exists():
    print(f"\n❌ ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
    print("Please update RUN_FOLDER to your actual run folder!")
    exit(1)

# ============================================================================
# LOAD ARCHITECTURE (SAME AS TRAINING)
# ============================================================================

print("\n" + "="*80)
print("🏗️  LOADING ARCHITECTURE")
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
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
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

    def forward(self, input_ids, attention_mask, return_attention=False):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states
        current_hidden = outputs.last_hidden_state

        attention_maps = {}
        gate_values = []

        for layer_idx in self.fusion_layers:
            if str(layer_idx) in self.fusion_modules:
                layer_hidden = hidden_states[layer_idx]
                fused_hidden, attn, gate = self.fusion_modules[str(layer_idx)](
                    layer_hidden, self.concept_embeddings, attention_mask
                )
                current_hidden = fused_hidden
                gate_values.append(gate.item())

                if return_attention:
                    attention_maps[f'layer_{layer_idx}'] = attn

        cls_hidden = self.dropout(current_hidden[:, 0, :])
        concept_scores = torch.sigmoid(self.concept_head(cls_hidden))
        diagnosis_logits = self.diagnosis_head(cls_hidden)

        result = {
            'logits': diagnosis_logits,
            'concept_scores': concept_scores,
            'hidden_states': current_hidden,
            'cls_hidden': cls_hidden,
            'avg_gate': np.mean(gate_values) if gate_values else 0.0
        }

        if return_attention:
            result['attention_maps'] = attention_maps

        return result


class ConceptDataset(Dataset):
    def __init__(self, texts, labels, concept_labels, tokenizer, max_length=384):
        self.texts = texts
        self.labels = labels
        self.concept_labels = concept_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx]),
            'concept_labels': torch.FloatTensor(self.concept_labels[idx])
        }

print("✅ Architecture loaded")

# ============================================================================
# LOAD TRAINED MODEL & DATA
# ============================================================================

print("\n" + "="*80)
print("📦 LOADING TRAINED MODEL & DATA")
print("="*80)

# Load checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
config = checkpoint['config']

TOP_50_CODES = config['top_50_codes']
print(f"✅ Loaded Top-50 codes: {len(TOP_50_CODES)}")

# Load global concepts
with open(SHARED_DATA_PATH / 'concept_list.json', 'r') as f:
    GLOBAL_CONCEPTS = json.load(f)
print(f"✅ Loaded {len(GLOBAL_CONCEPTS)} concepts")

# Initialize model
print(f"\n🔄 Loading BioClinicalBERT...")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

model = ShifaMind2Phase1(
    base_model,
    num_concepts=len(GLOBAL_CONCEPTS),
    num_classes=len(TOP_50_CODES),
    fusion_layers=[9, 11]
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

# Load validation data
with open(SHARED_DATA_PATH / 'val_split.pkl', 'rb') as f:
    df_val = pickle.load(f)

val_concept_labels = np.load(SHARED_DATA_PATH / 'val_concept_labels.npy')

print(f"✅ Loaded validation set: {len(df_val):,} samples")

# ============================================================================
# GET PREDICTIONS ON VALIDATION SET
# ============================================================================

print("\n" + "="*80)
print("🔮 GENERATING PREDICTIONS")
print("="*80)

val_dataset = ConceptDataset(
    df_val['text'].tolist(),
    df_val['labels'].tolist(),
    val_concept_labels,
    tokenizer
)

val_loader = DataLoader(
    val_dataset,
    batch_size=256,
    num_workers=0,
    pin_memory=True
)

all_probs = []
all_labels = []

USE_AMP = torch.cuda.is_available()

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Getting predictions"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        if USE_AMP:
            with autocast():
                outputs = model(input_ids, attention_mask)
        else:
            outputs = model(input_ids, attention_mask)

        probs = torch.sigmoid(outputs['logits']).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

all_probs = np.vstack(all_probs)
all_labels = np.vstack(all_labels)

print(f"✅ Predictions shape: {all_probs.shape}")
print(f"   Labels shape: {all_labels.shape}")

# ============================================================================
# THRESHOLD TUNING (PER-LABEL)
# ============================================================================

print("\n" + "="*80)
print("🎯 THRESHOLD TUNING (PER-LABEL)")
print("="*80)

THRESHOLD_CANDIDATES = np.arange(0.05, 0.96, 0.05)  # 0.05 to 0.95 in steps of 0.05

optimal_thresholds = {}
best_f1_scores = {}

print(f"Testing {len(THRESHOLD_CANDIDATES)} thresholds per label: {THRESHOLD_CANDIDATES[0]:.2f} to {THRESHOLD_CANDIDATES[-1]:.2f}")

for label_idx, code in enumerate(tqdm(TOP_50_CODES, desc="Tuning thresholds")):
    label_probs = all_probs[:, label_idx]
    label_true = all_labels[:, label_idx]

    best_f1 = 0.0
    best_threshold = 0.5

    for threshold in THRESHOLD_CANDIDATES:
        preds = (label_probs >= threshold).astype(int)
        f1 = f1_score(label_true, preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    optimal_thresholds[code] = float(best_threshold)
    best_f1_scores[code] = float(best_f1)

print(f"\n✅ Optimal thresholds found for {len(optimal_thresholds)} labels")

# ============================================================================
# EVALUATE WITH OPTIMAL THRESHOLDS
# ============================================================================

print("\n" + "="*80)
print("📊 EVALUATION: OPTIMAL THRESHOLDS vs FIXED 0.5")
print("="*80)

# Fixed threshold (0.5)
preds_fixed = (all_probs >= 0.5).astype(int)
f1_macro_fixed = f1_score(all_labels, preds_fixed, average='macro', zero_division=0)
f1_micro_fixed = f1_score(all_labels, preds_fixed, average='micro', zero_division=0)
precision_fixed = precision_score(all_labels, preds_fixed, average='macro', zero_division=0)
recall_fixed = recall_score(all_labels, preds_fixed, average='macro', zero_division=0)

# Optimal thresholds (per-label)
preds_optimal = np.zeros_like(all_probs, dtype=int)
for label_idx, code in enumerate(TOP_50_CODES):
    threshold = optimal_thresholds[code]
    preds_optimal[:, label_idx] = (all_probs[:, label_idx] >= threshold).astype(int)

f1_macro_optimal = f1_score(all_labels, preds_optimal, average='macro', zero_division=0)
f1_micro_optimal = f1_score(all_labels, preds_optimal, average='micro', zero_division=0)
precision_optimal = precision_score(all_labels, preds_optimal, average='macro', zero_division=0)
recall_optimal = recall_score(all_labels, preds_optimal, average='macro', zero_division=0)

print("\n" + "="*80)
print("🎉 THRESHOLD TUNING RESULTS")
print("="*80)

print("\n📊 Fixed Threshold (0.5):")
print(f"   Macro F1:    {f1_macro_fixed:.4f}")
print(f"   Micro F1:    {f1_micro_fixed:.4f}")
print(f"   Precision:   {precision_fixed:.4f}")
print(f"   Recall:      {recall_fixed:.4f}")

print("\n🎯 Optimal Thresholds (Per-Label):")
print(f"   Macro F1:    {f1_macro_optimal:.4f} (+{f1_macro_optimal - f1_macro_fixed:+.4f})")
print(f"   Micro F1:    {f1_micro_optimal:.4f} (+{f1_micro_optimal - f1_micro_fixed:+.4f})")
print(f"   Precision:   {precision_optimal:.4f} (+{precision_optimal - precision_fixed:+.4f})")
print(f"   Recall:      {recall_optimal:.4f} (+{recall_optimal - recall_fixed:+.4f})")

improvement = ((f1_macro_optimal - f1_macro_fixed) / f1_macro_fixed) * 100
print(f"\n🚀 Improvement: {improvement:+.2f}% relative gain in Macro F1!")

# ============================================================================
# TOP IMPROVEMENTS
# ============================================================================

print("\n📊 Top-10 Largest Improvements:")
improvements = {}
for code in TOP_50_CODES:
    idx = TOP_50_CODES.index(code)
    f1_fixed = f1_score(all_labels[:, idx], preds_fixed[:, idx], zero_division=0)
    f1_optimal = best_f1_scores[code]
    improvements[code] = f1_optimal - f1_fixed

top_improvements = sorted(improvements.items(), key=lambda x: x[1], reverse=True)[:10]

for rank, (code, improvement) in enumerate(top_improvements, 1):
    threshold = optimal_thresholds[code]
    f1_opt = best_f1_scores[code]
    print(f"   {rank}. {code}: +{improvement:.4f} (threshold={threshold:.2f}, F1={f1_opt:.4f})")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("💾 SAVING RESULTS")
print("="*80)

# Save optimal thresholds
thresholds_path = RESULTS_PATH / 'optimal_thresholds.json'
with open(thresholds_path, 'w') as f:
    json.dump(optimal_thresholds, f, indent=2)

print(f"✅ Optimal thresholds saved to: {thresholds_path}")

# Save comparison results
results = {
    'fixed_threshold': {
        'threshold': 0.5,
        'macro_f1': float(f1_macro_fixed),
        'micro_f1': float(f1_micro_fixed),
        'precision': float(precision_fixed),
        'recall': float(recall_fixed)
    },
    'optimal_thresholds': {
        'macro_f1': float(f1_macro_optimal),
        'micro_f1': float(f1_micro_optimal),
        'precision': float(precision_optimal),
        'recall': float(recall_optimal),
        'improvement_pct': float(improvement),
        'thresholds': optimal_thresholds,
        'per_label_f1': best_f1_scores
    }
}

results_path = RESULTS_PATH / 'threshold_tuning_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Results saved to: {results_path}")

# Save per-label comparison
comparison_df = pd.DataFrame({
    'icd_code': TOP_50_CODES,
    'optimal_threshold': [optimal_thresholds[c] for c in TOP_50_CODES],
    'f1_optimal': [best_f1_scores[c] for c in TOP_50_CODES],
    'f1_fixed_0.5': [f1_score(all_labels[:, i], preds_fixed[:, i], zero_division=0) for i in range(len(TOP_50_CODES))],
    'improvement': [improvements[c] for c in TOP_50_CODES]
})
comparison_df = comparison_df.sort_values('improvement', ascending=False)

csv_path = RESULTS_PATH / 'threshold_comparison.csv'
comparison_df.to_csv(csv_path, index=False)

print(f"✅ Comparison CSV saved to: {csv_path}")

print("\n" + "="*80)
print("✅ THRESHOLD TUNING COMPLETE!")
print("="*80)
print(f"\n📊 Summary:")
print(f"   Fixed (0.5):  Macro F1 = {f1_macro_fixed:.4f}")
print(f"   Optimal:      Macro F1 = {f1_macro_optimal:.4f} ({improvement:+.2f}%)")
print(f"   Best label:   {top_improvements[0][0]} (+{top_improvements[0][1]:.4f})")
print(f"\n💾 All results saved to: {RESULTS_PATH}")
print("\nAlhamdulillah! 🤲")
