#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND v302 PHASE 1 (FIXED) - Using v301 Architecture
================================================================================
CRITICAL FIX: Added alignment loss that was missing in first attempt

Uses already-extracted concepts from previous run (no re-extraction needed)
Implements v301's exact architecture:
- ConceptBottleneckCrossAttention with fusion at layers 9, 11
- MultiObjectiveLoss with alignment term
- Learnable concept embeddings
- Batch size 8, max_length 384

================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score
)
from tqdm.auto import tqdm

print("="*80)
print("🚀 SHIFAMIND v302 PHASE 1 (FIXED) - WITH ALIGNMENT LOSS")
print("="*80)
print("Using v301 architecture with v5's 131 refined concepts")
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
OUTPUT_BASE = BASE_PATH / '11_ShifaMind_v302'

# Find most recent v302 run with extracted concepts
run_folders = sorted([d for d in OUTPUT_BASE.glob('run_*') if d.is_dir()], reverse=True)
if not run_folders:
    print("❌ No v302 run found! Please run concept extraction first.")
    sys.exit(1)

RUN_PATH = run_folders[0]
CONCEPTS_PATH = RUN_PATH / 'phase_1_concepts'
MODELS_PATH = RUN_PATH / 'phase_1_models'
RESULTS_PATH = RUN_PATH / 'phase_1_results'
SHARED_DATA_PATH = RUN_PATH / 'shared_data'

# Create new models folder for fixed version
MODELS_PATH_FIXED = RUN_PATH / 'phase_1_models_fixed'
RESULTS_PATH_FIXED = RUN_PATH / 'phase_1_results_fixed'
MODELS_PATH_FIXED.mkdir(exist_ok=True)
RESULTS_PATH_FIXED.mkdir(exist_ok=True)

print(f"📁 Using run: {RUN_PATH.name}")
print(f"📁 Loading concepts from: {CONCEPTS_PATH}")
print(f"📁 Saving fixed models to: {MODELS_PATH_FIXED}")
print(f"📁 Saving fixed results to: {RESULTS_PATH_FIXED}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")

# Hyperparameters (matching v301)
BATCH_SIZE = 8  # v301 uses 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
MAX_LENGTH = 384  # v301 uses 384
SEED = 42

# Loss weights (matching v301)
LAMBDA_DX = 1.0
LAMBDA_ALIGN = 0.5  # CRITICAL: Was missing!
LAMBDA_CONCEPT = 0.3

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================================
# LOAD ALREADY-EXTRACTED DATA
# ============================================================================

print("\n" + "="*80)
print("📋 LOADING ALREADY-EXTRACTED CONCEPTS")
print("="*80)

# Load concept vocabulary
with open(CONCEPTS_PATH / 'v5_concept_vocabulary.json', 'r') as f:
    concept_data = json.load(f)
    CONCEPT_VOCAB = concept_data['concepts']
    NUM_CONCEPTS = len(CONCEPT_VOCAB)

print(f"✅ Loaded concept vocabulary: {NUM_CONCEPTS} concepts")

# Load concept matrices (already extracted!)
train_concept_matrix = np.load(CONCEPTS_PATH / 'train_concept_matrix.npy')
val_concept_matrix = np.load(CONCEPTS_PATH / 'val_concept_matrix.npy')
test_concept_matrix = np.load(CONCEPTS_PATH / 'test_concept_matrix.npy')

print(f"✅ Loaded concept matrices:")
print(f"   Train: {train_concept_matrix.shape}")
print(f"   Val:   {val_concept_matrix.shape}")
print(f"   Test:  {test_concept_matrix.shape}")

# Load Top-50 codes
with open(SHARED_DATA_PATH / 'top50_icd10_info.json', 'r') as f:
    top50_info = json.load(f)
    TOP_50_CODES = top50_info['top_50_codes']
    NUM_LABELS = len(TOP_50_CODES)

print(f"✅ Loaded {NUM_LABELS} ICD-10 codes")

# Load splits from v301
OLD_RUN_PATH = BASE_PATH / '10_ShifaMind'
old_run_folders = sorted([d for d in OLD_RUN_PATH.glob('run_*') if d.is_dir()], reverse=True)
OLD_SHARED = old_run_folders[0] / 'shared_data'

with open(OLD_SHARED / 'train_split.pkl', 'rb') as f:
    df_train = pickle.load(f)
with open(OLD_SHARED / 'val_split.pkl', 'rb') as f:
    df_val = pickle.load(f)
with open(OLD_SHARED / 'test_split.pkl', 'rb') as f:
    df_test = pickle.load(f)

print(f"✅ Loaded data splits:")
print(f"   Train: {len(df_train):,}")
print(f"   Val:   {len(df_val):,}")
print(f"   Test:  {len(df_test):,}")

# ============================================================================
# V301 ARCHITECTURE (EXACT COPY)
# ============================================================================

print("\n" + "="*80)
print("🏗️  BUILDING V301 ARCHITECTURE")
print("="*80)

# Install transformers
try:
    from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
except ImportError:
    print("Installing transformers...")
    os.system('pip install -q transformers')
    from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

class ConceptBottleneckCrossAttention(nn.Module):
    """Multiplicative concept bottleneck with cross-attention (from v301)"""
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
    """ShifaMind2 Phase 1: Concept Bottleneck (v301 architecture)"""
    def __init__(self, base_model, num_concepts, num_classes, fusion_layers=[9, 11]):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        self.num_concepts = num_concepts
        self.fusion_layers = fusion_layers

        # Learnable concept embeddings
        self.concept_embeddings = nn.Parameter(
            torch.randn(num_concepts, self.hidden_size) * 0.02
        )

        # Fusion modules at specified layers
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

        # Apply fusion at specified layers
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


class MultiObjectiveLoss(nn.Module):
    """Multi-objective loss with ALIGNMENT (from v301)"""
    def __init__(self, lambda_dx=1.0, lambda_align=0.5, lambda_concept=0.3):
        super().__init__()
        self.lambda_dx = lambda_dx
        self.lambda_align = lambda_align
        self.lambda_concept = lambda_concept
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, dx_labels, concept_labels):
        # 1. Diagnosis loss
        loss_dx = self.bce(outputs['logits'], dx_labels)

        # 2. ALIGNMENT LOSS (KEY!)
        dx_probs = torch.sigmoid(outputs['logits'])
        concept_scores = outputs['concept_scores']
        loss_align = torch.abs(
            dx_probs.unsqueeze(-1) - concept_scores.unsqueeze(1)
        ).mean()

        # 3. Concept loss
        concept_logits = torch.logit(concept_scores.clamp(1e-7, 1-1e-7))
        loss_concept = self.bce(concept_logits, concept_labels)

        # Total loss
        total_loss = (
            self.lambda_dx * loss_dx +
            self.lambda_align * loss_align +
            self.lambda_concept * loss_concept
        )

        components = {
            'total': total_loss.item(),
            'dx': loss_dx.item(),
            'align': loss_align.item(),
            'concept': loss_concept.item()
        }

        return total_loss, components

print("✅ Architecture defined (v301)")

# ============================================================================
# DATASET
# ============================================================================

print("\n" + "="*80)
print("📦 CREATING DATASETS")
print("="*80)

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

# Load tokenizer and model
MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
print(f"\nLoading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
print("✅ BioClinicalBERT loaded")

# Build model
model = ShifaMind2Phase1(
    base_model,
    num_concepts=NUM_CONCEPTS,
    num_classes=NUM_LABELS,
    fusion_layers=[9, 11]
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n✅ Model built: {total_params:,} parameters")
print(f"   Concepts: {NUM_CONCEPTS}")
print(f"   Diagnoses: {NUM_LABELS}")
print(f"   Fusion layers: [9, 11]")

# Create datasets
train_dataset = ConceptDataset(
    df_train['text'].tolist(),
    df_train['labels'].tolist(),
    train_concept_matrix,
    tokenizer,
    MAX_LENGTH
)
val_dataset = ConceptDataset(
    df_val['text'].tolist(),
    df_val['labels'].tolist(),
    val_concept_matrix,
    tokenizer,
    MAX_LENGTH
)
test_dataset = ConceptDataset(
    df_test['text'].tolist(),
    df_test['labels'].tolist(),
    test_concept_matrix,
    tokenizer,
    MAX_LENGTH
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

print(f"\n✅ Datasets created:")
print(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
print(f"   Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
print(f"   Test:  {len(test_dataset)} samples, {len(test_loader)} batches")

# ============================================================================
# TRAINING
# ============================================================================

print("\n" + "="*80)
print("🚀 TRAINING WITH ALIGNMENT LOSS")
print("="*80)

criterion = MultiObjectiveLoss(
    lambda_dx=LAMBDA_DX,
    lambda_align=LAMBDA_ALIGN,
    lambda_concept=LAMBDA_CONCEPT
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=len(train_loader) * NUM_EPOCHS
)

print(f"\n✅ Training setup:")
print(f"   Loss: Diagnosis + {LAMBDA_ALIGN}*Alignment + {LAMBDA_CONCEPT}*Concept")
print(f"   Optimizer: AdamW (lr={LEARNING_RATE})")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Epochs: {NUM_EPOCHS}")

def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()

    all_dx_preds = []
    all_dx_labels = []
    all_concept_preds = []
    all_concept_labels = []

    total_loss = 0
    loss_components = defaultdict(float)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            dx_labels = batch['labels'].to(device)
            concept_labels = batch['concept_labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss, components = criterion(outputs, dx_labels, concept_labels)

            total_loss += loss.item()
            for key, val in components.items():
                loss_components[key] += val

            all_dx_preds.append(torch.sigmoid(outputs['logits']).cpu().numpy())
            all_dx_labels.append(dx_labels.cpu().numpy())
            all_concept_preds.append(outputs['concept_scores'].cpu().numpy())
            all_concept_labels.append(concept_labels.cpu().numpy())

    all_dx_preds = np.vstack(all_dx_preds)
    all_dx_labels = np.vstack(all_dx_labels)
    all_concept_preds = np.vstack(all_concept_preds)
    all_concept_labels = np.vstack(all_concept_labels)

    dx_pred_binary = (all_dx_preds > 0.5).astype(int)
    concept_pred_binary = (all_concept_preds > 0.5).astype(int)

    dx_f1 = f1_score(all_dx_labels, dx_pred_binary, average='macro', zero_division=0)
    concept_f1 = f1_score(all_concept_labels, concept_pred_binary, average='macro', zero_division=0)

    return {
        'loss': total_loss / len(dataloader),
        'dx_f1': dx_f1,
        'concept_f1': concept_f1,
        'loss_dx': loss_components['dx'] / len(dataloader),
        'loss_align': loss_components['align'] / len(dataloader),
        'loss_concept': loss_components['concept'] / len(dataloader)
    }

# Training loop
history = {
    'train_loss': [],
    'val_loss': [],
    'val_dx_f1': [],
    'val_concept_f1': []
}

best_f1 = 0
best_epoch = 0

from collections import defaultdict

print(f"\n{'='*80}")
print(f"Starting training...")
print(f"{'='*80}\n")

for epoch in range(NUM_EPOCHS):
    model.train()

    train_loss = 0
    loss_components = defaultdict(float)

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')

    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        dx_labels = batch['labels'].to(device)
        concept_labels = batch['concept_labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss, components = criterion(outputs, dx_labels, concept_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        for key, val in components.items():
            loss_components[key] += val

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dx': f'{components["dx"]:.4f}',
            'align': f'{components["align"]:.4f}',
            'concept': f'{components["concept"]:.4f}'
        })

    avg_train_loss = train_loss / len(train_loader)

    print(f"\n📊 Epoch {epoch+1} Losses:")
    print(f"   Total:     {avg_train_loss:.4f}")
    print(f"   Diagnosis: {loss_components['dx']/len(train_loader):.4f}")
    print(f"   Alignment: {loss_components['align']/len(train_loader):.4f}")
    print(f"   Concept:   {loss_components['concept']/len(train_loader):.4f}")

    # Validation
    print(f"\n   Validating...")
    val_metrics = evaluate(model, val_loader, criterion, device)

    print(f"\n📈 Validation:")
    print(f"   Diagnosis F1: {val_metrics['dx_f1']:.4f}")
    print(f"   Concept F1:   {val_metrics['concept_f1']:.4f}")

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(val_metrics['loss'])
    history['val_dx_f1'].append(val_metrics['dx_f1'])
    history['val_concept_f1'].append(val_metrics['concept_f1'])

    # Save best model
    if val_metrics['dx_f1'] > best_f1:
        best_f1 = val_metrics['dx_f1']
        best_epoch = epoch + 1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dx_f1': val_metrics['dx_f1'],
            'val_metrics': val_metrics
        }, MODELS_PATH_FIXED / 'phase1_best.pt')
        print(f"   ✅ Saved best model (F1: {best_f1:.4f})")

    print()

print(f"\n{'='*80}")
print(f"✅ Training complete!")
print(f"   Best epoch: {best_epoch}")
print(f"   Best val F1: {best_f1:.4f}")
print(f"{'='*80}\n")

# ============================================================================
# FINAL TEST EVALUATION
# ============================================================================

print("="*80)
print("📊 FINAL TEST EVALUATION")
print("="*80)

checkpoint = torch.load(MODELS_PATH_FIXED / 'phase1_best.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_dx_preds, all_dx_labels = [], []
all_concept_preds, all_concept_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        dx_labels = batch['labels'].to(device)
        concept_labels = batch['concept_labels'].to(device)

        outputs = model(input_ids, attention_mask)

        all_dx_preds.append(torch.sigmoid(outputs['logits']).cpu())
        all_dx_labels.append(dx_labels.cpu())
        all_concept_preds.append(outputs['concept_scores'].cpu())
        all_concept_labels.append(concept_labels.cpu())

all_dx_preds = torch.cat(all_dx_preds, dim=0).numpy()
all_dx_labels = torch.cat(all_dx_labels, dim=0).numpy()
all_concept_preds = torch.cat(all_concept_preds, dim=0).numpy()
all_concept_labels = torch.cat(all_concept_labels, dim=0).numpy()

dx_pred_binary = (all_dx_preds > 0.5).astype(int)
concept_pred_binary = (all_concept_preds > 0.5).astype(int)

macro_f1 = f1_score(all_dx_labels, dx_pred_binary, average='macro', zero_division=0)
micro_f1 = f1_score(all_dx_labels, dx_pred_binary, average='micro', zero_division=0)
macro_precision = precision_score(all_dx_labels, dx_pred_binary, average='macro', zero_division=0)
macro_recall = recall_score(all_dx_labels, dx_pred_binary, average='macro', zero_division=0)

per_class_f1 = [
    f1_score(all_dx_labels[:, i], dx_pred_binary[:, i], zero_division=0)
    for i in range(NUM_LABELS)
]

concept_f1 = f1_score(all_concept_labels, concept_pred_binary, average='macro', zero_division=0)

print("\n" + "="*80)
print("🎉 SHIFAMIND v302 PHASE 1 (FIXED) - FINAL RESULTS")
print("="*80)

print("\n🎯 Diagnosis Performance:")
print(f"   Macro F1:    {macro_f1:.4f}")
print(f"   Micro F1:    {micro_f1:.4f}")
print(f"   Precision:   {macro_precision:.4f}")
print(f"   Recall:      {macro_recall:.4f}")

print(f"\n🧠 Concept Performance:")
print(f"   Concept F1:  {concept_f1:.4f}")

print(f"\n📈 Comparison with v301:")
print(f"   v301 Phase 1: F1 = 0.2801")
print(f"   v302 Phase 1: F1 = {macro_f1:.4f}")
improvement = (macro_f1 - 0.2801) / 0.2801 * 100
print(f"   Change:       {improvement:+.1f}%")

# Save results
results = {
    'phase': 'ShifaMind v302 Phase 1 (Fixed)',
    'diagnosis_metrics': {
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'precision': float(macro_precision),
        'recall': float(macro_recall),
        'per_class_f1': {code: float(f1) for code, f1 in zip(TOP_50_CODES, per_class_f1)}
    },
    'concept_metrics': {
        'concept_f1': float(concept_f1),
        'num_concepts': NUM_CONCEPTS
    },
    'training_history': history,
    'architecture': 'v301 with alignment loss',
    'loss_weights': {
        'lambda_dx': LAMBDA_DX,
        'lambda_align': LAMBDA_ALIGN,
        'lambda_concept': LAMBDA_CONCEPT
    }
}

with open(RESULTS_PATH_FIXED / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {RESULTS_PATH_FIXED / 'results.json'}")
print(f"💾 Best model saved to: {MODELS_PATH_FIXED / 'phase1_best.pt'}")

print("\n" + "="*80)
print("✅ SHIFAMIND v302 PHASE 1 (FIXED) COMPLETE!")
print("="*80)

if macro_f1 >= 0.2801:
    print("\n🎉 SUCCESS! v302 matches or exceeds v301 baseline!")
else:
    print("\n⚠️  Still below v301 baseline. May need more concepts or different architecture.")

print("\nAlhamdulillah! 🤲")
