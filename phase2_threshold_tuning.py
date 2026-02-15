#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND v302 PHASE 2: THRESHOLD TUNING (POST-HOC OPTIMIZATION)
================================================================================
Author: Mohammed Sameer Syed
University of Arizona - MS in AI Capstone

Optimizes classification thresholds per-label to maximize F1 scores for Phase 2.
Finds optimal thresholds for each of the Top-50 ICD-10 codes independently.

Handles Phase 2 GAT + UMLS architecture with graph-enhanced concepts.
================================================================================
"""

print("="*80)
print("🎯 SHIFAMIND v302 PHASE 2 - THRESHOLD TUNING")
print("="*80)

# ============================================================================
# IMPORTS
# ============================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")

# ============================================================================
# CONFIGURATION - AUTO-DETECT LATEST RUN
# ============================================================================

print("\n" + "="*80)
print("⚙️  CONFIGURATION")
print("="*80)

# Auto-detect the LATEST Phase 2 run
BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
OUTPUT_BASE = BASE_PATH / '11_ShifaMind_v302'

run_folders = sorted([d for d in OUTPUT_BASE.glob('run_*') if d.is_dir()], reverse=True)
if not run_folders:
    print("❌ No Phase 2 run found!")
    exit(1)

RUN_FOLDER = run_folders[0]
CHECKPOINT_PATH = RUN_FOLDER / 'phase_2_models' / 'phase2_best.pt'
RESULTS_PATH = RUN_FOLDER / 'phase_2_results'
SHARED_DATA_PATH = RUN_FOLDER / 'shared_data'
GRAPH_PATH = RUN_FOLDER / 'phase_2_graph'

print(f"\n📁 Run Folder: {RUN_FOLDER.name}")
print(f"📁 Checkpoint: {CHECKPOINT_PATH}")

if not CHECKPOINT_PATH.exists():
    print(f"\n❌ ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
    print("Please train Phase 2 first!")
    exit(1)

# ============================================================================
# LOAD TORCH GEOMETRIC
# ============================================================================

try:
    import torch_geometric
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data
    print("✅ torch_geometric found")
except ImportError:
    print("Installing torch_geometric...")
    os.system('pip install -q torch-geometric')
    import torch_geometric
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data

# ============================================================================
# LOAD ARCHITECTURE (SAME AS PHASE 2 TRAINING)
# ============================================================================

print("\n" + "="*80)
print("🏗️  LOADING ARCHITECTURE")
print("="*80)

# Load checkpoint to get config
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
config = checkpoint['config']

TOP_50_CODES = config['top_50_codes']
NUM_LABELS = len(TOP_50_CODES)

GRAPH_HIDDEN_DIM = config['graph_hidden_dim']
GAT_HEADS = config['gat_heads']
GAT_LAYERS = config['gat_layers']

print(f"✅ Loaded config:")
print(f"   Diagnoses: {NUM_LABELS}")
print(f"   GAT hidden dim: {GRAPH_HIDDEN_DIM}")
print(f"   GAT heads: {GAT_HEADS}")
print(f"   GAT layers: {GAT_LAYERS}")

# Load concept list
with open(SHARED_DATA_PATH / 'concept_list.json', 'r') as f:
    ALL_CONCEPTS = json.load(f)
NUM_CONCEPTS = len(ALL_CONCEPTS)

print(f"   Concepts: {NUM_CONCEPTS}")

# ============================================================================
# DEFINE ARCHITECTURES
# ============================================================================

class GATEncoder(nn.Module):
    """GAT encoder for learning concept embeddings from knowledge graph"""
    def __init__(self, in_channels, hidden_channels, num_layers=2, heads=4, dropout=0.3):
        super().__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        # First layer: in -> hidden
        self.convs.append(GATConv(
            in_channels,
            hidden_channels // heads,  # Output per head
            heads=heads,
            dropout=dropout,
            concat=True
        ))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(
                hidden_channels,
                hidden_channels // heads,
                heads=heads,
                dropout=dropout,
                concat=True
            ))

        # Last layer: hidden -> hidden (average heads)
        if num_layers > 1:
            self.convs.append(GATConv(
                hidden_channels,
                hidden_channels,
                heads=1,
                dropout=dropout,
                concat=False
            ))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = self.dropout(x)

        return x


class ShifaMind302Phase2(nn.Module):
    """ShifaMind v302 Phase 2: GAT + UMLS Knowledge Graph"""
    def __init__(self, bert_model, gat_encoder, graph_data, num_concepts, num_diagnoses, graph_hidden_dim):
        super().__init__()

        self.bert = bert_model
        self.gat = gat_encoder
        self.hidden_size = 768
        self.graph_hidden = graph_hidden_dim
        self.num_concepts = num_concepts
        self.num_diagnoses = num_diagnoses

        # Store graph
        self.register_buffer('graph_x', graph_data.x)
        self.register_buffer('graph_edge_index', graph_data.edge_index)
        self.graph_node_to_idx = graph_data.node_to_idx
        self.graph_idx_to_node = graph_data.idx_to_node

        # Project graph embeddings to BERT dimension
        self.graph_proj = nn.Linear(self.graph_hidden, self.hidden_size)

        # Concept fusion: combine BERT + GAT embeddings
        self.concept_fusion = nn.Sequential(
            nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Cross-attention: text attends to enhanced concepts
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Multiplicative gating
        self.gate_net = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(self.hidden_size)

        # Output heads
        self.concept_head = nn.Linear(self.hidden_size, num_concepts)
        self.diagnosis_head = nn.Linear(self.hidden_size, num_diagnoses)

        self.dropout = nn.Dropout(0.1)

    def get_graph_concept_embeddings(self):
        """Run GAT and extract concept embeddings"""
        # Run GAT on full graph
        graph_embeddings = self.gat(self.graph_x, self.graph_edge_index)

        # Extract concept node embeddings
        concept_embeds = []
        for concept in ALL_CONCEPTS:
            if concept in self.graph_node_to_idx:
                idx = self.graph_node_to_idx[concept]
                concept_embeds.append(graph_embeddings[idx])
            else:
                # Fallback: zeros
                concept_embeds.append(torch.zeros(self.graph_hidden, device=self.graph_x.device))

        concept_embeds = torch.stack(concept_embeds)  # [num_concepts, graph_hidden]
        concept_embeds = self.graph_proj(concept_embeds)  # [num_concepts, 768]

        return concept_embeds

    def forward(self, input_ids, attention_mask, concept_embeddings_bert):
        """Forward pass with BERT + GAT fusion"""
        batch_size = input_ids.shape[0]

        # 1. Encode text with BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, 768]

        # 2. Get GAT-enhanced concept embeddings
        gat_concepts = self.get_graph_concept_embeddings()  # [num_concepts, 768]

        # 3. Fuse BERT + GAT concept embeddings
        bert_concepts = concept_embeddings_bert.unsqueeze(0).expand(batch_size, -1, -1)
        gat_concepts_batched = gat_concepts.unsqueeze(0).expand(batch_size, -1, -1)

        fused_input = torch.cat([bert_concepts, gat_concepts_batched], dim=-1)  # [batch, num_concepts, 1536]
        enhanced_concepts = self.concept_fusion(fused_input)  # [batch, num_concepts, 768]

        # 4. Cross-attention: text attends to enhanced concepts
        context, attn_weights = self.cross_attention(
            query=hidden_states,
            key=enhanced_concepts,
            value=enhanced_concepts,
            need_weights=True
        )  # context: [batch, seq_len, 768]

        # 5. Multiplicative bottleneck gating
        pooled_text = hidden_states.mean(dim=1)  # [batch, 768]
        pooled_context = context.mean(dim=1)  # [batch, 768]

        gate_input = torch.cat([pooled_text, pooled_context], dim=-1)
        gate = self.gate_net(gate_input)  # [batch, 768]

        bottleneck_output = gate * pooled_context
        bottleneck_output = self.layer_norm(bottleneck_output)

        # 6. Output heads
        cls_hidden = self.dropout(pooled_text)
        concept_logits = self.concept_head(cls_hidden)
        concept_scores = torch.sigmoid(concept_logits)
        diagnosis_logits = self.diagnosis_head(bottleneck_output)

        return {
            'logits': diagnosis_logits,
            'concept_logits': concept_logits,
            'concept_scores': concept_scores,
            'gate_values': gate,
            'attention_weights': attn_weights,
            'bottleneck_output': bottleneck_output
        }


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

print("✅ Architecture defined")

# ============================================================================
# LOAD TRAINED MODEL & DATA
# ============================================================================

print("\n" + "="*80)
print("📦 LOADING TRAINED MODEL & DATA")
print("="*80)

# Load graph data
graph_data = torch.load(GRAPH_PATH / 'graph_data.pt', map_location=device)
print(f"✅ Loaded graph data:")
print(f"   Nodes: {graph_data.x.shape[0]}")
print(f"   Edges: {graph_data.edge_index.shape[1]}")

# Initialize BioClinicalBERT
print(f"\n🔄 Loading BioClinicalBERT...")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
print("✅ BioClinicalBERT loaded")

# Initialize GAT
gat_encoder = GATEncoder(
    in_channels=768,
    hidden_channels=GRAPH_HIDDEN_DIM,
    num_layers=GAT_LAYERS,
    heads=GAT_HEADS,
    dropout=0.3
).to(device)

# Initialize model
model = ShifaMind302Phase2(
    bert_model=bert_model,
    gat_encoder=gat_encoder,
    graph_data=graph_data,
    num_concepts=NUM_CONCEPTS,
    num_diagnoses=NUM_LABELS,
    graph_hidden_dim=GRAPH_HIDDEN_DIM
).to(device)

# Load trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

# Load concept embeddings
concept_embeddings = checkpoint['concept_embeddings']

# Load validation data from Phase 1 run
PHASE1_RUN_PATH = BASE_PATH / '10_ShifaMind'
phase1_runs = sorted([d for d in PHASE1_RUN_PATH.glob('run_*') if d.is_dir()], reverse=True)
if not phase1_runs:
    print("❌ No Phase 1 run found!")
    exit(1)

OLD_SHARED = phase1_runs[0] / 'shared_data'
with open(OLD_SHARED / 'val_split.pkl', 'rb') as f:
    df_val = pickle.load(f)

val_concept_labels = np.load(OLD_SHARED / 'val_concept_labels.npy')

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
                outputs = model(input_ids, attention_mask, concept_embeddings)
        else:
            outputs = model(input_ids, attention_mask, concept_embeddings)

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

improvement = ((f1_macro_optimal - f1_macro_fixed) / f1_macro_fixed) * 100 if f1_macro_fixed > 0 else 0
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
