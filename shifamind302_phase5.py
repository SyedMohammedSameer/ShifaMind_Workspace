#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND v302 PHASE 5: Fair Apples-to-Apples Comparison
================================================================================
Author: Mohammed Sameer Syed
University of Arizona - MS in AI Capstone

FAIR EVALUATION PROTOCOL:
- Re-evaluate v302 Phases 1, 2, 3 with unified threshold tuning
- Compare against v301 baseline models
- Three evaluation methods for EVERY model:
  1. Fixed threshold @ 0.5 (baseline)
  2. Tuned threshold (optimized on validation ONLY)
  3. Top-k predictions (k=5)

PRIMARY METRIC: Test Macro-F1 @ Tuned Threshold
(Ensures fairness across common/rare diagnoses)

MODULAR DESIGN:
- Each function (5.1 - 5.4) can be called independently
- Optimized for Google Colab (A100/L4)
- Prevents runtime disconnections with modular execution

Expected Runtime: ~20-35 minutes (evaluation only, no training)
================================================================================
"""

print("="*80)
print("🚀 SHIFAMIND v302 PHASE 5 - FAIR COMPARISON")
print("="*80)

# ============================================================================
# IMPORTS & SETUP
# ============================================================================

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
from sentence_transformers import SentenceTransformer

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("⚠️  Installing FAISS...")
    import subprocess
    subprocess.run(['pip', 'install', '-q', 'faiss-cpu'], check=True)
    import faiss
    FAISS_AVAILABLE = True

import json
import pickle
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Optional
import sys
import torch_geometric
from torch_geometric.nn import GATConv

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")

if torch.cuda.is_available():
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================================
# CONFIGURATION
# ============================================================================

print("\n" + "="*80)
print("⚙️  CONFIGURATION")
print("="*80)

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
SHIFAMIND302_BASE = BASE_PATH / '11_ShifaMind_v302'
SHIFAMIND301_BASE = BASE_PATH / '10_ShifaMind'

# Find latest v302 run
run_folders_302 = sorted([d for d in SHIFAMIND302_BASE.glob('run_*') if d.is_dir()], reverse=True)
if not run_folders_302:
    print("❌ No v302 run found!")
    sys.exit(1)

OUTPUT_BASE_302 = run_folders_302[0]
print(f"📁 v302 Run folder: {OUTPUT_BASE_302.name}")

# Find latest v301 run (for baseline comparison)
run_folders_301 = sorted([d for d in SHIFAMIND301_BASE.glob('run_*') if d.is_dir()], reverse=True)
if not run_folders_301:
    print("❌ No v301 run found!")
    sys.exit(1)

OUTPUT_BASE_301 = run_folders_301[0]
print(f"📁 v301 Run folder: {OUTPUT_BASE_301.name}")

# Paths
SHARED_DATA_PATH = OUTPUT_BASE_301 / 'shared_data'  # Use v301 data (same splits)
RESULTS_PATH = OUTPUT_BASE_302 / 'phase_5_results'
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Load config
PHASE1_CHECKPOINT = OUTPUT_BASE_301 / 'checkpoints' / 'phase1' / 'phase1_best.pt'
checkpoint = torch.load(PHASE1_CHECKPOINT, map_location='cpu', weights_only=False)
TOP_50_CODES = checkpoint['config']['top_50_codes']

with open(SHARED_DATA_PATH / 'concept_list.json', 'r') as f:
    ALL_CONCEPTS = json.load(f)

NUM_CONCEPTS = len(ALL_CONCEPTS)
NUM_DIAGNOSES = len(TOP_50_CODES)

print(f"✅ Configuration loaded:")
print(f"   Diagnoses: {NUM_DIAGNOSES}")
print(f"   Concepts: {NUM_CONCEPTS}")

# Calculate Top-k
train_labels = np.load(SHARED_DATA_PATH / 'train_concept_labels.npy')
with open(SHARED_DATA_PATH / 'val_split.pkl', 'rb') as f:
    df_val_temp = pickle.load(f)
avg_labels_per_sample = np.array([sum(row) for row in df_val_temp['labels'].tolist()]).mean()
TOP_K = int(round(avg_labels_per_sample))
print(f"   Top-k: {TOP_K}")

# ============================================================================
# UNIFIED EVALUATION FUNCTIONS
# ============================================================================

print("\n" + "="*80)
print("📊 UNIFIED EVALUATION PROTOCOL")
print("="*80)

def fix_checkpoint_keys(state_dict, rename_base_to_bert=True, skip_concept_embeddings=True):
    """Fix key names from checkpoint to match model architecture

    Args:
        state_dict: The checkpoint state dict
        rename_base_to_bert: If True, rename base_model.* to bert.* (for Phase 3)
                             If False, keep as base_model.* (for Phase 1)
        skip_concept_embeddings: If True, skip concept_embeddings from state dict (for Phase 3)
                                 If False, include it (for Phase 1)
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # Skip concept_embeddings only for Phase 3 (loaded separately)
        if key == 'concept_embeddings' and skip_concept_embeddings:
            continue

        # Rename base_model.* to bert.* for Phase 3
        if rename_base_to_bert and key.startswith('base_model.'):
            new_key = key.replace('base_model.', 'bert.')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def tune_global_threshold(probs_val, y_val):
    """Find optimal threshold on validation set"""
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
    """Evaluate with fixed threshold"""
    preds = (probs > threshold).astype(int)
    return {
        'macro_f1': float(f1_score(y_true, preds, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y_true, preds, average='micro', zero_division=0))
    }

def eval_with_topk(probs, y_true, k):
    """Evaluate with top-k predictions"""
    preds = np.zeros_like(probs)
    for i in range(len(probs)):
        top_k_indices = np.argsort(probs[i])[-k:]
        preds[i, top_k_indices] = 1
    return {
        'macro_f1': float(f1_score(y_true, preds, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y_true, preds, average='micro', zero_division=0))
    }

print("✅ Evaluation protocol ready")

# ============================================================================
# DATASET
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
# RAG COMPONENTS
# ============================================================================

class SimpleRAG:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', top_k=3, threshold=0.7):
        self.encoder = SentenceTransformer(model_name)
        self.top_k = top_k
        self.threshold = threshold
        self.index = None
        self.documents = []

    def build_index(self, documents: List[Dict]):
        self.documents = documents
        texts = [doc['text'] for doc in documents]

        embeddings = self.encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def retrieve(self, query: str) -> str:
        if self.index is None:
            return ""

        query_embedding = self.encoder.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, self.top_k)

        relevant_texts = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= self.threshold:
                relevant_texts.append(self.documents[idx]['text'])

        return " ".join(relevant_texts) if relevant_texts else ""

# ============================================================================
# GAT ENCODER
# ============================================================================

class GATEncoder(nn.Module):
    """GAT encoder for concept embeddings"""
    def __init__(self, in_channels, hidden_channels, num_heads=4, num_layers=2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.3)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False, dropout=0.3)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = self.dropout(F.elu(self.conv1(x, edge_index)))
        x = self.conv2(x, edge_index)
        return x

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class ConceptBottleneckCrossAttention(nn.Module):
    """Multiplicative concept bottleneck with cross-attention (Phase 1)"""
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

class ShifaMind302Phase1(nn.Module):
    """Phase 1: Concept Bottleneck only (EXACT v301 architecture)"""
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

    def forward(self, input_ids, attention_mask, concept_embeddings_external=None, input_texts=None):
        """Forward pass - concept_embeddings_external and input_texts are for API compatibility (unused)"""
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

class ShifaMind302Phase2(nn.Module):
    """Phase 2: Concept Bottleneck + GAT (no RAG)"""
    def __init__(self, bert_model, gat_encoder, graph_data, num_concepts, num_diagnoses, graph_hidden=256):
        super().__init__()
        self.bert = bert_model
        self.gat = gat_encoder
        self.hidden_size = 768
        self.graph_hidden = graph_hidden
        self.num_concepts = num_concepts
        self.num_diagnoses = num_diagnoses

        if graph_data is not None:
            self.register_buffer('graph_x', graph_data.x)
            self.register_buffer('graph_edge_index', graph_data.edge_index)
            self.graph_node_to_idx = graph_data.node_to_idx
            self.graph_idx_to_node = graph_data.idx_to_node
        else:
            self.graph_x = None
            self.graph_edge_index = None

        if gat_encoder is not None:
            self.graph_proj = nn.Linear(self.graph_hidden, self.hidden_size)
            self.concept_fusion = nn.Sequential(
                nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.gate_net = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.concept_head = nn.Linear(self.hidden_size, num_concepts)
        self.diagnosis_head = nn.Linear(self.hidden_size, num_diagnoses)
        self.dropout = nn.Dropout(0.1)

    def get_graph_concept_embeddings(self):
        if self.gat is None or self.graph_x is None:
            return None

        graph_embeddings = self.gat(self.graph_x, self.graph_edge_index)

        concept_embeds = []
        for concept in ALL_CONCEPTS:
            if concept in self.graph_node_to_idx:
                idx = self.graph_node_to_idx[concept]
                concept_embeds.append(graph_embeddings[idx])
            else:
                concept_embeds.append(torch.zeros(self.graph_hidden, device=self.graph_x.device))

        concept_embeds = torch.stack(concept_embeds)
        concept_embeds = self.graph_proj(concept_embeds)

        return concept_embeds

    def forward(self, input_ids, attention_mask, concept_embeddings_bert, input_texts=None):
        batch_size = input_ids.shape[0]

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled_bert = hidden_states.mean(dim=1)

        # Concept fusion (GAT + BERT)
        if self.concept_fusion is not None:
            gat_concepts = self.get_graph_concept_embeddings()
            bert_concepts = concept_embeddings_bert.unsqueeze(0).expand(batch_size, -1, -1)
            gat_concepts_batched = gat_concepts.unsqueeze(0).expand(batch_size, -1, -1)

            fused_input = torch.cat([bert_concepts, gat_concepts_batched], dim=-1)
            enhanced_concepts = self.concept_fusion(fused_input)
        else:
            enhanced_concepts = concept_embeddings_bert.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention
        pooled_bert_seq = pooled_bert.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)

        context, attn_weights = self.cross_attention(
            query=pooled_bert_seq,
            key=enhanced_concepts,
            value=enhanced_concepts,
            need_weights=True
        )

        pooled_context = context.mean(dim=1)

        gate_input = torch.cat([pooled_bert, pooled_context], dim=-1)
        gate = self.gate_net(gate_input)

        bottleneck_output = gate * pooled_context
        bottleneck_output = self.layer_norm(bottleneck_output)

        cls_hidden = self.dropout(pooled_bert)
        concept_logits = self.concept_head(cls_hidden)
        concept_scores = torch.sigmoid(concept_logits)
        diagnosis_logits = self.diagnosis_head(bottleneck_output)

        return {
            'logits': diagnosis_logits,
            'concept_logits': concept_logits,
            'concept_scores': concept_scores,
            'gate_values': gate
        }

class ShifaMind302Phase3(nn.Module):
    """Phase 3: Full model with Concept Bottleneck + GAT + RAG"""
    def __init__(self, bert_model, gat_encoder, rag_retriever, graph_data,
                 num_concepts, num_diagnoses, graph_hidden=256, rag_gate_max=0.4):
        super().__init__()
        self.bert = bert_model
        self.gat = gat_encoder
        self.rag = rag_retriever
        self.hidden_size = 768
        self.graph_hidden = graph_hidden
        self.num_concepts = num_concepts
        self.num_diagnoses = num_diagnoses
        self.rag_gate_max = rag_gate_max

        if graph_data is not None:
            self.register_buffer('graph_x', graph_data.x)
            self.register_buffer('graph_edge_index', graph_data.edge_index)
            self.graph_node_to_idx = graph_data.node_to_idx
            self.graph_idx_to_node = graph_data.idx_to_node
        else:
            self.graph_x = None
            self.graph_edge_index = None

        rag_dim = 384
        self.rag_projection = nn.Linear(rag_dim, self.hidden_size)
        self.rag_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid()
        )

        if gat_encoder is not None:
            self.graph_proj = nn.Linear(self.graph_hidden, self.hidden_size)
            self.concept_fusion = nn.Sequential(
                nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
        else:
            self.graph_proj = None
            self.concept_fusion = None

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.gate_net = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.concept_head = nn.Linear(self.hidden_size, num_concepts)
        self.diagnosis_head = nn.Linear(self.hidden_size, num_diagnoses)
        self.dropout = nn.Dropout(0.1)

    def get_graph_concept_embeddings(self):
        if self.gat is None or self.graph_x is None:
            return None

        graph_embeddings = self.gat(self.graph_x, self.graph_edge_index)

        concept_embeds = []
        for concept in ALL_CONCEPTS:
            if concept in self.graph_node_to_idx:
                idx = self.graph_node_to_idx[concept]
                concept_embeds.append(graph_embeddings[idx])
            else:
                concept_embeds.append(torch.zeros(self.graph_hidden, device=self.graph_x.device))

        concept_embeds = torch.stack(concept_embeds)
        concept_embeds = self.graph_proj(concept_embeds)

        return concept_embeds

    def forward(self, input_ids, attention_mask, concept_embeddings_bert, input_texts=None):
        batch_size = input_ids.shape[0]

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled_bert = hidden_states.mean(dim=1)

        # RAG fusion
        if self.rag is not None and input_texts is not None:
            rag_texts = [self.rag.retrieve(text) for text in input_texts]

            rag_embeddings = []
            for rag_text in rag_texts:
                if rag_text:
                    emb = self.rag.encoder.encode([rag_text], convert_to_numpy=True)[0]
                else:
                    emb = np.zeros(384)
                rag_embeddings.append(emb)

            rag_embeddings = torch.tensor(np.array(rag_embeddings), dtype=torch.float32).to(pooled_bert.device)
            rag_context = self.rag_projection(rag_embeddings)

            gate_input = torch.cat([pooled_bert, rag_context], dim=-1)
            gate = self.rag_gate(gate_input) * self.rag_gate_max

            bert_with_rag = pooled_bert + gate * rag_context
        else:
            bert_with_rag = pooled_bert

        # Concept fusion (GAT + BERT)
        if self.concept_fusion is not None:
            gat_concepts = self.get_graph_concept_embeddings()
            bert_concepts = concept_embeddings_bert.unsqueeze(0).expand(batch_size, -1, -1)
            gat_concepts_batched = gat_concepts.unsqueeze(0).expand(batch_size, -1, -1)

            fused_input = torch.cat([bert_concepts, gat_concepts_batched], dim=-1)
            enhanced_concepts = self.concept_fusion(fused_input)
        else:
            enhanced_concepts = concept_embeddings_bert.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention
        bert_with_rag_seq = bert_with_rag.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)

        context, attn_weights = self.cross_attention(
            query=bert_with_rag_seq,
            key=enhanced_concepts,
            value=enhanced_concepts,
            need_weights=True
        )

        pooled_context = context.mean(dim=1)

        gate_input = torch.cat([bert_with_rag, pooled_context], dim=-1)
        gate = self.gate_net(gate_input)

        bottleneck_output = gate * pooled_context
        bottleneck_output = self.layer_norm(bottleneck_output)

        cls_hidden = self.dropout(bert_with_rag)
        concept_logits = self.concept_head(cls_hidden)
        concept_scores = torch.sigmoid(concept_logits)
        diagnosis_logits = self.diagnosis_head(bottleneck_output)

        return {
            'logits': diagnosis_logits,
            'concept_logits': concept_logits,
            'concept_scores': concept_scores,
            'gate_values': gate
        }

print("✅ Model architectures loaded")

# ============================================================================
# PHASE 5.1: LOAD v302 CHECKPOINTS
# ============================================================================

def phase_5_1_load_v302_checkpoints():
    """
    Load v302 Phase 1, 2, 3 checkpoints and prepare for evaluation

    Returns:
        dict: {
            'phase1': (model, concept_embeddings),
            'phase2': (model, concept_embeddings),
            'phase3': (model, concept_embeddings)
        }
    """
    print("\n" + "="*80)
    print("📥 PHASE 5.1: LOADING v302 CHECKPOINTS")
    print("="*80)

    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    models_dict = {}

    # ========================================================================
    # PHASE 1: Load from v301 (inherited)
    # ========================================================================
    print("\n🔵 Loading Phase 1 (Concept Bottleneck only)...")
    phase1_checkpoint_path = OUTPUT_BASE_301 / 'checkpoints' / 'phase1' / 'phase1_best.pt'
    print(f"   📍 Checkpoint path: {phase1_checkpoint_path}")

    if phase1_checkpoint_path.exists():
        base_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device)
        model_p1 = ShifaMind302Phase1(base_model, NUM_CONCEPTS, NUM_DIAGNOSES).to(device)

        checkpoint = torch.load(phase1_checkpoint_path, map_location=device, weights_only=False)

        # Fix key names from checkpoint (keep base_model.* and include concept_embeddings for Phase 1)
        fixed_state_dict = fix_checkpoint_keys(checkpoint['model_state_dict'],
                                                rename_base_to_bert=False,
                                                skip_concept_embeddings=False)
        model_p1.load_state_dict(fixed_state_dict)

        # Get concept embeddings from the loaded model
        concept_embeddings_p1 = model_p1.concept_embeddings.detach()
        model_p1.eval()

        models_dict['phase1'] = (model_p1, concept_embeddings_p1, None, None)
        print(f"   ✅ Phase 1 loaded from v301")
    else:
        print(f"   ⚠️  Phase 1 checkpoint not found")

    # ========================================================================
    # PHASE 2: Load from v302 (GAT added)
    # ========================================================================
    print("\n🔵 Loading Phase 2 (CB + GAT, no RAG)...")
    phase2_checkpoint_path = OUTPUT_BASE_302 / 'phase_2_models' / 'phase2_best.pt'

    if phase2_checkpoint_path.exists():
        # Load graph data
        graph_data_path = OUTPUT_BASE_302 / 'phase_2_graph' / 'graph_data.pt'
        if graph_data_path.exists():
            graph_data = torch.load(graph_data_path, map_location='cpu', weights_only=False)
            print(f"   ✅ Graph data loaded: {graph_data.x.shape[0]} nodes")
        else:
            graph_data = None
            print("   ⚠️  Graph data not found")

        # Load checkpoint
        checkpoint = torch.load(phase2_checkpoint_path, map_location=device, weights_only=False)

        # Get graph_hidden_dim from config (Phase 2 doesn't have rag_config)
        if 'rag_config' in checkpoint['config']:
            graph_hidden = checkpoint['config']['rag_config'].get('graph_hidden_dim', 256)
        elif 'graph_hidden_dim' in checkpoint['config']:
            graph_hidden = checkpoint['config']['graph_hidden_dim']
        else:
            graph_hidden = 256  # Default
            print(f"   ⚠️  graph_hidden_dim not found in config, using default: {graph_hidden}")

        # Create GAT encoder
        gat_encoder = GATEncoder(
            in_channels=768,
            hidden_channels=graph_hidden,
            num_heads=4,
            num_layers=2
        ).to(device)

        # Create model
        bert_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device)
        model_p2 = ShifaMind302Phase2(
            bert_model=bert_model,
            gat_encoder=gat_encoder,
            graph_data=graph_data,
            num_concepts=NUM_CONCEPTS,
            num_diagnoses=NUM_DIAGNOSES,
            graph_hidden=graph_hidden
        ).to(device)

        model_p2.load_state_dict(checkpoint['model_state_dict'], strict=False)
        concept_embeddings_p2 = checkpoint['concept_embeddings'].to(device)
        model_p2.eval()

        models_dict['phase2'] = (model_p2, concept_embeddings_p2, None, None)
        print(f"   ✅ Phase 2 loaded from v302")
    else:
        print(f"   ⚠️  Phase 2 checkpoint not found at {phase2_checkpoint_path}")

    # ========================================================================
    # PHASE 3: Load from v302 (Full model with RAG)
    # ========================================================================
    print("\n🔵 Loading Phase 3 (Full: CB + GAT + RAG)...")
    phase3_checkpoint_path = OUTPUT_BASE_302 / 'phase_3_models' / 'phase3_best.pt'

    if phase3_checkpoint_path.exists():
        # Load graph data
        graph_data_path = OUTPUT_BASE_302 / 'phase_2_graph' / 'graph_data.pt'
        if graph_data_path.exists():
            graph_data = torch.load(graph_data_path, map_location='cpu', weights_only=False)
        else:
            graph_data = None

        # Load RAG
        evidence_path = OUTPUT_BASE_302 / 'phase_3_evidence' / 'evidence_corpus.json'
        if evidence_path.exists() and FAISS_AVAILABLE:
            with open(evidence_path, 'r') as f:
                evidence_corpus = json.load(f)
            rag = SimpleRAG(top_k=3, threshold=0.7)
            rag.build_index(evidence_corpus)
            print(f"   ✅ RAG loaded: {len(evidence_corpus)} passages")
        else:
            rag = SimpleRAG(top_k=3, threshold=0.7)
            print("   ⚠️  RAG corpus not found - using empty RAG")

        # Load checkpoint
        checkpoint = torch.load(phase3_checkpoint_path, map_location=device, weights_only=False)

        # Get graph_hidden_dim from config
        if 'rag_config' in checkpoint['config']:
            graph_hidden = checkpoint['config']['rag_config'].get('graph_hidden_dim', 256)
        elif 'graph_hidden_dim' in checkpoint['config']:
            graph_hidden = checkpoint['config']['graph_hidden_dim']
        else:
            graph_hidden = 256  # Default
            print(f"   ⚠️  graph_hidden_dim not found in config, using default: {graph_hidden}")

        # Create GAT encoder
        gat_encoder = GATEncoder(
            in_channels=768,
            hidden_channels=graph_hidden,
            num_heads=4,
            num_layers=2
        ).to(device)

        # Create model
        bert_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device)
        model_p3 = ShifaMind302Phase3(
            bert_model=bert_model,
            gat_encoder=gat_encoder,
            rag_retriever=rag,
            graph_data=graph_data,
            num_concepts=NUM_CONCEPTS,
            num_diagnoses=NUM_DIAGNOSES,
            graph_hidden=graph_hidden,
            rag_gate_max=0.4
        ).to(device)

        model_p3.load_state_dict(checkpoint['model_state_dict'], strict=False)
        concept_embeddings_p3 = checkpoint['concept_embeddings'].to(device)
        model_p3.eval()

        models_dict['phase3'] = (model_p3, concept_embeddings_p3, rag, graph_data)
        print(f"   ✅ Phase 3 loaded from v302")
    else:
        print(f"   ⚠️  Phase 3 checkpoint not found at {phase3_checkpoint_path}")

    print(f"\n✅ Loaded {len(models_dict)} v302 phases")
    return models_dict

# ============================================================================
# PHASE 5.2: EVALUATE v302 WITH THRESHOLD TUNING
# ============================================================================

def phase_5_2_evaluate_v302_with_tuning(models_dict):
    """
    Evaluate all v302 phases with unified protocol:
    1. Fixed threshold @ 0.5
    2. Tuned threshold (optimized on validation)
    3. Top-k predictions

    Args:
        models_dict: Output from phase_5_1_load_v302_checkpoints()

    Returns:
        dict: Results for each phase with all 3 evaluation methods
    """
    print("\n" + "="*80)
    print("📊 PHASE 5.2: EVALUATE v302 WITH THRESHOLD TUNING")
    print("="*80)

    # Load data
    print("\n📦 Loading data splits...")
    with open(SHARED_DATA_PATH / 'val_split.pkl', 'rb') as f:
        df_val = pickle.load(f)
    with open(SHARED_DATA_PATH / 'test_split.pkl', 'rb') as f:
        df_test = pickle.load(f)

    print(f"   Val: {len(df_val)}, Test: {len(df_test)}")

    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    val_dataset = EvalDataset(df_val, tokenizer)
    test_dataset = EvalDataset(df_test, tokenizer)

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    results = {}

    # Evaluate each phase
    for phase_name in ['phase1', 'phase2', 'phase3']:
        if phase_name not in models_dict:
            print(f"\n⚠️  Skipping {phase_name} (not loaded)")
            continue

        model, concept_embeddings, rag, graph_data = models_dict[phase_name]
        has_rag = (rag is not None and phase_name == 'phase3')

        phase_display = {
            'phase1': 'Phase 1 (CB only)',
            'phase2': 'Phase 2 (CB + GAT)',
            'phase3': 'Phase 3 (CB + GAT + RAG)'
        }

        print(f"\n🔵 Evaluating {phase_display[phase_name]}...")

        # Get predictions on validation set
        print("   Getting validation predictions...")
        all_probs_val = []
        all_labels_val = []

        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="  Val", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']

                if has_rag:
                    texts = batch['text']
                    outputs = model(input_ids, attention_mask, concept_embeddings, input_texts=texts)
                else:
                    # For Phase 1 and Phase 2, pass concept_embeddings (Phase 1 ignores it for API compatibility)
                    outputs = model(input_ids, attention_mask, concept_embeddings)

                logits = outputs['logits']
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs_val.append(probs)
                all_labels_val.append(labels.numpy())

        probs_val = np.vstack(all_probs_val)
        y_val = np.vstack(all_labels_val)

        # Tune threshold on validation
        print("   Tuning threshold on validation set...")
        tuned_threshold = tune_global_threshold(probs_val, y_val)

        # Get predictions on test set
        print("   Getting test predictions...")
        all_probs_test = []
        all_labels_test = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="  Test", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']

                if has_rag:
                    texts = batch['text']
                    outputs = model(input_ids, attention_mask, concept_embeddings, input_texts=texts)
                else:
                    # For Phase 1 and Phase 2, pass concept_embeddings (Phase 1 ignores it for API compatibility)
                    outputs = model(input_ids, attention_mask, concept_embeddings)

                logits = outputs['logits']
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs_test.append(probs)
                all_labels_test.append(labels.numpy())

        probs_test = np.vstack(all_probs_test)
        y_test = np.vstack(all_labels_test)

        # Evaluate with all 3 methods
        print("   Computing metrics...")
        val_results = {
            'fixed_05': eval_with_threshold(probs_val, y_val, 0.5),
            'tuned': eval_with_threshold(probs_val, y_val, tuned_threshold),
            'topk': eval_with_topk(probs_val, y_val, TOP_K)
        }

        test_results = {
            'fixed_05': eval_with_threshold(probs_test, y_test, 0.5),
            'tuned': eval_with_threshold(probs_test, y_test, tuned_threshold),
            'topk': eval_with_topk(probs_test, y_test, TOP_K)
        }

        results[phase_name] = {
            'validation': val_results,
            'test': test_results,
            'tuned_threshold': float(tuned_threshold)
        }

        print(f"   ✅ Results:")
        print(f"      Test Fixed@0.5:      Macro F1 = {test_results['fixed_05']['macro_f1']:.4f}, Micro F1 = {test_results['fixed_05']['micro_f1']:.4f}")
        print(f"      Test Tuned@{tuned_threshold:.2f}:   Macro F1 = {test_results['tuned']['macro_f1']:.4f}, Micro F1 = {test_results['tuned']['micro_f1']:.4f}")
        print(f"      Test Top-{TOP_K}:         Macro F1 = {test_results['topk']['macro_f1']:.4f}, Micro F1 = {test_results['topk']['micro_f1']:.4f}")

        # Clean up
        del model
        torch.cuda.empty_cache()

    print(f"\n✅ Evaluation complete for {len(results)} phases")
    return results

# ============================================================================
# PHASE 5.3: LOAD v301 BASELINE RESULTS
# ============================================================================

def phase_5_3_load_baseline_results():
    """
    Load pre-computed baseline results from v301
    (All baselines were already trained and evaluated with unified protocol)

    Returns:
        dict: Baseline model results with tuned thresholds
    """
    print("\n" + "="*80)
    print("📂 PHASE 5.3: LOADING v301 BASELINE RESULTS")
    print("="*80)

    # Try phase5_complete first (has all baselines)
    baseline_results_path = OUTPUT_BASE_301 / 'results' / 'phase5_complete' / 'complete_comparison.json'

    if not baseline_results_path.exists():
        # Fallback to phase5_fair (only has ShifaMind phases, no baselines)
        baseline_results_path = OUTPUT_BASE_301 / 'results' / 'phase5_fair' / 'fair_evaluation_results.json'
        print(f"   ⚠️  phase5_complete not found, trying phase5_fair...")

    if not baseline_results_path.exists():
        print(f"   ⚠️  Baseline results not found at: {baseline_results_path}")
        print("   Returning empty baseline results")
        return {}

    print(f"   📍 Loading from: {baseline_results_path}")

    with open(baseline_results_path, 'r') as f:
        data = json.load(f)

    # Extract baseline models (not ShifaMind phases)
    baseline_results = {}
    if 'models' in data:
        for model_name, results in data['models'].items():
            if 'ShifaMind' not in model_name and 'Phase' not in model_name:
                baseline_results[model_name] = results
    else:
        # If no 'models' key, the entire file might be the results dict
        for model_name, results in data.items():
            if 'ShifaMind' not in model_name and 'Phase' not in model_name:
                baseline_results[model_name] = results

    print(f"✅ Loaded {len(baseline_results)} baseline models:")
    for model_name in baseline_results:
        if 'test' in baseline_results[model_name] and 'tuned' in baseline_results[model_name]['test']:
            test_macro = baseline_results[model_name]['test']['tuned']['macro_f1']
            print(f"   - {model_name}: Test Macro F1 @ Tuned = {test_macro:.4f}")

    return baseline_results

# ============================================================================
# PHASE 5.4: CREATE FINAL COMPARISON TABLE
# ============================================================================

def phase_5_4_create_comparison_table(v302_results, baseline_results):
    """
    Create final comparison table with all models

    Args:
        v302_results: Output from phase_5_2_evaluate_v302_with_tuning()
        baseline_results: Output from phase_5_3_load_baseline_results()

    Returns:
        DataFrame: Complete comparison table sorted by Test Macro F1 @ Tuned
    """
    print("\n" + "="*80)
    print("📊 PHASE 5.4: CREATING FINAL COMPARISON TABLE")
    print("="*80)

    # Combine all results
    all_results = {}

    # Add v302 phases
    phase_names = {
        'phase1': 'ShifaMind v302 Phase 1 (CB only)',
        'phase2': 'ShifaMind v302 Phase 2 (CB + GAT)',
        'phase3': 'ShifaMind v302 Phase 3 (CB + GAT + RAG)'
    }

    for phase_key, results in v302_results.items():
        model_name = phase_names[phase_key]
        all_results[model_name] = results

    # Add baselines
    all_results.update(baseline_results)

    # Sort by Test Macro F1 @ Tuned
    sorted_models = sorted(
        all_results.items(),
        key=lambda x: x[1]['test']['tuned']['macro_f1'],
        reverse=True
    )

    # Create comparison table
    print("\n" + "="*120)
    print(f"{'Model':<50} {'Test Macro@0.5':<17} {'Test Macro@Tuned':<19} {'Test Macro@Top-k':<17} {'Category':<15}")
    print("="*120)

    table_data = []
    for model_name, results in sorted_models:
        test_fixed = results['test']['fixed_05']['macro_f1']
        test_tuned = results['test']['tuned']['macro_f1']
        test_topk = results['test']['topk']['macro_f1']

        # Categorize
        if 'ShifaMind' in model_name or 'Phase' in model_name:
            category = 'v302 Ablation'
        else:
            category = 'Baseline'

        print(f"{model_name:<50} {test_fixed:<17.4f} {test_tuned:<17.4f} {test_topk:<17.4f} {category:<15}")

        table_data.append({
            'Model': model_name,
            'Test_Macro_Fixed_0.5': test_fixed,
            'Test_Macro_Tuned': test_tuned,
            'Test_Macro_Top_k': test_topk,
            'Tuned_Threshold': results['tuned_threshold'],
            'Category': category
        })

    print("="*120)

    # Save results
    comparison_df = pd.DataFrame(table_data)
    comparison_df.to_csv(RESULTS_PATH / 'phase5_comparison_table.csv', index=False)

    final_results = {
        'evaluation_protocol': {
            'description': 'Unified 3-method evaluation for all models',
            'methods': ['Fixed threshold (0.5)', 'Tuned threshold (on validation)', f'Top-k (k={TOP_K})'],
            'primary_metric': 'Test Macro-F1 @ Tuned Threshold',
            'tuning_set': 'Validation only (NEVER test)',
            'justification': 'Macro-F1 ensures fairness across common/rare diagnoses',
            'top_k': TOP_K
        },
        'models': all_results,
        'comparison_table': table_data
    }

    with open(RESULTS_PATH / 'phase5_complete_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n💾 Results saved:")
    print(f"   - {RESULTS_PATH / 'phase5_comparison_table.csv'}")
    print(f"   - {RESULTS_PATH / 'phase5_complete_results.json'}")

    # Print summary
    best_model = sorted_models[0][0]
    best_score = sorted_models[0][1]['test']['tuned']['macro_f1']

    print("\n" + "="*80)
    print("✅ PHASE 5 COMPLETE!")
    print("="*80)
    print(f"\nBEST MODEL: {best_model}")
    print(f"Test Macro-F1 @ Tuned Threshold: {best_score:.4f}")
    print(f"\nAll models evaluated with IDENTICAL protocol:")
    print(f"  - Same data splits")
    print(f"  - Same evaluation metrics")
    print(f"  - Same threshold tuning procedure")
    print(f"  - Primary metric: Test Macro-F1 @ Tuned Threshold")
    print("\nAlhamdulillah! 🤲")

    return comparison_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎯 MAIN EXECUTION")
    print("="*80)
    print("\nYou can call these functions independently:")
    print("  1. models_dict = phase_5_1_load_v302_checkpoints()")
    print("  2. v302_results = phase_5_2_evaluate_v302_with_tuning(models_dict)")
    print("  3. baseline_results = phase_5_3_load_baseline_results()")
    print("  4. comparison_df = phase_5_4_create_comparison_table(v302_results, baseline_results)")
    print("\nOr run all together:")
    print("  # models_dict = phase_5_1_load_v302_checkpoints()")
    print("  # v302_results = phase_5_2_evaluate_v302_with_tuning(models_dict)")
    print("  # baseline_results = phase_5_3_load_baseline_results()")
    print("  # comparison_df = phase_5_4_create_comparison_table(v302_results, baseline_results)")
    print("\nUncomment the lines above to run automatically.")
    print("="*80)
