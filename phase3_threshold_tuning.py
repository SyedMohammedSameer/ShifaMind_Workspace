#!/usr/bin/env python3
"""
ShifaMind v302 - Phase 3 Threshold Tuning & Final Evaluation
Loads trained Phase 3 model, runs test evaluation, and tunes thresholds
"""

import os
import sys
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, hamming_loss, classification_report,
    multilabel_confusion_matrix, roc_auc_score
)
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import torch_geometric.nn as gnn
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
import faiss

# ============================================================================
# CONFIGURATION
# ============================================================================

print("\n" + "="*80)
print("🎯 SHIFAMIND v302 PHASE 3 - THRESHOLD TUNING & FINAL EVALUATION")
print("="*80)

# Paths - EXACT same logic as phase3_training_optimized.py
BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
SHIFAMIND_V302_BASE = BASE_PATH / '11_ShifaMind_v302'

# Find most recent Phase 3 run
phase3_runs = sorted([d for d in SHIFAMIND_V302_BASE.glob('run_*_phase3') if d.is_dir()])
if not phase3_runs:
    raise FileNotFoundError("❌ No Phase 3 runs found!")

PHASE3_RUN = phase3_runs[-1]
print(f"\n📁 Loading from Phase 3 run: {PHASE3_RUN.name}")

CHECKPOINT_PATH = PHASE3_RUN / 'phase_3_models'
RESULTS_PATH = PHASE3_RUN / 'phase_3_results'
EVIDENCE_PATH = PHASE3_RUN / 'evidence_store'

# Find Phase 2 run (most recent in v302 folder, excluding phase3)
phase2_runs = sorted([d for d in SHIFAMIND_V302_BASE.glob('run_*') if d.is_dir() and '_phase3' not in d.name])
if not phase2_runs:
    raise FileNotFoundError("❌ No Phase 2 runs found!")
PHASE2_RUN = phase2_runs[-1]
print(f"📁 Phase 2 run: {PHASE2_RUN.name}")

GRAPH_PATH = PHASE2_RUN / 'phase_2_graph'
PHASE2_CHECKPOINT = PHASE2_RUN / 'phase_2_models' / 'phase2_best.pt'

# Phase 1 shared data - look in 10_ShifaMind folder (NOT 11_ShifaMind_v302!)
PHASE1_BASE = BASE_PATH / '10_ShifaMind'
phase1_folders = sorted([d for d in PHASE1_BASE.glob('run_*') if d.is_dir()], reverse=True)
if not phase1_folders:
    raise FileNotFoundError("❌ No Phase 1 run found in 10_ShifaMind!")
PHASE1_RUN = phase1_folders[0]
print(f"📁 Phase 1 shared data: {PHASE1_RUN.name} (from 10_ShifaMind)")

SHARED_DATA_PATH = PHASE1_RUN / 'shared_data'

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🔥 GPU: {gpu_name}")
    print(f"💾 VRAM: {gpu_memory:.1f} GB")

# Constants
NUM_CONCEPTS = 111
NUM_DIAGNOSES = 50
SEED = 42
BATCH_SIZE = 192  # Large batch for inference

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================================
# LOAD SHARED DATA
# ============================================================================

print("\n" + "="*80)
print("📋 LOADING DATA")
print("="*80)

# Load splits
with open(SHARED_DATA_PATH / 'train_split.pkl', 'rb') as f:
    df_train = pickle.load(f)
with open(SHARED_DATA_PATH / 'val_split.pkl', 'rb') as f:
    df_val = pickle.load(f)
with open(SHARED_DATA_PATH / 'test_split.pkl', 'rb') as f:
    df_test = pickle.load(f)

# Load concept/diagnosis mappings - EXACT same as phase3_training_optimized.py
# Load concept list from Phase 1 run
with open(SHARED_DATA_PATH / 'concept_list.json', 'r') as f:
    ALL_CONCEPTS = json.load(f)

# Load Top-50 codes from original run (same as Phase 3 training)
ORIGINAL_RUN = BASE_PATH / '10_ShifaMind' / 'run_20260102_203225'
ORIGINAL_SHARED = ORIGINAL_RUN / 'shared_data'
with open(ORIGINAL_SHARED / 'top50_icd10_info.json', 'r') as f:
    top50_info = json.load(f)
    top_50_codes = top50_info['top_50_codes']

print(f"\n✅ Loaded data:")
print(f"   Train: {len(df_train):,} samples")
print(f"   Val:   {len(df_val):,} samples")
print(f"   Test:  {len(df_test):,} samples")
print(f"   Concepts: {len(ALL_CONCEPTS)}")
print(f"   Diagnoses: {len(top_50_codes)}")

# ============================================================================
# DATASET
# ============================================================================

class RAGDataset(Dataset):
    def __init__(self, df, tokenizer, concept_labels, top50_codes):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.concept_labels = concept_labels
        self.top50_codes = top50_codes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text'])

        # Multi-label format
        if isinstance(self.top50_codes, list):
            labels = row[self.top50_codes].values.astype(np.float32)
        else:
            labels = row['labels']

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.FloatTensor(labels),
            'text': text
        }

# ============================================================================
# RAG RETRIEVER
# ============================================================================

class RAGRetriever:
    def __init__(self, corpus_path, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        print(f"\n🤖 Initializing RAG with {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        self.encoder = self.encoder.to(device)
        print(f"✅ RAG encoder loaded on {device}")

        # Load corpus
        with open(corpus_path, 'r') as f:
            self.corpus = json.load(f)

        print(f"📚 Loaded {len(self.corpus)} evidence passages")

        # Build FAISS index
        self._build_index()

    def _build_index(self):
        print("\n🔨 Building FAISS index...")
        texts = [item['text'] for item in self.corpus]
        embeddings = self.encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        # Try GPU FAISS first
        try:
            res = faiss.StandardGpuResources()
            index_flat = faiss.IndexFlatIP(embeddings.shape[1])
            self.index = faiss.index_cpu_to_gpu(res, 0, index_flat)
            self.index.add(embeddings.astype('float32'))
            print(f"✅ GPU FAISS index built: {embeddings.shape[1]} dims, {len(embeddings)} vectors")
        except (AttributeError, RuntimeError):
            # Fallback to CPU
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings.astype('float32'))
            print(f"✅ CPU FAISS index built: {embeddings.shape[1]} dims, {len(embeddings)} vectors")

    def retrieve(self, query_text, top_k=3):
        """Retrieve top-k relevant passages"""
        query_emb = self.encoder.encode([query_text], convert_to_numpy=True)
        scores, indices = self.index.search(query_emb.astype('float32'), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                'text': self.corpus[idx]['text'],
                'score': float(score),
                'diagnosis': self.corpus[idx].get('diagnosis', 'N/A'),
                'source': self.corpus[idx].get('source', 'unknown')
            })

        return results

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class GATEncoder(nn.Module):
    """GAT encoder for learning concept embeddings from knowledge graph"""
    def __init__(self, in_channels, hidden_channels, num_layers=2, heads=4, dropout=0.3):
        super().__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        # First layer: in -> hidden
        self.convs.append(gnn.GATConv(
            in_channels,
            hidden_channels // heads,  # Output per head
            heads=heads,
            dropout=dropout,
            concat=True
        ))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(gnn.GATConv(
                hidden_channels,
                hidden_channels // heads,
                heads=heads,
                dropout=dropout,
                concat=True
            ))

        # Last layer: hidden -> hidden (average heads)
        if num_layers > 1:
            self.convs.append(gnn.GATConv(
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


class ShifaMindPhase2GAT(nn.Module):
    """Phase 2 model - EXACT architecture"""
    def __init__(self, bert_model, gat_encoder, graph_data, num_concepts, num_diagnoses):
        super().__init__()

        self.bert = bert_model
        self.gat = gat_encoder
        self.hidden_size = 768
        self.graph_hidden = 256
        self.num_concepts = num_concepts
        self.num_diagnoses = num_diagnoses

        # Store graph
        self.register_buffer('graph_x', graph_data.x)
        self.register_buffer('graph_edge_index', graph_data.edge_index)
        self.graph_node_to_idx = graph_data.node_to_idx
        self.graph_idx_to_node = graph_data.idx_to_node

        # Project graph embeddings to BERT dimension
        self.graph_proj = nn.Linear(self.graph_hidden, self.hidden_size)

        # Concept fusion
        self.concept_fusion = nn.Sequential(
            nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Cross-attention
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
        graph_embeddings = self.gat(self.graph_x, self.graph_edge_index)

        concept_embeds = []
        for concept in ALL_CONCEPTS:
            if concept in self.graph_node_to_idx:
                idx = self.graph_node_to_idx[concept]
                concept_embeds.append(graph_embeddings[idx])
            else:
                concept_embeds.append(torch.zeros(self.graph_hidden, device=self.graph_x.device))

        return torch.stack(concept_embeds)

    def forward(self, input_ids, attention_mask, concept_embeddings_bert):
        batch_size = input_ids.shape[0]

        # 1. Encode text with BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # 2. Get GAT-enhanced concept embeddings
        gat_concepts = self.get_graph_concept_embeddings()
        gat_concepts = self.graph_proj(gat_concepts)  # Project to 768-dim

        # 3. Fuse BERT + GAT concept embeddings
        bert_concepts = concept_embeddings_bert.unsqueeze(0).expand(batch_size, -1, -1)
        gat_concepts_batched = gat_concepts.unsqueeze(0).expand(batch_size, -1, -1)

        fused_input = torch.cat([bert_concepts, gat_concepts_batched], dim=-1)
        enhanced_concepts = self.concept_fusion(fused_input)

        # 4. Cross-attention
        context, attn_weights = self.cross_attention(
            query=hidden_states,
            key=enhanced_concepts,
            value=enhanced_concepts,
            need_weights=True
        )

        # 5. Multiplicative bottleneck gating
        pooled_text = hidden_states.mean(dim=1)
        pooled_context = context.mean(dim=1)

        gate_input = torch.cat([pooled_text, pooled_context], dim=-1)
        gate = self.gate_net(gate_input)

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


class ShifaMindPhase3RAG(nn.Module):
    """Phase 3: Phase 2 + RAG integration"""
    def __init__(self, phase2_model, rag_retriever, hidden_size=768):
        super().__init__()

        self.phase2_model = phase2_model
        self.rag = rag_retriever
        self.hidden_size = hidden_size

        # RAG components
        rag_dim = 384  # all-MiniLM-L6-v2 embedding size
        self.rag_projection = nn.Linear(rag_dim, hidden_size)

        self.rag_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Final diagnosis head
        self.diagnosis_head = nn.Linear(hidden_size, phase2_model.num_diagnoses)

    def forward(self, input_ids, attention_mask, concept_embeddings_bert, texts=None, use_rag=True):
        # Get Phase 2 outputs
        phase2_out = self.phase2_model(input_ids, attention_mask, concept_embeddings_bert)
        bottleneck = phase2_out['bottleneck_output']  # [batch, 768]

        if not use_rag or texts is None:
            # No RAG - just use Phase 2 diagnosis head
            return {
                **phase2_out,
                'logits': self.diagnosis_head(bottleneck),
                'rag_gate_values': torch.zeros(bottleneck.shape[0], 1, device=bottleneck.device)
            }

        # RAG retrieval
        batch_size = bottleneck.shape[0]
        rag_embeddings = []

        for text in texts:
            # Retrieve evidence
            evidence = self.rag.retrieve(text, top_k=3)

            # Encode retrieved passages
            evidence_texts = [e['text'] for e in evidence]
            evidence_embs = self.rag.encoder.encode(evidence_texts, convert_to_numpy=False, convert_to_tensor=True)
            evidence_embs = evidence_embs.to(bottleneck.device)

            # Pool evidence embeddings
            pooled_evidence = evidence_embs.mean(dim=0)  # [384]
            rag_embeddings.append(pooled_evidence)

        rag_embeddings = torch.stack(rag_embeddings)  # [batch, 384]
        rag_projected = self.rag_projection(rag_embeddings)  # [batch, 768]

        # Gated fusion of Phase 2 + RAG
        fusion_input = torch.cat([bottleneck, rag_projected], dim=-1)  # [batch, 1536]
        rag_gate_value = self.rag_gate(fusion_input)  # [batch, 1]

        # Weighted combination
        combined = rag_gate_value * rag_projected + (1 - rag_gate_value) * bottleneck

        # Final fusion
        final_repr = self.final_fusion(torch.cat([bottleneck, combined], dim=-1))

        # Diagnosis prediction
        diagnosis_logits = self.diagnosis_head(final_repr)

        return {
            **phase2_out,
            'logits': diagnosis_logits,
            'rag_gate_values': rag_gate_value
        }

# ============================================================================
# LOAD MODEL
# ============================================================================

print("\n" + "="*80)
print("🏗️  LOADING PHASE 3 MODEL")
print("="*80)

# Load graph
print("\n📊 Loading Phase 2 graph data...")
graph_data = torch.load(GRAPH_PATH / 'graph_data.pt', map_location='cpu', weights_only=False)
graph_data = graph_data.to(device)
print(f"✅ Loaded graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")

# Load BERT
print("\n🤖 Loading BioClinicalBERT...")
tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
bert_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

# Build GAT
print("🔨 Building GAT encoder...")
gat_encoder = GATEncoder(
    in_channels=graph_data.num_node_features,
    hidden_channels=256,
    num_layers=2,
    heads=4,
    dropout=0.3
)

# Initialize Phase 2 model
print("🏗️  Initializing Phase 2 model...")
phase2_model = ShifaMindPhase2GAT(
    bert_model=bert_model,
    gat_encoder=gat_encoder,
    graph_data=graph_data,
    num_concepts=NUM_CONCEPTS,
    num_diagnoses=NUM_DIAGNOSES
)

# Load trained concept embeddings from Phase 2
print("\n📥 Loading Phase 2 checkpoint...")
phase2_checkpoint = torch.load(PHASE2_CHECKPOINT, map_location='cpu', weights_only=False)

if 'concept_embeddings' in phase2_checkpoint:
    concept_emb_tensor = phase2_checkpoint['concept_embeddings']
    print(f"✅ Loaded trained concept embeddings: {concept_emb_tensor.shape}")
    concept_embeddings_bert = nn.Embedding(NUM_CONCEPTS, 768)
    concept_embeddings_bert.weight = nn.Parameter(concept_emb_tensor)
else:
    raise KeyError("❌ No concept_embeddings in Phase 2 checkpoint!")

concept_embeddings_bert = concept_embeddings_bert.to(device)

# Load Phase 2 weights
phase2_model.load_state_dict(phase2_checkpoint['model_state_dict'], strict=True)
phase2_model = phase2_model.to(device)
print("✅ Phase 2 model loaded")

# Load RAG retriever
print("\n📚 Loading RAG retriever...")
rag_retriever = RAGRetriever(EVIDENCE_PATH / 'evidence_corpus.json')

# Initialize Phase 3 model
print("\n🏗️  Initializing Phase 3 model...")
model = ShifaMindPhase3RAG(phase2_model, rag_retriever, hidden_size=768)
model = model.to(device)

# Load Phase 3 checkpoint
print("\n📥 Loading Phase 3 best model...")
checkpoint = torch.load(CHECKPOINT_PATH / 'best_model.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Loaded best model from epoch {checkpoint['epoch']}")
print(f"   Best Val F1: {checkpoint['best_f1']:.4f}")

# ============================================================================
# PREPARE DATASETS
# ============================================================================

print("\n" + "="*80)
print("📦 PREPARING DATASETS")
print("="*80)

val_dataset = RAGDataset(df_val, tokenizer, ALL_CONCEPTS, top_50_codes)
test_dataset = RAGDataset(df_test, tokenizer, ALL_CONCEPTS, top_50_codes)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"\n✅ Datasets ready:")
print(f"   Val batches:  {len(val_loader)}")
print(f"   Test batches: {len(test_loader)}")

# ============================================================================
# FINAL TEST EVALUATION (with default threshold 0.5)
# ============================================================================

print("\n" + "="*80)
print("📊 FINAL TEST EVALUATION (threshold=0.5)")
print("="*80)

all_preds = []
all_labels = []
all_probs = []

print("\n🔍 Evaluating on test set...")
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        texts = batch['text']

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            concept_embeddings_bert=concept_embeddings_bert.weight,
            texts=texts,
            use_rag=True
        )

        probs = torch.sigmoid(outputs['logits'])
        preds = (probs > 0.5).float()

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)
all_probs = np.vstack(all_probs)

# Compute metrics
test_f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
test_f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
test_f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
test_precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
test_recall = recall_score(all_labels, all_preds, average='micro', zero_division=0)
test_accuracy = accuracy_score(all_labels, all_preds)

print(f"\n📊 Test Results (threshold=0.5):")
print(f"   F1 (micro):    {test_f1_micro:.4f}")
print(f"   F1 (macro):    {test_f1_macro:.4f}")
print(f"   F1 (weighted): {test_f1_weighted:.4f}")
print(f"   Precision:     {test_precision:.4f}")
print(f"   Recall:        {test_recall:.4f}")
print(f"   Accuracy:      {test_accuracy:.4f}")

# ============================================================================
# THRESHOLD TUNING ON VALIDATION SET
# ============================================================================

print("\n" + "="*80)
print("🎯 THRESHOLD TUNING ON VALIDATION SET")
print("="*80)

# Get validation predictions
val_preds_all = []
val_labels_all = []
val_probs_all = []

print("\n🔍 Getting validation predictions...")
with torch.no_grad():
    for batch in tqdm(val_loader, desc="Validation"):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        texts = batch['text']

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            concept_embeddings_bert=concept_embeddings_bert.weight,
            texts=texts,
            use_rag=True
        )

        probs = torch.sigmoid(outputs['logits'])

        val_probs_all.append(probs.cpu().numpy())
        val_labels_all.append(labels.cpu().numpy())

val_probs_all = np.vstack(val_probs_all)
val_labels_all = np.vstack(val_labels_all)

# Tune threshold
print("\n🔍 Searching for optimal threshold...")
thresholds = np.arange(0.1, 0.9, 0.05)
best_threshold = 0.5
best_f1 = 0.0

threshold_results = []

for threshold in tqdm(thresholds, desc="Tuning"):
    val_preds_thresh = (val_probs_all > threshold).astype(int)
    f1 = f1_score(val_labels_all, val_preds_thresh, average='micro', zero_division=0)

    threshold_results.append({
        'threshold': float(threshold),
        'f1_micro': float(f1)
    })

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\n✅ Optimal threshold found: {best_threshold:.2f}")
print(f"   Val F1 (micro): {best_f1:.4f}")

# Save threshold tuning results
with open(RESULTS_PATH / 'threshold_tuning.json', 'w') as f:
    json.dump({
        'best_threshold': float(best_threshold),
        'best_val_f1': float(best_f1),
        'all_thresholds': threshold_results
    }, f, indent=2)

print(f"\n💾 Saved threshold tuning results to: {RESULTS_PATH / 'threshold_tuning.json'}")

# ============================================================================
# FINAL TEST EVALUATION WITH TUNED THRESHOLD
# ============================================================================

print("\n" + "="*80)
print(f"📊 FINAL TEST EVALUATION (threshold={best_threshold:.2f})")
print("="*80)

# Apply tuned threshold
test_preds_tuned = (all_probs > best_threshold).astype(int)

# Compute metrics
tuned_f1_micro = f1_score(all_labels, test_preds_tuned, average='micro', zero_division=0)
tuned_f1_macro = f1_score(all_labels, test_preds_tuned, average='macro', zero_division=0)
tuned_f1_weighted = f1_score(all_labels, test_preds_tuned, average='weighted', zero_division=0)
tuned_precision = precision_score(all_labels, test_preds_tuned, average='micro', zero_division=0)
tuned_recall = recall_score(all_labels, test_preds_tuned, average='micro', zero_division=0)
tuned_accuracy = accuracy_score(all_labels, test_preds_tuned)

print(f"\n📊 Final Test Results (tuned threshold={best_threshold:.2f}):")
print(f"   F1 (micro):    {tuned_f1_micro:.4f}")
print(f"   F1 (macro):    {tuned_f1_macro:.4f}")
print(f"   F1 (weighted): {tuned_f1_weighted:.4f}")
print(f"   Precision:     {tuned_precision:.4f}")
print(f"   Recall:        {tuned_recall:.4f}")
print(f"   Accuracy:      {tuned_accuracy:.4f}")

print("\n📈 Improvement from threshold tuning:")
print(f"   ΔF1 (micro):    {tuned_f1_micro - test_f1_micro:+.4f}")
print(f"   ΔF1 (macro):    {tuned_f1_macro - test_f1_macro:+.4f}")
print(f"   ΔPrecision:     {tuned_precision - test_precision:+.4f}")
print(f"   ΔRecall:        {tuned_recall - test_recall:+.4f}")

# ============================================================================
# SAVE FINAL RESULTS
# ============================================================================

print("\n" + "="*80)
print("💾 SAVING RESULTS")
print("="*80)

final_results = {
    'model': 'ShifaMind v302 Phase 3 (RAG)',
    'timestamp': datetime.now().isoformat(),
    'test_samples': len(df_test),

    'results_default_threshold': {
        'threshold': 0.5,
        'f1_micro': float(test_f1_micro),
        'f1_macro': float(test_f1_macro),
        'f1_weighted': float(test_f1_weighted),
        'precision': float(test_precision),
        'recall': float(test_recall),
        'accuracy': float(test_accuracy)
    },

    'threshold_tuning': {
        'best_threshold': float(best_threshold),
        'val_f1': float(best_f1)
    },

    'results_tuned_threshold': {
        'threshold': float(best_threshold),
        'f1_micro': float(tuned_f1_micro),
        'f1_macro': float(tuned_f1_macro),
        'f1_weighted': float(tuned_f1_weighted),
        'precision': float(tuned_precision),
        'recall': float(tuned_recall),
        'accuracy': float(tuned_accuracy)
    },

    'improvement': {
        'delta_f1_micro': float(tuned_f1_micro - test_f1_micro),
        'delta_f1_macro': float(tuned_f1_macro - test_f1_macro),
        'delta_precision': float(tuned_precision - test_precision),
        'delta_recall': float(tuned_recall - test_recall)
    }
}

# Save results
with open(RESULTS_PATH / 'final_test_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print(f"✅ Saved final results to: {RESULTS_PATH / 'final_test_results.json'}")

# Per-diagnosis results
per_diagnosis_metrics = []
for i, dx_code in enumerate(top_50_codes):
    dx_f1 = f1_score(all_labels[:, i], test_preds_tuned[:, i], zero_division=0)
    dx_precision = precision_score(all_labels[:, i], test_preds_tuned[:, i], zero_division=0)
    dx_recall = recall_score(all_labels[:, i], test_preds_tuned[:, i], zero_division=0)

    per_diagnosis_metrics.append({
        'diagnosis': dx_code,
        'f1': float(dx_f1),
        'precision': float(dx_precision),
        'recall': float(dx_recall),
        'support': int(all_labels[:, i].sum())
    })

with open(RESULTS_PATH / 'per_diagnosis_metrics.json', 'w') as f:
    json.dump(per_diagnosis_metrics, f, indent=2)

print(f"✅ Saved per-diagnosis metrics to: {RESULTS_PATH / 'per_diagnosis_metrics.json'}")

# Save predictions
np.save(RESULTS_PATH / 'test_predictions.npy', test_preds_tuned)
np.save(RESULTS_PATH / 'test_probabilities.npy', all_probs)
np.save(RESULTS_PATH / 'test_labels.npy', all_labels)

print(f"✅ Saved predictions to: {RESULTS_PATH}")

print("\n" + "="*80)
print("✅ PHASE 3 EVALUATION COMPLETE!")
print("="*80)
print(f"\n🎯 Best threshold: {best_threshold:.2f}")
print(f"📊 Final F1 (micro): {tuned_f1_micro:.4f}")
print(f"📊 Final F1 (macro): {tuned_f1_macro:.4f}")
print(f"🎉 All results saved to: {RESULTS_PATH}")
print("\n" + "="*80)
