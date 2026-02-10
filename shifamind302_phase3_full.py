#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND v302 PHASE 3 FULL: RAG + GAT (Winner Config)
================================================================================
Author: Mohammed Sameer Syed
University of Arizona - MS in AI Capstone

Full training with winning configuration from pilot study:
- v301 Conservative: λ_dx=2.0, λ_align=0.5, λ_concept=0.3
- Learning rate: 5e-6
- RAG gate max: 0.4
- 5 epochs training
- Test set evaluation

Architecture (3-way fusion):
1. BERT text encoding
2. RAG retrieval & gated fusion
3. GAT concept embeddings from UMLS graph
4. Concept fusion: BERT concepts + GAT concepts
5. Cross-attention with RAG-enhanced text
6. Output heads

Target: F1 > 0.40 (beat LAAT baseline)

================================================================================
"""

print("="*80)
print("🚀 SHIFAMIND v302 PHASE 3 FULL TRAINING")
print("="*80)
print("\nUsing winning configuration: v301 Conservative")
print("   λ_dx=2.0, λ_align=0.5, λ_concept=0.3")
print("   LR=5e-6, RAG_gate_max=0.4")

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
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer, AutoModel,
    get_linear_schedule_with_warmup
)

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
from typing import Dict, List
from collections import defaultdict
import sys
import time

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")

# ============================================================================
# CONFIGURATION
# ============================================================================

print("\n" + "="*80)
print("⚙️  CONFIGURATION")
print("="*80)

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
SHIFAMIND302_BASE = BASE_PATH / '11_ShifaMind_v302'

run_folders = sorted([d for d in SHIFAMIND302_BASE.glob('run_*') if d.is_dir()], reverse=True)

if not run_folders:
    print("❌ No v302 Phase 2 run found!")
    sys.exit(1)

OUTPUT_BASE = run_folders[0]
print(f"📁 Using run folder: {OUTPUT_BASE.name}")

# Load Phase 2 config
PHASE2_CHECKPOINT = OUTPUT_BASE / 'phase_2_models' / 'phase2_best.pt'
if not PHASE2_CHECKPOINT.exists():
    print(f"❌ Phase 2 checkpoint not found!")
    sys.exit(1)

checkpoint = torch.load(PHASE2_CHECKPOINT, map_location='cpu', weights_only=False)
phase2_config = checkpoint['config']
TOP_50_CODES = phase2_config['top_50_codes']
NUM_CONCEPTS = phase2_config['num_concepts']
NUM_DIAGNOSES = phase2_config['num_diagnoses']
GRAPH_HIDDEN_DIM = phase2_config['graph_hidden_dim']
GAT_HEADS = phase2_config['gat_heads']
GAT_LAYERS = phase2_config['gat_layers']

print(f"✅ Loaded Phase 2 config:")
print(f"   Top-50 codes: {len(TOP_50_CODES)}")
print(f"   Concepts: {NUM_CONCEPTS}")

# Load v301 data
SHIFAMIND301_BASE = BASE_PATH / '10_ShifaMind'
v301_run_folders = sorted([d for d in SHIFAMIND301_BASE.glob('run_*') if d.is_dir()], reverse=True)
if v301_run_folders:
    V301_SHARED_DATA = v301_run_folders[0] / 'shared_data'
else:
    print("❌ v301 run folder not found!")
    sys.exit(1)

# Paths
CHECKPOINT_PATH = OUTPUT_BASE / 'phase_3_models'
RESULTS_PATH = OUTPUT_BASE / 'phase_3_results'
EVIDENCE_PATH = OUTPUT_BASE / 'phase_3_evidence'

for path in [CHECKPOINT_PATH, RESULTS_PATH, EVIDENCE_PATH]:
    path.mkdir(parents=True, exist_ok=True)

with open(V301_SHARED_DATA / 'concept_list.json', 'r') as f:
    ALL_CONCEPTS = json.load(f)

# Winning hyperparameters
LAMBDA_DX = 2.0
LAMBDA_ALIGN = 0.5
LAMBDA_CONCEPT = 0.3
LEARNING_RATE = 5e-6
RAG_GATE_MAX = 0.4
RAG_TOP_K = 3
RAG_THRESHOLD = 0.7
PROTOTYPES_PER_DIAGNOSIS = 20
EPOCHS = 5
BATCH_SIZE = 8

print(f"\n⚖️  Hyperparameters:")
print(f"   λ_dx={LAMBDA_DX}, λ_align={LAMBDA_ALIGN}, λ_concept={LAMBDA_CONCEPT}")
print(f"   LR={LEARNING_RATE}, RAG_gate_max={RAG_GATE_MAX}")
print(f"   Epochs={EPOCHS}")

# ============================================================================
# LOAD OR BUILD EVIDENCE CORPUS
# ============================================================================

print("\n" + "="*80)
print("📚 LOADING EVIDENCE CORPUS")
print("="*80)

evidence_corpus_path = EVIDENCE_PATH / 'evidence_corpus.json'

if evidence_corpus_path.exists():
    print(f"✅ Loading existing corpus from: {evidence_corpus_path}")
    with open(evidence_corpus_path, 'r') as f:
        evidence_corpus = json.load(f)
    print(f"   Total passages: {len(evidence_corpus)}")
else:
    print("⚠️  Evidence corpus not found. Please run shifamind302_phase3.py first to build corpus.")
    sys.exit(1)

# ============================================================================
# BUILD FAISS RETRIEVER
# ============================================================================

print("\n" + "="*80)
print("🔍 BUILDING FAISS RETRIEVER")
print("="*80)

class SimpleRAG:
    """Simple RAG using FAISS + sentence-transformers"""
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', top_k=3, threshold=0.7):
        print(f"\n🤖 Initializing RAG with {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        self.top_k = top_k
        self.threshold = threshold
        self.index = None
        self.documents = []
        print(f"✅ RAG encoder loaded")

    def build_index(self, documents: List[Dict]):
        print(f"\n🔨 Building FAISS index from {len(documents)} documents...")
        self.documents = documents
        texts = [doc['text'] for doc in documents]

        embeddings = self.encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings = embeddings.astype('float32')

        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print(f"✅ FAISS index built: {self.index.ntotal} vectors")

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

rag = SimpleRAG(top_k=RAG_TOP_K, threshold=RAG_THRESHOLD)
rag.build_index(evidence_corpus)

# ============================================================================
# LOAD PHASE 2 COMPONENTS
# ============================================================================

print("\n" + "="*80)
print("📥 LOADING PHASE 2 COMPONENTS")
print("="*80)

import torch_geometric
from torch_geometric.nn import GATConv

graph_data_path = OUTPUT_BASE / 'phase_2_graph' / 'graph_data.pt'
graph_data = torch.load(graph_data_path, map_location='cpu', weights_only=False)
print(f"✅ Loaded graph data: {graph_data.x.shape[0]} nodes, {graph_data.edge_index.shape[1]} edges")

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

gat_encoder = GATEncoder(
    in_channels=768,
    hidden_channels=GRAPH_HIDDEN_DIM,
    num_heads=GAT_HEADS,
    num_layers=GAT_LAYERS
).to(device)

print(f"✅ GAT encoder initialized")

# ============================================================================
# BUILD PHASE 3 MODEL
# ============================================================================

print("\n" + "="*80)
print("🏗️  BUILDING PHASE 3 MODEL")
print("="*80)

class ShifaMind302Phase3(nn.Module):
    """ShifaMind v302 Phase 3: RAG + GAT fusion"""
    def __init__(self, bert_model, gat_encoder, rag_retriever, graph_data,
                 num_concepts, num_diagnoses, rag_gate_max=0.4):
        super().__init__()

        self.bert = bert_model
        self.gat = gat_encoder
        self.rag = rag_retriever
        self.hidden_size = 768
        self.graph_hidden = GRAPH_HIDDEN_DIM
        self.num_concepts = num_concepts
        self.num_diagnoses = num_diagnoses
        self.rag_gate_max = rag_gate_max

        self.register_buffer('graph_x', graph_data.x)
        self.register_buffer('graph_edge_index', graph_data.edge_index)
        self.graph_node_to_idx = graph_data.node_to_idx
        self.graph_idx_to_node = graph_data.idx_to_node

        rag_dim = 384
        self.rag_projection = nn.Linear(rag_dim, self.hidden_size)

        self.rag_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid()
        )

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

        if input_texts is not None:
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

        gat_concepts = self.get_graph_concept_embeddings()

        bert_concepts = concept_embeddings_bert.unsqueeze(0).expand(batch_size, -1, -1)
        gat_concepts_batched = gat_concepts.unsqueeze(0).expand(batch_size, -1, -1)

        fused_input = torch.cat([bert_concepts, gat_concepts_batched], dim=-1)
        enhanced_concepts = self.concept_fusion(fused_input)

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
            'gate_values': gate,
            'attention_weights': attn_weights
        }

tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
bert_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device)
concept_embedding_layer = nn.Embedding(NUM_CONCEPTS, 768).to(device)

model = ShifaMind302Phase3(
    bert_model=bert_model,
    gat_encoder=gat_encoder,
    rag_retriever=rag,
    graph_data=graph_data,
    num_concepts=NUM_CONCEPTS,
    num_diagnoses=NUM_DIAGNOSES,
    rag_gate_max=RAG_GATE_MAX
).to(device)

# Load Phase 2 checkpoint
print(f"\n📥 Loading Phase 2 checkpoint...")
checkpoint_data = torch.load(PHASE2_CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
print("✅ Loaded Phase 2 weights (partial)")

print(f"\n✅ Phase 3 model initialized")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("📦 LOADING DATA")
print("="*80)

with open(V301_SHARED_DATA / 'train_split.pkl', 'rb') as f:
    df_train = pickle.load(f)
with open(V301_SHARED_DATA / 'val_split.pkl', 'rb') as f:
    df_val = pickle.load(f)
with open(V301_SHARED_DATA / 'test_split.pkl', 'rb') as f:
    df_test = pickle.load(f)

train_concept_labels = np.load(V301_SHARED_DATA / 'train_concept_labels.npy')
val_concept_labels = np.load(V301_SHARED_DATA / 'val_concept_labels.npy')
test_concept_labels = np.load(V301_SHARED_DATA / 'test_concept_labels.npy')

print(f"✅ Data loaded:")
print(f"   Train: {len(df_train)}")
print(f"   Val:   {len(df_val)}")
print(f"   Test:  {len(df_test)}")

class RAGDataset(Dataset):
    def __init__(self, df, tokenizer, concept_labels):
        self.texts = df['text'].tolist()
        self.labels = df['labels'].tolist()
        self.tokenizer = tokenizer
        self.concept_labels = concept_labels

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
            'labels': torch.tensor(self.labels[idx], dtype=torch.float),
            'concept_labels': torch.tensor(self.concept_labels[idx], dtype=torch.float)
        }

train_dataset = RAGDataset(df_train, tokenizer, train_concept_labels)
val_dataset = RAGDataset(df_val, tokenizer, val_concept_labels)
test_dataset = RAGDataset(df_test, tokenizer, test_concept_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# ============================================================================
# TRAINING SETUP
# ============================================================================

print("\n" + "="*80)
print("⚙️  TRAINING SETUP")
print("="*80)

class MultiObjectiveLoss(nn.Module):
    def __init__(self, lambda_dx, lambda_align, lambda_concept):
        super().__init__()
        self.lambda_dx = lambda_dx
        self.lambda_align = lambda_align
        self.lambda_concept = lambda_concept
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, dx_labels, concept_labels):
        loss_dx = self.bce(outputs['logits'], dx_labels)

        dx_probs = torch.sigmoid(outputs['logits'])
        concept_scores = outputs['concept_scores']
        loss_align = torch.abs(dx_probs.unsqueeze(-1) - concept_scores.unsqueeze(1)).mean()

        loss_concept = self.bce(outputs['concept_logits'], concept_labels)

        total_loss = (
            self.lambda_dx * loss_dx +
            self.lambda_align * loss_align +
            self.lambda_concept * loss_concept
        )

        return total_loss, {
            'loss_dx': loss_dx.item(),
            'loss_align': loss_align.item(),
            'loss_concept': loss_concept.item(),
            'total_loss': total_loss.item()
        }

criterion = MultiObjectiveLoss(LAMBDA_DX, LAMBDA_ALIGN, LAMBDA_CONCEPT)

optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(concept_embedding_layer.parameters()),
    lr=LEARNING_RATE
)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

print(f"✅ Training setup complete")
print(f"   Optimizer: AdamW (lr={LEARNING_RATE})")
print(f"   Scheduler: Linear warmup (10% of steps)")

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("\n" + "="*80)
print("🏋️  TRAINING PHASE 3")
print("="*80)

best_val_f1 = 0.0
history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

concept_embeddings = concept_embedding_layer.weight.detach()

for epoch in range(EPOCHS):
    print(f"\n📍 Epoch {epoch+1}/{EPOCHS}")
    epoch_start_time = time.time()

    model.train()
    train_losses = []

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        concept_labels = batch['concept_labels'].to(device)
        texts = batch['text']

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask, concept_embeddings, input_texts=texts)
        loss, loss_components = criterion(outputs, labels, concept_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_train_loss = np.mean(train_losses)
    history['train_loss'].append(avg_train_loss)

    # Validation
    model.eval()
    val_losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            concept_labels = batch['concept_labels'].to(device)
            texts = batch['text']

            outputs = model(input_ids, attention_mask, concept_embeddings, input_texts=texts)
            loss, _ = criterion(outputs, labels, concept_labels)

            val_losses.append(loss.item())

            preds = (torch.sigmoid(outputs['logits']) > 0.5).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    avg_val_loss = np.mean(val_losses)
    val_f1 = f1_score(all_labels, all_preds, average='macro')

    history['val_loss'].append(avg_val_loss)
    history['val_f1'].append(val_f1)

    epoch_time = time.time() - epoch_start_time

    print(f"   Train Loss: {avg_train_loss:.4f}")
    print(f"   Val Loss:   {avg_val_loss:.4f}")
    print(f"   Val F1:     {val_f1:.4f}")
    print(f"   Time:       {epoch_time/60:.1f} min")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_val_f1,
            'concept_embeddings': concept_embeddings,
            'evidence_corpus': evidence_corpus,
            'config': {
                'num_concepts': NUM_CONCEPTS,
                'num_diagnoses': NUM_DIAGNOSES,
                'rag_config': {
                    'top_k': RAG_TOP_K,
                    'threshold': RAG_THRESHOLD,
                    'gate_max': RAG_GATE_MAX
                },
                'top_50_codes': TOP_50_CODES,
                'hyperparams': {
                    'lambda_dx': LAMBDA_DX,
                    'lambda_align': LAMBDA_ALIGN,
                    'lambda_concept': LAMBDA_CONCEPT,
                    'learning_rate': LEARNING_RATE
                }
            }
        }, CHECKPOINT_PATH / 'phase3_best.pt')
        print(f"   ✅ Saved best model (F1: {best_val_f1:.4f})")

# ============================================================================
# FINAL TEST EVALUATION
# ============================================================================

print("\n" + "="*80)
print("📊 FINAL TEST EVALUATION")
print("="*80)

checkpoint = torch.load(CHECKPOINT_PATH / 'phase3_best.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        texts = batch['text']

        outputs = model(input_ids, attention_mask, concept_embeddings, input_texts=texts)

        probs = torch.sigmoid(outputs['logits']).cpu().numpy()
        preds = (probs > 0.5).astype(int)

        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

macro_f1 = f1_score(all_labels, all_preds, average='macro')
micro_f1 = f1_score(all_labels, all_preds, average='micro')
macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

print("\n" + "="*80)
print("🎉 SHIFAMIND v302 PHASE 3 - FINAL RESULTS")
print("="*80)

print(f"\n🎯 Diagnosis Performance:")
print(f"   Macro F1:    {macro_f1:.4f}")
print(f"   Micro F1:    {micro_f1:.4f}")
print(f"   Precision:   {macro_precision:.4f}")
print(f"   Recall:      {macro_recall:.4f}")

print(f"\n📊 Top-10 Best Performing Diagnoses:")
top_10_best = sorted(zip(TOP_50_CODES, per_class_f1), key=lambda x: x[1], reverse=True)[:10]
for rank, (code, f1) in enumerate(top_10_best, 1):
    print(f"   {rank}. {code}: F1={f1:.4f}")

print(f"\n📈 Full Pipeline Progression:")
print(f"   v302 Phase 2 (GAT + UMLS):     F1 = 0.3121")
print(f"   v302 Phase 3 (+ RAG):          F1 = {macro_f1:.4f}")
improvement = (macro_f1 - 0.3121) / 0.3121 * 100
print(f"   Improvement:                   {improvement:+.1f}%")

print(f"\n📈 Comparison to v301:")
print(f"   v301 Phase 2:                  F1 = 0.2536")
print(f"   v301 Phase 3:                  F1 = 0.3831")
print(f"   v302 Phase 3:                  F1 = {macro_f1:.4f}")

if macro_f1 > 0.3831:
    delta = macro_f1 - 0.3831
    print(f"   🎉 BEAT v301 by {delta:.4f}!")

# Save results
results = {
    'phase': 'ShifaMind v302 Phase 3 - RAG + GAT (Full)',
    'timestamp': OUTPUT_BASE.name.replace('run_', ''),
    'run_folder': str(OUTPUT_BASE),
    'diagnosis_metrics': {
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'precision': float(macro_precision),
        'recall': float(macro_recall),
        'per_class_f1': {code: float(f1) for code, f1 in zip(TOP_50_CODES, per_class_f1)}
    },
    'architecture': '3-way Fusion: BERT + GAT + RAG',
    'hyperparameters': {
        'lambda_dx': LAMBDA_DX,
        'lambda_align': LAMBDA_ALIGN,
        'lambda_concept': LAMBDA_CONCEPT,
        'learning_rate': LEARNING_RATE,
        'rag_gate_max': RAG_GATE_MAX
    },
    'rag_config': {
        'method': 'FAISS + sentence-transformers',
        'top_k': RAG_TOP_K,
        'threshold': RAG_THRESHOLD,
        'corpus_size': len(evidence_corpus)
    },
    'training_history': history,
    'comparison': {
        'v302_phase2': 0.3121,
        'v301_phase3': 0.3831,
        'v302_phase3': float(macro_f1)
    }
}

with open(RESULTS_PATH / 'phase3_final_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {RESULTS_PATH / 'phase3_final_results.json'}")
print(f"💾 Best model saved to: {CHECKPOINT_PATH / 'phase3_best.pt'}")

print("\n" + "="*80)
print("✅ SHIFAMIND v302 PHASE 3 COMPLETE!")
print("="*80)
print(f"\n📍 Final Macro F1: {macro_f1:.4f}")

if macro_f1 >= 0.40:
    print("\n🏆 LIKELY BEAT LAAT BASELINE! Congratulations!")
elif macro_f1 >= 0.38:
    print("\n🎯 Very close to target! May have beaten LAAT!")
else:
    print("\n📌 Consider hyperparameter tuning or HeteroGNN improvements")

print("\nAlhamdulillah! 🤲")
