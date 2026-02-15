#!/usr/bin/env python3
"""
================================================================================
🎯 SHIFAMIND v302 PHASE 3: THRESHOLD OPTIMIZATION (MAXIMUM GPU OPTIMIZED)
================================================================================
Find optimal classification thresholds for each diagnosis code
Uses validation set to maximize F1 score per diagnosis
================================================================================
"""

print("="*80)
print("🎯 SHIFAMIND v302 PHASE 3 - THRESHOLD OPTIMIZATION")
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
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from transformers import AutoTokenizer, AutoModel

from sentence_transformers import SentenceTransformer

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("⚠️  FAISS not available - installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'faiss-cpu'])
    import faiss
    FAISS_AVAILABLE = True

import json
import pickle
from pathlib import Path
from tqdm.auto import tqdm
import sys

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# PATHS & CONFIGURATION
# ============================================================================

print("\n" + "="*80)
print("⚙️  CONFIGURATION")
print("="*80)

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
SHIFAMIND_V302_BASE = BASE_PATH / '11_ShifaMind_v302'

# Find newest Phase 3 run folder
run_folders = sorted([d for d in SHIFAMIND_V302_BASE.glob('run_*_phase3') if d.is_dir()], reverse=True)
if not run_folders:
    print("❌ No Phase 3 run found!")
    sys.exit(1)

RUN_FOLDER = run_folders[0]
print(f"\n📁 Loading from: {RUN_FOLDER.name}")

CHECKPOINT_PATH = RUN_FOLDER / 'phase_3_models'
RESULTS_PATH = RUN_FOLDER / 'phase_3_results'
SHARED_DATA_PATH = RUN_FOLDER / 'shared_data'
EVIDENCE_PATH = RUN_FOLDER / 'evidence_store'

# Find the data folder (should be from the Phase 2 run)
phase2_runs = sorted([d for d in SHIFAMIND_V302_BASE.glob('run_2026*') if d.is_dir() and 'phase3' not in d.name], reverse=True)
if phase2_runs:
    DATA_FOLDER = phase2_runs[0] / 'shared_data'
else:
    print("❌ No Phase 2 data folder found!")
    sys.exit(1)

print(f"📁 Data folder: {DATA_FOLDER}")

# Load checkpoint
BEST_MODEL_PATH = CHECKPOINT_PATH / 'best_model.pth'
if not BEST_MODEL_PATH.exists():
    print(f"❌ Best model not found at {BEST_MODEL_PATH}")
    sys.exit(1)

checkpoint = torch.load(BEST_MODEL_PATH, map_location='cpu')
config = checkpoint['config']

TOP_50_CODES = config['top_50_codes']
ALL_CONCEPTS = config['all_concepts']
NUM_LABELS = len(TOP_50_CODES)
NUM_CONCEPTS = len(ALL_CONCEPTS)

print(f"\n✅ Configuration loaded:")
print(f"   Diagnoses: {NUM_LABELS}")
print(f"   Concepts: {NUM_CONCEPTS}")

# GPU settings
print(f"\n🖥️  Device: {device}")

if device.type == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"🔥 GPU: {gpu_name}")
    print(f"💾 VRAM: {total_vram:.1f} GB")

    # MAXIMUM batch size for inference
    if total_vram >= 90:
        BATCH_SIZE = 256  # 16x original!
        NUM_WORKERS = 12
    elif total_vram >= 70:
        BATCH_SIZE = 192
        NUM_WORKERS = 10
    elif total_vram >= 40:
        BATCH_SIZE = 128
        NUM_WORKERS = 8
    else:
        BATCH_SIZE = 64
        NUM_WORKERS = 4
else:
    BATCH_SIZE = 16
    NUM_WORKERS = 2

USE_FP16 = device.type == 'cuda'

print(f"\n⚙️  Inference settings:")
print(f"   Batch size:  {BATCH_SIZE}")
print(f"   num_workers: {NUM_WORKERS}")
print(f"   FP16:        {USE_FP16}")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("📋 LOADING DATA")
print("="*80)

# Load validation and test splits
with open(DATA_FOLDER / 'val_split.pkl', 'rb') as f:
    df_val = pickle.load(f)
with open(DATA_FOLDER / 'test_split.pkl', 'rb') as f:
    df_test = pickle.load(f)

val_concept_labels = np.load(DATA_FOLDER / 'val_concept_labels.npy')
test_concept_labels = np.load(DATA_FOLDER / 'test_concept_labels.npy')

print(f"\n✅ Data loaded:")
print(f"   Val:  {len(df_val):,} samples")
print(f"   Test: {len(df_test):,} samples")

# ============================================================================
# LOAD MODEL ARCHITECTURE
# ============================================================================

print("\n" + "="*80)
print("🏗️  LOADING MODEL")
print("="*80)

# Import GAT
try:
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data
except ImportError:
    print("⚠️  Installing torch_geometric...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'torch_geometric'])
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data

class ShifaMindPhase2GAT(nn.Module):
    """Phase 2 model with GAT"""
    def __init__(self, num_concepts, num_labels, hidden_size=768, gat_heads=4, gat_layers=2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_concepts = num_concepts
        self.num_labels = num_labels

        self.bert = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.concept_embeddings = nn.Embedding(num_concepts, hidden_size)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.gat_layers = nn.ModuleList()
        for i in range(gat_layers):
            in_channels = hidden_size if i == 0 else hidden_size * gat_heads
            out_channels = hidden_size
            self.gat_layers.append(
                GATConv(in_channels, out_channels, heads=gat_heads, dropout=0.1, concat=True)
            )

        self.gat_projection = nn.Linear(hidden_size * gat_heads, hidden_size)

        self.concept_head = nn.Linear(hidden_size, num_concepts)
        self.diagnosis_head = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask, graph_data=None):
        batch_size = input_ids.shape[0]

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled_bert = hidden_states.mean(dim=1)

        concept_embs = self.concept_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1)

        bert_expanded = pooled_bert.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
        concept_context, concept_attn = self.cross_attention(
            query=bert_expanded,
            key=concept_embs,
            value=concept_embs,
            need_weights=True
        )

        pooled_context = concept_context.mean(dim=1)

        gate_input = torch.cat([pooled_bert, pooled_context], dim=-1)
        gate = self.gate_net(gate_input)

        bottleneck_output = gate * pooled_context
        bottleneck_output = self.layer_norm(bottleneck_output)

        concept_logits = self.concept_head(pooled_bert)

        if graph_data is not None:
            x = graph_data.x
            edge_index = graph_data.edge_index

            for gat in self.gat_layers:
                x = gat(x, edge_index)
                x = F.elu(x)

            x = self.gat_projection(x)
            graph_output = x.mean(dim=0, keepdim=True).expand(batch_size, -1)
        else:
            graph_output = torch.zeros_like(bottleneck_output)

        combined = torch.cat([bottleneck_output, graph_output], dim=-1)
        diagnosis_logits = self.diagnosis_head(combined)

        return {
            'logits': diagnosis_logits,
            'concept_logits': concept_logits,
            'concept_scores': torch.sigmoid(concept_logits),
            'gate_values': gate,
            'concept_attn': concept_attn
        }

class SimpleRAG:
    """Simple RAG using FAISS"""
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', top_k=3, threshold=0.7):
        self.encoder = SentenceTransformer(model_name)
        self.encoder.to(device)
        self.top_k = top_k
        self.threshold = threshold
        self.index = None
        self.documents = []

    def build_index(self, documents: list):
        self.documents = documents
        texts = [doc['text'] for doc in documents]

        embeddings = self.encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True, batch_size=256)
        embeddings = embeddings.astype('float32')

        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        if device.type == 'cuda':
            res = faiss.StandardGpuResources()
            cpu_index = faiss.IndexFlatIP(dimension)
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
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

class ShifaMindPhase3RAG(nn.Module):
    """Phase 3: Phase 2 + RAG"""
    def __init__(self, phase2_model, rag_retriever, hidden_size=768):
        super().__init__()

        self.phase2_model = phase2_model
        self.rag = rag_retriever
        self.hidden_size = hidden_size

        rag_dim = 384
        self.rag_projection = nn.Linear(rag_dim, hidden_size)

        self.rag_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, graph_data=None, input_texts=None):
        batch_size = input_ids.shape[0]

        bert_outputs = self.phase2_model.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = bert_outputs.last_hidden_state
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
            gate = self.rag_gate(gate_input)
            gate = gate * config['rag_config']['gate_max']

            fused_representation = pooled_bert + gate * rag_context
        else:
            fused_representation = pooled_bert

        # Phase 2 processing
        concept_embs = self.phase2_model.concept_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1)

        bert_expanded = fused_representation.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
        concept_context, concept_attn = self.phase2_model.cross_attention(
            query=bert_expanded,
            key=concept_embs,
            value=concept_embs,
            need_weights=True
        )

        pooled_context = concept_context.mean(dim=1)

        gate_input = torch.cat([fused_representation, pooled_context], dim=-1)
        gate = self.phase2_model.gate_net(gate_input)

        bottleneck_output = gate * pooled_context
        bottleneck_output = self.phase2_model.layer_norm(bottleneck_output)

        concept_logits = self.phase2_model.concept_head(fused_representation)

        if graph_data is not None:
            x = graph_data.x
            edge_index = graph_data.edge_index

            for gat in self.phase2_model.gat_layers:
                x = gat(x, edge_index)
                x = F.elu(x)

            x = self.phase2_model.gat_projection(x)
            graph_output = x.mean(dim=0, keepdim=True).expand(batch_size, -1)
        else:
            graph_output = torch.zeros_like(bottleneck_output)

        combined = torch.cat([bottleneck_output, graph_output], dim=-1)
        diagnosis_logits = self.phase2_model.diagnosis_head(combined)

        return {
            'logits': diagnosis_logits,
            'concept_logits': concept_logits,
            'concept_scores': torch.sigmoid(concept_logits),
            'gate_values': gate,
            'concept_attn': concept_attn
        }

# Load evidence corpus
print("\n📚 Loading evidence corpus...")
with open(EVIDENCE_PATH / 'evidence_corpus.json', 'r') as f:
    evidence_corpus = json.load(f)

print(f"✅ Loaded {len(evidence_corpus)} evidence passages")

# Build RAG
print("\n🔍 Building RAG retriever...")
rag = SimpleRAG(
    top_k=config['rag_config']['top_k'],
    threshold=config['rag_config']['threshold']
)
rag.build_index(evidence_corpus)

# Initialize models
print("\n🏗️  Initializing models...")
phase2_model = ShifaMindPhase2GAT(
    num_concepts=NUM_CONCEPTS,
    num_labels=NUM_LABELS,
    hidden_size=768,
    gat_heads=4,
    gat_layers=2
)

model = ShifaMindPhase3RAG(
    phase2_model=phase2_model,
    rag_retriever=rag,
    hidden_size=768
).to(device)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Model loaded successfully")

# Load graph
print("\n📊 Loading knowledge graph...")
# Find Phase 2 run with graph
phase2_run_with_graph = None
for run in phase2_runs:
    graph_path = run / 'phase_2_graph' / 'umls_graph.pt'
    if graph_path.exists():
        phase2_run_with_graph = run
        break

if phase2_run_with_graph is None:
    print("⚠️  No knowledge graph found - running without graph")
    graph_data = None
else:
    GRAPH_PATH = phase2_run_with_graph / 'phase_2_graph'
    graph_data = torch.load(GRAPH_PATH / 'umls_graph.pt', map_location=device)
    print(f"✅ Graph loaded: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")

# ============================================================================
# DATASET
# ============================================================================

print("\n" + "="*80)
print("📦 PREPARING DATASETS")
print("="*80)

class RAGDataset(Dataset):
    def __init__(self, df, tokenizer, concept_labels):
        self.texts = df['text'].tolist()
        self.labels = df['labels'].tolist()
        self.tokenizer = tokenizer
        self.concept_labels = concept_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'text': text,
            'labels': torch.tensor(self.labels[idx], dtype=torch.float),
            'concept_labels': torch.tensor(self.concept_labels[idx], dtype=torch.float)
        }

tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

val_dataset = RAGDataset(df_val, tokenizer, val_concept_labels)
test_dataset = RAGDataset(df_test, tokenizer, test_concept_labels)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print(f"✅ Datasets ready")

# ============================================================================
# GET PREDICTIONS ON VALIDATION SET
# ============================================================================

print("\n" + "="*80)
print("🔮 GENERATING PREDICTIONS ON VALIDATION SET")
print("="*80)

all_val_probs = []
all_val_labels = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Validation"):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        texts = batch['text']

        if USE_FP16:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask, graph_data, input_texts=texts)
        else:
            outputs = model(input_ids, attention_mask, graph_data, input_texts=texts)

        probs = torch.sigmoid(outputs['logits']).cpu().numpy()

        all_val_probs.append(probs)
        all_val_labels.append(labels.cpu().numpy())

all_val_probs = np.vstack(all_val_probs)
all_val_labels = np.vstack(all_val_labels)

print(f"\n✅ Validation predictions: {all_val_probs.shape}")

# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

print("\n" + "="*80)
print("🎯 OPTIMIZING THRESHOLDS PER DIAGNOSIS")
print("="*80)

optimal_thresholds = []
threshold_results = []

print("\n🔍 Searching for optimal thresholds...")

for label_idx in tqdm(range(NUM_LABELS), desc="Optimizing"):
    y_true = all_val_labels[:, label_idx]
    y_prob = all_val_probs[:, label_idx]

    # Skip if no positive samples
    if y_true.sum() == 0:
        optimal_thresholds.append(0.5)
        threshold_results.append({
            'code': TOP_50_CODES[label_idx],
            'optimal_threshold': 0.5,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'support': 0
        })
        continue

    # Try thresholds from 0.1 to 0.9
    thresholds = np.arange(0.1, 0.91, 0.05)
    best_f1 = 0.0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    optimal_thresholds.append(best_threshold)

    # Calculate metrics at optimal threshold
    y_pred_opt = (y_prob >= best_threshold).astype(int)
    precision = precision_score(y_true, y_pred_opt, zero_division=0)
    recall = recall_score(y_true, y_pred_opt, zero_division=0)

    threshold_results.append({
        'code': TOP_50_CODES[label_idx],
        'optimal_threshold': float(best_threshold),
        'f1': float(best_f1),
        'precision': float(precision),
        'recall': float(recall),
        'support': int(y_true.sum())
    })

optimal_thresholds = np.array(optimal_thresholds)

print(f"\n✅ Threshold optimization complete!")
print(f"\n📊 Threshold Statistics:")
print(f"   Mean:   {optimal_thresholds.mean():.3f}")
print(f"   Median: {np.median(optimal_thresholds):.3f}")
print(f"   Min:    {optimal_thresholds.min():.3f}")
print(f"   Max:    {optimal_thresholds.max():.3f}")

# Top/Bottom codes by threshold
sorted_by_threshold = sorted(threshold_results, key=lambda x: x['optimal_threshold'], reverse=True)
print(f"\n🔝 Top 5 Highest Thresholds:")
for result in sorted_by_threshold[:5]:
    print(f"   {result['code']}: {result['optimal_threshold']:.3f} (F1={result['f1']:.3f})")

print(f"\n🔽 Top 5 Lowest Thresholds:")
for result in sorted_by_threshold[-5:]:
    print(f"   {result['code']}: {result['optimal_threshold']:.3f} (F1={result['f1']:.3f})")

# ============================================================================
# EVALUATE ON TEST SET WITH OPTIMAL THRESHOLDS
# ============================================================================

print("\n" + "="*80)
print("📊 EVALUATING ON TEST SET WITH OPTIMAL THRESHOLDS")
print("="*80)

all_test_probs = []
all_test_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test"):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        texts = batch['text']

        if USE_FP16:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask, graph_data, input_texts=texts)
        else:
            outputs = model(input_ids, attention_mask, graph_data, input_texts=texts)

        probs = torch.sigmoid(outputs['logits']).cpu().numpy()

        all_test_probs.append(probs)
        all_test_labels.append(labels.cpu().numpy())

all_test_probs = np.vstack(all_test_probs)
all_test_labels = np.vstack(all_test_labels)

# Apply optimal thresholds
all_test_preds_optimized = (all_test_probs >= optimal_thresholds).astype(int)

# Also get default threshold predictions
all_test_preds_default = (all_test_probs >= 0.5).astype(int)

# Calculate metrics
macro_f1_optimized = f1_score(all_test_labels, all_test_preds_optimized, average='macro', zero_division=0)
micro_f1_optimized = f1_score(all_test_labels, all_test_preds_optimized, average='micro', zero_division=0)
macro_precision_optimized = precision_score(all_test_labels, all_test_preds_optimized, average='macro', zero_division=0)
macro_recall_optimized = recall_score(all_test_labels, all_test_preds_optimized, average='macro', zero_division=0)

macro_f1_default = f1_score(all_test_labels, all_test_preds_default, average='macro', zero_division=0)
micro_f1_default = f1_score(all_test_labels, all_test_preds_default, average='micro', zero_division=0)

print(f"\n🎯 Test Set Performance:")
print(f"\n   WITH OPTIMAL THRESHOLDS:")
print(f"   Macro F1:        {macro_f1_optimized:.4f}")
print(f"   Micro F1:        {micro_f1_optimized:.4f}")
print(f"   Macro Precision: {macro_precision_optimized:.4f}")
print(f"   Macro Recall:    {macro_recall_optimized:.4f}")

print(f"\n   WITH DEFAULT THRESHOLD (0.5):")
print(f"   Macro F1:        {macro_f1_default:.4f}")
print(f"   Micro F1:        {micro_f1_default:.4f}")

improvement = macro_f1_optimized - macro_f1_default
print(f"\n   📈 IMPROVEMENT: +{improvement:.4f} ({improvement/macro_f1_default*100:.1f}%)")

# Per-class F1
per_class_f1_optimized = f1_score(all_test_labels, all_test_preds_optimized, average=None, zero_division=0)

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("💾 SAVING RESULTS")
print("="*80)

results = {
    'optimal_thresholds': optimal_thresholds.tolist(),
    'threshold_details': threshold_results,
    'test_metrics_optimized': {
        'macro_f1': float(macro_f1_optimized),
        'micro_f1': float(micro_f1_optimized),
        'macro_precision': float(macro_precision_optimized),
        'macro_recall': float(macro_recall_optimized),
        'per_class_f1': {code: float(f1) for code, f1 in zip(TOP_50_CODES, per_class_f1_optimized)}
    },
    'test_metrics_default': {
        'macro_f1': float(macro_f1_default),
        'micro_f1': float(micro_f1_default)
    },
    'improvement': {
        'macro_f1_gain': float(improvement),
        'percent_improvement': float(improvement / macro_f1_default * 100)
    },
    'top_50_codes': TOP_50_CODES
}

with open(RESULTS_PATH / 'threshold_optimization_results.json', 'w') as f:
    json.dump(results, f, indent=2)

np.save(RESULTS_PATH / 'optimal_thresholds.npy', optimal_thresholds)
np.save(RESULTS_PATH / 'test_predictions_optimized.npy', all_test_preds_optimized)

print(f"✅ Results saved to: {RESULTS_PATH}")

# Save summary
summary_df = pd.DataFrame(threshold_results)
summary_df = summary_df.sort_values('f1', ascending=False)
summary_df.to_csv(RESULTS_PATH / 'threshold_summary.csv', index=False)

print(f"✅ Threshold summary saved to: {RESULTS_PATH / 'threshold_summary.csv'}")

print("\n" + "="*80)
print("✅ THRESHOLD OPTIMIZATION COMPLETE!")
print("="*80)
print(f"\n📊 Final Results:")
print(f"   Test Macro F1 (Optimized): {macro_f1_optimized:.4f}")
print(f"   Test Macro F1 (Default):   {macro_f1_default:.4f}")
print(f"   Improvement:               +{improvement:.4f} ({improvement/macro_f1_default*100:.1f}%)")
print(f"\n💾 Files saved:")
print(f"   • {RESULTS_PATH / 'threshold_optimization_results.json'}")
print(f"   • {RESULTS_PATH / 'optimal_thresholds.npy'}")
print(f"   • {RESULTS_PATH / 'test_predictions_optimized.npy'}")
print(f"   • {RESULTS_PATH / 'threshold_summary.csv'}")
print("\nAlhamdulillah! 🤲")
