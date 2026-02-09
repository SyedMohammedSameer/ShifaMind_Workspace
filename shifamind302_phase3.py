#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND v302 PHASE 3: RAG with FAISS + GAT
================================================================================
Author: Mohammed Sameer Syed
University of Arizona - MS in AI Capstone

Architecture:
- Load Phase 2 checkpoint (GAT + UMLS knowledge graph)
- Build FAISS index with sentence-transformers
- 3-way fusion: BERT + GAT + RAG
- Evidence corpus: clinical knowledge + MIMIC prototypes (1,050 passages)
- Gated fusion for RAG integration

Changes from v301 Phase 3:
1. ✅ 3-way fusion (BERT + GAT + RAG) instead of 2-way (BERT + RAG)
2. ✅ Loads GAT-enhanced Phase 2 model
3. ✅ Preserves v302's concept fusion architecture
4. ✅ Pilot study for hyperparameter selection

Target: Beat LAAT baseline and complete end-to-end pipeline

================================================================================
"""

print("="*80)
print("🚀 SHIFAMIND v302 PHASE 3: RAG + GAT")
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

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")

# ============================================================================
# CONFIGURATION: LOAD FROM PHASE 2
# ============================================================================

print("\n" + "="*80)
print("⚙️  CONFIGURATION: LOADING FROM PHASE 2")
print("="*80)

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
SHIFAMIND302_BASE = BASE_PATH / '11_ShifaMind_v302'

run_folders = sorted([d for d in SHIFAMIND302_BASE.glob('run_*') if d.is_dir()], reverse=True)

if not run_folders:
    print("❌ No v302 Phase 2 run found!")
    sys.exit(1)

OUTPUT_BASE = run_folders[0]
print(f"📁 Using run folder: {OUTPUT_BASE.name}")

PHASE2_CHECKPOINT = OUTPUT_BASE / 'phase_2_models' / 'phase2_best.pt'
if not PHASE2_CHECKPOINT.exists():
    print(f"❌ Phase 2 checkpoint not found at: {PHASE2_CHECKPOINT}")
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
print(f"   Diagnoses: {NUM_DIAGNOSES}")

# Load v301 data paths (for compatibility)
SHIFAMIND301_BASE = BASE_PATH / '10_ShifaMind'
v301_run_folders = sorted([d for d in SHIFAMIND301_BASE.glob('run_*') if d.is_dir()], reverse=True)
if v301_run_folders:
    V301_SHARED_DATA = v301_run_folders[0] / 'shared_data'
else:
    print("❌ v301 run folder not found for data loading!")
    sys.exit(1)

# Phase 3 paths
CHECKPOINT_PATH = OUTPUT_BASE / 'phase_3_models'
RESULTS_PATH = OUTPUT_BASE / 'phase_3_results'
EVIDENCE_PATH = OUTPUT_BASE / 'phase_3_evidence'

for path in [CHECKPOINT_PATH, RESULTS_PATH, EVIDENCE_PATH]:
    path.mkdir(parents=True, exist_ok=True)

with open(V301_SHARED_DATA / 'concept_list.json', 'r') as f:
    ALL_CONCEPTS = json.load(f)

print(f"\n🧠 Concepts: {len(ALL_CONCEPTS)}")

# RAG hyperparameters
RAG_TOP_K = 3
RAG_THRESHOLD = 0.7
PROTOTYPES_PER_DIAGNOSIS = 20

# Pilot study configs
PILOT_CONFIGS = {
    'v301_conservative': {
        'name': 'v301 Conservative (Proven)',
        'lambda_dx': 2.0,
        'lambda_align': 0.5,
        'lambda_concept': 0.3,
        'learning_rate': 5e-6,
        'rag_gate_max': 0.4
    },
    'v302_optimized': {
        'name': 'v302 Optimized (Tuned for stronger Phase 2)',
        'lambda_dx': 1.5,
        'lambda_align': 0.5,
        'lambda_concept': 0.3,
        'learning_rate': 1e-5,
        'rag_gate_max': 0.5
    }
}

BATCH_SIZE = 8

print(f"\n🔬 Pilot Study Configurations:")
for key, config in PILOT_CONFIGS.items():
    print(f"\n   {config['name']}:")
    print(f"      λ_dx={config['lambda_dx']}, λ_align={config['lambda_align']}, λ_concept={config['lambda_concept']}")
    print(f"      LR={config['learning_rate']}, RAG_gate_max={config['rag_gate_max']}")

# ============================================================================
# BUILD EVIDENCE CORPUS (1,050 PASSAGES)
# ============================================================================

print("\n" + "="*80)
print("📚 BUILDING EVIDENCE CORPUS (1,050 PASSAGES)")
print("="*80)

def build_evidence_corpus_top50(top_50_codes):
    """
    Build evidence corpus for Top-50 diagnoses
    1. Clinical knowledge (50 passages)
    2. Case prototypes from MIMIC (1,000 passages = 20 per diagnosis)
    """
    print("\n📖 Building evidence corpus...")

    corpus = []

    # Part 1: Clinical knowledge base
    clinical_knowledge_base = {
        # Respiratory (J codes)
        'J': 'Respiratory conditions: assess cough, dyspnea, chest imaging, oxygen saturation',
        'J18': 'Pneumonia diagnosis requires fever, cough, infiltrates on imaging',
        'J44': 'COPD: chronic airflow limitation, emphysema, chronic bronchitis',
        'J96': 'Respiratory failure: hypoxia, hypercapnia, requires oxygen support',

        # Cardiac (I codes)
        'I': 'Cardiovascular disease: assess chest pain, dyspnea, edema, cardiac markers',
        'I50': 'Heart failure: dyspnea, edema, elevated BNP, reduced EF on echo',
        'I25': 'Ischemic heart disease: angina, troponin, EKG changes',
        'I21': 'MI: acute chest pain, troponin elevation, ST changes',
        'I10': 'Hypertension: elevated BP >140/90, cardiovascular risk assessment',

        # Infection (A codes)
        'A': 'Infectious disease: fever, cultures, antibiotics',
        'A41': 'Sepsis: organ dysfunction, hypotension, lactate >2, positive cultures',

        # Renal (N codes)
        'N': 'Renal disease: creatinine, BUN, urine output',
        'N17': 'Acute kidney injury: rapid creatinine rise, oliguria',
        'N18': 'Chronic kidney disease: GFR <60, proteinuria',

        # Metabolic (E codes)
        'E': 'Endocrine/metabolic: glucose, electrolytes, hormone levels',
        'E11': 'Type 2 diabetes: hyperglycemia, A1c >6.5%, insulin resistance',
        'E87': 'Electrolyte disorders: sodium, potassium, calcium imbalance',
        'E78': 'Hyperlipidemia: elevated cholesterol, LDL, triglycerides',

        # GI (K codes)
        'K': 'GI disease: abdominal pain, nausea, imaging',
        'K80': 'Cholelithiasis: RUQ pain, ultrasound showing stones',

        # Mental health (F codes)
        'F': 'Mental health: psychiatric assessment, mood, cognition',

        # Injury (S/T codes)
        'S': 'Injury/trauma: mechanism, imaging, stabilization',
        'T': 'Poisoning/external causes: toxicology, supportive care',

        # Z codes (status)
        'Z': 'Healthcare encounter status codes',
        'Z79': 'Long-term medication use',
        'Z86': 'Personal history of disease',
        'Z87': 'Personal history of other conditions',
        'Z95': 'Presence of cardiac/vascular implants',
    }

    print("\n📝 Adding clinical knowledge...")
    for code in top_50_codes:
        matched = False
        for key, knowledge in clinical_knowledge_base.items():
            if code.startswith(key):
                corpus.append({
                    'text': f"{code}: {knowledge}",
                    'diagnosis': code,
                    'source': 'clinical_knowledge'
                })
                matched = True
                break

        if not matched:
            corpus.append({
                'text': f"{code}: Diagnosis code requiring clinical correlation",
                'diagnosis': code,
                'source': 'clinical_knowledge'
            })

    print(f"   Added {len(corpus)} clinical knowledge passages")

    # Part 2: MIMIC prototypes
    print(f"\n🏥 Sampling {PROTOTYPES_PER_DIAGNOSIS} case prototypes per diagnosis...")

    with open(V301_SHARED_DATA / 'train_split.pkl', 'rb') as f:
        df_train = pickle.load(f)

    for idx, dx_code in enumerate(top_50_codes):
        # Find positive samples
        if 'labels' in df_train.columns:
            code_idx = top_50_codes.index(dx_code)
            positive_samples = df_train[df_train['labels'].apply(
                lambda x: x[code_idx] == 1 if isinstance(x, list) and len(x) > code_idx else False
            )]
        else:
            positive_samples = pd.DataFrame()

        n_samples = min(len(positive_samples), PROTOTYPES_PER_DIAGNOSIS)
        if n_samples > 0:
            sampled = positive_samples.sample(n=n_samples, random_state=SEED)

            for _, row in sampled.iterrows():
                text = str(row['text'])[:500]
                corpus.append({
                    'text': text,
                    'diagnosis': dx_code,
                    'source': 'mimic_prototype'
                })

        if (idx + 1) % 10 == 0:
            print(f"   Processed {idx + 1}/{len(top_50_codes)} diagnoses...")

    print(f"\n✅ Evidence corpus built:")
    print(f"   Total passages: {len(corpus)}")
    print(f"   Clinical knowledge: {len([c for c in corpus if c['source'] == 'clinical_knowledge'])}")
    print(f"   MIMIC prototypes: {len([c for c in corpus if c['source'] == 'mimic_prototype'])}")

    return corpus

evidence_corpus = build_evidence_corpus_top50(TOP_50_CODES)

with open(EVIDENCE_PATH / 'evidence_corpus.json', 'w') as f:
    json.dump(evidence_corpus, f, indent=2)

print(f"💾 Saved corpus to: {EVIDENCE_PATH / 'evidence_corpus.json'}")

# ============================================================================
# FAISS RETRIEVER
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

        print("   Encoding documents...")
        embeddings = self.encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings = embeddings.astype('float32')

        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print(f"✅ FAISS index built:")
        print(f"   Dimension: {dimension}")
        print(f"   Total vectors: {self.index.ntotal}")

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

# Load graph data
import torch_geometric
from torch_geometric.nn import GATConv

graph_data_path = OUTPUT_BASE / 'phase_2_graph' / 'graph_data.pt'
if not graph_data_path.exists():
    print(f"❌ Graph data not found at: {graph_data_path}")
    sys.exit(1)

graph_data = torch.load(graph_data_path, map_location='cpu')
print(f"✅ Loaded graph data:")
print(f"   Nodes: {graph_data.x.shape[0]}")
print(f"   Features: {graph_data.x.shape[1]}")
print(f"   Edges: {graph_data.edge_index.shape[1]}")

# Build GAT encoder (same architecture as Phase 2)
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

print(f"✅ GAT encoder initialized:")
print(f"   Parameters: {sum(p.numel() for p in gat_encoder.parameters()):,}")

# ============================================================================
# SHIFAMIND v302 PHASE 3 MODEL (3-WAY FUSION)
# ============================================================================

print("\n" + "="*80)
print("🏗️  BUILDING PHASE 3 MODEL (3-WAY FUSION)")
print("="*80)

class ShifaMind302Phase3(nn.Module):
    """
    ShifaMind v302 Phase 3: RAG + GAT fusion

    Architecture (3-way fusion):
    1. BERT text encoding
    2. RAG retrieval & fusion with BERT
    3. GAT concept embeddings from graph
    4. Concept fusion: BERT concepts + GAT concepts
    5. Cross-attention with RAG-enhanced text
    6. Output heads
    """
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

        # Store graph
        self.register_buffer('graph_x', graph_data.x)
        self.register_buffer('graph_edge_index', graph_data.edge_index)
        self.graph_node_to_idx = graph_data.node_to_idx
        self.graph_idx_to_node = graph_data.idx_to_node

        # RAG projection (384 → 768)
        rag_dim = 384
        self.rag_projection = nn.Linear(rag_dim, self.hidden_size)

        # RAG gating
        self.rag_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid()
        )

        # Graph projection (256 → 768)
        self.graph_proj = nn.Linear(self.graph_hidden, self.hidden_size)

        # Concept fusion: BERT + GAT (from Phase 2)
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
        # Run GAT
        graph_embeddings = self.gat(self.graph_x, self.graph_edge_index)

        # Extract concept nodes
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
        """
        3-way fusion forward pass

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            concept_embeddings_bert: [num_concepts, 768] - learned BERT concept embeddings
            input_texts: List[str] - for RAG retrieval
        """
        batch_size = input_ids.shape[0]

        # Step 1: BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, 768]
        pooled_bert = hidden_states.mean(dim=1)  # [batch, 768]

        # Step 2: RAG retrieval & fusion (if texts provided)
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

            # Gated RAG fusion
            gate_input = torch.cat([pooled_bert, rag_context], dim=-1)
            gate = self.rag_gate(gate_input) * self.rag_gate_max

            bert_with_rag = pooled_bert + gate * rag_context  # RAG-enhanced BERT
        else:
            bert_with_rag = pooled_bert

        # Step 3: Get GAT concept embeddings
        gat_concepts = self.get_graph_concept_embeddings()  # [num_concepts, 768]

        # Step 4: Fuse BERT concepts + GAT concepts (v302 Phase 2 fusion)
        bert_concepts = concept_embeddings_bert.unsqueeze(0).expand(batch_size, -1, -1)
        gat_concepts_batched = gat_concepts.unsqueeze(0).expand(batch_size, -1, -1)

        fused_input = torch.cat([bert_concepts, gat_concepts_batched], dim=-1)
        enhanced_concepts = self.concept_fusion(fused_input)  # [batch, num_concepts, 768]

        # Step 5: Cross-attention with RAG-enhanced text
        bert_with_rag_seq = bert_with_rag.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)

        context, attn_weights = self.cross_attention(
            query=bert_with_rag_seq,
            key=enhanced_concepts,
            value=enhanced_concepts,
            need_weights=True
        )

        pooled_context = context.mean(dim=1)

        # Step 6: Multiplicative gating
        gate_input = torch.cat([bert_with_rag, pooled_context], dim=-1)
        gate = self.gate_net(gate_input)

        bottleneck_output = gate * pooled_context
        bottleneck_output = self.layer_norm(bottleneck_output)

        # Step 7: Output heads
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

# Initialize base model
tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
bert_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device)
concept_embedding_layer = nn.Embedding(NUM_CONCEPTS, 768).to(device)

print(f"✅ Base models initialized")

# ============================================================================
# PILOT STUDY: TRAIN 1 EPOCH WITH EACH CONFIG
# ============================================================================

print("\n" + "="*80)
print("🔬 PILOT STUDY: COMPARING HYPERPARAMETER CONFIGURATIONS")
print("="*80)

# Load datasets
with open(V301_SHARED_DATA / 'train_split.pkl', 'rb') as f:
    df_train = pickle.load(f)
with open(V301_SHARED_DATA / 'val_split.pkl', 'rb') as f:
    df_val = pickle.load(f)

train_concept_labels = np.load(V301_SHARED_DATA / 'train_concept_labels.npy')
val_concept_labels = np.load(V301_SHARED_DATA / 'val_concept_labels.npy')

print(f"\n📊 Data loaded:")
print(f"   Train: {len(df_train)} samples")
print(f"   Val:   {len(df_val)} samples")

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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

print(f"✅ Datasets created:")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")

# Loss function
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

def train_one_epoch(model, train_loader, optimizer, criterion, concept_embeddings, device):
    """Train for one epoch"""
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

        train_losses.append(loss.item())
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return np.mean(train_losses)

def evaluate(model, val_loader, criterion, concept_embeddings, device):
    """Evaluate model"""
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

    val_f1 = f1_score(all_labels, all_preds, average='macro')

    return np.mean(val_losses), val_f1

# Run pilot study
pilot_results = {}

for config_name, config in PILOT_CONFIGS.items():
    print(f"\n{'='*80}")
    print(f"🧪 TESTING: {config['name']}")
    print(f"{'='*80}")

    # Create fresh model
    model = ShifaMind302Phase3(
        bert_model=bert_model,
        gat_encoder=gat_encoder,
        rag_retriever=rag,
        graph_data=graph_data,
        num_concepts=NUM_CONCEPTS,
        num_diagnoses=NUM_DIAGNOSES,
        rag_gate_max=config['rag_gate_max']
    ).to(device)

    # Load Phase 2 checkpoint
    print(f"\n📥 Loading Phase 2 checkpoint...")
    checkpoint_data = torch.load(PHASE2_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
    print("✅ Loaded Phase 2 weights (partial)")

    # Setup training
    criterion = MultiObjectiveLoss(
        lambda_dx=config['lambda_dx'],
        lambda_align=config['lambda_align'],
        lambda_concept=config['lambda_concept']
    )

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(concept_embedding_layer.parameters()),
        lr=config['learning_rate']
    )

    concept_embeddings = concept_embedding_layer.weight.detach()

    # Train 1 epoch
    print(f"\n🏋️  Training 1 epoch...")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, concept_embeddings, device)

    # Validate
    print(f"\n📊 Validating...")
    val_loss, val_f1 = evaluate(model, val_loader, criterion, concept_embeddings, device)

    pilot_results[config_name] = {
        'config': config,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_f1': val_f1
    }

    print(f"\n📈 Results:")
    print(f"   Train Loss: {train_loss:.4f}")
    print(f"   Val Loss:   {val_loss:.4f}")
    print(f"   Val F1:     {val_f1:.4f}")

# ============================================================================
# PILOT STUDY RESULTS
# ============================================================================

print(f"\n{'='*80}")
print(f"🏆 PILOT STUDY RESULTS")
print(f"{'='*80}\n")

for config_name, results in pilot_results.items():
    print(f"{results['config']['name']}:")
    print(f"   Val F1: {results['val_f1']:.4f}")
    print(f"   Train Loss: {results['train_loss']:.4f}")
    print(f"   Val Loss: {results['val_loss']:.4f}")
    print()

# Select winner
winner_name = max(pilot_results.keys(), key=lambda x: pilot_results[x]['val_f1'])
winner = pilot_results[winner_name]

print(f"🥇 WINNER: {winner['config']['name']}")
print(f"   Val F1: {winner['val_f1']:.4f} (after 1 epoch)")
print(f"\n✅ Will use this configuration for remaining 4 epochs")

# Save pilot results
with open(RESULTS_PATH / 'pilot_study_results.json', 'w') as f:
    # Convert to serializable format
    serializable_results = {}
    for k, v in pilot_results.items():
        serializable_results[k] = {
            'config': v['config'],
            'train_loss': float(v['train_loss']),
            'val_loss': float(v['val_loss']),
            'val_f1': float(v['val_f1'])
        }
    json.dump({
        'pilot_results': serializable_results,
        'winner': winner_name
    }, f, indent=2)

print(f"\n💾 Pilot results saved to: {RESULTS_PATH / 'pilot_study_results.json'}")
print(f"\n{'='*80}")
print(f"✅ PILOT STUDY COMPLETE!")
print(f"{'='*80}")
print(f"\nNext: Continue training with {winner['config']['name']} for 4 more epochs")
print(f"Expected final F1: ~0.38-0.42 (based on v301's +51% gain from Phase 2)")
print("\nAlhamdulillah! 🤲")
