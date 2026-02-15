#!/usr/bin/env python3
"""
================================================================================
🚀 SHIFAMIND v302 PHASE 3: RAG with FAISS (MAXIMUM GPU OPTIMIZED)
================================================================================
Using FAISS + sentence-transformers for RAG
Training from NEWEST Phase 2 checkpoint with BioClinicalBERT + GAT
MAXIMUM GPU optimization for 96GB VRAM!

Architecture:
1. Load Phase 2 model (BioClinicalBERT + Concept Bottleneck + GAT)
2. Build FAISS evidence store with clinical knowledge + MIMIC prototypes
3. Gated RAG fusion (40% cap)
4. Fine-tune with diagnosis-focused training

Target: Diagnosis F1 > 0.80
================================================================================
"""

print("="*80)
print("🚀 SHIFAMIND v302 PHASE 3 - RAG WITH FAISS (MAXIMUM GPU OPTIMIZED)")
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
    print("⚠️  FAISS not available - installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'faiss-cpu'])
    import faiss
    FAISS_AVAILABLE = True

import json
import pickle
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime
import sys

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# PATHS & CONFIGURATION
# ============================================================================

print("\n" + "="*80)
print("⚙️  CONFIGURATION")
print("="*80)

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
SHIFAMIND_V302_BASE = BASE_PATH / '11_ShifaMind_v302'

# Find newest run folder
run_folders = sorted([d for d in SHIFAMIND_V302_BASE.glob('run_*') if d.is_dir()], reverse=True)
if not run_folders:
    print("❌ No Phase 2 run found!")
    sys.exit(1)

OLD_RUN = run_folders[0]
print(f"\n📁 Loading from Phase 2 run: {OLD_RUN.name}")

# Phase 2 checkpoint path
PHASE2_CHECKPOINT = OLD_RUN / 'phase_2_models' / 'best_model.pth'
if not PHASE2_CHECKPOINT.exists():
    print(f"❌ Phase 2 checkpoint not found at {PHASE2_CHECKPOINT}")
    sys.exit(1)

OLD_SHARED = OLD_RUN / 'shared_data'

# Create new Phase 3 folders
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_FOLDER = SHIFAMIND_V302_BASE / f"run_{timestamp}_phase3"
CHECKPOINT_PATH = RUN_FOLDER / 'phase_3_models'
RESULTS_PATH = RUN_FOLDER / 'phase_3_results'
EVIDENCE_PATH = RUN_FOLDER / 'evidence_store'
SHARED_DATA_PATH = RUN_FOLDER / 'shared_data'

for path in [CHECKPOINT_PATH, RESULTS_PATH, EVIDENCE_PATH, SHARED_DATA_PATH]:
    path.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Run folder: {RUN_FOLDER}")
print(f"📁 Checkpoints: {CHECKPOINT_PATH}")
print(f"📁 Results: {RESULTS_PATH}")
print(f"📁 Evidence: {EVIDENCE_PATH}")

# ============================================================================
# GPU OPTIMIZATION SETTINGS
# ============================================================================

print(f"\n🖥️  Device: {device}")

if device.type == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"🔥 GPU: {gpu_name}")
    print(f"💾 VRAM: {total_vram:.1f} GB")

    # MAXIMUM GPU OPTIMIZATION
    # For 96GB VRAM - go BIG!
    if total_vram >= 90:
        TRAIN_BATCH_SIZE = 96      # 12x original! (was 8)
        VAL_BATCH_SIZE = 192       # 12x original! (was 16)
        NUM_WORKERS = 12
        PREFETCH_FACTOR = 3
        LEARNING_RATE = 6e-5       # Scale with batch size
        GRADIENT_ACCUM_STEPS = 1
    elif total_vram >= 70:
        TRAIN_BATCH_SIZE = 64
        VAL_BATCH_SIZE = 128
        NUM_WORKERS = 10
        PREFETCH_FACTOR = 3
        LEARNING_RATE = 4e-5
        GRADIENT_ACCUM_STEPS = 1
    elif total_vram >= 40:
        TRAIN_BATCH_SIZE = 32
        VAL_BATCH_SIZE = 64
        NUM_WORKERS = 8
        PREFETCH_FACTOR = 2
        LEARNING_RATE = 2e-5
        GRADIENT_ACCUM_STEPS = 2
    else:
        TRAIN_BATCH_SIZE = 16
        VAL_BATCH_SIZE = 32
        NUM_WORKERS = 4
        PREFETCH_FACTOR = 2
        LEARNING_RATE = 1e-5
        GRADIENT_ACCUM_STEPS = 4
else:
    TRAIN_BATCH_SIZE = 4
    VAL_BATCH_SIZE = 8
    NUM_WORKERS = 2
    PREFETCH_FACTOR = 2
    LEARNING_RATE = 5e-6
    GRADIENT_ACCUM_STEPS = 8

# Training settings
EPOCHS = 5
USE_FP16 = device.type == 'cuda'

# RAG settings
RAG_TOP_K = 3
RAG_THRESHOLD = 0.7
RAG_GATE_MAX = 0.4
PROTOTYPES_PER_DIAGNOSIS = 20

# Loss weights
LAMBDA_DX = 2.0
LAMBDA_ALIGN = 0.5
LAMBDA_CONCEPT = 0.3

print(f"\n⚙️  Hyperparameters (MAXIMUM GPU OPTIMIZATION):")
print(f"   Train batch size: {TRAIN_BATCH_SIZE} ({TRAIN_BATCH_SIZE//8}x original!)")
print(f"   Val batch size:   {VAL_BATCH_SIZE} ({VAL_BATCH_SIZE//16}x original!)")
print(f"   Gradient accum:   {GRADIENT_ACCUM_STEPS}")
print(f"   num_workers:      {NUM_WORKERS}")
print(f"   prefetch_factor:  {PREFETCH_FACTOR}")
print(f"   Learning rate:    {LEARNING_RATE}")
print(f"   Epochs:           {EPOCHS}")
print(f"   FP16 precision:   {USE_FP16}")

print(f"\n⚖️  Loss Weights:")
print(f"   λ_dx:      {LAMBDA_DX}")
print(f"   λ_align:   {LAMBDA_ALIGN}")
print(f"   λ_concept: {LAMBDA_CONCEPT}")

print(f"\n📚 RAG Configuration:")
print(f"   Top-K:     {RAG_TOP_K}")
print(f"   Threshold: {RAG_THRESHOLD}")
print(f"   Gate Max:  {RAG_GATE_MAX}")

# ============================================================================
# LOAD DATA FROM PHASE 2
# ============================================================================

print("\n" + "="*80)
print("📋 LOADING DATA FROM PHASE 2")
print("="*80)

# Load splits
with open(OLD_SHARED / 'train_split.pkl', 'rb') as f:
    df_train = pickle.load(f)
with open(OLD_SHARED / 'val_split.pkl', 'rb') as f:
    df_val = pickle.load(f)
with open(OLD_SHARED / 'test_split.pkl', 'rb') as f:
    df_test = pickle.load(f)

# Load concept labels
train_concept_labels = np.load(OLD_SHARED / 'train_concept_labels.npy')
val_concept_labels = np.load(OLD_SHARED / 'val_concept_labels.npy')
test_concept_labels = np.load(OLD_SHARED / 'test_concept_labels.npy')

# Load Top-50 codes (from original run)
ORIGINAL_RUN = BASE_PATH / '10_ShifaMind' / 'run_20260102_203225'
ORIGINAL_SHARED = ORIGINAL_RUN / 'shared_data'

with open(ORIGINAL_SHARED / 'top50_icd10_info.json', 'r') as f:
    top50_info = json.load(f)
    TOP_50_CODES = top50_info['top_50_codes']

# Load concept list
with open(OLD_SHARED / 'concept_list.json', 'r') as f:
    ALL_CONCEPTS = json.load(f)

NUM_CONCEPTS = len(ALL_CONCEPTS)
NUM_LABELS = len(TOP_50_CODES)

print(f"\n✅ Loaded data:")
print(f"   Train: {len(df_train):,} samples")
print(f"   Val:   {len(df_val):,} samples")
print(f"   Test:  {len(df_test):,} samples")
print(f"   Concepts: {NUM_CONCEPTS}")
print(f"   Diagnoses: {NUM_LABELS}")

# Copy shared data to new run folder
import shutil
for filename in ['top50_icd10_info.json', 'concept_list.json']:
    src = ORIGINAL_SHARED / filename if filename == 'top50_icd10_info.json' else OLD_SHARED / filename
    dst = SHARED_DATA_PATH / filename
    shutil.copy(src, dst)

# ============================================================================
# BUILD EVIDENCE CORPUS
# ============================================================================

print("\n" + "="*80)
print("📚 BUILDING EVIDENCE CORPUS")
print("="*80)

def build_evidence_corpus_top50(top_50_codes, df_train):
    """
    Build evidence corpus for Top-50 diagnoses
    1. Clinical knowledge (curated)
    2. Case prototypes from MIMIC
    """
    print("\n📖 Building evidence corpus...")

    corpus = []

    # Clinical knowledge base
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
        'I48': 'Atrial fibrillation: irregular rhythm, palpitations, stroke risk',
        'I10': 'Hypertension: elevated BP, end-organ damage assessment',

        # Infection (A codes)
        'A': 'Infectious disease: fever, cultures, antibiotics',
        'A41': 'Sepsis: organ dysfunction, hypotension, lactate >2, positive cultures',

        # Renal (N codes)
        'N': 'Renal disease: creatinine, BUN, urine output',
        'N17': 'Acute kidney injury: rapid creatinine rise, oliguria',
        'N18': 'Chronic kidney disease: GFR <60, proteinuria',
        'N39': 'Urinary tract disorders: dysuria, frequency, positive culture',

        # Metabolic (E codes)
        'E': 'Endocrine/metabolic: glucose, electrolytes, hormone levels',
        'E11': 'Type 2 diabetes: hyperglycemia, A1c >6.5%, insulin resistance',
        'E87': 'Electrolyte disorders: sodium, potassium, calcium imbalance',
        'E86': 'Volume depletion: dehydration, hypovolemia',

        # GI (K codes)
        'K': 'GI disease: abdominal pain, nausea, imaging',
        'K80': 'Cholelithiasis: RUQ pain, ultrasound showing stones',
        'K21': 'GERD: heartburn, acid reflux, esophagitis',

        # Mental health (F codes)
        'F': 'Mental health: psychiatric assessment, mood, cognition',
        'F32': 'Depression: low mood, anhedonia, sleep disturbance',
        'F41': 'Anxiety: excessive worry, panic, physical symptoms',

        # Injury (S/T codes)
        'S': 'Injury/trauma: mechanism, imaging, stabilization',
        'T': 'Poisoning/external causes: toxicology, supportive care',

        # Neoplasm (C/D codes)
        'C': 'Malignancy: histology, staging, treatment planning',
        'D': 'Benign neoplasm: imaging, biopsy if indicated',

        # Blood (D5-D7)
        'D6': 'Anemia: CBC, iron studies, transfusion if severe',

        # Neurological (G codes)
        'G': 'Neurological: mental status, focal deficits, imaging',
        'G89': 'Pain syndromes: assessment, multimodal analgesia',
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

    # Case prototypes from MIMIC
    print(f"\n🏥 Sampling {PROTOTYPES_PER_DIAGNOSIS} case prototypes per diagnosis...")

    for idx, dx_code in enumerate(top_50_codes):
        # Find positive samples
        code_column_exists = dx_code in df_train.columns
        if code_column_exists:
            positive_samples = df_train[df_train[dx_code] == 1]
        else:
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
                text = str(row['text'])[:500]  # Truncate for efficiency
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

evidence_corpus = build_evidence_corpus_top50(TOP_50_CODES, df_train)

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
        self.encoder.to(device)  # Put on GPU!
        self.top_k = top_k
        self.threshold = threshold
        self.index = None
        self.documents = []
        print(f"✅ RAG encoder loaded on {device}")

    def build_index(self, documents: list):
        print(f"\n🔨 Building FAISS index from {len(documents)} documents...")
        self.documents = documents
        texts = [doc['text'] for doc in documents]

        print("   Encoding documents...")
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=256  # Batch encoding for speed
        )
        embeddings = embeddings.astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        # Use GPU index if available
        if device.type == 'cuda':
            print("   Building GPU FAISS index...")
            res = faiss.StandardGpuResources()
            cpu_index = faiss.IndexFlatIP(dimension)
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        print(f"✅ FAISS index built:")
        print(f"   Dimension: {dimension}")
        print(f"   Total vectors: {self.index.ntotal}")
        print(f"   Device: {'GPU' if device.type == 'cuda' else 'CPU'}")

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
# LOAD PHASE 2 MODEL ARCHITECTURE (GAT-based)
# ============================================================================

print("\n" + "="*80)
print("🏗️  LOADING PHASE 2 MODEL ARCHITECTURE")
print("="*80)

# Import GAT components
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
    """Phase 2 model with GAT knowledge graph"""
    def __init__(self, num_concepts, num_labels, hidden_size=768, gat_heads=4, gat_layers=2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_concepts = num_concepts
        self.num_labels = num_labels

        # BioClinicalBERT encoder
        self.bert = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

        # Concept embeddings
        self.concept_embeddings = nn.Embedding(num_concepts, hidden_size)

        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Gating mechanism
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(gat_layers):
            in_channels = hidden_size if i == 0 else hidden_size * gat_heads
            out_channels = hidden_size
            self.gat_layers.append(
                GATConv(in_channels, out_channels, heads=gat_heads, dropout=0.1, concat=True)
            )

        # Final projection after GAT
        self.gat_projection = nn.Linear(hidden_size * gat_heads, hidden_size)

        # Heads
        self.concept_head = nn.Linear(hidden_size, num_concepts)
        self.diagnosis_head = nn.Linear(hidden_size * 2, num_labels)  # Concat bottleneck + graph

    def forward(self, input_ids, attention_mask, graph_data=None):
        batch_size = input_ids.shape[0]

        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled_bert = hidden_states.mean(dim=1)

        # Concept bottleneck
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

        # Concept predictions
        concept_logits = self.concept_head(pooled_bert)

        # Graph convolution
        if graph_data is not None:
            x = graph_data.x  # [num_nodes, hidden_size]
            edge_index = graph_data.edge_index

            for gat in self.gat_layers:
                x = gat(x, edge_index)
                x = F.elu(x)

            x = self.gat_projection(x)
            graph_output = x.mean(dim=0, keepdim=True).expand(batch_size, -1)
        else:
            graph_output = torch.zeros_like(bottleneck_output)

        # Combine bottleneck + graph
        combined = torch.cat([bottleneck_output, graph_output], dim=-1)
        diagnosis_logits = self.diagnosis_head(combined)

        return {
            'logits': diagnosis_logits,
            'concept_logits': concept_logits,
            'concept_scores': torch.sigmoid(concept_logits),
            'gate_values': gate,
            'concept_attn': concept_attn
        }

# ============================================================================
# PHASE 3 MODEL (with RAG)
# ============================================================================

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
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, graph_data=None, input_texts=None):
        batch_size = input_ids.shape[0]

        # Get BERT representation
        bert_outputs = self.phase2_model.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = bert_outputs.last_hidden_state
        pooled_bert = hidden_states.mean(dim=1)

        # RAG retrieval and fusion
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

            # Gated fusion
            gate_input = torch.cat([pooled_bert, rag_context], dim=-1)
            gate = self.rag_gate(gate_input)
            gate = gate * RAG_GATE_MAX  # Cap at 40%

            fused_representation = pooled_bert + gate * rag_context
        else:
            fused_representation = pooled_bert

        # Replace BERT output with fused representation
        # Create modified hidden states
        fused_hidden = hidden_states.clone()
        fused_hidden = fused_hidden + (fused_representation - pooled_bert).unsqueeze(1)

        # Run through Phase 2 concept bottleneck and GAT
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

        # Concept predictions
        concept_logits = self.phase2_model.concept_head(fused_representation)

        # Graph convolution
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

        # Combine bottleneck + graph
        combined = torch.cat([bottleneck_output, graph_output], dim=-1)
        diagnosis_logits = self.phase2_model.diagnosis_head(combined)

        return {
            'logits': diagnosis_logits,
            'concept_logits': concept_logits,
            'concept_scores': torch.sigmoid(concept_logits),
            'gate_values': gate,
            'concept_attn': concept_attn
        }

# Initialize Phase 2 model
print("\n🏗️  Initializing Phase 2 model...")
phase2_model = ShifaMindPhase2GAT(
    num_concepts=NUM_CONCEPTS,
    num_labels=NUM_LABELS,
    hidden_size=768,
    gat_heads=4,
    gat_layers=2
)

# Load Phase 2 checkpoint
print(f"\n📥 Loading Phase 2 checkpoint from {PHASE2_CHECKPOINT}...")
checkpoint = torch.load(PHASE2_CHECKPOINT, map_location='cpu')
phase2_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
print("✅ Loaded Phase 2 weights")

# Initialize Phase 3 model with RAG
print("\n🏗️  Initializing Phase 3 model with RAG...")
model = ShifaMindPhase3RAG(
    phase2_model=phase2_model,
    rag_retriever=rag,
    hidden_size=768
).to(device)

print(f"\n✅ ShifaMind Phase 3 model initialized")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Load graph data
print("\n📊 Loading knowledge graph...")
GRAPH_PATH = OLD_RUN / 'phase_2_graph'
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

train_dataset = RAGDataset(df_train, tokenizer, train_concept_labels)
val_dataset = RAGDataset(df_val, tokenizer, val_concept_labels)
test_dataset = RAGDataset(df_test, tokenizer, test_concept_labels)

# Use pin_memory for faster GPU transfer
train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    pin_memory=True
)

print(f"\n✅ Datasets ready:")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches:   {len(val_loader)}")
print(f"   Test batches:  {len(test_loader)}")

# ============================================================================
# LOSS & OPTIMIZER
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
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUM_STEPS
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# FP16 training
scaler = torch.cuda.amp.GradScaler() if USE_FP16 else None

print(f"\n✅ Training setup complete")
print(f"   Optimizer: AdamW (lr={LEARNING_RATE}, weight_decay=0.01)")
print(f"   Scheduler: Linear warmup ({warmup_steps} steps) + decay ({total_steps} total)")
print(f"   Mixed precision: {USE_FP16}")

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("\n" + "="*80)
print("🏋️  TRAINING PHASE 3 (RAG-ENHANCED)")
print("="*80)

best_val_f1 = 0.0
history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_precision': [], 'val_recall': []}

for epoch in range(EPOCHS):
    print(f"\n{'='*80}")
    print(f"📍 Epoch {epoch+1}/{EPOCHS}")
    print(f"{'='*80}")

    # ========================================================================
    # TRAINING
    # ========================================================================

    model.train()
    train_losses = []
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        concept_labels = batch['concept_labels'].to(device, non_blocking=True)
        texts = batch['text']

        # Forward pass with mixed precision
        if USE_FP16:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask, graph_data, input_texts=texts)
                loss, loss_components = criterion(outputs, labels, concept_labels)
                loss = loss / GRADIENT_ACCUM_STEPS

            scaler.scale(loss).backward()
        else:
            outputs = model(input_ids, attention_mask, graph_data, input_texts=texts)
            loss, loss_components = criterion(outputs, labels, concept_labels)
            loss = loss / GRADIENT_ACCUM_STEPS
            loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % GRADIENT_ACCUM_STEPS == 0:
            if USE_FP16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

        train_losses.append(loss.item() * GRADIENT_ACCUM_STEPS)

        # Update progress bar
        if device.type == 'cuda':
            gpu_mem = torch.cuda.memory_allocated() / (1024**3)
            gpu_max = torch.cuda.max_memory_allocated() / (1024**3)
            pbar.set_postfix({
                'loss': f"{loss.item() * GRADIENT_ACCUM_STEPS:.4f}",
                'GPU': f"{gpu_mem:.1f}/{gpu_max:.1f}GB"
            })
        else:
            pbar.set_postfix({'loss': f"{loss.item() * GRADIENT_ACCUM_STEPS:.4f}"})

    avg_train_loss = np.mean(train_losses)
    history['train_loss'].append(avg_train_loss)

    print(f"\n📊 Training complete:")
    print(f"   Avg Loss: {avg_train_loss:.4f}")
    if device.type == 'cuda':
        print(f"   Peak GPU Memory: {torch.cuda.max_memory_allocated() / (1024**3):.1f} GB")
        torch.cuda.reset_peak_memory_stats()

    # ========================================================================
    # VALIDATION
    # ========================================================================

    model.eval()
    val_losses = []
    all_preds = []
    all_labels = []

    print(f"\n🔍 Validating...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            concept_labels = batch['concept_labels'].to(device, non_blocking=True)
            texts = batch['text']

            if USE_FP16:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask, graph_data, input_texts=texts)
                    loss, _ = criterion(outputs, labels, concept_labels)
            else:
                outputs = model(input_ids, attention_mask, graph_data, input_texts=texts)
                loss, _ = criterion(outputs, labels, concept_labels)

            val_losses.append(loss.item())

            preds = (torch.sigmoid(outputs['logits']) > 0.5).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    avg_val_loss = np.mean(val_losses)
    val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    history['val_loss'].append(avg_val_loss)
    history['val_f1'].append(val_f1)
    history['val_precision'].append(val_precision)
    history['val_recall'].append(val_recall)

    print(f"\n📊 Validation Results:")
    print(f"   Loss:      {avg_val_loss:.4f}")
    print(f"   F1:        {val_f1:.4f}")
    print(f"   Precision: {val_precision:.4f}")
    print(f"   Recall:    {val_recall:.4f}")

    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1': best_val_f1,
            'history': history,
            'config': {
                'num_concepts': NUM_CONCEPTS,
                'num_labels': NUM_LABELS,
                'top_50_codes': TOP_50_CODES,
                'all_concepts': ALL_CONCEPTS,
                'rag_config': {
                    'top_k': RAG_TOP_K,
                    'threshold': RAG_THRESHOLD,
                    'gate_max': RAG_GATE_MAX
                },
                'training_config': {
                    'batch_size': TRAIN_BATCH_SIZE,
                    'learning_rate': LEARNING_RATE,
                    'epochs': EPOCHS,
                    'lambda_dx': LAMBDA_DX,
                    'lambda_align': LAMBDA_ALIGN,
                    'lambda_concept': LAMBDA_CONCEPT
                },
                'timestamp': timestamp
            }
        }, CHECKPOINT_PATH / 'best_model.pth')

        print(f"   ✅ Saved best model (F1: {best_val_f1:.4f})")

# ============================================================================
# FINAL EVALUATION
# ============================================================================

print("\n" + "="*80)
print("📊 FINAL EVALUATION ON TEST SET")
print("="*80)

# Load best model
checkpoint = torch.load(CHECKPOINT_PATH / 'best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

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

        if USE_FP16:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask, graph_data, input_texts=texts)
        else:
            outputs = model(input_ids, attention_mask, graph_data, input_texts=texts)

        probs = torch.sigmoid(outputs['logits']).cpu().numpy()
        preds = (probs > 0.5).astype(int)

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

all_probs = np.vstack(all_probs)
all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

# Calculate metrics
macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

print(f"\n🎯 Test Set Performance:")
print(f"   Macro F1:        {macro_f1:.4f}")
print(f"   Micro F1:        {micro_f1:.4f}")
print(f"   Macro Precision: {macro_precision:.4f}")
print(f"   Macro Recall:    {macro_recall:.4f}")

# Save results
results = {
    'phase': 'ShifaMind v302 Phase 3 - RAG with FAISS',
    'timestamp': timestamp,
    'run_folder': str(RUN_FOLDER),
    'test_metrics': {
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'per_class_f1': {code: float(f1) for code, f1 in zip(TOP_50_CODES, per_class_f1)}
    },
    'validation_metrics': {
        'best_f1': float(best_val_f1),
        'final_f1': float(history['val_f1'][-1])
    },
    'architecture': 'BioClinicalBERT + Concept Bottleneck + GAT + FAISS RAG',
    'rag_config': {
        'method': 'FAISS + sentence-transformers',
        'model': 'all-MiniLM-L6-v2',
        'top_k': RAG_TOP_K,
        'threshold': RAG_THRESHOLD,
        'gate_max': RAG_GATE_MAX,
        'corpus_size': len(evidence_corpus)
    },
    'training_config': {
        'batch_size': TRAIN_BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
        'lambda_dx': LAMBDA_DX,
        'lambda_align': LAMBDA_ALIGN,
        'lambda_concept': LAMBDA_CONCEPT,
        'fp16': USE_FP16,
        'gradient_accum_steps': GRADIENT_ACCUM_STEPS
    },
    'training_history': history
}

with open(RESULTS_PATH / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save predictions
np.save(RESULTS_PATH / 'test_predictions.npy', all_preds)
np.save(RESULTS_PATH / 'test_probabilities.npy', all_probs)
np.save(RESULTS_PATH / 'test_labels.npy', all_labels)

print(f"\n💾 Results saved to: {RESULTS_PATH}")
print(f"💾 Best model saved to: {CHECKPOINT_PATH / 'best_model.pth'}")

print("\n" + "="*80)
print("✅ SHIFAMIND v302 PHASE 3 COMPLETE!")
print("="*80)
print(f"\n📍 Run folder: {RUN_FOLDER}")
print(f"   Test Macro F1: {macro_f1:.4f}")
print(f"   Test Micro F1: {micro_f1:.4f}")
print(f"   Best Val F1:   {best_val_f1:.4f}")
print("\nNext: Run phase3_threshold_optimization.py for optimal thresholds")
print("\nAlhamdulillah! 🤲")
