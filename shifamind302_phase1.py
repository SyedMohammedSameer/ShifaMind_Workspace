#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND v302 - PHASE 1: CONCEPT BOTTLENECK WITH REFINED CONCEPTS
================================================================================
Author: ShifaMind Research Team
Version: 302 (Phase A - Week 2 Full Implementation)
Purpose: Train concept bottleneck on full dataset using v5 refined concepts

Changes from v301:
- Uses v5's 131 refined concepts (11.7% density, 15.4 concepts/sample)
- ScispaCy + UMLS concept extraction (4.5 hours estimated)
- Same BioClinicalBERT architecture with multiplicative gating
- Saves to 11_ShifaMind_v302 folder

Expected improvement: 0.28 → 0.30+ F1 (better concepts = better bottleneck)
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
from collections import defaultdict, Counter
from datetime import datetime
import time
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, roc_auc_score, hamming_loss
)
from tqdm.auto import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

print("="*80)
print("🚀 SHIFAMIND v302 - PHASE 1: CONCEPT BOTTLENECK")
print("="*80)
print("Using v5 refined concepts: 131 concepts, 11.7% density")
print("Training from scratch on full dataset (80K samples)")
print()

# Paths
BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
UMLS_PATH = BASE_PATH / '01_Raw_Datasets' / 'Extracted' / 'umls-2025AA-metathesaurus-full' / '2025AA' / 'META'

# New output folder for v302
OUTPUT_BASE = BASE_PATH / '11_ShifaMind_v302'
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_PATH = OUTPUT_BASE / f'run_{RUN_TIMESTAMP}'
RUN_PATH.mkdir(parents=True, exist_ok=True)

# Create subfolders
SHARED_DATA_PATH = RUN_PATH / 'shared_data'
CONCEPTS_PATH = RUN_PATH / 'phase_1_concepts'
MODELS_PATH = RUN_PATH / 'phase_1_models'
RESULTS_PATH = RUN_PATH / 'phase_1_results'

for path in [SHARED_DATA_PATH, CONCEPTS_PATH, MODELS_PATH, RESULTS_PATH]:
    path.mkdir(exist_ok=True)

print(f"📁 Output folder: {RUN_PATH}")
print(f"📁 Concepts: {CONCEPTS_PATH}")
print(f"📁 Models: {MODELS_PATH}")
print(f"📁 Results: {RESULTS_PATH}")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
MAX_LENGTH = 512
SEED = 42

# Set seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================================
# STEP 1: LOAD DATA AND V5 CONCEPT VOCABULARY
# ============================================================================

print("\n" + "="*80)
print("📋 STEP 1: LOADING DATA AND V5 CONCEPT VOCABULARY")
print("="*80)

# Find most recent v301 run for data
OLD_RUN_PATH = BASE_PATH / '10_ShifaMind'
run_folders = sorted([d for d in OLD_RUN_PATH.glob('run_*') if d.is_dir()], reverse=True)
if not run_folders:
    print("❌ No v301 run found! Please run shifamind301.py first.")
    sys.exit(1)

OLD_SHARED = run_folders[0] / 'shared_data'
OLD_PILOT = run_folders[0] / 'phase_a_pilot'

print(f"📁 Loading data from: {run_folders[0].name}")

# Load splits
with open(OLD_SHARED / 'train_split.pkl', 'rb') as f:
    df_train = pickle.load(f)
with open(OLD_SHARED / 'val_split.pkl', 'rb') as f:
    df_val = pickle.load(f)
with open(OLD_SHARED / 'test_split.pkl', 'rb') as f:
    df_test = pickle.load(f)

print(f"✅ Loaded splits:")
print(f"   Train: {len(df_train):,} samples")
print(f"   Val:   {len(df_val):,} samples")
print(f"   Test:  {len(df_test):,} samples")

# Load Top-50 codes
with open(OLD_SHARED / 'top50_icd10_info.json', 'r') as f:
    top50_info = json.load(f)
    TOP_50_CODES = top50_info['top_50_codes']

print(f"✅ Loaded {len(TOP_50_CODES)} ICD-10 codes")

# Save to new location
with open(SHARED_DATA_PATH / 'top50_icd10_info.json', 'w') as f:
    json.dump(top50_info, f, indent=2)

# Load v5 concept vocabulary from pilot
with open(OLD_PILOT / 'candidate_concepts.json', 'r') as f:
    v5_concepts_data = json.load(f)
    CONCEPT_VOCAB = v5_concepts_data['concepts']
    CONCEPT_TO_CUI = v5_concepts_data['concept_to_cui']
    DIAGNOSIS_TO_CONCEPTS = v5_concepts_data['diagnosis_to_concepts']

print(f"\n✅ Loaded v5 concept vocabulary:")
print(f"   Total concepts: {len(CONCEPT_VOCAB)}")
print(f"   UMLS mapped: {len(CONCEPT_TO_CUI)}/{len(CONCEPT_VOCAB)} ({len(CONCEPT_TO_CUI)/len(CONCEPT_VOCAB)*100:.1f}%)")
print(f"   Diagnoses covered: {len(DIAGNOSIS_TO_CONCEPTS)}/{len(TOP_50_CODES)}")
print(f"\n🔍 Sample concepts: {CONCEPT_VOCAB[:10]}")

# Save to new location
with open(CONCEPTS_PATH / 'v5_concept_vocabulary.json', 'w') as f:
    json.dump({
        'concepts': CONCEPT_VOCAB,
        'concept_to_cui': CONCEPT_TO_CUI,
        'diagnosis_to_concepts': DIAGNOSIS_TO_CONCEPTS,
        'num_concepts': len(CONCEPT_VOCAB),
        'source': 'v5 pilot with 11.7% density, 15.4 concepts/sample'
    }, f, indent=2)

NUM_CONCEPTS = len(CONCEPT_VOCAB)
NUM_LABELS = len(TOP_50_CODES)

# ============================================================================
# STEP 2: INSTALL SCISPACY AND LOAD MODEL
# ============================================================================

print("\n" + "="*80)
print("🔧 STEP 2: LOADING SCISPACY FOR CONCEPT EXTRACTION")
print("="*80)

def check_and_install_scispacy():
    """Check and install ScispaCy"""
    try:
        import spacy
        import scispacy
        from negspacy.negation import Negex
        print("✅ scispacy and negspacy found")
    except ImportError:
        print("Installing scispacy and negspacy...")
        os.system('pip install -q scispacy negspacy')
        import spacy
        import scispacy
        from negspacy.negation import Negex

    # Check if large model is installed
    try:
        nlp_test = spacy.load("en_core_sci_lg")
        print("✅ en_core_sci_lg found")
        return True
    except:
        print("⚠️  Installing en_core_sci_lg (2-3 minutes)...")
        result = os.system('pip install -q https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz')
        if result != 0:
            print("❌ Installation failed")
            return False
        print("✅ en_core_sci_lg installed")
        return True

if not check_and_install_scispacy():
    print("\n❌ Failed to install ScispaCy. Please install manually:")
    print("  pip install scispacy negspacy")
    print("  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz")
    sys.exit(1)

import spacy
from negspacy.negation import Negex

print("\nLoading en_core_sci_lg...")
nlp = spacy.load("en_core_sci_lg")
nlp.add_pipe("negex")
print(f"✅ ScispaCy loaded with pipeline: {nlp.pipe_names}")

# ============================================================================
# STEP 3: CONCEPT EXTRACTION FUNCTION
# ============================================================================

print("\n" + "="*80)
print("🔬 STEP 3: CONCEPT EXTRACTION SETUP")
print("="*80)

def extract_concepts_from_text(text, concept_vocab, nlp_model):
    """Extract concepts using ScispaCy NER + keyword matching"""
    text = str(text)[:5000]  # Truncate for speed

    doc = nlp_model(text.lower())

    concept_labels = {concept: {'present': 0, 'negated': 0} for concept in concept_vocab}

    # Method 1: NER with negation
    for ent in doc.ents:
        ent_text = ent.text.lower().strip()
        is_negated = ent._.negex if hasattr(ent._, 'negex') else False

        if ent_text in concept_vocab:
            concept_labels[ent_text]['present'] = 1
            if is_negated:
                concept_labels[ent_text]['negated'] = 1

        for concept in concept_vocab:
            if concept in ent_text or ent_text in concept:
                concept_labels[concept]['present'] = 1
                if is_negated:
                    concept_labels[concept]['negated'] = 1

    # Method 2: Keyword matching
    text_lower = text.lower()
    for concept in concept_vocab:
        pattern = r'\b' + re.escape(concept) + r'\b'
        if re.search(pattern, text_lower):
            concept_labels[concept]['present'] = 1

            # Simple negation check
            match_pos = text_lower.find(concept)
            context_start = max(0, match_pos - 50)
            context = text_lower[context_start:match_pos]

            negation_terms = ['no ', 'denies', 'deny', 'negative', 'without', 'absent',
                            'ruled out', 'rule out', 'r/o', 'not ']
            if any(neg in context for neg in negation_terms):
                concept_labels[concept]['negated'] = 1

    return concept_labels

print("✅ Concept extraction function ready")
print(f"   Method: ScispaCy NER + keyword matching with negation")
print(f"   Vocabulary size: {len(CONCEPT_VOCAB)} concepts")

# ============================================================================
# STEP 4: EXTRACT CONCEPTS FROM FULL TRAINING SET
# ============================================================================

print("\n" + "="*80)
print("🔄 STEP 4: EXTRACTING CONCEPTS FROM FULL TRAINING SET")
print("="*80)

# Check if already extracted
concept_file = CONCEPTS_PATH / 'train_concept_labels.pkl'

if concept_file.exists():
    print(f"⚠️  Found existing concept extraction at {concept_file}")
    response = input("   Use existing? (y/n): ")
    if response.lower() == 'y':
        with open(concept_file, 'rb') as f:
            train_concept_data = pickle.load(f)
            train_concept_labels = train_concept_data['concept_labels']
        print(f"✅ Loaded {len(train_concept_labels)} pre-extracted concept labels")
    else:
        concept_file.unlink()
        train_concept_labels = None
else:
    train_concept_labels = None

if train_concept_labels is None:
    print(f"🔄 Extracting concepts from {len(df_train):,} training samples...")
    print("   (Estimated time: 4.5 hours at 5 samples/sec)")
    print()

    train_concept_labels = []
    extraction_times = []

    start_time = time.time()

    for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Extracting"):
        sample_start = time.time()
        text = row['text']
        concept_dict = extract_concepts_from_text(text, CONCEPT_VOCAB, nlp)
        train_concept_labels.append(concept_dict)
        extraction_times.append(time.time() - sample_start)

        # Progress update every 1000 samples
        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            avg_time = np.mean(extraction_times)
            remaining = (len(df_train) - idx - 1) * avg_time
            print(f"\n   Progress: {idx+1:,}/{len(df_train):,} ({(idx+1)/len(df_train)*100:.1f}%)")
            print(f"   Elapsed: {elapsed/3600:.1f}h | Remaining: {remaining/3600:.1f}h | Speed: {1/avg_time:.1f} samples/sec")

    total_time = time.time() - start_time
    print(f"\n✅ Extraction complete!")
    print(f"   Total time: {total_time/3600:.1f} hours")
    print(f"   Avg time per sample: {np.mean(extraction_times):.2f}s")

    # Save
    with open(concept_file, 'wb') as f:
        pickle.dump({
            'concept_labels': train_concept_labels,
            'concepts': CONCEPT_VOCAB,
            'extraction_time_hours': total_time / 3600
        }, f)
    print(f"💾 Saved to {concept_file}")

# Extract concepts from val and test sets
print(f"\n🔄 Extracting concepts from validation set ({len(df_val):,} samples)...")
val_concept_labels = []
for idx, row in tqdm(df_val.iterrows(), total=len(df_val), desc="Val"):
    concept_dict = extract_concepts_from_text(row['text'], CONCEPT_VOCAB, nlp)
    val_concept_labels.append(concept_dict)

with open(CONCEPTS_PATH / 'val_concept_labels.pkl', 'wb') as f:
    pickle.dump({
        'concept_labels': val_concept_labels,
        'concepts': CONCEPT_VOCAB
    }, f)

print(f"\n🔄 Extracting concepts from test set ({len(df_test):,} samples)...")
test_concept_labels = []
for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Test"):
    concept_dict = extract_concepts_from_text(row['text'], CONCEPT_VOCAB, nlp)
    test_concept_labels.append(concept_dict)

with open(CONCEPTS_PATH / 'test_concept_labels.pkl', 'wb') as f:
    pickle.dump({
        'concept_labels': test_concept_labels,
        'concepts': CONCEPT_VOCAB
    }, f)

print("✅ All concept labels extracted and saved")

# ============================================================================
# STEP 5: CONCEPT ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("📊 STEP 5: CONCEPT EXTRACTION ANALYSIS")
print("="*80)

# Convert to matrix for analysis
concept_matrix = np.zeros((len(df_train), NUM_CONCEPTS))
negation_matrix = np.zeros((len(df_train), NUM_CONCEPTS))

for i, concept_dict in enumerate(train_concept_labels):
    for j, concept in enumerate(CONCEPT_VOCAB):
        concept_matrix[i, j] = concept_dict[concept]['present']
        negation_matrix[i, j] = concept_dict[concept]['negated']

# Statistics
concept_counts = concept_matrix.sum(axis=0)
concept_freq = concept_counts / len(df_train)
concepts_per_sample = concept_matrix.sum(axis=1)

avg_concepts = concepts_per_sample.mean()
median_concepts = np.median(concepts_per_sample)
density = avg_concepts / NUM_CONCEPTS

print(f"\n🔍 Sparsity Analysis:")
print(f"   Total concepts: {NUM_CONCEPTS}")
print(f"   Avg concepts per sample: {avg_concepts:.1f}")
print(f"   Median: {median_concepts:.0f}")
print(f"   Density: {density*100:.1f}%")

coverage = (concepts_per_sample > 0).sum() / len(df_train)
print(f"\n📈 Coverage:")
print(f"   Samples with ≥1 concept: {(concepts_per_sample > 0).sum()}/{len(df_train)} ({coverage*100:.1f}%)")

# Top concepts
top_20_idx = np.argsort(concept_counts)[-20:][::-1]
print(f"\n🔝 Top-20 Concepts:")
for idx in top_20_idx:
    print(f"   {CONCEPT_VOCAB[idx]:40s}: {int(concept_counts[idx]):6d} ({concept_freq[idx]*100:5.1f}%)")

# Negation
negation_rate = negation_matrix.sum() / (concept_matrix.sum() + 1e-10)
print(f"\n🚫 Negation:")
print(f"   Total concepts: {int(concept_matrix.sum())}")
print(f"   Negated: {int(negation_matrix.sum())}")
print(f"   Rate: {negation_rate*100:.1f}%")

# Save analysis
analysis = {
    'num_concepts': NUM_CONCEPTS,
    'avg_concepts_per_sample': float(avg_concepts),
    'median_concepts_per_sample': float(median_concepts),
    'density': float(density),
    'coverage': float(coverage),
    'negation_rate': float(negation_rate),
    'top_20_concepts': [(CONCEPT_VOCAB[idx], int(concept_counts[idx]), float(concept_freq[idx]))
                        for idx in top_20_idx]
}

with open(CONCEPTS_PATH / 'concept_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)

print(f"\n💾 Saved analysis to {CONCEPTS_PATH / 'concept_analysis.json'}")

# ============================================================================
# STEP 6: PREPARE CONCEPT MATRICES FOR TRAINING
# ============================================================================

print("\n" + "="*80)
print("🔧 STEP 6: PREPARING CONCEPT MATRICES FOR TRAINING")
print("="*80)

# Convert val and test to matrices
val_concept_matrix = np.zeros((len(df_val), NUM_CONCEPTS))
for i, concept_dict in enumerate(val_concept_labels):
    for j, concept in enumerate(CONCEPT_VOCAB):
        val_concept_matrix[i, j] = concept_dict[concept]['present']

test_concept_matrix = np.zeros((len(df_test), NUM_CONCEPTS))
for i, concept_dict in enumerate(test_concept_labels):
    for j, concept in enumerate(CONCEPT_VOCAB):
        test_concept_matrix[i, j] = concept_dict[concept]['present']

print(f"✅ Concept matrices ready:")
print(f"   Train: {concept_matrix.shape}")
print(f"   Val:   {val_concept_matrix.shape}")
print(f"   Test:  {test_concept_matrix.shape}")

# Save as numpy arrays for faster loading
np.save(CONCEPTS_PATH / 'train_concept_matrix.npy', concept_matrix)
np.save(CONCEPTS_PATH / 'val_concept_matrix.npy', val_concept_matrix)
np.save(CONCEPTS_PATH / 'test_concept_matrix.npy', test_concept_matrix)

print(f"💾 Saved concept matrices to {CONCEPTS_PATH}")

# ============================================================================
# STEP 7: BUILD BIOCLINICALBERT CONCEPT BOTTLENECK MODEL
# ============================================================================

print("\n" + "="*80)
print("🏗️  STEP 7: BUILDING BIOCLINICALBERT CONCEPT BOTTLENECK")
print("="*80)

# Install transformers
try:
    from transformers import AutoTokenizer, AutoModel
    print("✅ transformers found")
except ImportError:
    print("Installing transformers...")
    os.system('pip install -q transformers')
    from transformers import AutoTokenizer, AutoModel

# Load BioClinicalBERT
MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
print(f"\nLoading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("✅ Tokenizer loaded")

class ConceptBottleneckModel(nn.Module):
    """
    BioClinicalBERT + Concept Bottleneck + Diagnosis Prediction
    Uses MULTIPLICATIVE gating (not additive) as per shifamind301.py
    """
    def __init__(self, model_name, num_concepts, num_labels, freeze_bert=False):
        super(ConceptBottleneckModel, self).__init__()

        # BioClinicalBERT encoder
        self.bert = AutoModel.from_pretrained(model_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        hidden_size = self.bert.config.hidden_size  # 768

        # Concept prediction head
        self.concept_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_concepts),
            nn.Sigmoid()  # Multi-label
        )

        # Concept bottleneck with multiplicative gating
        self.concept_gate = nn.Sequential(
            nn.Linear(num_concepts, num_concepts),
            nn.Sigmoid()
        )

        # Diagnosis prediction from gated concepts
        self.diagnosis_head = nn.Sequential(
            nn.Linear(num_concepts, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels),
            nn.Sigmoid()  # Multi-label
        )

    def forward(self, input_ids, attention_mask):
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Predict concepts
        concepts = self.concept_head(cls_output)  # [batch, num_concepts]

        # Multiplicative gating (KEY: not additive!)
        gate = self.concept_gate(concepts)
        gated_concepts = concepts * gate

        # Predict diagnoses from gated concepts
        diagnoses = self.diagnosis_head(gated_concepts)

        return diagnoses, concepts, gated_concepts

model = ConceptBottleneckModel(
    model_name=MODEL_NAME,
    num_concepts=NUM_CONCEPTS,
    num_labels=NUM_LABELS,
    freeze_bert=False
)

model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n✅ Model built:")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Architecture: BioClinicalBERT → Concepts ({NUM_CONCEPTS}) → Gate → Diagnoses ({NUM_LABELS})")
print(f"   Gating: MULTIPLICATIVE (concepts * gate)")

# ============================================================================
# STEP 8: DATASET AND DATALOADER
# ============================================================================

print("\n" + "="*80)
print("📦 STEP 8: CREATING DATASETS AND DATALOADERS")
print("="*80)

class MIMICDataset(Dataset):
    def __init__(self, dataframe, concept_matrix, tokenizer, max_length=512):
        self.texts = dataframe['text'].values
        self.labels = np.array([row for row in dataframe['labels'].values])
        self.concepts = concept_matrix
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        concepts = self.concepts[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.FloatTensor(labels),
            'concepts': torch.FloatTensor(concepts)
        }

train_dataset = MIMICDataset(df_train, concept_matrix, tokenizer, MAX_LENGTH)
val_dataset = MIMICDataset(df_val, val_concept_matrix, tokenizer, MAX_LENGTH)
test_dataset = MIMICDataset(df_test, test_concept_matrix, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"✅ Datasets created:")
print(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
print(f"   Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
print(f"   Test:  {len(test_dataset)} samples, {len(test_loader)} batches")

# ============================================================================
# STEP 9: TRAINING SETUP
# ============================================================================

print("\n" + "="*80)
print("⚙️  STEP 9: TRAINING SETUP")
print("="*80)

# Loss functions
diagnosis_criterion = nn.BCELoss()
concept_criterion = nn.BCELoss()

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Scheduler
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=1.0,
    end_factor=0.1,
    total_iters=total_steps
)

print(f"✅ Training configuration:")
print(f"   Optimizer: AdamW (lr={LEARNING_RATE})")
print(f"   Scheduler: LinearLR (1.0 → 0.1)")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Total steps: {total_steps}")
print(f"   Loss: BCELoss (diagnosis + concepts)")

# ============================================================================
# STEP 10: TRAINING LOOP
# ============================================================================

print("\n" + "="*80)
print("🚀 STEP 10: TRAINING")
print("="*80)

def evaluate(model, dataloader, device):
    """Evaluate model on validation/test set"""
    model.eval()

    all_preds = []
    all_labels = []
    all_concept_preds = []
    all_concept_labels = []

    total_loss = 0
    total_diag_loss = 0
    total_concept_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            concepts = batch['concepts'].to(device)

            diag_preds, concept_preds, _ = model(input_ids, attention_mask)

            diag_loss = diagnosis_criterion(diag_preds, labels)
            concept_loss = concept_criterion(concept_preds, concepts)
            loss = diag_loss + 0.5 * concept_loss  # Weight concept loss

            total_loss += loss.item()
            total_diag_loss += diag_loss.item()
            total_concept_loss += concept_loss.item()

            all_preds.append(diag_preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_concept_preds.append(concept_preds.cpu().numpy())
            all_concept_labels.append(concepts.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_concept_preds = np.vstack(all_concept_preds)
    all_concept_labels = np.vstack(all_concept_labels)

    # Diagnosis metrics
    diag_preds_binary = (all_preds >= 0.5).astype(int)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, diag_preds_binary, average='macro', zero_division=0
    )

    # Concept metrics
    concept_preds_binary = (all_concept_preds >= 0.5).astype(int)
    concept_precision, concept_recall, concept_f1, _ = precision_recall_fscore_support(
        all_concept_labels, concept_preds_binary, average='macro', zero_division=0
    )

    return {
        'loss': total_loss / len(dataloader),
        'diag_loss': total_diag_loss / len(dataloader),
        'concept_loss': total_concept_loss / len(dataloader),
        'precision': precision_macro,
        'recall': recall_macro,
        'f1': f1_macro,
        'concept_precision': concept_precision,
        'concept_recall': concept_recall,
        'concept_f1': concept_f1,
    }

# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'val_f1': [],
    'val_concept_f1': []
}

best_val_f1 = 0
best_epoch = 0

print(f"\n{'='*80}")
print(f"Starting training for {NUM_EPOCHS} epochs...")
print(f"{'='*80}\n")

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0
    train_diag_loss = 0
    train_concept_loss = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')

    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        concepts = batch['concepts'].to(device)

        optimizer.zero_grad()

        diag_preds, concept_preds, _ = model(input_ids, attention_mask)

        diag_loss = diagnosis_criterion(diag_preds, labels)
        concept_loss = concept_criterion(concept_preds, concepts)
        loss = diag_loss + 0.5 * concept_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        train_diag_loss += diag_loss.item()
        train_concept_loss += concept_loss.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'diag': f'{diag_loss.item():.4f}',
            'concept': f'{concept_loss.item():.4f}'
        })

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    print(f"\n   Evaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device)

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(val_metrics['loss'])
    history['val_f1'].append(val_metrics['f1'])
    history['val_concept_f1'].append(val_metrics['concept_f1'])

    print(f"\n   Epoch {epoch+1} Results:")
    print(f"   ├─ Train Loss: {avg_train_loss:.4f}")
    print(f"   ├─ Val Loss:   {val_metrics['loss']:.4f}")
    print(f"   ├─ Val F1 (Diagnosis): {val_metrics['f1']:.4f}")
    print(f"   └─ Val F1 (Concepts):  {val_metrics['concept_f1']:.4f}")

    # Save best model
    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        best_epoch = epoch + 1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_metrics['f1'],
            'val_metrics': val_metrics
        }, MODELS_PATH / 'best_model.pt')
        print(f"   ✅ Best model saved (F1: {best_val_f1:.4f})")

    print()

print(f"\n{'='*80}")
print(f"✅ Training complete!")
print(f"   Best epoch: {best_epoch}")
print(f"   Best val F1: {best_val_f1:.4f}")
print(f"{'='*80}\n")

# Save training history
with open(RESULTS_PATH / 'training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

# ============================================================================
# STEP 11: FINAL EVALUATION ON TEST SET
# ============================================================================

print("\n" + "="*80)
print("📊 STEP 11: FINAL EVALUATION ON TEST SET")
print("="*80)

# Load best model
checkpoint = torch.load(MODELS_PATH / 'best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"✅ Loaded best model from epoch {checkpoint['epoch']+1}")

# Evaluate on test set
print("\nEvaluating on test set...")
test_metrics = evaluate(model, test_loader, device)

print(f"\n📊 Test Set Results:")
print(f"{'='*80}")
print(f"Diagnosis Prediction:")
print(f"   Precision (Macro): {test_metrics['precision']:.4f}")
print(f"   Recall (Macro):    {test_metrics['recall']:.4f}")
print(f"   F1 Score (Macro):  {test_metrics['f1']:.4f}")
print(f"\nConcept Prediction:")
print(f"   Precision (Macro): {test_metrics['concept_precision']:.4f}")
print(f"   Recall (Macro):    {test_metrics['concept_recall']:.4f}")
print(f"   F1 Score (Macro):  {test_metrics['concept_f1']:.4f}")
print(f"{'='*80}\n")

# Save test results
with open(RESULTS_PATH / 'test_metrics.json', 'w') as f:
    json.dump(test_metrics, f, indent=2)

# ============================================================================
# STEP 12: COMPARISON WITH V301 BASELINE
# ============================================================================

print("\n" + "="*80)
print("📈 STEP 12: COMPARISON WITH V301 BASELINE")
print("="*80)

v301_f1 = 0.2801  # From shifamind301.py Phase 1
v302_f1 = test_metrics['f1']
improvement = (v302_f1 - v301_f1) / v301_f1 * 100

print(f"\nPhase 1 Comparison:")
print(f"{'='*80}")
print(f"ShifaMind v301 (Phase 1):  F1 = {v301_f1:.4f}")
print(f"ShifaMind v302 (Phase 1):  F1 = {v302_f1:.4f}")
print(f"Improvement:               {improvement:+.1f}%")
print(f"{'='*80}\n")

comparison = {
    'v301_f1': v301_f1,
    'v302_f1': v302_f1,
    'improvement_percent': improvement,
    'v302_concepts': NUM_CONCEPTS,
    'v302_density': float(density),
    'v302_concepts_per_sample': float(avg_concepts)
}

with open(RESULTS_PATH / 'v301_vs_v302_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ SHIFAMIND v302 PHASE 1 COMPLETE!")
print("="*80)

print(f"\n📊 Final Summary:")
print(f"   ✓ Concepts: {NUM_CONCEPTS} (v5 refined)")
print(f"   ✓ Density: {density*100:.1f}%")
print(f"   ✓ Concepts/sample: {avg_concepts:.1f}")
print(f"   ✓ Test F1: {v302_f1:.4f}")
print(f"   ✓ Improvement over v301: {improvement:+.1f}%")

print(f"\n📁 Output:")
print(f"   ✓ Run folder: {RUN_PATH}")
print(f"   ✓ Best model: {MODELS_PATH / 'best_model.pt'}")
print(f"   ✓ Results: {RESULTS_PATH}")

print(f"\n🚀 Next Steps:")
if v302_f1 > v301_f1:
    print(f"   ✅ Phase 1 improved! Ready to integrate Phases 2-4")
else:
    print(f"   ⚠️  Phase 1 did not improve. Review concept extraction")

print("\nAlhamdulillah! 🤲")
print("="*80)
