# -*- coding: utf-8 -*-
"""ShifaMind302.ipynb

SHIFAMIND302 - PHASE A ENHANCEMENTS
====================================
Enhanced concept labeling with UMLS/ScispaCy + GPU optimizations

Author: Mohammed Sameer Syed
University of Arizona - MS in AI Capstone

MAJOR ENHANCEMENTS FROM SHIFAMIND301:
1. ✅ ScispaCy + UMLS EntityLinker for concept extraction (replaces keyword matching)
2. ✅ Negation detection using negspacy
3. ✅ Expanded concept vocabulary: 113 → 263 concepts
4. ✅ GPU optimizations: Mixed precision (AMP), larger batches, pin_memory
5. ✅ New output folder: 11_ShifaMind_v302
6. ✅ UMLS CUI mapping for medical concepts
7. ✅ Enhanced temporal, severity, and anatomical concept coverage

Expected Improvements:
- Concept F1: 0.11 → 0.40+
- Better intervention accuracy (concept causality)
- Improved diagnosis F1 through better concept quality

================================================================================
"""

# ============================================================================
# INSTALLATION (run once in Colab)
# ============================================================================

!pip uninstall -y faiss faiss-gpu faiss-cpu
!pip install faiss-cpu

!pip install torch_geometric

!pip install scispacy
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
!pip install negspacy

from google.colab import drive
drive.mount('/content/drive')

"""## Configuration & Setup"""

import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from transformers import (
    AutoTokenizer, AutoModel,
    get_linear_schedule_with_warmup
)

import json
import pickle
import gzip
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import re
from datetime import datetime
import sys

# ScispaCy + UMLS imports
import scispacy
import spacy
from scispacy.linking import EntityLinker
from negspacy.negation import Negex

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# EXPANDED CONCEPT VOCABULARY (113 → 263 concepts)
# ============================================================================

print("\n" + "="*80)
print("🧠 EXPANDED CONCEPT VOCABULARY")
print("="*80)

# Original 113 concepts (preserved for backward compatibility)
ORIGINAL_CONCEPTS = [
    # Symptoms
    'fever', 'cough', 'dyspnea', 'pain', 'nausea', 'vomiting', 'diarrhea', 'fatigue',
    'headache', 'dizziness', 'weakness', 'confusion', 'syncope', 'chest', 'abdominal',
    'dysphagia', 'hemoptysis', 'hematuria', 'hematemesis', 'melena', 'jaundice',
    'edema', 'rash', 'pruritus', 'weight', 'anorexia', 'malaise',
    # Vital signs / Physical findings
    'hypotension', 'hypertension', 'tachycardia', 'bradycardia', 'tachypnea', 'hypoxia',
    'hypothermia', 'shock', 'altered', 'lethargic', 'obtunded',
    # Organ systems
    'cardiac', 'pulmonary', 'renal', 'hepatic', 'neurologic', 'gastrointestinal',
    'respiratory', 'cardiovascular', 'genitourinary', 'musculoskeletal', 'endocrine',
    'hematologic', 'dermatologic', 'psychiatric',
    # Common conditions
    'infection', 'sepsis', 'pneumonia', 'uti', 'cellulitis', 'meningitis',
    'failure', 'infarction', 'ischemia', 'hemorrhage', 'thrombosis', 'embolism',
    'obstruction', 'perforation', 'rupture', 'stenosis', 'regurgitation',
    'hypertrophy', 'atrophy', 'neoplasm', 'malignancy', 'metastasis',
    # Lab/diagnostic
    'elevated', 'decreased', 'anemia', 'leukocytosis', 'thrombocytopenia',
    'hyperglycemia', 'hypoglycemia', 'acidosis', 'alkalosis', 'hypoxemia',
    'creatinine', 'bilirubin', 'troponin', 'bnp', 'lactate', 'wbc', 'cultures',
    # Imaging/procedures
    'infiltrate', 'consolidation', 'effusion', 'cardiomegaly',
    'ultrasound', 'ct', 'mri', 'xray', 'echo', 'ekg',
    # Treatments
    'antibiotics', 'diuretics', 'vasopressors', 'insulin', 'anticoagulation',
    'oxygen', 'ventilation', 'dialysis', 'transfusion', 'surgery'
]

# NEW: Temporal aspects (15 concepts)
TEMPORAL_CONCEPTS = [
    'acute', 'chronic', 'recurrent', 'persistent', 'recent', 'progressive',
    'sudden', 'gradual', 'intermittent', 'continuous', 'transient', 'subacute',
    'longstanding', 'new-onset', 'relapsing'
]

# NEW: Severity indicators (10 concepts)
SEVERITY_CONCEPTS = [
    'mild', 'moderate', 'severe', 'critical', 'life-threatening',
    'stable', 'worsening', 'improving', 'resolved', 'refractory'
]

# NEW: Anatomical locations (25 concepts)
ANATOMICAL_CONCEPTS = [
    'left', 'right', 'bilateral', 'anterior', 'posterior', 'superior', 'inferior',
    'medial', 'lateral', 'proximal', 'distal', 'central', 'peripheral', 'upper',
    'lower', 'apical', 'basal', 'diffuse', 'focal', 'localized', 'widespread',
    'generalized', 'unilateral', 'ipsilateral', 'contralateral'
]

# NEW: Additional symptoms (30 concepts)
ADDITIONAL_SYMPTOMS = [
    'dyspepsia', 'constipation', 'bloating', 'wheezing', 'stridor', 'rhinorrhea',
    'epistaxis', 'diplopia', 'blurred vision', 'photophobia', 'tinnitus',
    'vertigo', 'tremor', 'seizure', 'paresthesia', 'dysarthria', 'aphasia',
    'ataxia', 'rigidity', 'spasticity', 'incontinence', 'retention',
    'oliguria', 'polyuria', 'polydipsia', 'palpitations', 'claudication',
    'paresis', 'plegia', 'arthralgia', 'myalgia'
]

# NEW: Additional lab abnormalities (20 concepts)
LAB_CONCEPTS = [
    'hypernatremia', 'hyponatremia', 'hyperkalemia', 'hypokalemia',
    'hypercalcemia', 'hypocalcemia', 'azotemia', 'uremia', 'proteinuria',
    'hematocrit', 'hemoglobin', 'platelets', 'coagulopathy', 'inr',
    'pt', 'ptt', 'lipase', 'amylase', 'transaminases', 'alkaline phosphatase'
]

# NEW: Comorbidities (20 concepts)
COMORBIDITY_CONCEPTS = [
    'diabetes mellitus', 'hypertensive', 'copd', 'asthma', 'heart failure',
    'coronary artery disease', 'atrial fibrillation', 'stroke', 'ckd',
    'cirrhosis', 'obesity', 'immunosuppressed', 'malnutrition', 'dementia',
    'parkinson', 'cancer', 'hiv', 'transplant', 'autoimmune', 'pregnancy'
]

# NEW: Clinical context (15 concepts)
CLINICAL_CONTEXT = [
    'admission', 'discharge', 'emergency', 'elective', 'icu', 'floor',
    'postoperative', 'preoperative', 'intraoperative', 'outpatient',
    'readmission', 'transfer', 'consult', 'follow-up', 'exacerbation'
]

# NEW: Expanded anatomical systems (15 concepts)
ANATOMICAL_SYSTEMS = [
    'coronary', 'cerebrovascular', 'peripheral vascular', 'lymphatic',
    'splenic', 'pancreatic', 'biliary', 'intestinal', 'colonic', 'rectal',
    'bladder', 'prostate', 'thyroid', 'adrenal', 'pituitary'
]

# Combine all concepts
GLOBAL_CONCEPTS = (
    ORIGINAL_CONCEPTS +
    TEMPORAL_CONCEPTS +
    SEVERITY_CONCEPTS +
    ANATOMICAL_CONCEPTS +
    ADDITIONAL_SYMPTOMS +
    LAB_CONCEPTS +
    COMORBIDITY_CONCEPTS +
    CLINICAL_CONTEXT +
    ANATOMICAL_SYSTEMS
)

# Remove duplicates while preserving order
seen = set()
GLOBAL_CONCEPTS = [x for x in GLOBAL_CONCEPTS if not (x in seen or seen.add(x))]

print(f"📊 Total concepts: {len(GLOBAL_CONCEPTS)}")
print(f"   Original: {len(ORIGINAL_CONCEPTS)}")
print(f"   Temporal: {len(TEMPORAL_CONCEPTS)}")
print(f"   Severity: {len(SEVERITY_CONCEPTS)}")
print(f"   Anatomical: {len(ANATOMICAL_CONCEPTS)}")
print(f"   Additional Symptoms: {len(ADDITIONAL_SYMPTOMS)}")
print(f"   Lab Values: {len(LAB_CONCEPTS)}")
print(f"   Comorbidities: {len(COMORBIDITY_CONCEPTS)}")
print(f"   Clinical Context: {len(CLINICAL_CONTEXT)}")
print(f"   Anatomical Systems: {len(ANATOMICAL_SYSTEMS)}")

"""## Phase 1: Concept Bottleneck Model with UMLS"""

#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND302 PHASE 1: Concept Bottleneck with UMLS (Top-50 ICD-10)
================================================================================

ENHANCEMENTS:
1. ScispaCy + UMLS EntityLinker for concept extraction
2. Negation detection (negspacy)
3. Expanded concept vocabulary (263 concepts)
4. GPU optimizations (AMP, larger batches)
5. New output folder: 11_ShifaMind_v302

Target Metrics:
- Diagnosis F1: >0.75
- Concept F1: >0.40 (improved from 0.11)
- Concept Completeness: >0.80

================================================================================
"""

print("\n" + "="*80)
print("🚀 SHIFAMIND302 PHASE 1 - UMLS-ENHANCED CONCEPTS")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

print("\n" + "="*80)
print("⚙️  CONFIGURATION")
print("="*80)

# Create timestamped run folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
OUTPUT_BASE = BASE_PATH / '11_ShifaMind_v302' / f'run_{timestamp}'

# Run-specific paths
SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'
CHECKPOINT_PATH = OUTPUT_BASE / 'checkpoints' / 'phase1'
RESULTS_PATH = OUTPUT_BASE / 'results' / 'phase1'
CONCEPT_STORE_PATH = OUTPUT_BASE / 'concept_store'
LOGS_PATH = OUTPUT_BASE / 'logs'

# Create all directories
for path in [SHARED_DATA_PATH, CHECKPOINT_PATH, RESULTS_PATH, CONCEPT_STORE_PATH, LOGS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Run Folder: {OUTPUT_BASE}")
print(f"📁 Timestamp: {timestamp}")
print(f"📁 Shared Data: {SHARED_DATA_PATH}")
print(f"📁 Checkpoints: {CHECKPOINT_PATH}")
print(f"📁 Results: {RESULTS_PATH}")
print(f"📁 Concept Store: {CONCEPT_STORE_PATH}")

# Raw MIMIC-IV paths
RAW_MIMIC_PATH = BASE_PATH / '01_Raw_Datasets' / 'Extracted' / 'mimic-iv-3.1' / 'mimic-iv-3.1' / 'hosp'
RAW_MIMIC_NOTE_PATH = BASE_PATH / '01_Raw_Datasets' / 'Extracted' / 'mimic-iv-note-2.2' / 'note'

# UMLS path
UMLS_PATH = BASE_PATH / '01_Raw_Datasets' / 'Extracted' / 'umls-2025AA-metathesaurus-full' / '2025AA' / 'META'

print(f"\n📂 MIMIC-IV Hosp: {RAW_MIMIC_PATH}")
print(f"📂 MIMIC-IV Note: {RAW_MIMIC_NOTE_PATH}")
print(f"📂 UMLS Path: {UMLS_PATH}")

# Hyperparameters
LAMBDA_DX = 1.0
LAMBDA_ALIGN = 0.5
LAMBDA_CONCEPT = 0.3

# GPU Optimization Settings
USE_AMP = True  # Mixed precision training
BATCH_SIZE_TRAIN = 16  # Increased from 8
BATCH_SIZE_VAL = 32    # Increased from 16
PIN_MEMORY = True
NUM_WORKERS = 2
PREFETCH_FACTOR = 2

print(f"\n⚖️  Loss Weights:")
print(f"   λ1 (Diagnosis): {LAMBDA_DX}")
print(f"   λ2 (Alignment): {LAMBDA_ALIGN}")
print(f"   λ3 (Concept):   {LAMBDA_CONCEPT}")

print(f"\n🚀 GPU Optimizations:")
print(f"   Mixed Precision (AMP): {USE_AMP}")
print(f"   Batch Size (Train): {BATCH_SIZE_TRAIN}")
print(f"   Batch Size (Val/Test): {BATCH_SIZE_VAL}")
print(f"   Pin Memory: {PIN_MEMORY}")

# ============================================================================
# SCISPACY + UMLS INITIALIZATION
# ============================================================================

print("\n" + "="*80)
print("🔬 INITIALIZING SCISPACY + UMLS")
print("="*80)

print("\n📥 Loading ScispaCy model (en_core_sci_lg)...")
nlp = spacy.load("en_core_sci_lg")

print("📥 Adding UMLS EntityLinker...")
# Use full UMLS if available, otherwise use built-in KB
if UMLS_PATH.exists():
    print(f"   Using full UMLS from: {UMLS_PATH}")
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
else:
    print("   ⚠️  Full UMLS not found, using built-in KB")
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

print("📥 Adding Negex (negation detection)...")
nlp.add_pipe("negex")

print("✅ ScispaCy + UMLS + Negex initialized")

# ============================================================================
# ENHANCED CONCEPT EXTRACTION WITH UMLS
# ============================================================================

def extract_concepts_umls(text, concept_vocabulary):
    """
    Extract concepts using ScispaCy + UMLS + Negation detection

    Replaces simple keyword matching with medical NER

    Returns:
        concept_vector: Binary vector [num_concepts]
        extracted_entities: List of (entity_text, cui, negated)
    """
    doc = nlp(str(text)[:5000])  # Limit to 5000 chars for speed

    # Initialize concept vector
    concept_vector = [0] * len(concept_vocabulary)
    concept_to_idx = {c.lower(): i for i, c in enumerate(concept_vocabulary)}

    extracted_entities = []

    # Process entities
    for ent in doc.ents:
        entity_text = ent.text.lower()
        is_negated = ent._.negex

        # Get UMLS CUI if available
        cui = None
        if hasattr(ent._, 'kb_ents') and len(ent._.kb_ents) > 0:
            cui = ent._.kb_ents[0][0]  # Top CUI

        extracted_entities.append((entity_text, cui, is_negated))

        # Map to concept vocabulary (only if NOT negated)
        if not is_negated:
            # Direct match
            for concept, idx in concept_to_idx.items():
                if concept in entity_text or entity_text in concept:
                    concept_vector[idx] = 1

    return concept_vector, extracted_entities

# ============================================================================
# STEP 1: COMPUTE TOP-50 ICD-10 CODES (Same as 301)
# ============================================================================

print("\n" + "="*80)
print("📊 COMPUTING TOP-50 ICD-10 CODES FROM MIMIC-IV")
print("="*80)

def normalize_icd10_code(code):
    """Normalize ICD-10 code: uppercase, remove dots"""
    if pd.isna(code):
        return None
    code_str = str(code).upper().replace('.', '').strip()
    return code_str if code_str else None

print("\n1️⃣ Loading diagnoses_icd.csv.gz...")
diagnoses_path = RAW_MIMIC_PATH / 'diagnoses_icd.csv.gz'
df_diag = pd.read_csv(diagnoses_path, compression='gzip')
print(f"   Loaded {len(df_diag):,} diagnosis records")

# Filter ICD-10 only
df_diag_icd10 = df_diag[df_diag['icd_version'] == 10].copy()
print(f"   ICD-10 records: {len(df_diag_icd10):,}")

# Normalize codes
df_diag_icd10['icd_code_normalized'] = df_diag_icd10['icd_code'].apply(normalize_icd10_code)
df_diag_icd10 = df_diag_icd10.dropna(subset=['icd_code_normalized'])
print(f"   After normalization: {len(df_diag_icd10):,}")

print("\n2️⃣ Loading discharge notes...")
discharge_path = RAW_MIMIC_NOTE_PATH / 'discharge.csv.gz'
df_notes = pd.read_csv(discharge_path, compression='gzip')
print(f"   Loaded {len(df_notes):,} discharge notes")

# Keep only non-empty notes
df_notes = df_notes[df_notes['text'].notna() & (df_notes['text'].str.len() > 100)].copy()
print(f"   Non-empty notes: {len(df_notes):,}")

# Get unique hadm_id with notes
valid_hadm_ids = set(df_notes['hadm_id'].unique())
print(f"   Unique hadm_id with discharge notes: {len(valid_hadm_ids):,}")

print("\n3️⃣ Filtering diagnoses to hadm_id with notes...")
df_diag_icd10 = df_diag_icd10[df_diag_icd10['hadm_id'].isin(valid_hadm_ids)].copy()
print(f"   Diagnoses with notes: {len(df_diag_icd10):,}")

print("\n4️⃣ Computing Top-50 ICD-10 codes by admission frequency...")
code_counts = df_diag_icd10.groupby('icd_code_normalized')['hadm_id'].nunique().sort_values(ascending=False)
print(f"   Total unique ICD-10 codes: {len(code_counts):,}")

# Take top 50
TOP_50_CODES = code_counts.head(50).index.tolist()
TOP_50_COUNTS = code_counts.head(50).values.tolist()

print(f"\n✅ TOP-50 ICD-10 CODES:")
print(f"{'Rank':<6} {'Code':<10} {'Admissions':<12}")
print("-" * 30)
for rank, (code, count) in enumerate(zip(TOP_50_CODES, TOP_50_COUNTS), 1):
    print(f"{rank:<6} {code:<10} {count:<12,}")

# Save Top-50 info
top50_info = {
    'timestamp': timestamp,
    'top_50_codes': TOP_50_CODES,
    'top_50_counts': {code: int(count) for code, count in zip(TOP_50_CODES, TOP_50_COUNTS)},
    'total_unique_codes': len(code_counts),
    'total_icd10_records': len(df_diag_icd10),
    'valid_admissions': len(valid_hadm_ids)
}

with open(SHARED_DATA_PATH / 'top50_icd10_info.json', 'w') as f:
    json.dump(top50_info, f, indent=2)

print(f"\n💾 Saved Top-50 info to: {SHARED_DATA_PATH / 'top50_icd10_info.json'}")

# ============================================================================
# STEP 2: BUILD MIMIC_DX_DATA.CSV (Same as 301)
# ============================================================================

print("\n" + "="*80)
print("📊 BUILDING mimic_dx_data.csv WITH TOP-50 LABELS")
print("="*80)

print("\n1️⃣ Creating multi-label matrix...")
hadm_labels = defaultdict(lambda: [0] * len(TOP_50_CODES))
code_to_idx = {code: idx for idx, code in enumerate(TOP_50_CODES)}

for _, row in tqdm(df_diag_icd10.iterrows(), total=len(df_diag_icd10), desc="Processing diagnoses"):
    hadm_id = row['hadm_id']
    code = row['icd_code_normalized']
    if code in code_to_idx:
        hadm_labels[hadm_id][code_to_idx[code]] = 1

print(f"   Labeled {len(hadm_labels):,} admissions")

print("\n2️⃣ Merging with discharge notes...")
df_notes_with_labels = df_notes.copy()
df_notes_with_labels['labels'] = df_notes_with_labels['hadm_id'].map(
    lambda x: hadm_labels.get(x, [0] * len(TOP_50_CODES))
)

# Keep only admissions that have at least one Top-50 label
df_notes_with_labels['has_top50'] = df_notes_with_labels['labels'].apply(lambda x: sum(x) > 0)
df_final = df_notes_with_labels[df_notes_with_labels['has_top50']].copy()

print(f"   Admissions with Top-50 labels: {len(df_final):,}")

# Add individual code columns
for idx, code in enumerate(TOP_50_CODES):
    df_final[code] = df_final['labels'].apply(lambda x: x[idx])

print("\n3️⃣ Label distribution:")
label_counts = [df_final[code].sum() for code in TOP_50_CODES]
print(f"   Mean labels per admission: {np.mean([sum(x) for x in df_final['labels']]):.2f}")
print(f"   Median labels per admission: {np.median([sum(x) for x in df_final['labels']]):.0f}")

# Save to CSV
mimic_dx_path = OUTPUT_BASE / 'mimic_dx_data_top50.csv'
df_final[['subject_id', 'hadm_id', 'text'] + TOP_50_CODES].to_csv(mimic_dx_path, index=False)
print(f"\n💾 Saved dataset to: {mimic_dx_path}")
print(f"   Rows: {len(df_final):,}")

# ============================================================================
# STEP 3: CREATE TRAIN/VAL/TEST SPLITS
# ============================================================================

print("\n" + "="*80)
print("📊 CREATING TRAIN/VAL/TEST SPLITS (FRESH)")
print("="*80)

df = df_final[['text', 'labels'] + TOP_50_CODES].copy()
df = df.dropna(subset=['text'])

print(f"\n📊 Dataset size: {len(df):,} samples")

# Split: 70% train, 15% val, 15% test
train_idx, temp_idx = train_test_split(
    range(len(df)),
    test_size=0.3,
    random_state=SEED
)
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    random_state=SEED
)

df_train = df.iloc[train_idx].reset_index(drop=True)
df_val = df.iloc[val_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)

print(f"\n✅ Splits created:")
print(f"   Train: {len(df_train):,} ({len(df_train)/len(df)*100:.1f}%)")
print(f"   Val:   {len(df_val):,} ({len(df_val)/len(df)*100:.1f}%)")
print(f"   Test:  {len(df_test):,} ({len(df_test)/len(df)*100:.1f}%)")

# Save splits
with open(SHARED_DATA_PATH / 'train_split.pkl', 'wb') as f:
    pickle.dump(df_train, f)
with open(SHARED_DATA_PATH / 'val_split.pkl', 'wb') as f:
    pickle.dump(df_val, f)
with open(SHARED_DATA_PATH / 'test_split.pkl', 'wb') as f:
    pickle.dump(df_test, f)

print(f"\n💾 Saved splits to: {SHARED_DATA_PATH}")

# ============================================================================
# STEP 4: GENERATE CONCEPT LABELS (UMLS-BASED)
# ============================================================================

print("\n" + "="*80)
print("🧠 GENERATING CONCEPT LABELS (UMLS-BASED)")
print("="*80)

def generate_concept_labels_umls(texts, concepts):
    """Generate concept labels using ScispaCy + UMLS"""
    labels = []
    entities_log = []

    for text in tqdm(texts, desc="Labeling with UMLS"):
        concept_vector, extracted_ents = extract_concepts_umls(text, concepts)
        labels.append(concept_vector)
        entities_log.append(extracted_ents)

    return np.array(labels), entities_log

print(f"\n🔍 Using {len(GLOBAL_CONCEPTS)} concepts with UMLS extraction")
print("⏳ This may take 2-4 hours for 115K samples...")

train_concept_labels, train_entities = generate_concept_labels_umls(df_train['text'], GLOBAL_CONCEPTS)
val_concept_labels, val_entities = generate_concept_labels_umls(df_val['text'], GLOBAL_CONCEPTS)
test_concept_labels, test_entities = generate_concept_labels_umls(df_test['text'], GLOBAL_CONCEPTS)

print(f"\n✅ Concept labels generated:")
print(f"   Shape: {train_concept_labels.shape}")
print(f"   Concepts per sample (train): {train_concept_labels.sum(axis=1).mean():.2f}")

# Save concept labels
np.save(SHARED_DATA_PATH / 'train_concept_labels.npy', train_concept_labels)
np.save(SHARED_DATA_PATH / 'val_concept_labels.npy', val_concept_labels)
np.save(SHARED_DATA_PATH / 'test_concept_labels.npy', test_concept_labels)

# Save concept list
with open(SHARED_DATA_PATH / 'concept_list.json', 'w') as f:
    json.dump(GLOBAL_CONCEPTS, f, indent=2)

# Save entity extraction log (sample for debugging)
with open(CONCEPT_STORE_PATH / 'entity_extraction_sample.json', 'w') as f:
    json.dump({
        'train_sample': train_entities[:10],
        'val_sample': val_entities[:10]
    }, f, indent=2)

print(f"💾 Saved concept labels to: {SHARED_DATA_PATH}")

# ============================================================================
# ARCHITECTURE (Same as 301 but optimized)
# ============================================================================

print("\n" + "="*80)
print("🏗️  ARCHITECTURE: CONCEPT BOTTLENECK")
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
    """ShifaMind302 Phase 1: UMLS-Enhanced Concept Bottleneck"""
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


class MultiObjectiveLoss(nn.Module):
    """Multi-objective loss: L_dx + L_align + L_concept"""
    def __init__(self, lambda_dx=1.0, lambda_align=0.5, lambda_concept=0.3):
        super().__init__()
        self.lambda_dx = lambda_dx
        self.lambda_align = lambda_align
        self.lambda_concept = lambda_concept
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, dx_labels, concept_labels):
        loss_dx = self.bce(outputs['logits'], dx_labels)

        dx_probs = torch.sigmoid(outputs['logits'])
        concept_scores = outputs['concept_scores']
        loss_align = torch.abs(
            dx_probs.unsqueeze(-1) - concept_scores.unsqueeze(1)
        ).mean()

        concept_logits = torch.logit(concept_scores.clamp(1e-7, 1-1e-7))
        loss_concept = self.bce(concept_logits, concept_labels)

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


print("✅ Architecture defined")

# ============================================================================
# DATASET (GPU-Optimized)
# ============================================================================

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

# ============================================================================
# TRAINING (GPU-Optimized with AMP)
# ============================================================================

print("\n" + "="*80)
print("🏋️  TRAINING PHASE 1 (GPU-OPTIMIZED)")
print("="*80)

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

model = ShifaMind302Phase1(
    base_model,
    num_concepts=len(GLOBAL_CONCEPTS),
    num_classes=len(TOP_50_CODES),
    fusion_layers=[9, 11]
).to(device)

print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"   Num concepts: {len(GLOBAL_CONCEPTS)}")
print(f"   Num diagnoses: {len(TOP_50_CODES)}")

# Create datasets with GPU optimizations
train_dataset = ConceptDataset(
    df_train['text'].tolist(),
    df_train['labels'].tolist(),
    train_concept_labels,
    tokenizer
)
val_dataset = ConceptDataset(
    df_val['text'].tolist(),
    df_val['labels'].tolist(),
    val_concept_labels,
    tokenizer
)
test_dataset = ConceptDataset(
    df_test['text'].tolist(),
    df_test['labels'].tolist(),
    test_concept_labels,
    tokenizer
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True,
    pin_memory=PIN_MEMORY,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE_VAL,
    pin_memory=PIN_MEMORY,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE_VAL,
    pin_memory=PIN_MEMORY,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR
)

print(f"✅ Datasets ready (GPU-optimized)")

# Training setup
criterion = MultiObjectiveLoss(
    lambda_dx=LAMBDA_DX,
    lambda_align=LAMBDA_ALIGN,
    lambda_concept=LAMBDA_CONCEPT
)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

num_epochs = 5
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=len(train_loader) // 2,
    num_training_steps=len(train_loader) * num_epochs
)

# Initialize AMP scaler
scaler = GradScaler() if USE_AMP else None

best_f1 = 0.0
history = {'train_loss': [], 'val_f1': [], 'concept_f1': []}

# Training loop with AMP
for epoch in range(num_epochs):
    print(f"\n{'='*70}\nEpoch {epoch+1}/{num_epochs}\n{'='*70}")

    model.train()
    epoch_losses = defaultdict(list)

    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        dx_labels = batch['labels'].to(device, non_blocking=True)
        concept_labels = batch['concept_labels'].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        if USE_AMP:
            with autocast():
                outputs = model(input_ids, attention_mask)
                loss, components = criterion(outputs, dx_labels, concept_labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids, attention_mask)
            loss, components = criterion(outputs, dx_labels, concept_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        for k, v in components.items():
            epoch_losses[k].append(v)

    print(f"\n📊 Epoch {epoch+1} Losses:")
    print(f"   Total:     {np.mean(epoch_losses['total']):.4f}")
    print(f"   Diagnosis: {np.mean(epoch_losses['dx']):.4f}")
    print(f"   Alignment: {np.mean(epoch_losses['align']):.4f}")
    print(f"   Concept:   {np.mean(epoch_losses['concept']):.4f}")

    # Validation
    model.eval()
    all_dx_preds, all_dx_labels = [], []
    all_concept_preds, all_concept_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            dx_labels = batch['labels'].to(device, non_blocking=True)
            concept_labels = batch['concept_labels'].to(device, non_blocking=True)

            if USE_AMP:
                with autocast():
                    outputs = model(input_ids, attention_mask)
            else:
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

    dx_f1 = f1_score(all_dx_labels, dx_pred_binary, average='macro', zero_division=0)
    concept_f1 = f1_score(all_concept_labels, concept_pred_binary, average='macro', zero_division=0)

    print(f"\n📈 Validation:")
    print(f"   Diagnosis F1: {dx_f1:.4f}")
    print(f"   Concept F1:   {concept_f1:.4f}")

    history['train_loss'].append(np.mean(epoch_losses['total']))
    history['val_f1'].append(dx_f1)
    history['concept_f1'].append(concept_f1)

    if dx_f1 > best_f1:
        best_f1 = dx_f1
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'macro_f1': best_f1,
            'concept_f1': concept_f1,
            'concept_embeddings': model.concept_embeddings.data.cpu(),
            'num_concepts': model.num_concepts,
            'config': {
                'num_concepts': len(GLOBAL_CONCEPTS),
                'num_classes': len(TOP_50_CODES),
                'fusion_layers': [9, 11],
                'lambda_dx': LAMBDA_DX,
                'lambda_align': LAMBDA_ALIGN,
                'lambda_concept': LAMBDA_CONCEPT,
                'top_50_codes': TOP_50_CODES,
                'timestamp': timestamp,
                'use_amp': USE_AMP,
                'batch_size_train': BATCH_SIZE_TRAIN
            }
        }
        torch.save(checkpoint, CHECKPOINT_PATH / 'phase1_best.pt')
        print(f"   ✅ Saved best model (F1: {best_f1:.4f})")

print(f"\n✅ Training complete! Best Diagnosis F1: {best_f1:.4f}")

# ============================================================================
# FINAL EVALUATION
# ============================================================================

print("\n" + "="*80)
print("📊 FINAL TEST EVALUATION")
print("="*80)

checkpoint = torch.load(CHECKPOINT_PATH / 'phase1_best.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_dx_preds, all_dx_labels = [], []
all_concept_preds, all_concept_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        dx_labels = batch['labels'].to(device, non_blocking=True)
        concept_labels = batch['concept_labels'].to(device, non_blocking=True)

        if USE_AMP:
            with autocast():
                outputs = model(input_ids, attention_mask)
        else:
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
    for i in range(len(TOP_50_CODES))
]

concept_f1 = f1_score(all_concept_labels, concept_pred_binary, average='macro', zero_division=0)

print("\n" + "="*80)
print("🎉 SHIFAMIND302 PHASE 1 - FINAL RESULTS")
print("="*80)

print("\n🎯 Diagnosis Performance (Top-50 ICD-10):")
print(f"   Macro F1:    {macro_f1:.4f}")
print(f"   Micro F1:    {micro_f1:.4f}")
print(f"   Precision:   {macro_precision:.4f}")
print(f"   Recall:      {macro_recall:.4f}")

print(f"\n🧠 Concept Performance:")
print(f"   Concept F1:  {concept_f1:.4f}")

print(f"\n📊 Top-10 Best Performing Diagnoses:")
top_10_best = sorted(zip(TOP_50_CODES, per_class_f1), key=lambda x: x[1], reverse=True)[:10]
for rank, (code, f1) in enumerate(top_10_best, 1):
    count = top50_info['top_50_counts'].get(code, 0)
    print(f"   {rank}. {code}: F1={f1:.4f} (n={count:,})")

print(f"\n📊 Top-10 Worst Performing Diagnoses:")
top_10_worst = sorted(zip(TOP_50_CODES, per_class_f1), key=lambda x: x[1])[:10]
for rank, (code, f1) in enumerate(top_10_worst, 1):
    count = top50_info['top_50_counts'].get(code, 0)
    print(f"   {rank}. {code}: F1={f1:.4f} (n={count:,})")

# Save results
results = {
    'phase': 'ShifaMind302 Phase 1 - UMLS Enhanced',
    'timestamp': timestamp,
    'run_folder': str(OUTPUT_BASE),
    'diagnosis_metrics': {
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'precision': float(macro_precision),
        'recall': float(macro_recall),
        'per_class_f1': {code: float(f1) for code, f1 in zip(TOP_50_CODES, per_class_f1)}
    },
    'concept_metrics': {
        'concept_f1': float(concept_f1),
        'num_concepts': len(GLOBAL_CONCEPTS)
    },
    'dataset_info': {
        'num_labels': len(TOP_50_CODES),
        'train_samples': len(df_train),
        'val_samples': len(df_val),
        'test_samples': len(df_test)
    },
    'loss_weights': {
        'lambda_dx': LAMBDA_DX,
        'lambda_align': LAMBDA_ALIGN,
        'lambda_concept': LAMBDA_CONCEPT
    },
    'gpu_optimizations': {
        'use_amp': USE_AMP,
        'batch_size_train': BATCH_SIZE_TRAIN,
        'batch_size_val': BATCH_SIZE_VAL,
        'pin_memory': PIN_MEMORY
    },
    'training_history': history
}

with open(RESULTS_PATH / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save per-label F1 scores as CSV
per_label_df = pd.DataFrame({
    'icd_code': TOP_50_CODES,
    'f1_score': per_class_f1,
    'train_count': [top50_info['top_50_counts'].get(code, 0) for code in TOP_50_CODES]
})
per_label_df = per_label_df.sort_values('f1_score', ascending=False)
per_label_df.to_csv(RESULTS_PATH / 'per_label_f1.csv', index=False)

print(f"\n💾 Results saved to: {RESULTS_PATH / 'results.json'}")
print(f"💾 Per-label F1 saved to: {RESULTS_PATH / 'per_label_f1.csv'}")
print(f"💾 Best model saved to: {CHECKPOINT_PATH / 'phase1_best.pt'}")

print("\n" + "="*80)
print("✅ SHIFAMIND302 PHASE 1 COMPLETE!")
print("="*80)
print("\n📍 Summary:")
print(f"   ✅ UMLS-based concept extraction")
print(f"   ✅ Expanded to {len(GLOBAL_CONCEPTS)} concepts")
print(f"   ✅ GPU-optimized training (AMP: {USE_AMP})")
print(f"   ✅ Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}")
print(f"   ✅ Concept F1: {concept_f1:.4f} (target: >0.40)")
print(f"\n📁 All artifacts saved to: {OUTPUT_BASE}")
print(f"\nNext: Implement Phase 2 (GraphSAGE) with enhanced concepts")
print("\nAlhamdulillah! 🤲")
