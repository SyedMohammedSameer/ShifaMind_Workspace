"""
EMERGENCY: Recreate Phase 1 shared_data folder (NO TRAINING!)

This script recreates ONLY the shared_data files that Phase 2/3 need:
- train_split.pkl, val_split.pkl, test_split.pkl
- train_concept_labels.npy, val_concept_labels.npy, test_concept_labels.npy
- concept_list.json
- top50_icd10_info.json

Uses the EXACT same logic as Phase 1, but skips all training.
Takes ~2-3 minutes instead of ~20 minutes!
"""

import pandas as pd
import numpy as np
import pickle
import json
import re
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# PATHS
# ============================================================================

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')

# Original data (from old run)
ORIGINAL_RUN = BASE_PATH / '10_ShifaMind' / 'run_20260102_203225'
ORIGINAL_CSV = ORIGINAL_RUN / 'mimic_dx_data_top50.csv'

# Phase 1 run that lost its shared_data
PHASE1_RUN = BASE_PATH / '10_ShifaMind' / 'run_20260215_013437'
SHARED_DATA_PATH = PHASE1_RUN / 'shared_data'
SHARED_DATA_PATH.mkdir(parents=True, exist_ok=True)

print("="*80)
print("🚨 EMERGENCY: RECREATING PHASE 1 SHARED_DATA")
print("="*80)
print(f"\n📁 Original CSV: {ORIGINAL_CSV}")
print(f"📁 Recreating: {SHARED_DATA_PATH}")

# ============================================================================
# CHECK IF ORIGINAL CSV EXISTS
# ============================================================================

if not ORIGINAL_CSV.exists():
    print(f"\n❌ ERROR: Original CSV not found at {ORIGINAL_CSV}")
    print("\n💡 OPTIONS:")
    print("   1. Re-run Phase 1 completely")
    print("   2. Check if you have the CSV in another location")
    print("   3. Re-download MIMIC-IV data")
    raise FileNotFoundError(f"Missing: {ORIGINAL_CSV}")

# ============================================================================
# LOAD ORIGINAL DATA
# ============================================================================

print("\n📊 Loading original CSV...")
df = pd.read_csv(ORIGINAL_CSV)
print(f"✅ Loaded {len(df):,} samples")

# Get Top-50 codes from the data
top50_codes = sorted(df['icd10_code'].value_counts().head(50).index.tolist())
NUM_LABELS = len(top50_codes)

# ============================================================================
# DEFINE CONCEPTS (SAME AS PHASE 1)
# ============================================================================

ALL_CONCEPTS = [
    # Demographics (8)
    'age', 'male', 'female', 'elderly', 'adult', 'pediatric', 'gender', 'race',

    # Vital signs (10)
    'temperature', 'fever', 'hypothermia', 'blood pressure', 'hypertension',
    'hypotension', 'heart rate', 'tachycardia', 'bradycardia', 'respiratory rate',

    # Cardiovascular (12)
    'chest pain', 'myocardial infarction', 'heart failure', 'arrhythmia',
    'atrial fibrillation', 'cardiac arrest', 'coronary artery disease',
    'pericarditis', 'endocarditis', 'cardiomyopathy', 'valve disease', 'thrombosis',

    # Respiratory (11)
    'dyspnea', 'shortness of breath', 'cough', 'pneumonia', 'asthma',
    'chronic obstructive pulmonary disease', 'copd', 'pulmonary embolism',
    'respiratory failure', 'pleural effusion', 'pneumothorax',

    # Neurological (10)
    'altered mental status', 'confusion', 'seizure', 'stroke', 'headache',
    'syncope', 'dizziness', 'weakness', 'numbness', 'paralysis',

    # Gastrointestinal (10)
    'abdominal pain', 'nausea', 'vomiting', 'diarrhea', 'constipation',
    'gastrointestinal bleeding', 'liver disease', 'pancreatitis', 'bowel obstruction', 'ascites',

    # Renal/Genitourinary (8)
    'acute kidney injury', 'chronic kidney disease', 'renal failure',
    'urinary tract infection', 'hematuria', 'dysuria', 'oliguria', 'anuria',

    # Metabolic/Endocrine (10)
    'diabetes', 'hyperglycemia', 'hypoglycemia', 'diabetic ketoacidosis',
    'electrolyte imbalance', 'hyponatremia', 'hypernatremia', 'hypokalemia',
    'hyperkalemia', 'thyroid disorder',

    # Hematologic (8)
    'anemia', 'thrombocytopenia', 'coagulopathy', 'bleeding', 'leukocytosis',
    'leukopenia', 'neutropenia', 'pancytopenia',

    # Infectious (8)
    'sepsis', 'infection', 'bacteremia', 'fever of unknown origin',
    'cellulitis', 'abscess', 'osteomyelitis', 'meningitis',

    # Trauma/Injury (6)
    'trauma', 'fracture', 'fall', 'injury', 'laceration', 'burn',

    # Oncologic (5)
    'cancer', 'malignancy', 'metastasis', 'chemotherapy', 'radiation',

    # Psychiatric (5)
    'depression', 'anxiety', 'psychosis', 'suicidal ideation', 'delirium'
]

NUM_CONCEPTS = len(ALL_CONCEPTS)

print(f"\n🧠 Using {NUM_CONCEPTS} concepts")
print(f"🏷️  Using {NUM_LABELS} diagnoses")

# ============================================================================
# CREATE SPLITS (SAME RANDOM STATE AS ORIGINAL!)
# ============================================================================

print("\n📊 Creating train/val/test splits...")

# Use same random state for reproducibility
from sklearn.model_selection import train_test_split

# 70/15/15 split
df_train, df_temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df['icd10_code'])
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp['icd10_code'])

print(f"✅ Splits created:")
print(f"   Train: {len(df_train):,} ({len(df_train)/len(df)*100:.1f}%)")
print(f"   Val:   {len(df_val):,} ({len(df_val)/len(df)*100:.1f}%)")
print(f"   Test:  {len(df_test):,} ({len(df_test)/len(df)*100:.1f}%)")

# Save splits
print(f"\n💾 Saving splits...")
with open(SHARED_DATA_PATH / 'train_split.pkl', 'wb') as f:
    pickle.dump(df_train, f)
with open(SHARED_DATA_PATH / 'val_split.pkl', 'wb') as f:
    pickle.dump(df_val, f)
with open(SHARED_DATA_PATH / 'test_split.pkl', 'wb') as f:
    pickle.dump(df_test, f)

print(f"✅ Saved to: {SHARED_DATA_PATH}")

# ============================================================================
# GENERATE CONCEPT LABELS (KEYWORD-BASED)
# ============================================================================

print("\n🧠 Generating concept labels...")

def label_concepts(text, concepts):
    """Simple keyword matching for concept labeling"""
    if pd.isna(text):
        return np.zeros(len(concepts), dtype=np.float32)

    text_lower = text.lower()
    labels = np.zeros(len(concepts), dtype=np.float32)

    for i, concept in enumerate(concepts):
        # Simple keyword match
        if concept.lower() in text_lower:
            labels[i] = 1.0

    return labels

# Label all splits
print("Labeling train set...")
train_concept_labels = np.array([
    label_concepts(text, ALL_CONCEPTS)
    for text in tqdm(df_train['text'], desc="Train")
])

print("Labeling val set...")
val_concept_labels = np.array([
    label_concepts(text, ALL_CONCEPTS)
    for text in tqdm(df_val['text'], desc="Val")
])

print("Labeling test set...")
test_concept_labels = np.array([
    label_concepts(text, ALL_CONCEPTS)
    for text in tqdm(df_test['text'], desc="Test")
])

print(f"\n✅ Concept labels generated:")
print(f"   Shape: {train_concept_labels.shape}")
print(f"   Avg concepts/sample: {train_concept_labels.mean(axis=0).sum():.2f}")

# Save concept labels
print(f"\n💾 Saving concept labels...")
np.save(SHARED_DATA_PATH / 'train_concept_labels.npy', train_concept_labels)
np.save(SHARED_DATA_PATH / 'val_concept_labels.npy', val_concept_labels)
np.save(SHARED_DATA_PATH / 'test_concept_labels.npy', test_concept_labels)

print(f"✅ Saved to: {SHARED_DATA_PATH}")

# ============================================================================
# SAVE METADATA
# ============================================================================

print("\n💾 Saving metadata...")

# Top-50 info
top50_info = {
    'top_50_codes': top50_codes,
    'code_names': {code: f"ICD-10: {code}" for code in top50_codes}
}
with open(SHARED_DATA_PATH / 'top50_icd10_info.json', 'w') as f:
    json.dump(top50_info, f, indent=2)

# Concept list
with open(SHARED_DATA_PATH / 'concept_list.json', 'w') as f:
    json.dump(ALL_CONCEPTS, f, indent=2)

print(f"✅ Saved metadata")

# ============================================================================
# VERIFY
# ============================================================================

print("\n" + "="*80)
print("✅ SHARED_DATA RECREATED SUCCESSFULLY!")
print("="*80)

print("\n📁 Created files:")
print(f"   ✅ {SHARED_DATA_PATH / 'train_split.pkl'}")
print(f"   ✅ {SHARED_DATA_PATH / 'val_split.pkl'}")
print(f"   ✅ {SHARED_DATA_PATH / 'test_split.pkl'}")
print(f"   ✅ {SHARED_DATA_PATH / 'train_concept_labels.npy'}")
print(f"   ✅ {SHARED_DATA_PATH / 'val_concept_labels.npy'}")
print(f"   ✅ {SHARED_DATA_PATH / 'test_concept_labels.npy'}")
print(f"   ✅ {SHARED_DATA_PATH / 'top50_icd10_info.json'}")
print(f"   ✅ {SHARED_DATA_PATH / 'concept_list.json'}")

print("\n🚀 NOW YOU CAN RUN PHASE 3:")
print("!python /content/drive/MyDrive/ShifaMind/phase3_training_optimized.py")
