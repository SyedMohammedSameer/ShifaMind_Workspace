#!/usr/bin/env python3
"""
================================================================================
SHIFAMIND PHASE A - WEEK 1: PILOT STUDY
utilphase1.py - Diagnosis-driven concept extraction with ScispaCy
================================================================================
Author: ShifaMind Research Team
Purpose: Test concept extraction on 5K samples before full implementation
Expected runtime: 30-60 minutes
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
from tqdm.auto import tqdm
import time
import re

print("="*80)
print("🚀 SHIFAMIND PHASE A - WEEK 1 PILOT (v3 - AGGRESSIVE)")
print("="*80)
print("Version 3: Aggressive filtering - removed all history patterns & common terms")
print("Target: 120-150 concepts, 8-12% density, <40% max frequency")

# ============================================================================
# DEPENDENCY CHECK & INSTALLATION
# ============================================================================

def check_and_install_dependencies():
    """Check and install required packages"""
    print("\n📦 Checking dependencies...")

    try:
        import spacy
        import scispacy
        print("✅ scispacy found")
    except ImportError:
        print("Installing scispacy...")
        os.system('pip install -q scispacy')
        import spacy
        import scispacy

    try:
        from negspacy.negation import Negex
        print("✅ negspacy found")
    except ImportError:
        print("Installing negspacy...")
        os.system('pip install -q negspacy')

    # Check if large model is installed
    try:
        nlp_test = spacy.load("en_core_sci_lg")
        print("✅ en_core_sci_lg found")
        return True
    except:
        print("⚠️  en_core_sci_lg not found. Installing (this may take 2-3 minutes)...")
        # Use the large model which is more stable
        result = os.system('pip install -q https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz')

        if result != 0:
            print("❌ Installation failed. Trying alternative method...")
            os.system('pip install --no-deps https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz')

        try:
            nlp_test = spacy.load("en_core_sci_lg")
            print("✅ en_core_sci_lg installed successfully")
            return True
        except:
            print("❌ Could not install en_core_sci_lg")
            return False

if not check_and_install_dependencies():
    print("\n" + "="*80)
    print("❌ DEPENDENCY INSTALLATION FAILED")
    print("="*80)
    print("\nPlease manually run:")
    print("  pip install scispacy negspacy")
    print("  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz")
    sys.exit(1)

# Now import after ensuring installation
import spacy
from negspacy.negation import Negex

device = 'cuda' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu'
print(f"\n🖥️  Device: {device}")

# ============================================================================
# CONFIGURATION
# ============================================================================

print("\n" + "="*80)
print("⚙️  CONFIGURATION")
print("="*80)

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
UMLS_PATH = BASE_PATH / '01_Raw_Datasets' / 'Extracted' / 'umls-2025AA-metathesaurus-full' / '2025AA' / 'META'
SHIFAMIND2_BASE = BASE_PATH / '10_ShifaMind'

# Find most recent run folder
run_folders = sorted([d for d in SHIFAMIND2_BASE.glob('run_*') if d.is_dir()], reverse=True)
if not run_folders:
    print("❌ No existing run found! Please run shifamind301.py first.")
    sys.exit(1)

OUTPUT_BASE = run_folders[0]
SHARED_DATA_PATH = OUTPUT_BASE / 'shared_data'
PILOT_PATH = OUTPUT_BASE / 'phase_a_pilot'
PILOT_PATH.mkdir(exist_ok=True)

print(f"📁 Using run folder: {OUTPUT_BASE.name}")
print(f"📁 Pilot results: {PILOT_PATH}")
print(f"📁 UMLS path: {UMLS_PATH}")

# Pilot size
PILOT_SIZE = 5000
SEED = 42
np.random.seed(SEED)

# ============================================================================
# STEP 1: DIAGNOSIS-DRIVEN CONCEPT VOCABULARY
# ============================================================================

print("\n" + "="*80)
print("📋 STEP 1: DIAGNOSIS-DRIVEN CONCEPT SELECTION")
print("="*80)

# Load Top-50 codes
with open(SHARED_DATA_PATH / 'top50_icd10_info.json', 'r') as f:
    top50_info = json.load(f)
    TOP_50_CODES = top50_info['top_50_codes']

print(f"✅ Loaded {len(TOP_50_CODES)} ICD-10 codes")

# REFINED ICD-10 clinical knowledge base (v3: aggressive filtering)
# v3 changes: Removed ALL "history" patterns, high-frequency concepts (>50%),
# generic symptoms (edema, fever, nausea, vomiting, chest pain),
# generic procedures (blood transfusion, chest x-ray), location terms
ICD10_CLINICAL_KNOWLEDGE = {
    # Cardiovascular (I codes)
    'I10': ['hypertension', 'hypertensive'],
    'I110': ['hypertensive heart disease', 'left ventricular hypertrophy'],
    'I130': ['hypertensive heart kidney disease', 'hypertensive renal disease'],
    'I2510': ['atherosclerotic heart disease', 'coronary artery disease', 'atherosclerosis', 'angina'],
    'I252': ['old myocardial infarction', 'prior myocardial infarction'],
    'I129': ['hypertensive chronic kidney disease', 'hypertensive renal disease'],
    'I480': ['atrial fibrillation', 'atrial flutter', 'irregular heart rhythm'],
    'I4891': ['heart failure', 'cardiac failure', 'congestive heart failure'],
    'I5032': ['chronic heart failure', 'chronic systolic heart failure', 'reduced ejection fraction'],

    # Metabolic (E codes)
    'E785': ['hyperlipidemia', 'high cholesterol', 'elevated cholesterol', 'dyslipidemia'],
    'E78': ['hyperlipidemia', 'dyslipidemia'],
    'E039': ['hypothyroidism', 'thyroid disorder', 'low thyroid'],
    'E119': ['type 2 diabetes', 'diabetes mellitus', 'diabetes type 2', 'diabetic'],
    'E1122': ['diabetic chronic kidney disease', 'diabetic nephropathy'],
    'E669': ['obesity', 'morbid obesity', 'overweight'],
    'E871': ['hyponatremia', 'low sodium'],
    'E872': ['metabolic acidosis', 'acidosis'],

    # History/Status codes (Z codes) - removed all "history" patterns
    'Z87891': ['nicotine dependence', 'tobacco use'],
    'Z7901': ['long term aspirin use', 'aspirin therapy'],
    'Z794': ['long term medication use'],
    'Z7902': ['long term anticoagulant', 'warfarin therapy', 'anticoagulation therapy'],
    'Z955': ['coronary angioplasty status', 'coronary stent'],
    'Z951': ['cardiac pacemaker', 'permanent pacemaker'],
    'Z8673': ['cerebrovascular accident'],
    'Z86718': ['pulmonary embolism'],
    'Z66': ['body mass index', 'elevated body mass index'],
    'Z23': ['vaccination', 'immunization'],

    # GI (K codes)
    'K219': ['gastroesophageal reflux', 'reflux disease', 'heartburn'],
    'K5900': ['constipation', 'chronic constipation'],

    # Mental Health (F codes)
    'F329': ['major depressive disorder', 'major depression', 'clinical depression'],
    'F419': ['anxiety disorder', 'generalized anxiety'],
    'F17210': ['nicotine dependence', 'tobacco dependence'],

    # Renal (N codes)
    'N179': ['acute kidney injury', 'acute renal failure'],
    'N183': ['chronic kidney disease stage 3', 'chronic renal insufficiency'],
    'N189': ['chronic kidney disease', 'chronic renal disease'],
    'N390': ['urinary tract infection', 'bladder infection'],
    'N400': ['benign prostatic hyperplasia', 'prostate enlargement'],

    # Respiratory (J codes)
    'J45909': ['asthma', 'bronchial asthma', 'reactive airway disease'],
    'J449': ['chronic obstructive pulmonary disease', 'emphysema', 'chronic bronchitis'],
    'J9601': ['acute respiratory failure', 'respiratory failure'],
    'J189': ['pneumonia', 'bacterial pneumonia', 'pulmonary infiltrate'],

    # Neurologic (G codes)
    'G4733': ['obstructive sleep apnea', 'sleep apnea syndrome'],
    'G4700': ['insomnia', 'sleep disorder'],
    'G8929': ['hemiplegia', 'paralysis'],

    # Hematologic (D codes)
    'D649': ['anemia unspecified', 'low hemoglobin'],
    'D62': ['hemorrhagic anemia'],
    'D696': ['thrombocytopenia', 'low platelets'],

    # External causes (Y codes)
    'Y929': ['adverse drug effect', 'medication side effect'],
}

# Build comprehensive concept vocabulary
print("\n🧠 Building concept vocabulary from ICD-10 descriptions...")

candidate_concepts = set()
diagnosis_to_concepts = defaultdict(list)

# Extract from ICD-10 knowledge base
for code, concepts in ICD10_CLINICAL_KNOWLEDGE.items():
    for concept in concepts:
        concept_clean = concept.lower().strip()
        candidate_concepts.add(concept_clean)
        diagnosis_to_concepts[code].append(concept_clean)

# REFINED common clinical concepts (v3: aggressive filtering)
# Removed: history patterns, high-frequency concepts (>50%), generic symptoms/procedures
COMMON_CLINICAL_CONCEPTS = [
    # Specific vital signs/symptoms (multi-word preferred)
    'hypotension', 'tachycardia', 'bradycardia', 'tachypnea',
    'oxygen saturation', 'respiratory rate',
    'shortness of breath', 'altered mental status', 'weight loss',

    # Specific lab findings (with context)
    'elevated creatinine', 'elevated glucose', 'low hemoglobin',
    'elevated troponin', 'elevated bilirubin',
    'low sodium', 'low potassium', 'elevated lactate',
    'platelet count',

    # Specific imaging findings
    'pulmonary infiltrate', 'pulmonary consolidation', 'pleural effusion',
    'cardiomegaly', 'pulmonary edema', 'pulmonary opacity',

    # Specific diagnostic tests (full names)
    'computed tomography', 'echocardiogram',
    'electrocardiogram', 'stress test', 'cardiac catheterization',

    # Specific treatments (multi-word)
    'antibiotic therapy', 'diuretic therapy', 'beta blocker',
    'ace inhibitor', 'insulin therapy', 'anticoagulation therapy',
    'antiplatelet therapy', 'supplemental oxygen', 'mechanical ventilation',
    'hemodialysis', 'intravenous fluids',

    # Specific organ-related terms
    'cardiac disease', 'pulmonary disease', 'renal disease',
    'liver disease', 'cardiovascular disease', 'respiratory disease',
    'kidney disease', 'heart disease',

    # Specific conditions (full names)
    'sepsis', 'septic shock', 'bacterial infection',
    'respiratory failure', 'renal failure', 'liver failure',
    'chronic disease', 'acute disease',

    # Clinical terms (specific, non-generic)
    'dyspnea', 'wheezing', 'cough', 'diarrhea',
    'headache', 'dizziness', 'syncope', 'palpitations',
    'confusion', 'weakness', 'fatigue',
    'hemorrhage', 'bleeding', 'thrombosis',
    'hypertension', 'hypotension',
]

for concept in COMMON_CLINICAL_CONCEPTS:
    candidate_concepts.add(concept.lower().strip())

candidate_concepts = sorted(list(candidate_concepts))

print(f"✅ Generated {len(candidate_concepts)} candidate concepts")
print(f"\n📊 Concept distribution:")
print(f"   Diagnoses with mapped concepts: {len(diagnosis_to_concepts)}/{len(TOP_50_CODES)}")
print(f"   Avg concepts per diagnosis: {np.mean([len(v) for v in diagnosis_to_concepts.values()]):.1f}")
print(f"\n🔍 Sample concepts: {candidate_concepts[:20]}")

# ============================================================================
# STEP 2: FAST UMLS LOOKUP (OPTIONAL)
# ============================================================================

print("\n" + "="*80)
print("📚 STEP 2: BUILDING FAST UMLS CONCEPT LOOKUP")
print("="*80)

print("⚠️  Building lightweight UMLS dictionary...")

umls_concept_to_cui = {}
concept_to_cui = {}

mrconso_path = UMLS_PATH / 'MRCONSO.RRF'
if mrconso_path.exists():
    print(f"📖 Loading UMLS concepts from MRCONSO.RRF...")
    print("   (Processing 3M concepts, ~20-30 seconds)")

    start_time = time.time()
    concept_count = 0

    with open(mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if concept_count % 500000 == 0 and concept_count > 0:
                print(f"   Processed {concept_count:,} concepts...")

            parts = line.strip().split('|')
            if len(parts) < 15:
                continue

            cui = parts[0]
            language = parts[1]
            concept_str = parts[14].lower().strip()

            if language != 'ENG':
                continue

            if concept_str not in umls_concept_to_cui:
                umls_concept_to_cui[concept_str] = cui

            concept_count += 1

            if concept_count >= 3000000:
                break

    elapsed = time.time() - start_time
    print(f"✅ Built UMLS dictionary: {len(umls_concept_to_cui):,} concepts in {elapsed:.1f}s")

    # Map our concepts to CUIs
    for concept in candidate_concepts:
        if concept in umls_concept_to_cui:
            concept_to_cui[concept] = umls_concept_to_cui[concept]

    print(f"✅ Mapped {len(concept_to_cui)}/{len(candidate_concepts)} concepts to UMLS CUIs")
else:
    print(f"⚠️  UMLS MRCONSO.RRF not found at {mrconso_path}")
    print("   Proceeding without CUI mapping")

# Save concept vocabulary
with open(PILOT_PATH / 'candidate_concepts.json', 'w') as f:
    json.dump({
        'concepts': candidate_concepts,
        'concept_to_cui': concept_to_cui,
        'diagnosis_to_concepts': {k: v for k, v in diagnosis_to_concepts.items()}
    }, f, indent=2)

print(f"💾 Saved to {PILOT_PATH / 'candidate_concepts.json'}")

# ============================================================================
# STEP 3: LOAD SCISPACY (LARGE MODEL)
# ============================================================================

print("\n" + "="*80)
print("🔧 STEP 3: LOADING SCISPACY (LARGE MODEL)")
print("="*80)

print("Loading en_core_sci_lg...")
try:
    nlp = spacy.load("en_core_sci_lg")
    print("✅ ScispaCy loaded successfully")
except Exception as e:
    print(f"❌ Failed to load: {e}")
    sys.exit(1)

# Add negation detection
print("Adding negation detection...")
try:
    nlp.add_pipe("negex")
    print("✅ Negation detection added")
except:
    print("⚠️  Could not add negex (may already exist)")

print(f"\n📊 Pipeline: {nlp.pipe_names}")

# ============================================================================
# STEP 4: EXTRACT CONCEPTS FROM PILOT
# ============================================================================

print("\n" + "="*80)
print("🔬 STEP 4: CONCEPT EXTRACTION ON 5K PILOT")
print("="*80)

# Load training data
with open(SHARED_DATA_PATH / 'train_split.pkl', 'rb') as f:
    df_train = pickle.load(f)

print(f"✅ Loaded {len(df_train):,} training samples")

# Sample pilot
pilot_indices = np.random.choice(len(df_train), size=min(PILOT_SIZE, len(df_train)), replace=False)
df_pilot = df_train.iloc[pilot_indices].reset_index(drop=True)

print(f"📊 Pilot dataset: {len(df_pilot)} samples")

def extract_concepts_from_text(text, candidate_concepts, nlp_model):
    """Extract concepts using ScispaCy NER + keyword matching"""
    text = str(text)[:5000]  # Truncate for speed

    doc = nlp_model(text.lower())

    concept_labels = {concept: {'present': 0, 'negated': 0} for concept in candidate_concepts}

    # Method 1: NER with negation
    for ent in doc.ents:
        ent_text = ent.text.lower().strip()
        is_negated = ent._.negex if hasattr(ent._, 'negex') else False

        if ent_text in candidate_concepts:
            concept_labels[ent_text]['present'] = 1
            if is_negated:
                concept_labels[ent_text]['negated'] = 1

        for concept in candidate_concepts:
            if concept in ent_text or ent_text in concept:
                concept_labels[concept]['present'] = 1
                if is_negated:
                    concept_labels[concept]['negated'] = 1

    # Method 2: Keyword matching
    text_lower = text.lower()
    for concept in candidate_concepts:
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

print("\n🔄 Extracting concepts...")
print("   (Using ScispaCy NER + keyword matching with negation)")

pilot_concept_labels = []
extraction_times = []

for idx, row in tqdm(df_pilot.iterrows(), total=len(df_pilot), desc="Processing"):
    start_time = time.time()
    text = row['text']
    concept_dict = extract_concepts_from_text(text, candidate_concepts, nlp)
    pilot_concept_labels.append(concept_dict)
    extraction_times.append(time.time() - start_time)

avg_time = np.mean(extraction_times)
total_time = np.sum(extraction_times)

print(f"\n⏱️  Extraction Performance:")
print(f"   Total time: {total_time/60:.1f} minutes")
print(f"   Avg time per sample: {avg_time:.2f}s")
print(f"   Throughput: {1/avg_time:.1f} samples/sec")

# Estimate full dataset time
full_dataset_size = len(df_train)
estimated_full_time = (avg_time * full_dataset_size) / 3600
print(f"\n📊 Estimated time for {full_dataset_size:,} samples: {estimated_full_time:.1f} hours")

if estimated_full_time > 3:
    print("   ⚠️  >3 hours - Consider parallelization")
else:
    print("   ✅ <3 hours - Feasible")

# ============================================================================
# STEP 5: ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("📊 STEP 5: PILOT ANALYSIS")
print("="*80)

# Convert to matrices
concept_matrix = np.zeros((len(df_pilot), len(candidate_concepts)))
negation_matrix = np.zeros((len(df_pilot), len(candidate_concepts)))

for i, concept_dict in enumerate(pilot_concept_labels):
    for j, concept in enumerate(candidate_concepts):
        concept_matrix[i, j] = concept_dict[concept]['present']
        negation_matrix[i, j] = concept_dict[concept]['negated']

# Statistics
concept_counts = concept_matrix.sum(axis=0)
concept_freq = concept_counts / len(df_pilot)
concepts_per_sample = concept_matrix.sum(axis=1)

avg_concepts = concepts_per_sample.mean()
median_concepts = np.median(concepts_per_sample)
sparsity = avg_concepts / len(candidate_concepts)

print(f"\n🔍 Sparsity Analysis:")
print(f"   Total concepts: {len(candidate_concepts)}")
print(f"   Avg concepts per sample: {avg_concepts:.1f}")
print(f"   Median: {median_concepts:.0f}")
print(f"   Density: {sparsity*100:.1f}%")

if sparsity < 0.03:
    print("   ⚠️  Very sparse (<3%)")
elif sparsity < 0.08:
    print("   ✅ Good (3-8%)")
else:
    print("   ⚠️  High density (>8%)")

# Coverage
samples_with_concepts = (concepts_per_sample > 0).sum()
coverage = samples_with_concepts / len(df_pilot)

print(f"\n📈 Coverage:")
print(f"   Samples with ≥1 concept: {samples_with_concepts}/{len(df_pilot)} ({coverage*100:.1f}%)")

if coverage >= 0.85:
    print("   ✅ Excellent (>85%)")
elif coverage >= 0.7:
    print("   ⚠️  Moderate (70-85%)")
else:
    print("   ❌ Poor (<70%)")

# Frequency distribution
rare_concepts = (concept_freq < 0.005).sum()
common_concepts = (concept_freq > 0.5).sum()
useful_concepts = len(candidate_concepts) - rare_concepts - common_concepts

print(f"\n📊 Frequency Distribution:")
print(f"   Rare (<0.5%): {rare_concepts}")
print(f"   Useful (0.5-50%): {useful_concepts}")
print(f"   Common (>50%): {common_concepts}")

# Top concepts
top_20_idx = np.argsort(concept_counts)[-20:][::-1]
print(f"\n🔝 Top-20 Concepts:")
for idx in top_20_idx:
    print(f"   {candidate_concepts[idx]:35s}: {int(concept_counts[idx]):4d} ({concept_freq[idx]*100:5.1f}%)")

# Negation
negation_rate = negation_matrix.sum() / (concept_matrix.sum() + 1e-10)
print(f"\n🚫 Negation:")
print(f"   Total concepts: {int(concept_matrix.sum())}")
print(f"   Negated: {int(negation_matrix.sum())}")
print(f"   Rate: {negation_rate*100:.1f}%")

# Diagnosis alignment
print(f"\n🎯 Diagnosis-Concept Alignment:")
diagnosis_concept_coverage = {}

for idx, row in df_pilot.iterrows():
    labels = row['labels']
    concepts = pilot_concept_labels[idx]

    for dx_idx, label in enumerate(labels):
        if label == 1:
            dx_code = TOP_50_CODES[dx_idx]
            if dx_code not in diagnosis_concept_coverage:
                diagnosis_concept_coverage[dx_code] = []

            expected_concepts = diagnosis_to_concepts.get(dx_code, [])
            if expected_concepts:
                found = sum(1 for c in expected_concepts if concepts[c]['present'] == 1)
                diagnosis_concept_coverage[dx_code].append(found / len(expected_concepts))

print(f"   Diagnoses in pilot: {len(diagnosis_concept_coverage)}")
for dx_code in sorted(diagnosis_concept_coverage.keys())[:10]:
    avg_cov = np.mean(diagnosis_concept_coverage[dx_code])
    n = len(diagnosis_concept_coverage[dx_code])
    print(f"   {dx_code}: {avg_cov*100:5.1f}% coverage ({n} samples)")

# ============================================================================
# STEP 6: RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("💡 STEP 6: RECOMMENDATIONS")
print("="*80)

recommendations = []

if sparsity < 0.03:
    recommendations.append("❌ REDUCE concepts (too sparse)")
    recommended_concepts = 150
elif sparsity > 0.1:
    recommendations.append("⚠️  Can expand to 300+")
    recommended_concepts = 300
else:
    recommendations.append("✅ Concept count is good")
    recommended_concepts = len(candidate_concepts)

if coverage < 0.7:
    recommendations.append("❌ IMPROVE extraction (low coverage)")
elif coverage < 0.85:
    recommendations.append("⚠️  Add more common concepts")
else:
    recommendations.append("✅ Coverage is excellent")

if estimated_full_time > 5:
    recommendations.append("❌ PARALLELIZE (>5 hours)")
elif estimated_full_time > 2:
    recommendations.append("⚠️  Consider parallelization")
else:
    recommendations.append("✅ Extraction speed acceptable")

if rare_concepts > useful_concepts:
    recommendations.append(f"🔧 FILTER {rare_concepts} rare concepts")
    recommended_concepts = useful_concepts + common_concepts

print("\n".join(recommendations))

print(f"\n📋 Recommendation: {recommended_concepts} concepts")

# Filter
if rare_concepts > 0:
    filtered_concepts = [candidate_concepts[i] for i in range(len(candidate_concepts))
                        if concept_freq[i] >= 0.005]
    print(f"   Filtered: {len(candidate_concepts)} → {len(filtered_concepts)}")
else:
    filtered_concepts = candidate_concepts

# Save results
with open(PILOT_PATH / 'filtered_concepts.json', 'w') as f:
    json.dump({
        'concepts': filtered_concepts,
        'count': len(filtered_concepts),
        'pilot_stats': {
            'avg_concepts_per_sample': float(avg_concepts),
            'sparsity': float(sparsity),
            'coverage': float(coverage),
            'negation_rate': float(negation_rate)
        }
    }, f, indent=2)

with open(PILOT_PATH / 'pilot_concept_labels.pkl', 'wb') as f:
    pickle.dump({
        'concept_labels': pilot_concept_labels,
        'concepts': candidate_concepts,
        'pilot_indices': pilot_indices
    }, f)

print(f"\n💾 Saved to {PILOT_PATH}/")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ WEEK 1 PILOT COMPLETE!")
print("="*80)

print(f"\n📊 Summary:")
print(f"   ✓ {len(df_pilot)} samples processed")
print(f"   ✓ {len(filtered_concepts)} concepts after filtering")
print(f"   ✓ {avg_concepts:.1f} avg concepts/sample")
print(f"   ✓ {sparsity*100:.1f}% density")
print(f"   ✓ {coverage*100:.1f}% coverage")
print(f"   ✓ {total_time/60:.1f} min total ({avg_time:.2f}s/sample)")
print(f"   ✓ {estimated_full_time:.1f} hours estimated for full dataset")

print(f"\n🚀 Next Steps:")
if all(['✅' in r for r in recommendations]):
    print("   ✅ ALL CHECKS PASSED!")
    print("   Ready for Week 2 full implementation")
else:
    print("   ⚠️  Review recommendations above")
    print("   Adjust and re-run if needed")

print("\nAlhamdulillah! 🤲")
