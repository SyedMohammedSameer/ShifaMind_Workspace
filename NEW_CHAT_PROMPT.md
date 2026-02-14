# ShifaMind v302 Phase 5 GPU-Dependent Performance Issue

## CRITICAL PROBLEM

The **same trained checkpoint** produces **different evaluation results on different GPUs**:

| GPU | Phase 1 Macro F1 @ 0.5 | Best Threshold | Val Micro-F1 |
|-----|------------------------|----------------|--------------|
| **H100** | **0.2934** ✅ (original) | 0.25 | 0.5363 |
| **A100** | **0.2169** ❌ (26% drop) | 0.22 | 0.4747 |
| **L4** | **0.1912** ❌ (35% drop) | 0.21 | 0.4757 |

**Same checkpoint, same code, same data** - only GPU differs. This is NOT a code bug.

---

## WHAT HAS BEEN VERIFIED (NOT THE ISSUE)

✅ **ICD code ordering**: Checkpoint codes perfectly match DataFrame column order
✅ **Checkpoint loading**: All keys load successfully with `strict=True`
✅ **Concept embeddings**: Loading correctly (verified mean values match)
✅ **BERT weights**: Loading correctly (verified values change after load)
✅ **Evaluation code**: Using v301's exact evaluation code still gives wrong results
✅ **Data consistency**: test_split.pkl matches checkpoint's TOP_50_CODES exactly
✅ **Checkpoint selection**: Tried multiple checkpoints (F1=0.2797, 0.2906) - all show GPU dependency

**Conclusion**: This is NOT a code/data mismatch issue. It's GPU-architecture specific.

---

## PROJECT STRUCTURE

```
/content/drive/MyDrive/ShifaMind/10_ShifaMind/
├── shifamind301.py              # Phase 1 training code (original)
├── shifamindextra.py            # Phase 5 evaluation (gave 0.2934 on H100)
├── shifamind302_phase5.py       # Phase 5 modular code (gives 0.2169 on A100)
├── check_icd_ordering.py        # Diagnostic script (verified ICD codes OK)
└── run_20260102_203225/         # Best performing run
    ├── checkpoints/
    │   └── phase1/
    │       └── phase1_best.pt   # Training F1: 0.2797
    └── shared_data/
        ├── top50_icd10_info.json
        ├── test_split.pkl       # 17,266 samples
        ├── val_split.pkl
        └── concept_embeddings.npy

Other runs:
- run_20260102_061815/  # Training F1: 0.2906 (better, but still shows GPU issue)
- run_20260102_053548/  # Training F1: 0.1082 (poor)
```

---

## MODEL ARCHITECTURE (ShifaMind2Phase1)

```python
class ShifaMind2Phase1(nn.Module):
    def __init__(self, num_diagnoses=50, num_concepts=113):
        super().__init__()
        self.base_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        hidden_size = self.base_model.config.hidden_size  # 768

        # Concept prediction head
        self.concept_classifier = nn.Linear(hidden_size, num_concepts)

        # Multi-scale fusion
        self.fusion_layer = nn.Linear(hidden_size * 4, hidden_size)

        # Diagnosis prediction with concept fusion
        self.diagnosis_classifier = nn.Linear(
            hidden_size + num_concepts,  # 768 + 113 = 881
            num_diagnoses  # 50
        )

    def forward(self, input_ids, attention_mask,
                concept_embeddings_external=None, input_texts=None):
        # Get BERT outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Multi-scale fusion (last 4 layers)
        hidden_states = outputs.hidden_states
        last_four = torch.cat([
            hidden_states[-1][:, 0, :],  # [CLS] from last layer
            hidden_states[-2][:, 0, :],
            hidden_states[-3][:, 0, :],
            hidden_states[-4][:, 0, :]
        ], dim=-1)

        fused = self.fusion_layer(last_four)  # [batch, 768]

        # Predict concepts
        concept_scores = torch.sigmoid(self.concept_classifier(fused))

        # Concatenate and predict diagnoses
        combined = torch.cat([fused, concept_scores], dim=-1)
        diagnosis_logits = self.diagnosis_classifier(combined)

        return {
            'logits': diagnosis_logits,
            'concept_scores': concept_scores
        }
```

**Key details**:
- Input: Discharge notes (max 512 tokens)
- BERT: Bio_ClinicalBERT (768-dim)
- Multi-label classification: 50 ICD-10 codes
- Intermediate task: 113 medical concepts
- Loss: BCE for both tasks

---

## CHECKPOINT STRUCTURE

```python
checkpoint = {
    'model_state_dict': {
        'base_model.embeddings.word_embeddings.weight': Tensor[...],
        'base_model.encoder.layer.0.attention.self.query.weight': Tensor[...],
        # ... all BERT weights ...
        'concept_classifier.weight': Tensor[113, 768],
        'concept_classifier.bias': Tensor[113],
        'fusion_layer.weight': Tensor[768, 3072],
        'fusion_layer.bias': Tensor[768],
        'diagnosis_classifier.weight': Tensor[50, 881],
        'diagnosis_classifier.bias': Tensor[50]
    },
    'config': {
        'top_50_codes': ['E785', 'I10', 'Z87891', ...],  # 50 codes
        'timestamp': '20260102_203225',
        'lambda_dx': 1.0,
        'lambda_align': 0.5,
        'lambda_concept': 0.3
    },
    'macro_f1': 0.2797,
    'micro_f1': 0.3957,
    'epoch': 15
}
```

---

## EVALUATION CODE (get_probs_from_model)

```python
def get_probs_from_model(model, loader, has_rag=False, concept_embeddings=None):
    """Get probabilities from model predictions"""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Getting predictions", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()

            if has_rag and concept_embeddings is not None:
                texts = batch['text']
                outputs = model(input_ids, attention_mask,
                              concept_embeddings, input_texts=texts)
                logits = outputs['logits']
            else:
                outputs = model(input_ids, attention_mask)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)

    return np.vstack(all_probs), np.vstack(all_labels)
```

**Threshold tuning**:
```python
def tune_threshold(probs_val, y_val):
    best_threshold = 0.5
    best_f1 = 0
    for threshold in np.arange(0.1, 0.9, 0.01):
        preds = (probs_val > threshold).astype(int)
        f1 = f1_score(y_val, preds, average='micro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold
```

---

## ORIGINAL RESULTS (H100 - CORRECT)

From shifamindextra.py on H100:

```
📊 PHASE 1 RESULTS (No RAG)
────────────────────────────────────────
Fixed@0.5 : Macro-F1 = 0.2934, Micro-F1 = 0.4210

🎯 THRESHOLD TUNING
   Best threshold: 0.25 (val micro-F1: 0.5363)

Tuned@0.25: Macro-F1 = 0.4360, Micro-F1 = 0.5325
```

---

## CURRENT RESULTS (A100/L4 - WRONG)

From shifamind302_phase5.py on A100:

```
📊 PHASE 1 RESULTS (No RAG)
────────────────────────────────────────
Fixed@0.5 : Macro-F1 = 0.2169, Micro-F1 = 0.3717

🎯 THRESHOLD TUNING
   Best threshold: 0.22 (val micro-F1: 0.4747)

Tuned@0.22: Macro-F1 = 0.3838, Micro-F1 = 0.4719
```

**Performance drop**: -26% macro F1, -12% micro F1

---

## KEY OBSERVATIONS

1. **Threshold difference**: H100 finds 0.25 optimal, A100 finds 0.22
2. **Val micro-F1 difference**: H100 gets 0.5363, A100 gets 0.4747 (12% drop)
3. **Test macro-F1 difference**: H100 gets 0.2934, A100 gets 0.2169 (26% drop)
4. **Pattern**: Lower-tier GPUs (L4) perform worse than mid-tier (A100)

---

## POTENTIAL CAUSES TO INVESTIGATE

### 1. **GPU Tensor Core Precision**
- H100: 4th gen Tensor Cores (FP8, TF32, FP16, FP32)
- A100: 3rd gen Tensor Cores (TF32, FP16, FP32)
- L4: 4th gen but optimized for inference (FP8, FP16)

**Hypothesis**: TF32 vs FP32 accumulation in matrix multiplications

### 2. **cuDNN/CUDA Version Differences**
- H100 may use newer cuDNN with different implementations
- LayerNorm, Softmax, GELU might have different numerical paths

### 3. **PyTorch Matmul Precision**
```python
# Default behavior changed in PyTorch 1.12+
torch.backends.cuda.matmul.allow_tf32 = True  # Default on Ampere+
```

### 4. **Non-Deterministic Operations**
- `torch.sigmoid()` on different GPUs
- Attention softmax accumulation order
- Multi-scale fusion concatenation

### 5. **Memory Layout Differences**
- H100 has different memory bandwidth (3TB/s vs 2TB/s)
- This affects how tensors are laid out and accessed

---

## EXPERIMENTS TO RUN

### Experiment 1: Force FP32 Precision
```python
torch.set_float32_matmul_precision('highest')  # Force FP32
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Re-run evaluation
```

### Experiment 2: Deterministic Mode
```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Re-run evaluation
```

### Experiment 3: Explicit FP64 Evaluation
```python
# Load model in float64
model = model.double()
# Convert inputs to float64
input_ids = input_ids.long()  # Keep as long
logits = model(input_ids, attention_mask).float()  # Cast back
```

### Experiment 4: Profile Numerical Differences
```python
# Add hooks to log intermediate activations
def hook_fn(module, input, output):
    print(f"{module.__class__.__name__}: mean={output.mean():.6f}, std={output.std():.6f}")

model.fusion_layer.register_forward_hook(hook_fn)
model.diagnosis_classifier.register_forward_hook(hook_fn)
```

### Experiment 5: Compare Logits Distribution
```python
# On H100 and A100, save raw logits before sigmoid
logits_h100 = []  # Load from H100 run
logits_a100 = []  # Load from A100 run

# Compare distribution
diff = logits_h100 - logits_a100
print(f"Mean diff: {diff.mean()}")
print(f"Max diff: {diff.max()}")
print(f"Distribution: {np.percentile(np.abs(diff), [50, 90, 99])}")
```

---

## FILES TO USE

**Main evaluation script**: `shifamind302_phase5.py`
**Checkpoint**: `/content/drive/MyDrive/ShifaMind/10_ShifaMind/run_20260102_203225/checkpoints/phase1/phase1_best.pt`
**Test data**: `/content/drive/MyDrive/ShifaMind/10_ShifaMind/run_20260102_203225/shared_data/test_split.pkl`

---

## ENVIRONMENT

- **Platform**: Google Colab
- **PyTorch**: 2.x (check exact version)
- **CUDA**: Varies by GPU (check with `torch.version.cuda`)
- **cuDNN**: Varies by GPU (check with `torch.backends.cudnn.version()`)

**Check command**:
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"cuDNN: {torch.backends.cudnn.version()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"TF32 allowed: {torch.backends.cuda.matmul.allow_tf32}")
```

---

## QUESTION FOR YOU

**Can you identify why the same checkpoint produces 26% lower performance on A100/L4 compared to H100?**

Focus on:
1. GPU-specific numerical precision issues
2. PyTorch/CUDA backend differences
3. Tensor Core implementation differences
4. Potential solutions to make results GPU-agnostic

**The goal**: Get 0.2934 macro F1 on A100/L4 (same as H100), or understand why it's impossible and document the GPU dependency.
