# 🚀 Quick Start: Clean Phase 1 Retrain

## What This Does

Complete Phase 1 retraining with:
- ✅ Deterministic splits (seed=42)
- ✅ Same train/val/test sizes as original
- ✅ Proper checkpointing with metrics
- ✅ Should achieve **≥0.4360 Macro F1**

---

## How to Run

### **In Google Colab:**

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Install dependencies (if needed)
!pip install -q transformers torch scikit-learn tqdm

# 3. Run the training script
!python /content/drive/MyDrive/ShifaMind/train_phase1_clean.py
```

That's it! The script handles everything:
- ✅ Loads `mimic_dx_data_top50.csv`
- ✅ Creates deterministic splits
- ✅ Saves splits BEFORE training starts
- ✅ Trains for 5 epochs
- ✅ Saves best checkpoint
- ✅ All in new timestamped folder: `run_YYYYMMDD_HHMMSS_clean`

---

## What to Expect

### **Training Time:**
- ~2-3 hours on Colab GPU (T4)
- ~5-6 hours on Colab CPU (not recommended)

### **Expected Output:**
```
🚀 SHIFAMIND PHASE 1 - CLEAN RETRAIN
================================================================================
🖥️  Device: cuda
🎲 Random Seed: 42 (deterministic mode enabled)

📁 Run folder: run_20260215_123456_clean
...

Epoch 1/5
================================================================================
Training: 100%|██████████| 5036/5036 [25:32<00:00]

📊 Epoch 1 Losses:
   Total:     0.3245
   Diagnosis: 0.2156
   Alignment: 0.0542
   Concept:   0.0547

Validating: 100%|██████████| 540/540 [02:15<00:00]

📈 Validation Metrics (threshold=0.5):
   Diagnosis Macro F1: 0.3421
   Concept Macro F1:   0.0875
   💾 Saved best checkpoint (Macro F1: 0.3421)

...

Epoch 5/5
================================================================================
...
📈 Validation Metrics (threshold=0.5):
   Diagnosis Macro F1: 0.4425  ← Should be ≥0.43
   Concept Macro F1:   0.1134
   💾 Saved best checkpoint (Macro F1: 0.4425)

================================================================================
✅ TRAINING COMPLETE!
================================================================================
🏆 Best Validation Macro F1: 0.4425
📁 Run folder: run_20260215_123456_clean
💾 Checkpoint: .../checkpoints/phase1/phase1_best.pt
```

---

## Verify Results

After training completes, verify the checkpoint:

```python
import torch

# Load checkpoint
checkpoint_path = '/content/drive/MyDrive/ShifaMind/10_ShifaMind/run_20260215_123456_clean/checkpoints/phase1/phase1_best.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(f"Best Macro F1: {checkpoint['macro_f1']:.4f}")  # Should be ≥0.4360
print(f"Concept F1:    {checkpoint['concept_f1']:.4f}")
print(f"Epoch:         {checkpoint['epoch']}")
```

---

## What Gets Saved

```
run_20260215_123456_clean/
├── shared_data/
│   ├── train_split.pkl          ← Deterministic splits
│   ├── val_split.pkl
│   ├── test_split.pkl
│   ├── split_info.json          ← Metadata (seed, sizes, indices)
│   ├── train_concept_labels.npy
│   ├── val_concept_labels.npy
│   ├── test_concept_labels.npy
│   ├── concept_list.json
│   └── top50_icd10_info.json
├── checkpoints/
│   └── phase1/
│       └── phase1_best.pt       ← Best model (Macro F1 ≥0.43)
└── results/
    └── phase1/
        └── training_history.json
```

---

## If Training Fails

**Common issues:**

1. **Out of Memory (OOM)**
   ```python
   # Reduce batch size in the script:
   # Change: BATCH_SIZE = 16
   # To:     BATCH_SIZE = 8
   ```

2. **Colab Disconnects**
   - Use Colab Pro for longer runtimes
   - Or split into manual epochs

3. **Import Errors**
   ```python
   !pip install transformers torch scikit-learn tqdm numpy pandas
   ```

---

## Next Steps After Training

1. ✅ Verify Macro F1 ≥ 0.4360
2. ✅ Update `shifamind302.py` to use new checkpoint
3. ✅ Run Phase 2 evaluation
4. ✅ Run Phase 3 (FAISS)

---

**Questions?** Check the source code - it's heavily commented! 🚀
