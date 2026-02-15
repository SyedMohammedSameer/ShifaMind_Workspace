# 🔄 How to Recover Original Data Splits

If the verification script shows that your current pickles don't match the checkpoint, here are ways to recover the originals:

---

## Method 1: Google Drive Version History ⭐ EASIEST

Google Drive keeps version history for files!

### Steps:
1. Go to Google Drive: `https://drive.google.com`
2. Navigate to: `MyDrive/ShifaMind/10_ShifaMind/run_20260102_203225/shared_data/`
3. **Right-click** on `val_split.pkl` → **Manage versions**
4. Look for versions from **January 2, 2026** (when the checkpoint was created)
5. Download the old version
6. Repeat for `test_split.pkl` and `train_split.pkl`
7. Replace current files with old versions

---

## Method 2: Check Other Run Folders

The verification script will list all run folders. If an older run has the right data:

```bash
# In the verification output, look for runs with:
# - Val size: 17,265 samples
# - Test size: 17,266 samples
# - Timestamp close to checkpoint (January 2, 2026)

# If found, copy from that run:
cp /content/drive/MyDrive/ShifaMind/10_ShifaMind/run_XXXXXX/shared_data/*.pkl \
   /content/drive/MyDrive/ShifaMind/10_ShifaMind/run_20260102_203225/shared_data/
```

---

## Method 3: Git LFS (if pickles were tracked)

If pickles were committed to git with LFS:

```bash
# Check git history
git log --all --full-history -- "*split.pkl"

# If found, checkout old version:
git checkout <commit-hash> -- path/to/val_split.pkl
```

---

## Method 4: Recreate from Source Data (HARDER)

If you still have the original MIMIC-IV data and preprocessing is deterministic:

The notebook shows splits were created with:
- **Seed: 42**
- **Train/Val/Test split: 60/20/20**
- Using `sklearn.model_selection.train_test_split`

If the source CSV (`mimic_dx_data_top50.csv`) is unchanged, running Phase 1 preprocessing with the same seed will recreate identical splits.

Check if this file exists and hasn't changed:
```bash
ls -lh /content/drive/MyDrive/ShifaMind/10_ShifaMind/run_20260102_203225/mimic_dx_data_top50.csv
```

---

## What to Check First:

1. ✅ **Run the verification script** (`verify_data_splits.py`)
2. ✅ **Check Google Drive version history** (Method 1)
3. ✅ **Look at other run folders** (Method 2)
4. ❌ Only if all else fails → **Retrain Phase 1** (Option 2)

---

## Expected File Hashes (if you need to verify):

If you find candidate pickle files, you can verify them:

```python
import pickle
import hashlib

# Load and check basic properties
with open('val_split.pkl', 'rb') as f:
    df = pickle.load(f)

print(f"Samples: {len(df)}")  # Should be 17,265
print(f"Columns: {list(df.columns)}")

# Check label distribution
import numpy as np
avg_labels = np.mean([sum(row) for row in df['labels'].tolist()])
print(f"Avg labels/sample: {avg_labels:.2f}")  # Should be ~5
```

**Expected values from original notebook:**
- Val samples: **17,265**
- Test samples: **17,266**
- Avg labels/sample: **~5** (Top-K = 5)

---

## Need Help?

If none of these methods work, we'll proceed with **Option 2: Clean Phase 1 Retrain**.

The retrain will:
- ✅ Use same seed (42)
- ✅ Create deterministic splits
- ✅ Save everything properly
- ✅ Match or exceed original 0.4360 performance
