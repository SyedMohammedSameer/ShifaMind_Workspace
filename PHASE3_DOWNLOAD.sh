#!/bin/bash
# Download the LATEST Phase 3 script with cache-busting

cd /content/drive/MyDrive/ShifaMind

# Force download with timestamp to bypass cache
TIMESTAMP=$(date +%s)

wget "https://raw.githubusercontent.com/SyedMohammedSameer/ShifaMind_Workspace/claude/review-pipeline-files-i2crP/phase3_training_optimized.py?t=$TIMESTAMP" \
     -O phase3_training_optimized.py \
     --no-cache \
     --no-check-certificate

echo "✅ Downloaded latest phase3_training_optimized.py"
echo "Now run: python /content/drive/MyDrive/ShifaMind/phase3_training_optimized.py"
