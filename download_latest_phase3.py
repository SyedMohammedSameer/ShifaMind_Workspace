"""
Download the LATEST Phase 3 script from GitHub (bypass cache!)
Run this in Colab BEFORE running Phase 3
"""

import requests
import time
from pathlib import Path

# GitHub raw URL with cache-busting timestamp
timestamp = int(time.time())
url = f"https://raw.githubusercontent.com/SyedMohammedSameer/ShifaMind_Workspace/claude/review-pipeline-files-i2crP/phase3_training_optimized.py?t={timestamp}"

# Download location
save_path = Path("/content/drive/MyDrive/ShifaMind/phase3_training_optimized.py")
save_path.parent.mkdir(parents=True, exist_ok=True)

print(f"🔄 Downloading latest Phase 3 script from GitHub...")
print(f"📥 URL: {url}")

# Download with no cache
response = requests.get(url, headers={'Cache-Control': 'no-cache'})
response.raise_for_status()

# Save
save_path.write_text(response.text)

print(f"✅ Downloaded latest script to: {save_path}")
print(f"📊 File size: {len(response.text):,} bytes")

# Verify the fix is in the file
content = response.text
if "'phase3' not in d.name" in content:
    print("✅ VERIFIED: Filter fix is present")
if "PHASE1_RUN = phase1_folders[0]" in content:
    print("✅ VERIFIED: Phase 1 path fix is present")
if "OLD_SHARED = PHASE1_RUN / 'shared_data'" in content:
    print("✅ VERIFIED: OLD_SHARED points to Phase 1")

print("\n🚀 NOW RUN:")
print("!python /content/drive/MyDrive/ShifaMind/phase3_training_optimized.py")
