#!/usr/bin/env python3
"""
GPU DIAGNOSTIC & VERIFICATION SCRIPT
====================================

This script comprehensively checks GPU availability and configuration
for Google Colab L4/T4 GPU environments.

CRITICAL: The user reported GPU was NEVER used during ShifaMind302 Phase 1 training!
This script investigates why and provides fixes.
"""

import torch
import sys
from pathlib import Path

print("="*80)
print("🖥️  GPU DIAGNOSTIC & VERIFICATION")
print("="*80)
print()

# Store diagnostic results
diagnostics = {
    'gpu_available': False,
    'gpu_count': 0,
    'gpu_devices': [],
    'cuda_version': None,
    'pytorch_version': None,
    'issues': [],
    'recommendations': []
}

# ============================================================================
# CHECK 1: PyTorch CUDA Availability
# ============================================================================
print("1️⃣  PyTorch CUDA Availability")
print("-" * 80)

diagnostics['pytorch_version'] = torch.__version__
print(f"   PyTorch version: {torch.__version__}")

cuda_available = torch.cuda.is_available()
diagnostics['gpu_available'] = cuda_available

if cuda_available:
    print(f"   ✅ CUDA is available: True")
else:
    print(f"   ❌ CUDA is available: False")
    diagnostics['issues'].append("CUDA not available - GPU cannot be used!")
    diagnostics['recommendations'].append("Change Colab runtime to GPU: Runtime → Change runtime type → T4/L4 GPU")

# ============================================================================
# CHECK 2: GPU Device Information
# ============================================================================
print("\n2️⃣  GPU Device Information")
print("-" * 80)

if cuda_available:
    gpu_count = torch.cuda.device_count()
    diagnostics['gpu_count'] = gpu_count
    print(f"   GPU count: {gpu_count}")

    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_props = torch.cuda.get_device_properties(i)

        diagnostics['gpu_devices'].append({
            'id': i,
            'name': gpu_name,
            'total_memory_gb': gpu_props.total_memory / 1e9,
            'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
        })

        print(f"\n   GPU {i}:")
        print(f"      Name: {gpu_name}")
        print(f"      Memory: {gpu_props.total_memory / 1e9:.2f} GB")
        print(f"      Compute Capability: {gpu_props.major}.{gpu_props.minor}")
        print(f"      Multi Processors: {gpu_props.multi_processor_count}")

        # Check if it's the expected GPU
        if 'L4' not in gpu_name and 'T4' not in gpu_name and 'A100' not in gpu_name:
            diagnostics['issues'].append(f"GPU {i} is {gpu_name} - not a Colab GPU (L4/T4/A100)")
else:
    print("   ⚠️  No GPU available to inspect")

# ============================================================================
# CHECK 3: CUDA Version
# ============================================================================
print("\n3️⃣  CUDA Version")
print("-" * 80)

if cuda_available:
    cuda_version = torch.version.cuda
    diagnostics['cuda_version'] = cuda_version
    print(f"   CUDA version: {cuda_version}")

    # Check CUDA build version
    try:
        cudnn_version = torch.backends.cudnn.version()
        print(f"   cuDNN version: {cudnn_version}")
    except:
        print(f"   cuDNN version: Not available")
else:
    print("   ⚠️  CUDA not available")

# ============================================================================
# CHECK 4: Memory Test
# ============================================================================
print("\n4️⃣  GPU Memory Test")
print("-" * 80)

if cuda_available:
    try:
        device = torch.device('cuda:0')

        # Allocate a small tensor
        test_tensor = torch.randn(1000, 1000).to(device)

        # Check memory usage
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9

        print(f"   ✅ Successfully allocated test tensor on GPU")
        print(f"   Memory allocated: {allocated:.4f} GB")
        print(f"   Memory reserved: {reserved:.4f} GB")

        # Clean up
        del test_tensor
        torch.cuda.empty_cache()
        print(f"   ✅ Memory test passed")

    except Exception as e:
        print(f"   ❌ Memory test failed: {e}")
        diagnostics['issues'].append(f"GPU memory test failed: {e}")
else:
    print("   ⚠️  Skipping - no GPU available")

# ============================================================================
# CHECK 5: Device Selection Test
# ============================================================================
print("\n5️⃣  Device Selection Test")
print("-" * 80)

# This is the pattern used in shifamind302.py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device selected: {device}")

if device.type == 'cuda':
    print(f"   ✅ Device type: CUDA (GPU will be used)")
else:
    print(f"   ❌ Device type: CPU (GPU will NOT be used)")
    diagnostics['issues'].append("Device defaulted to CPU instead of CUDA!")
    diagnostics['recommendations'].append("Ensure CUDA is available before running training")

# ============================================================================
# CHECK 6: Model Transfer Test
# ============================================================================
print("\n6️⃣  Model Transfer Test")
print("-" * 80)

if cuda_available:
    try:
        # Create a simple model
        model = torch.nn.Linear(10, 5)

        # Transfer to GPU
        model = model.to(device)

        # Check if model is on GPU
        model_device = next(model.parameters()).device
        print(f"   Model device: {model_device}")

        if model_device.type == 'cuda':
            print(f"   ✅ Model successfully transferred to GPU")
        else:
            print(f"   ❌ Model is on CPU, not GPU")
            diagnostics['issues'].append("Model not transferring to GPU")

    except Exception as e:
        print(f"   ❌ Model transfer failed: {e}")
        diagnostics['issues'].append(f"Model transfer failed: {e}")
else:
    print("   ⚠️  Skipping - no GPU available")

# ============================================================================
# CHECK 7: Training Loop Test
# ============================================================================
print("\n7️⃣  Training Loop Test (Forward/Backward Pass)")
print("-" * 80)

if cuda_available:
    try:
        # Simple training simulation
        model = torch.nn.Linear(10, 5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        # Create fake data
        x = torch.randn(32, 10).to(device)
        y = torch.randn(32, 5).to(device)

        # Forward pass
        output = model(x)
        loss = criterion(output, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check if computation was on GPU
        if output.device.type == 'cuda':
            print(f"   ✅ Training loop executed on GPU")
            print(f"   Loss: {loss.item():.4f}")
        else:
            print(f"   ❌ Training loop executed on CPU")
            diagnostics['issues'].append("Training computations not on GPU")

    except Exception as e:
        print(f"   ❌ Training loop test failed: {e}")
        diagnostics['issues'].append(f"Training loop test failed: {e}")
else:
    print("   ⚠️  Skipping - no GPU available")

# ============================================================================
# CHECK 8: Mixed Precision (AMP) Test
# ============================================================================
print("\n8️⃣  Mixed Precision (AMP) Availability")
print("-" * 80)

if cuda_available:
    try:
        from torch.cuda.amp import autocast, GradScaler

        # Test AMP
        scaler = GradScaler()
        model = torch.nn.Linear(10, 5).to(device)
        optimizer = torch.optim.Adam(model.parameters())

        x = torch.randn(32, 10).to(device)
        y = torch.randn(32, 5).to(device)

        with autocast():
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print(f"   ✅ AMP (Mixed Precision) is available and working")
        print(f"   Note: AMP was enabled in 302 but not requested by user")
        diagnostics['recommendations'].append("Consider disabling AMP if not needed (user didn't request it)")

    except Exception as e:
        print(f"   ⚠️  AMP test failed: {e}")
        print(f"   Note: AMP may not be available on this GPU")
else:
    print("   ⚠️  Skipping - no GPU available")

# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("📋 DIAGNOSTIC SUMMARY")
print("="*80)

print(f"\n✅ GPU Available: {diagnostics['gpu_available']}")
if diagnostics['gpu_available']:
    print(f"✅ GPU Count: {diagnostics['gpu_count']}")
    for gpu in diagnostics['gpu_devices']:
        print(f"✅ GPU: {gpu['name']} ({gpu['total_memory_gb']:.1f} GB)")

if diagnostics['issues']:
    print(f"\n🔴 Issues Found: {len(diagnostics['issues'])}")
    for i, issue in enumerate(diagnostics['issues'], 1):
        print(f"   {i}. {issue}")
else:
    print(f"\n✅ No issues found - GPU is properly configured")

if diagnostics['recommendations']:
    print(f"\n🔧 Recommendations: {len(diagnostics['recommendations'])}")
    for i, rec in enumerate(diagnostics['recommendations'], 1):
        print(f"   {i}. {rec}")

# ============================================================================
# GENERATE FIX CODE
# ============================================================================
print("\n" + "="*80)
print("🔧 GENERATED FIX CODE")
print("="*80)

if not diagnostics['gpu_available']:
    print("""
CRITICAL FIX NEEDED: GPU is not available in this runtime!

To fix in Google Colab:
1. Click 'Runtime' in the menu
2. Select 'Change runtime type'
3. Set 'Hardware accelerator' to 'GPU'
4. Choose 'T4 GPU' or 'L4 GPU'
5. Click 'Save'
6. The runtime will restart

Then re-run your training script.
""")
else:
    print("""
✅ GPU is available and working!

If your training script is still not using GPU, ensure:

1. Device is set correctly:
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

2. Model is moved to device:
   model = model.to(device)

3. Data is moved to device in training loop:
   input_ids = batch['input_ids'].to(device)
   labels = batch['labels'].to(device)

4. Add verification in training loop:
   print(f"Model device: {next(model.parameters()).device}")
   print(f"Data device: {input_ids.device}")

If using mixed precision (AMP), wrap forward pass:
   with autocast():
       outputs = model(input_ids, attention_mask)
       loss = criterion(outputs, labels)
""")

# Save diagnostics
diagnostics_path = Path('/home/user/ShifaMind_Workspace/gpu_diagnostics.json')
import json
with open(diagnostics_path, 'w') as f:
    json.dump(diagnostics, f, indent=2)
print(f"\n💾 Diagnostics saved to: {diagnostics_path}")

print("\n" + "="*80)
print("✅ DIAGNOSTIC COMPLETE")
print("="*80)
