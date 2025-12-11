# Phase 3 Fix: stringzilla Compilation Error

## Problem

The `inference` package (Roboflow SDK) has a dependency on `stringzilla` which failed to compile in Modal's Debian slim environment with this error:

```
include/stringzilla/memory.h:885:27: error: incompatible types when initializing type '__m512i'
```

**Root Cause**: The `stringzilla` library tries to use AVX-512 intrinsics (`_mm512_loadu_epi8`) that aren't properly supported by the GCC version in Modal's base Debian slim image.

## Solution Applied

### 1. Added Build Dependencies ([train.py](train.py):108-110)

Added essential build tools to compile C extensions:

```python
.apt_install(
    # ... existing dependencies ...

    # Build dependencies for stringzilla (inference dependency)
    "build-essential",
    "gcc",
    "g++",
)
```

### 2. Pre-install stringzilla with Binary Wheel ([train.py](train.py):112-115)

Install `stringzilla` separately BEFORE `inference` to use precompiled binary wheels when available:

```python
.pip_install(
    # Install stringzilla binary wheel first (avoid compilation)
    "stringzilla",
)
```

This approach:
- Attempts to use precompiled wheels from PyPI (no compilation needed)
- Falls back to compilation with proper build tools if wheel unavailable
- Ensures `stringzilla` is available before `inference` tries to install it

### 3. Kept inference Package ([train.py](train.py):136)

Reverted changes to keep using Roboflow's `inference` package:

```python
"inference",  # Roboflow inference SDK for RF-DETR
```

### 4. Reverted RF-DETR Model Wrapper ([models/rfdetr_model.py](models/rfdetr_model.py))

Restored original implementation using Roboflow's `get_model()` API:

```python
from inference import get_model

VARIANT_MAPPING = {
    'nano': 'rf-detr-nano',
    'small': 'rf-detr-small',
    'medium': 'rf-detr-medium',
}

self.model = get_model(model_id)
```

## Files Modified

1. **[train.py](train.py)**
   - Lines 108-110: Added build-essential, gcc, g++
   - Lines 112-115: Pre-install stringzilla
   - Line 136: Kept inference package

2. **[models/rfdetr_model.py](models/rfdetr_model.py)**
   - Lines 13-19: Reverted to use Roboflow inference
   - Lines 31-35: Reverted VARIANT_MAPPING to rf-detr-* models
   - Lines 48-63: Reverted __init__ to use get_model()
   - Lines 138-155: Reverted train() to use Roboflow API

## Testing

To verify the fix works:

```bash
# Test Modal image build (should succeed now)
modal run train.py::prepare_filtered_datasets --detach
```

## Alternative Solutions (If This Fails)

If the binary wheel approach still fails:

### Option A: Use stringzilla 2.x (older version)
```python
.pip_install("stringzilla==2.2.1")  # Before inference
```

### Option B: Set environment variable to disable AVX-512
```python
.env({"STRINGZILLA_DISABLE_AVX512": "1"})
```

### Option C: Use Modal's Python 3.10+ image
```python
modal.Image.debian_slim(python_version="3.10")
```

### Option D: Skip inference, use HuggingFace DETR directly
- Remove `inference` dependency
- Use `transformers` DETR models instead
- Implement custom training loop

## Expected Outcome

With these changes, the Modal image should build successfully and the RF-DETR integration should work using Roboflow's inference SDK.

## Next Steps

Once this fix is verified:

1. Test dataset preparation: `modal run train.py::prepare_filtered_datasets`
2. Test YOLO training: `modal run train.py::train_model --model-type yolo --variant yolov8m`
3. Test RF-DETR inference: Verify model loads correctly
4. Implement full RF-DETR training loop integration

---

**Date**: 2025-12-05
**Status**: Fix applied, awaiting Modal deployment test
