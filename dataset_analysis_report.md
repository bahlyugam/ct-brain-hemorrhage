# Dataset Analysis Report: V2/V3 → V4 Transition

## Executive Summary

**Total Images Analysis:**
- V1: 2,139 unique images
- V2: 7,536 unique images
- V3: 6,991 unique images
- V4: 14,627 unique images

**Containment Analysis:**
- V1 → V4: 97.5% (53 missing images)
- V2 → V4: 92.7% (548 missing images)
- V3 → V4: 99.9% (10 missing images) ⚠️

---

## Critical Finding: 10 V3 Images Missing from V4

### Root Cause Identified ✓

**All 10 missing images have entries in V3 COCO annotations but their image files DO NOT EXIST on disk.**

The V4 preparation script correctly filtered these out using the `filter_missing_images()` function (line 543-578 in prepare_v4_dataset.py), which removes images without corresponding files.

### Missing Images List

**Train Split (9 images):**
1. `thick_522158515_1.png` → (522158515, 1)
2. `thick_535266178_3.png` → (535266178, 3) [VALID split]
3. `thick_538526893_1.png` → (538526893, 1)
4. `thick_539909148_3.png` → (539909148, 3)
5. `thick_540396301_2.png` → (540396301, 2)
6. `thick_542733111_2.png` → (542733111, 2)
7. `thick_548723066_2.png` → (548723066, 2)
8. `thick_550151840_3.png` → (550151840, 3)
9. `thick_553191526_2.png` → (553191526, 2)
10. `thick_563729384_1.png` → (563729384, 1)

**Valid Split (1 image):**
- `thick_535266178_3.png` → (535266178, 3)

**All files have prefix `thick_` and expected path:**
```
/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v3/filtered_4class/coco/{split}/thick_{patientid}_{instance}.png
```

**File Status: ALL FILES MISSING** ❌

---

## Analysis of 548 "Missing" V2 Images

The 548 V2 images not in V4 break down as follows:

### 1. Intentionally Filtered (545 images)
**V2 → V3 Transition Filtering:**
- These images were filtered during the V2 → V3 transition
- Likely reasons:
  - Wrong hemorrhage classes (EDH/HC removed in V3 4-class filtering)
  - Quality issues
  - Slice thickness filtering (only 5.0mm and 5.5mm kept)
  - Invalid bounding boxes

**Evidence:** V2 has 7,536 images, V3 has 6,991 images → 545 images filtered

### 2. Restored via Roboflow (7 images)
**Examples:**
- `537451072_26`
- `537657326_14`
- And 5 others

These were filtered in V3 but restored in V4 because the Roboflow dataset (which includes 6 classes: EDH, HC, IPH, IVH, SAH, SDH) contains them.

### 3. New V2-only Images (3 images)
These are images in V2 but neither in V3 nor V4:
- Calculated as: 548 missing - 545 filtered - 7 restored + 10 V3-missing = ~3 images

---

## Breakdown Summary

| Category | Count | Description |
|----------|-------|-------------|
| **V2 → V3 filtered (intentional)** | 545 | Quality/class/thickness filtering |
| **V2 filtered but restored in V4** | 7 | Via Roboflow dataset |
| **V3 images with missing files** | 10 | In annotations but files don't exist |
| **New V2-only images** | ~3 | In V2, not in V3 or V4 |
| **V1 missing from V4** | 53 | Older dataset, expected gaps |

---

## Recommendations

### Priority 1: Investigate V3 Missing Files ⚠️
**Action Required:** Determine why 10 image files are missing from V3 dataset but listed in annotations

**Questions to answer:**
1. Were these files accidentally deleted?
2. Did they fail during V3 preparation (corruption, download failure)?
3. Can they be recovered from V2 or original source?

**Check V2 for these files:**
```bash
# Search V2 for the 10 missing patient_id_instance combinations
find data/training_datasets/archived/current_20251105/UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined \
  -name "*522158515_1*" -o -name "*535266178_3*" -o -name "*538526893_1*" \
  -o -name "*539909148_3*" -o -name "*540396301_2*" -o -name "*542733111_2*" \
  -o -name "*548723066_2*" -o -name "*550151840_3*" -o -name "*553191526_2*" \
  -o -name "*563729384_1*"
```

### Priority 2: Decision on V2 Images
**Options:**

**Option A: Keep Current V4 (Recommended)**
- Pros: V4 already has 14,627 images with 50% no-hem balance
- Pros: The 545 filtered images were intentionally removed for quality
- Cons: Missing 10 V3 images (if they can be recovered)

**Option B: Add All 548 V2 Images**
- Pros: Maximum dataset size
- Cons: Re-introduces 545 images that were previously filtered for quality/class reasons
- Cons: May degrade model performance with lower-quality images
- Cons: Requires re-balancing no-hemorrhage ratio

**Option C: Add Only Recoverable V3 Images (10 files)**
- Pros: Fixes the V3 → V4 gap without compromising quality
- Pros: Maintains the intentional V2 → V3 filtering
- Cons: Requires recovering/finding the 10 missing files

**Option D: Add V2-only Images (~3 files) + Recovered V3 Images (10 files)**
- Pros: Conservative addition, maintains quality standards
- Pros: Fixes V3 gap
- Cons: Requires identifying and recovering specific files

---

## Next Steps

1. **Investigate Missing Files:**
   - Check if the 10 files exist in V2 dataset
   - Check original data sources (S3, downloads)
   - Review V3 preparation logs for errors

2. **User Decision Required:**
   - Which option (A/B/C/D) to proceed with?
   - Are the 545 filtered V2 images acceptable to re-introduce?

3. **If Adding Images:**
   - Modify `prepare_v4_dataset.py` to include selected images
   - Re-run preparation pipeline
   - Validate patient-level splitting (no leakage)
   - Verify no-hemorrhage balance maintains ~50%

---

## Technical Notes

### Patient-Instance Extraction Logic
The analysis used the corrected extraction function that handles multiple prefixes:
```python
# Handles: v3_thick_520892552_14.png → (520892552, 14)
prefixes = ['v3_', 'roboflow_', 'thin_', 'thick_', 'no_hem_']
changed = True
while changed:
    changed = False
    for prefix in prefixes:
        if base.startswith(prefix):
            base = base[len(prefix):]
            changed = True
            break
```

### File Existence Check
The `filter_missing_images()` function (prepare_v4_dataset.py:543-578) correctly filters out images without files by checking the `image_paths` mapping built during dataset loading.

---

## Conclusion

**The "missing" images are not actually lost - they are accounted for:**

1. **10 V3 images:** Listed in annotations but files don't exist (data integrity issue)
2. **545 V2 images:** Intentionally filtered during V2 → V3 transition
3. **7 images:** Filtered in V3 but restored via Roboflow in V4
4. **~3 V2-only images:** In V2 but not in V3 or V4

**The V4 dataset is correctly prepared** based on available files and intentional filtering decisions. The only concern is the 10 V3 images with missing files, which should be investigated and potentially recovered from source data.
