# WandB Metrics Logging Guide

**Date**: 2025-12-08
**Training Run**: RF-DETR Medium with 3x Augmented Dataset (Anti-Overfitting)

---

## WandB Dashboard

**Project**: `brain-ct-hemorrhage`
**Run Name**: `rfdetr_medium_4class_aug3x_antioverfitting`
**Tags**: `augmented`, `3x`, `anti-overfitting`, `regularized`

**URL**: https://wandb.ai/your-team/brain-ct-hemorrhage

---

## Metrics Being Logged

### 1. Training Metrics (Every 10 Batches)

**Prefix**: `train/`

| Metric | Description | Expected Range | Goal |
|--------|-------------|----------------|------|
| `train/loss` | Total training loss | 8.0 → 4.0 | Decrease steadily |
| `train/lr` | Learning rate | 5e-5 → 0 | Cosine decay |
| `train/class_error` | Classification error % | 15% → <5% | Minimize |
| `train/loss_ce` | Cross-entropy loss | 2.0 → 0.5 | Decrease |
| `train/loss_bbox` | Bounding box L1 loss | 0.5 → 0.1 | Decrease |
| `train/loss_giou` | GIoU loss | 0.8 → 0.2 | Decrease |
| `train/time` | Time per batch (seconds) | ~2-3s | Monitor |

**Logged**: Every 10 batches during training
**Step**: Incremental batch step counter

---

### 2. Validation Metrics (Every Epoch)

**Prefix**: `val/`

| Metric | Description | Expected Range | Goal |
|--------|-------------|----------------|------|
| `val/loss` | Total validation loss | 6.0 → <6.0 | Stay close to train loss |
| `val/class_error` | Validation class error % | 20% → <15% | Gap from train <10% |
| `val/loss_ce` | Validation CE loss | - | Monitor |
| `val/loss_bbox` | Validation bbox loss | - | Gap from train <50% |
| `val/loss_giou` | Validation GIoU loss | - | Monitor |

**Logged**: At end of each epoch
**Step**: Epoch number

---

### 3. COCO Evaluation Metrics (Every Epoch)

**Prefix**: `val/`

| Metric | Description | Expected Value | Goal |
|--------|-------------|----------------|------|
| `val/mAP` | Mean Average Precision @ IoU=0.50:0.95 | 0.30 → 0.50+ | Primary metric |
| `val/mAP_50` | mAP @ IoU=0.50 | 0.50 → 0.70+ | Easier threshold |
| `val/mAP_75` | mAP @ IoU=0.75 | 0.35 → 0.55+ | Strict threshold |
| `val/mAP_small` | mAP for small objects | - | Monitor |
| `val/mAP_medium` | mAP for medium objects | - | Monitor |
| `val/mAP_large` | mAP for large objects | - | Monitor |

**Logged**: At end of each epoch after COCO evaluation
**Step**: Epoch number

**Note**: These are the official COCO metrics used to compare object detection models.

---

### 4. Test Set Metrics (After Training)

**Prefix**: `test/`

| Metric | Description | Target | Success Criteria |
|--------|-------------|--------|------------------|
| `test/loss` | Test set total loss | <6.0 | Gap from train <20% |
| `test/class_error` | Test set class error % | **<15%** | **Down from 50.68%!** |
| `test/loss_bbox` | Test set bbox loss | <0.3 | Gap from train <50% |
| `test/loss_giou` | Test set GIoU loss | - | Monitor |
| `test/mAP` | Test set mAP | >0.40 | Production ready |

**Logged**: Once at end of training after final test evaluation
**Step**: Final epoch number

---

## Key Metrics to Monitor

### Success Indicators (Anti-Overfitting)

1. **Train-Val Gap** ✅
   ```
   (val/loss - train/loss) / train/loss < 0.20  (20% gap)
   ```
   **Previous**: 80% gap (4.81 → 8.68)
   **Target**: <20% gap

2. **Test Class Error** ✅
   ```
   test/class_error < 15%
   ```
   **Previous**: 50.68% (random guess level)
   **Target**: <15%

3. **BBox Loss Gap** ✅
   ```
   (test/loss_bbox - train/loss_bbox) / train/loss_bbox < 0.50  (50% gap)
   ```
   **Previous**: 264% gap (0.14 → 0.51)
   **Target**: <50% gap

4. **Validation mAP** ✅
   ```
   val/mAP > 0.40
   ```
   Measures overall detection quality

---

## How to Access in WandB

### View Metrics Comparison

1. **Go to WandB Dashboard**: https://wandb.ai/your-team/brain-ct-hemorrhage

2. **Select Run**: `rfdetr_medium_4class_aug3x_antioverfitting`

3. **Key Charts to Add**:

   **Chart 1: Train vs Val Loss**
   - X-axis: Epoch
   - Y-axis: `train/loss`, `val/loss`
   - Should converge closely (not diverge!)

   **Chart 2: Class Error Comparison**
   - X-axis: Epoch
   - Y-axis: `train/class_error`, `val/class_error`
   - Gap should be <10%

   **Chart 3: BBox Loss Gap**
   - X-axis: Epoch
   - Y-axis: `train/loss_bbox`, `val/loss_bbox`
   - Should track closely

   **Chart 4: mAP Over Time**
   - X-axis: Epoch
   - Y-axis: `val/mAP`, `val/mAP_50`, `val/mAP_75`
   - Should increase steadily

   **Chart 5: Learning Rate Schedule**
   - X-axis: Step
   - Y-axis: `train/lr`
   - Should show cosine decay from 5e-5 → 0

---

## Comparison with Previous Run

### Baseline (No Augmentation, Overfit)

**Run**: `rfdetr_medium_4class` (Epoch 68)

| Metric | Baseline Train | Baseline Test | Gap |
|--------|----------------|---------------|-----|
| Loss | 4.81 | 8.68 | **+80%** |
| Class Error | 0% | 50.68% | **+∞** |
| BBox Loss | 0.14 | 0.51 | **+264%** |

### New Run (3x Augmented + Regularized)

**Run**: `rfdetr_medium_4class_aug3x_antioverfitting`

| Metric | Target Train | Target Val | Target Test | Gap |
|--------|--------------|------------|-------------|-----|
| Loss | ~5.5 | <6.5 | <6.0 | **<20%** |
| Class Error | <5% | <12% | <15% | **<10%** |
| BBox Loss | ~0.2 | <0.3 | <0.3 | **<50%** |
| mAP | - | >0.40 | >0.40 | - |

---

## Alert Conditions

### Early Stopping Trigger

If validation loss doesn't improve for **50 epochs**, training will stop automatically.

**Monitor**:
- `val/loss` plateau
- Early stopping message in logs

### Underfitting Warning

If training loss remains high (>6.0 after 50 epochs):
- ⚠ Regularization too strong
- Consider: Reduce weight_decay from 1e-3 to 5e-4

### Still Overfitting Warning

If test/class_error > 20% after training:
- ⚠ Need more augmentation or smaller model
- Consider: Try RF-DETR Small variant

---

## Expected Timeline

**Training Duration**: 40-50 hours on Modal A10G

**Checkpoints**:
- Epoch 10: Learning rate warmup complete
- Epoch 50: Should see clear convergence
- Epoch 100: May trigger early stopping if plateau
- Epoch 200: Maximum training duration

**Key Milestones**:
- By Epoch 20: `val/mAP` > 0.30
- By Epoch 50: `val/mAP` > 0.40
- By Epoch 100: `test/class_error` < 15%

---

## Summary

**What Changed from Baseline**:
1. ✅ Dataset multiplied 3x with aggressive augmentation
2. ✅ Regularization 10x stronger (weight_decay 1e-3)
3. ✅ Learning rate reduced 50% (5e-5)
4. ✅ Batch size increased 133% (batch=32, effective=64)
5. ✅ Dropout added (0.1)
6. ✅ Early stopping enabled (patience 50)
7. ✅ GPU upgraded to A100 40GB for larger batches

**Expected Improvements**:
- Train-test gap: 80% → **<20%**
- Test class error: 50.68% → **<15%**
- BBox loss gap: 264% → **<50%**
- Production-ready model with mAP > 0.40

---

**Status**: Ready for training
**Next**: Upload dataset → Start training → Monitor WandB
