# Debugging Overfitting Issues

This document describes how to test if the model can overfit on a small dataset.

## Quick Test

1. **Run the standalone overfitting test:**
   ```bash
   python test_overfit.py --config_path configs/repurpose_merge.yaml
   ```
   This will:
   - Load only 4 training samples
   - Train for 100 epochs with high learning rate
   - Report if the model can overfit

2. **Use SimpleMCTransformer in main training:**
   ```bash
   python main.py --config_path configs/repurpose_merge.yaml --use-simple-model
   ```
   This uses a minimal model architecture (just linear layers) to isolate transformer-related issues.

## What to Look For

### Success Indicators:
- Loss decreases by >90% after 100 epochs
- Model predictions change from uniform (~0.5) to confident values
- Gradient norms are non-zero

### Failure Indicators:
- Loss stays constant or decreases minimally
- Predictions remain uniform around 0.5
- Gradients vanish to near-zero

## Debugging Steps

1. **Check Data Pipeline:**
   - Verify labels contain positive examples
   - Ensure masks are correctly applied
   - Check feature dimensions match config

2. **Verify Loss Computation:**
   - Focal loss should penalize wrong predictions
   - Mask should not zero out all gradients
   - Check if loss is averaged correctly

3. **Model Architecture:**
   - SimpleMCTransformer removes transformer complexity
   - Just uses linear layers for direct debugging
   - If this fails, issue is in data/loss pipeline

## Next Steps

If SimpleMCTransformer can overfit:
- Gradually add complexity back
- Test with small transformer layers
- Add regularization carefully

If SimpleMCTransformer cannot overfit:
- Debug data loading pipeline
- Check loss function implementation
- Verify gradient flow with hooks