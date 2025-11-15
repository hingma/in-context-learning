# Analysis: Why "Averaging" Model Shows Different Values

## Summary

The "Averaging" baseline model shows **different numerical values** in `eval_colab.ipynb` vs `eval.ipynb`, but they represent the **same underlying performance**. The difference is due to **normalization**.

## The Key Difference

### eval_colab.ipynb (Cell 18)
```
Averaging: 10.0860 (final loss)
```
- Shows **raw squared error** values
- No normalization applied

### eval.ipynb (Cell 4 output plot)
```
Averaging: ~0.5 (from plot)
```
- Shows **normalized squared error** (divided by n_dims = 20)
- Uses `collect_results()` from `plot_utils.py` which normalizes all metrics

## The Normalization Logic

In `src/plot_utils.py` (lines 98-107):

```python
normalization = n_dims  # equals 20 for linear regression
if r.task == "sparse_linear_regression":
    normalization = int(r.kwargs.split("=")[-1])
if r.task == "decision_tree":
    normalization = 1

for k, v in m.items():
    v = v[:xlim]
    v = [vv / normalization for vv in v]  # Divide by normalization!
    m_processed[k] = v
```

## Verification

```
Raw value (eval_colab.ipynb):     10.0860
Normalized (eval.ipynb):          10.0860 / 20 = 0.504
Plot shows approximately:         ~0.5
```

✅ **The values match!**

## What is the "Averaging" Model?

From `src/models.py` (lines 267-292), the Averaging model implements:

```python
# For each test point i:
train_zs = train_xs * train_ys.unsqueeze(dim=-1)  # element-wise product
w_p = train_zs.mean(dim=1)  # average across training points
pred = test_x @ w_p  # linear prediction
```

This computes: **ŷ = x · mean(x_i * y_i)**

## Why Normalization?

The normalization allows **fair comparison across different problem dimensions**:

- **Zero estimator baseline**: Always predicts 0, giving squared error = n_dims (in expectation)
- **Normalizing by n_dims**: Makes the baseline = 1.0, so all other models can be compared relative to this
- **Interpretation**: 
  - Raw value of 10.0860 means 50% worse than perfect (n_dims = 20)
  - Normalized value of ~0.5 directly shows this: 0.5 < 1.0 (better than baseline)

## Conclusion

Both notebooks are showing the **correct** values for the Averaging model:
- **eval_colab.ipynb**: Raw values (10.0860)
- **eval.ipynb**: Normalized values (~0.5)

The model's **actual performance is identical** - it's just a different representation. The normalization in `eval.ipynb` makes it easier to compare across tasks with different dimensionalities.

## Additional Context

The "Averaging" baseline is a simple estimator that:
1. For the first point, predicts 0 (no training data)
2. For subsequent points, uses the average of (x_i * y_i) as a weight vector
3. Makes predictions by dotting the test point with this weight vector

This is a reasonable but simplistic approach compared to:
- **Least Squares (OLS)**: Optimal linear estimator (loss ≈ 0.0000)
- **3-Nearest Neighbors**: Uses local similarity (loss = 13.8574 raw, ~0.69 normalized)
- **Your trained model (QWEN2.5)**: Neural ICL learner (loss = 0.2405 raw, ~0.012 normalized)

The Averaging model performs better than k-NN but worse than OLS and your trained transformer.

