# OOD Results with Normalization - Example Output

## What You'll See After Running the Updated Notebook

### 1. OOD Individual Plots (Cell 20)
Each OOD scenario gets its own plot (unchanged - shows raw values):
```
Plotting OOD scenarios (each with baseline comparisons):

  ✓ random_quadrants: y-range [0.00, 456.52]
  ✓ orthogonal_train_test: y-range [0.00, 40.31]
  ✓ overlapping_train_test: y-range [0.00, 9946.75]
  ✓ half_subspace: y-range [0.00, 815.98]
  ✓ skewed: y-range [0.00, 6928.80]
```

### 2. **NEW!** OOD Comprehensive Summary Table (Cell 20)
Immediately after the plots, you'll see this new table:

```
==============================================================================================================
OOD ROBUSTNESS SUMMARY (with Normalized Values)
==============================================================================================================
Scenario                            Raw Loss        Normalized      vs Standard          Status         
--------------------------------------------------------------------------------------------------------------
Random Quadrants                    11.1649         0.5582          +4542.1%             ❌ Poor        
Orthogonal Train Test               0.0509          0.0025          -78.8%               ⚠️  Degraded   
Overlapping Train Test              0.1518          0.0076          -36.9%               ⚠️  Degraded   
Half Subspace                       1.0407          0.0520          +332.7%              ❌ Poor        
Skewed                             17.6777          0.8839          +7250.0%             ❌ Poor        
--------------------------------------------------------------------------------------------------------------
Standard (Reference)                0.2405          0.0120          baseline             ✅ Baseline    
==============================================================================================================

Note: Normalized values = Raw / n_dims (20). Status based on degradation vs standard.
```

### 3. **ENHANCED!** Scaling Summary (Cell 22)
The scaling analysis now includes normalized values:

```
====================================================================================================
SCALING ROBUSTNESS ANALYSIS (with Normalized Values)
====================================================================================================

Input (X) Scaling:
  Scale      Raw Loss        Normalized      vs Standard    
  ---------------------------------------------------------
  0.333        0.1954          0.0098                 -15.0%
  0.500        0.1110          0.0056                 -51.7%
  2.000        4.8295          0.2415               +2002.3%
  3.000        48.5379         2.4269              +21028.7%

Output (Y) Scaling:
  Scale      Raw Loss        Normalized      vs Standard    
  ---------------------------------------------------------
  0.333        0.0505          0.0025                 -78.0%
  0.500        0.0846          0.0042                 -63.2%
  2.000        3.1939          0.1597               +1290.3%
  3.000        25.0345         1.2517              +10797.6%

====================================================================================================
Note: Normalized values = Raw / n_dims (20) for fair comparison across dimensions.
```

### 4. Complete Evaluation Summary (Cell 24)
The final summary also shows normalized values:

```
================================================================================
COMPLETE EVALUATION SUMMARY
================================================================================

📊 PERFORMANCE METRICS:
  Standard (in-distribution) loss:
    Raw: 0.2405
    Normalized (÷20): 0.0120
  Improvement over baseline: 98.8%

🎯 BASELINE COMPARISONS:
  ⚠️ vs Least Squares:
      Raw: 0.0000 | Normalized: 0.0000
      Model advantage: -∞% worse
  ✅ vs Averaging:
      Raw: 10.0860 | Normalized: 0.5430
      Model advantage: +97.6% better
  ✅ vs 3-Nearest Neighbors:
      Raw: 13.8574 | Normalized: 0.6929
      Model advantage: +98.3% better

🌐 OUT-OF-DISTRIBUTION ROBUSTNESS:
  ❌ Random Quadrants:
      Raw: 11.1649 | Normalized: 0.5582
      Degradation: +4542.1%
  ❌ Orthogonal Train Test:
      Raw: 0.0509 | Normalized: 0.0025
      Degradation: -78.8%
  ...
```

## Key Benefits of This Format

### 1. **Immediate Insights**
You can now instantly see:
- ✅ **Good**: < 15% degradation from standard
- ⚠️  **Degraded**: 15-50% degradation
- ❌ **Poor**: > 50% degradation

### 2. **Normalized Comparison**
The normalized column lets you compare across dimensions:
```
Averaging:       0.5430 (normalized)
Random Quadrants: 0.5582 (normalized)
```
→ The model on random quadrants performs similarly to the Averaging baseline!

### 3. **Scale-Invariant Understanding**
With n_dims = 20:
```
Raw 10.0860 / 20 = 0.5430 normalized
```
If you had n_dims = 50, the baseline would be 50, and normalization would adjust accordingly.

### 4. **Quick Status Checks**
The status column gives you at-a-glance understanding:
```
Orthogonal Train Test  → ⚠️  Degraded    (-78.8%)  → Better than standard!
Random Quadrants       → ❌ Poor         (+4542%)  → Much worse
```

## Interpretation Guide

### Understanding the Numbers

**Raw Values:**
- Actual squared error
- Direct output from the loss function
- Scale: typically 0 to ~20 for n_dims=20

**Normalized Values:**
- Raw value ÷ n_dims
- Scale: typically 0 to ~1
- Makes comparison intuitive:
  - < 1.0 = Better than zero estimator
  - = 1.0 = Same as zero estimator
  - > 1.0 = Worse than zero estimator

**Status Indicators:**
- ✅ **Good**: Model maintains performance (< 15% change)
- ⚠️  **Degraded**: Some performance loss but still reasonable (15-50%)
- ❌ **Poor**: Significant performance degradation (> 50%)

### Example Analysis

Looking at the Averaging baseline:
```
Raw: 10.0860
Normalized: 0.5430
Baseline: 20.0 (raw) or 1.0 (normalized)

Interpretation:
- The Averaging model achieves 54.3% of the baseline error
- This means it's 45.7% better than always predicting zero
- In normalized terms: 1.0 - 0.5430 = 0.457 improvement
```

## Conclusion

Now **every section** of the evaluation shows both:
1. **Raw values** - for absolute magnitude
2. **Normalized values** - for relative comparison

This makes it easy to:
- Compare with literature (which uses normalized)
- Understand actual magnitudes (raw values)
- Quickly assess model robustness (status indicators)
- Compare across different experimental settings

The notebook now provides a **complete picture** of your model's performance! 🎉

