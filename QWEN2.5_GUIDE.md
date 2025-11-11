# Qwen2.5 Model Guide

This guide explains how to use the Qwen2.5-0.5B architecture for in-context learning tasks.

## Overview

We've added support for training Qwen2.5 models from scratch, similar to the existing GPT-2 implementation. Qwen2.5 is a modern transformer architecture that includes features like:

- Grouped-query attention (GQA) for improved efficiency
- Modern architectural improvements from the Qwen2 family
- Compatible with the same in-context learning framework

## Quick Start

### 1. Basic Training with Qwen2.5

Use the provided example configuration:

```bash
python src/train.py --config src/conf/qwen2.5_example.yaml --out_dir ./outputs/qwen2.5
```

### 2. Create Your Own Config

Create a new YAML file or modify an existing one:

```yaml
inherit: 
    - models/qwen2.5.yaml  # Use Qwen2.5 architecture
    - wandb.yaml

model:
    n_dims: 20
    n_positions: 101

training:
    task: linear_regression
    data: gaussian
    batch_size: 64
    learning_rate: 0.0001
    train_steps: 10000
    # ... other training parameters

out_dir: ./models/my_qwen_model
```

### 3. Test Run

Try a quick test run (100 steps):

```bash
python src/train.py --config src/conf/qwen2.5_example.yaml --out_dir ./outputs --test_run
```

## Model Configuration

The Qwen2.5 model is configured similarly to GPT-2 but with architecture-specific parameters:

```yaml
model:
    family: qwen2.5          # Use Qwen2.5 architecture
    n_embd: 896              # Hidden size (896 for 0.5B model)
    n_layer: 24              # Number of transformer layers
    n_head: 14               # Number of attention heads
    n_dims: 20               # Input dimension
    n_positions: 101         # Maximum context length
```

### Default Qwen2.5-0.5B Configuration

The `models/qwen2.5.yaml` provides architecture parameters approximating the Qwen2.5-0.5B model:

- **Hidden size**: 896
- **Layers**: 24
- **Attention heads**: 14
- **GQA**: Enabled (14 key-value heads)

## Differences from GPT-2

### Architecture
- **Grouped-Query Attention**: Qwen2.5 uses GQA which can be more memory-efficient
- **Modern normalization**: Updated layer normalization techniques
- **Intermediate size**: 4x hidden size (standard for modern transformers)

### Configuration
- Requires `num_key_value_heads` parameter (automatically set equal to `n_head`)
- Different default hyperparameters optimized for the Qwen architecture

## Training Tips

### Memory Usage

Qwen2.5-0.5B is larger than typical GPT-2 configurations used in this repo. If you encounter memory issues:

1. **Reduce batch size**:
   ```yaml
   training:
     batch_size: 32  # or 16
   ```

2. **Use smaller model**:
   ```yaml
   model:
     n_embd: 512    # Reduce from 896
     n_layer: 12    # Reduce from 24
     n_head: 8      # Reduce from 14
   ```

3. **Reduce context length**:
   ```yaml
   model:
     n_positions: 51  # Reduce from 101
   ```

### Performance

- Initial experiments show Qwen2.5 trains similarly to GPT-2
- May require different learning rates for optimal performance
- Curriculum learning helps with convergence

## Implementation Details

### Code Location

The Qwen2.5 implementation is in `src/models.py`:

```python
from transformers import Qwen2Model, Qwen2Config

class Qwen2TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        # ... initialization with Qwen2Config
```

### Import

Qwen2 models are imported from the `transformers` library:

```python
from transformers import Qwen2Model, Qwen2Config
```

Make sure you have a recent version of `transformers` installed:

```bash
pip install --upgrade transformers
```

## Example Configurations

### Small Test Model
```yaml
model:
    family: qwen2.5
    n_embd: 256
    n_layer: 6
    n_head: 4
    n_dims: 10
    n_positions: 51
```

### Medium Model (closer to 0.5B)
```yaml
model:
    family: qwen2.5
    n_embd: 896
    n_layer: 24
    n_head: 14
    n_dims: 20
    n_positions: 101
```

## Troubleshooting

### Import Error: No module named 'transformers'

Install or upgrade transformers:
```bash
pip install --upgrade transformers
```

### Import Error: cannot import name 'Qwen2Model'

Your transformers version is too old. Qwen2 support was added in recent versions:
```bash
pip install --upgrade transformers>=4.37.0
```

### CUDA Out of Memory

Qwen2.5-0.5B is larger than typical configs. Try:
1. Reducing `batch_size`
2. Reducing `n_embd`, `n_layer`
3. Using gradient checkpointing (future feature)

## Comparison with GPT-2

| Feature | GPT-2 | Qwen2.5 |
|---------|-------|---------|
| Attention | Multi-head | Grouped-query |
| Default layers | 12 | 24 |
| Default hidden | 256 | 896 |
| Memory efficiency | Standard | Better (GQA) |
| Training speed | Baseline | Similar |

## Further Reading

- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)
- [Hugging Face Qwen2 Documentation](https://huggingface.co/docs/transformers/model_doc/qwen2)
- See `TRAINING_README.md` for general training documentation

## Questions?

- Check the example config: `src/conf/qwen2.5_example.yaml`
- Review the model implementation: `src/models.py` (line ~139)
- See `TRAINING_README.md` for general training help

