# Training Guide

This guide covers how to train in-context learning models using the updated configuration system (without quinine).

## Table of Contents
- [Local Training](#local-training)
- [Google Colab Training](#google-colab-training)
- [Configuration Options](#configuration-options)

## Local Training

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Required packages: PyTorch, transformers, wandb, xgboost, pyyaml

### Installation
```bash
# Install dependencies
pip install torch transformers wandb xgboost pyyaml tqdm matplotlib seaborn
```

### Quick Start

1. **Create a configuration file** based on the example:
```bash
cp config_example.yaml my_config.yaml
```

2. **Edit the configuration** to set your parameters:
```yaml
model:
  family: gpt2
  n_positions: 256
  n_dims: 20
  n_embd: 256
  n_layer: 12
  n_head: 8

training:
  task: linear_regression
  train_steps: 10000
  batch_size: 64
  learning_rate: 3e-4
  # ... more options
```

3. **Run training**:
```bash
python src/train.py --config my_config.yaml --out_dir ./outputs
```

### Command-Line Options

```bash
python src/train.py \
  --config CONFIG_FILE      # Path to YAML config (required)
  --out_dir OUTPUT_DIR      # Output directory (default: ./outputs)
  --test_run                # Quick test with 100 steps (optional)
  --resume_id RUN_ID        # Resume from specific run (optional)
```

### Examples

**Basic training:**
```bash
python src/train.py --config configs/linear_regression.yaml --out_dir ./outputs
```

**Test run (100 steps):**
```bash
python src/train.py --config configs/linear_regression.yaml --out_dir ./outputs --test_run
```

**Resume training:**
```bash
python src/train.py --config configs/linear_regression.yaml --out_dir ./outputs --resume_id <uuid>
```

## Google Colab Training

For training on Google Colab with free GPU access:

### Option 1: Use the Training Notebook

1. Upload `train_colab_new.ipynb` to Google Colab
2. Set runtime to GPU (Runtime → Change runtime type → GPU)
3. Follow the notebook cells sequentially

The notebook includes:
- ✅ Automatic package installation
- ✅ GPU detection and setup
- ✅ W&B integration
- ✅ Inline configuration (no separate files needed)
- ✅ Model download after training

### Option 2: Manual Setup

```python
# In a Colab notebook cell:
!git clone https://github.com/yourusername/in-context-learning.git
%cd in-context-learning

# Install packages
!pip install transformers wandb xgboost pyyaml

# Run training
!python src/train.py --config config_example.yaml --out_dir ./outputs
```

## Configuration Options

### Model Configuration

```yaml
model:
  family: gpt2              # Model type: 'gpt2' or 'lstm'
  n_positions: 256          # Maximum sequence length
  n_dims: 20                # Latent dimension
  n_embd: 256               # Embedding dimension
  n_layer: 12               # Number of layers
  n_head: 8                 # Number of attention heads (GPT-2 only)
```

### Training Configuration

```yaml
training:
  task: linear_regression   # Task type (see below)
  task_kwargs: {}           # Task-specific arguments
  num_tasks: null           # Number of tasks (null = unlimited)
  num_training_examples: null  # Training examples (null = unlimited)
  
  data: gaussian            # Data distribution
  batch_size: 64            # Batch size
  learning_rate: 3e-4       # Learning rate
  train_steps: 10000        # Total training steps
  
  save_every_steps: 1000    # Checkpoint frequency
  keep_every_steps: -1      # Permanent checkpoints (-1 = disabled)
  resume_id: null           # Resume from run ID (null = new run)
```

### Available Tasks

1. **linear_regression**: Standard linear regression
2. **sparse_linear_regression**: Sparse linear regression with feature selection
3. **linear_classification**: Binary linear classification
4. **relu_2nn_regression**: 2-layer ReLU neural network regression
5. **decision_tree**: Decision tree learning

### Curriculum Learning

The curriculum gradually increases task difficulty:

```yaml
training:
  curriculum:
    dims:
      start: 5              # Initial dimensions
      end: 20               # Final dimensions
      inc: 1                # Increment per update
      interval: 100         # Update every N steps
    points:
      start: 10             # Initial data points
      end: 41               # Final data points
      inc: 1                # Increment per update
      interval: 100         # Update every N steps
```

### Weights & Biases Configuration

```yaml
wandb:
  project: in-context-training    # W&B project name
  entity: your-entity             # W&B entity/team
  notes: ""                       # Run notes
  name: null                      # Run name (null = auto)
  log_every_steps: 10             # Logging frequency
```

## Monitoring Training

### Weights & Biases

Training automatically logs to W&B (if not in test mode):
- Loss curves (overall and pointwise)
- Curriculum progress (n_points, n_dims)
- Excess loss over baseline
- System metrics (GPU, memory)

Access your W&B dashboard at: https://wandb.ai

### Local Logs

Training progress is displayed in the terminal with a progress bar:
```
loss 0.1234: 45%|████▌     | 4500/10000 [12:34<14:21,  6.38it/s]
```

## Output Files

Training creates the following outputs:

```
outputs/
└── <run-id>/
    ├── config.yaml         # Configuration used for this run
    ├── state.pt            # Latest checkpoint (resumable)
    └── model_<step>.pt     # Permanent checkpoints (if enabled)
```

### Loading a Trained Model

```python
import torch
from models import build_model
import yaml

# Load config
with open('outputs/<run-id>/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Build model
model = build_model(config['model'])

# Load weights
state = torch.load('outputs/<run-id>/state.pt')
model.load_state_dict(state['model_state_dict'])
model.eval()
```

## Tips and Best Practices

### GPU Memory

If you run out of GPU memory:
- Reduce `batch_size` (e.g., 64 → 32)
- Reduce `n_embd` (e.g., 256 → 128)
- Reduce `n_layer` (e.g., 12 → 6)

### Training Time

Typical training times:
- **Test run (100 steps)**: 1-2 minutes
- **Full training (10k steps)**: 1-3 hours (GPU-dependent)

### Checkpointing

- Set `save_every_steps` for regular checkpoints (resumable)
- Set `keep_every_steps > 0` for permanent model snapshots
- Use `resume_id` to continue interrupted training

### Curriculum Learning

The curriculum automatically adjusts task difficulty:
- Starts with few dimensions and data points
- Gradually increases to full complexity
- Helps model learn more effectively

## Troubleshooting

### Import Errors

If you get import errors, ensure you're in the project root:
```bash
cd /path/to/in-context-learning
python src/train.py --config ...
```

### CUDA Out of Memory

Reduce memory usage:
```yaml
training:
  batch_size: 32  # Reduce from 64
model:
  n_embd: 128     # Reduce from 256
```

### W&B Authentication

Login to W&B:
```bash
wandb login
# Paste your API key when prompted
```

Or set environment variable:
```bash
export WANDB_API_KEY=your_api_key
```

## Migration from Quinine

If you're migrating from the old quinine-based system:

1. **No more command-line configs** - Use YAML files instead
2. **Simpler syntax** - Standard YAML instead of quinine DSL
3. **Better version control** - Track configs in git
4. **Same training logic** - All model code unchanged

See `CONFIG_MIGRATION.md` for detailed migration guide.

## Questions?

- Check `config_example.yaml` for a complete example
- See `train_colab_new.ipynb` for an interactive walkthrough
- Read the source: `src/train.py`, `src/config.py`

