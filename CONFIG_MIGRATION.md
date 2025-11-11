# Configuration Migration Guide

This document describes the migration from `quinine` to a standard YAML-based configuration system.

## What Changed

### Old System (quinine)
- Used `QuinineArgumentParser` to parse config from CLI arguments
- Configuration schema defined using quinine's domain-specific language
- Required `quinine` and `funcy` dependencies

### New System (YAML + argparse)
- Uses standard Python `argparse` for CLI argument parsing
- Configuration stored in YAML files
- No external dependencies for config handling (only built-in Python libraries)
- More portable and easier to understand

## How to Use

### 1. Create a Configuration File

Create a YAML configuration file based on `config_example.yaml`:

```bash
cp config_example.yaml my_config.yaml
```

Edit the file to set your desired parameters.

### 2. Run Training

```bash
python src/train.py --config my_config.yaml --out_dir ./outputs
```

### Command-Line Arguments

- `--config`: Path to YAML configuration file (required)
- `--out_dir`: Output directory for checkpoints and logs (default: ./outputs)
- `--test_run`: Run in test mode with reduced steps (flag)
- `--resume_id`: Resume training from a specific run ID (optional)

### Example Commands

**Basic training:**
```bash
python src/train.py --config configs/linear_regression.yaml --out_dir ./outputs
```

**Test run:**
```bash
python src/train.py --config configs/linear_regression.yaml --out_dir ./outputs --test_run
```

**Resume training:**
```bash
python src/train.py --config configs/linear_regression.yaml --out_dir ./outputs --resume_id <run-uuid>
```

## Configuration Structure

See `config_example.yaml` for a complete example with all available options.

Required sections:
- `model`: Model architecture configuration
- `training`: Training parameters and curriculum
  - `curriculum`: Curriculum learning schedule for dims and points

Optional sections:
- `wandb`: Weights & Biases logging configuration

### Key Differences from Quinine

1. **File-based**: Configuration is now file-based rather than command-line based
2. **YAML format**: Uses standard YAML instead of quinine's custom format
3. **Simpler override**: Command-line arguments override config file values
4. **Better version control**: Config files can be easily versioned in git
5. **More readable**: YAML is more human-readable than command-line args

## Files Modified

- `src/train.py`: Updated to use new config system
- `src/schema.py`: Simplified to documentation only
- `src/config.py`: New module for config loading and validation
- `config_example.yaml`: Example configuration file

## Validation

The configuration is validated when loaded:
- Checks for required fields
- Validates task names against allowed list
- Validates model family (gpt2 or lstm)
- Ensures curriculum structure is correct

Validation errors will be raised with descriptive messages if the configuration is invalid.

