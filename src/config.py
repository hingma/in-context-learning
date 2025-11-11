import os
import yaml
import argparse
from typing import Any, Dict


class ConfigDict(dict):
    """Dictionary that supports attribute-style access."""
    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, dict):
                return ConfigDict(value)
            return value
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")


def validate_config(config: Dict[str, Any]) -> None:
    """Validate the configuration dictionary."""
    # Required top-level keys
    assert "model" in config, "Missing 'model' in config"
    assert "training" in config, "Missing 'training' in config"
    
    # Model validation
    model = config["model"]
    assert "family" in model, "Missing 'family' in model config"
    assert model["family"] in ["gpt2", "qwen2.5", "lstm"], f"Invalid model family: {model['family']}"
    assert "n_positions" in model, "Missing 'n_positions' in model config"
    assert "n_dims" in model, "Missing 'n_dims' in model config"
    assert "n_embd" in model, "Missing 'n_embd' in model config"
    assert "n_layer" in model, "Missing 'n_layer' in model config"
    assert "n_head" in model, "Missing 'n_head' in model config"
    
    # Training validation
    training = config["training"]
    assert "task" in training, "Missing 'task' in training config"
    valid_tasks = [
        "linear_regression",
        "sparse_linear_regression",
        "linear_classification",
        "relu_2nn_regression",
        "decision_tree",
    ]
    assert training["task"] in valid_tasks, f"Invalid task: {training['task']}"
    assert "curriculum" in training, "Missing 'curriculum' in training config"
    
    # Curriculum validation
    curriculum = training["curriculum"]
    assert "dims" in curriculum, "Missing 'dims' in curriculum config"
    assert "points" in curriculum, "Missing 'points' in curriculum config"
    
    for key in ["dims", "points"]:
        curr_item = curriculum[key]
        assert "start" in curr_item, f"Missing 'start' in curriculum.{key}"
        assert "end" in curr_item, f"Missing 'end' in curriculum.{key}"
        assert "inc" in curr_item, f"Missing 'inc' in curriculum.{key}"
        assert "interval" in curr_item, f"Missing 'interval' in curriculum.{key}"


def set_defaults(config: Dict[str, Any]) -> None:
    """Set default values for optional configuration parameters."""
    # Training defaults
    if "training" not in config:
        config["training"] = {}
    
    training = config["training"]
    training.setdefault("data", "gaussian")
    training.setdefault("batch_size", 64)
    training.setdefault("learning_rate", 3e-4)
    training.setdefault("train_steps", 1000)
    training.setdefault("save_every_steps", 1000)
    training.setdefault("keep_every_steps", -1)
    training.setdefault("resume_id", None)
    training.setdefault("num_tasks", None)
    training.setdefault("num_training_examples", None)
    training.setdefault("task_kwargs", {})
    
    # Wandb defaults
    if "wandb" not in config:
        config["wandb"] = {}
    
    wandb = config["wandb"]
    wandb.setdefault("project", "in-context-training")
    wandb.setdefault("entity", "in-context")
    wandb.setdefault("notes", "")
    wandb.setdefault("name", None)
    wandb.setdefault("log_every_steps", 10)
    
    # Top-level defaults
    config.setdefault("test_run", False)


def load_config(config_path: str) -> ConfigDict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    set_defaults(config)
    validate_config(config)
    
    return ConfigDict(config)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train in-context learning model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Run in test mode with reduced steps"
    )
    parser.add_argument(
        "--resume_id",
        type=str,
        default=None,
        help="Resume training from a specific run ID"
    )
    
    return parser.parse_args()


def get_config():
    """Load and merge configuration from file and command-line arguments."""
    args = parse_args()
    
    # Load config from file
    config = load_config(args.config)
    
    # Override with command-line arguments
    config.out_dir = args.out_dir
    config.test_run = args.test_run
    
    if args.resume_id is not None:
        config.training.resume_id = args.resume_id
    
    return config

