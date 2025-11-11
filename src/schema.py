"""
Configuration schema for in-context learning training.
This module defines the valid tasks and provides reference documentation.
Actual config loading and validation is now handled by config.py.
"""

# List of valid tasks
TASK_LIST = [
    "linear_regression",
    "sparse_linear_regression",
    "linear_classification",
    "relu_2nn_regression",
    "decision_tree",
]

# Configuration schema documentation
SCHEMA_DOC = """
Configuration Structure:

model:
  family: str (gpt2 or lstm)
  n_positions: int  # maximum context length
  n_dims: int  # latent dimension
  n_embd: int  # embedding dimension
  n_layer: int  # number of layers
  n_head: int  # number of attention heads

training:
  task: str (one of linear_regression, sparse_linear_regression, 
            linear_classification, relu_2nn_regression, decision_tree)
  task_kwargs: dict
  num_tasks: int or null (default: null)
  num_training_examples: int or null (default: null)
  data: str (default: "gaussian")
  batch_size: int (default: 64)
  learning_rate: float (default: 3e-4)
  train_steps: int (default: 1000)
  save_every_steps: int (default: 1000)
  keep_every_steps: int (default: -1)
  resume_id: str or null (default: null)
  curriculum:
    dims:
      start: int  # initial parameter
      end: int  # limit of final value
      inc: int  # how much to increment each time
      interval: int  # increment every how many steps
    points:
      start: int
      end: int
      inc: int
      interval: int

wandb:
  project: str (default: "in-context-training")
  entity: str (default: "in-context")
  notes: str (default: "")
  name: str or null (default: null)
  log_every_steps: int (default: 10)

out_dir: str (required)
test_run: bool (default: false)
"""
