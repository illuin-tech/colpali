#!/usr/bin/env python3
"""
Optuna-based hyperparameter optimization for ColIntern3.5.
Runs multiple trials with different hyperparameters and finds the best combination.
"""

import optuna
import json
import subprocess
import os
from pathlib import Path


def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    
    # Define search space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    gradient_accumulation = trial.suggest_categorical("gradient_accumulation", [2, 4, 8])
    lora_rank = trial.suggest_categorical("lora_rank", [16, 32, 64])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.2)
    
    # Create trial-specific output directory
    trial_name = f"trial_{trial.number}"
    output_dir = f"./experiments/optuna_trials/{trial_name}"
    
    # Calculate warmup steps (estimate based on dataset size)
    estimated_steps_per_epoch = 127000 // (batch_size * gradient_accumulation)
    warmup_steps = int(estimated_steps_per_epoch * warmup_ratio)
    
    # Create training command
    cmd = [
        "python", "scripts/configs/internvl3_5/train_colintern35_model.py",
        "--output-dir", output_dir,
        "--lr", str(learning_rate),
        "--peft",
        # Override batch size and other params via environment variables
    ]
    
    # Set environment variables for this trial
    env = os.environ.copy()
    env["TRIAL_BATCH_SIZE"] = str(batch_size)
    env["TRIAL_GRAD_ACCUM"] = str(gradient_accumulation)
    env["TRIAL_LORA_RANK"] = str(lora_rank)
    env["TRIAL_WARMUP_STEPS"] = str(warmup_steps)
    
    try:
        # Run training (shortened for optimization)
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            print(f"Trial {trial.number} failed: {result.stderr}")
            return float('inf')  # Return worst possible score
        
        # Extract validation loss from training logs
        log_file = Path(output_dir) / "trainer_state.json"
        if log_file.exists():
            with open(log_file) as f:
                trainer_state = json.load(f)
            
            # Get the best validation score
            log_history = trainer_state.get("log_history", [])
            eval_losses = [entry.get("eval_loss") for entry in log_history if "eval_loss" in entry]
            
            if eval_losses:
                return min(eval_losses)  # Return best validation loss
        
        return float('inf')  # No valid evaluation found
        
    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number} timed out")
        return float('inf')
    except Exception as e:
        print(f"Trial {trial.number} error: {e}")
        return float('inf')


def run_optuna_optimization(n_trials=20):
    """Run Optuna hyperparameter optimization."""
    
    # Create study
    study = optuna.create_study(
        direction="minimize",  # Minimize validation loss
        study_name="colintern35_hyperopt",
        storage="sqlite:///optuna_study.db",
        load_if_exists=True
    )
    
    print(f"🔍 Starting Optuna optimization with {n_trials} trials...")
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=24*3600)  # 24 hour timeout
    
    # Print results
    print("=" * 80)
    print("🏆 OPTUNA OPTIMIZATION RESULTS")
    print("=" * 80)
    
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (validation loss): {study.best_value:.4f}")
    
    print("\n📊 Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Generate training command with best parameters
    best_params = study.best_params
    cmd = f"""python scripts/configs/internvl3_5/train_colintern35_model.py \\
  --output-dir ./experiments/colintern3_5-best \\
  --lr {best_params['learning_rate']:.2e} \\
  --peft"""
    
    print(f"\n🚀 Optimal training command:")
    print(cmd)
    
    # Save results
    with open("optuna_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    
    return study.best_params


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10, help="Number of trials to run")
    args = parser.parse_args()
    
    run_optuna_optimization(args.trials)
