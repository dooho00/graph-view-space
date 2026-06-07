
import os
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np
import wandb

class WandbConfig:
    """Configuration class for wandb settings."""
    
    def __init__(self, 
                 project_name: str = "graph_view_transformation",
                 entity: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 notes: Optional[str] = None,
                 save_dir: str = "./wandb_logs"):
        self.project_name = project_name
        self.entity = entity
        self.experiment_name = experiment_name
        self.tags = tags or []
        self.notes = notes
        self.save_dir = save_dir
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)


class WandbLogger:
    """Minimal wandb logger for graph_view_transformation experiments."""
    
    def __init__(self, config: WandbConfig, hyperparameters: Optional[Dict] = None):
        self.config = config
        self.run = None
        self.hyperparameters = hyperparameters or {}
        self.is_initialized = False
        # Hold per-dataset Stage 2 summaries to compute aggregates
        self.stage2_summaries: Dict[str, Dict[str, Any]] = {}
        
    def init_wandb(self, mode: str = "online") -> bool:

        try:
            # Generate experiment name if not provided
            if not self.config.experiment_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.config.experiment_name = f"experiment_{timestamp}"
            
            self.run = wandb.init(
                project=self.config.project_name,
                entity=self.config.entity,
                name=self.config.experiment_name,
                tags=self.config.tags,
                notes=self.config.notes,
                dir=self.config.save_dir,
                config=self.hyperparameters,
                mode=mode,
                reinit=True
            )
            
            self.is_initialized = True
            print(f"✅ Wandb initialized successfully!")
            print(f"   Project: {self.config.project_name}")
            print(f"   Run: {self.config.experiment_name}")
            if self.run.url:
                print(f"   URL: {self.run.url}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize wandb: {e}")
            print("   Continuing without wandb logging...")
            self.is_initialized = False
            return False
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """Log hyperparameters to wandb."""
        if not self.is_initialized:
            return
            
        try:
            # Update config with new hyperparameters
            wandb.config.update(hyperparams)
            print(f"📊 Logged {len(hyperparams)} hyperparameters to wandb")
        except Exception as e:
            print(f"⚠️ Failed to log hyperparameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to wandb."""
        if not self.is_initialized:
            return
            
        try:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
        except Exception as e:
            print(f"⚠️ Failed to log metrics: {e}")
    
    def log_stage2_results(self, dataset_name: str, summary: Dict[str, Any]) -> None:
        """Log Stage 2 (evaluation) results for a specific dataset."""
        if not self.is_initialized or not summary:
            return
        
        try:
            # Flatten the summary dictionary for logging
            metrics = {}
            for method, results in summary.items():
                if isinstance(results, dict):
                    for metric, value in results.items():
                        if isinstance(value, (int, float)):
                            metrics[f"stage2/{dataset_name}/{method}/{metric}"] = value
                        else:
                            metrics[f"stage2/{dataset_name}/{method}/{metric}"] = str(value)
                else:
                    if isinstance(results, (int, float)):
                        metrics[f"stage2/{dataset_name}/{method}"] = results
                    else:
                        metrics[f"stage2/{dataset_name}/{method}"] = str(results)

            # Store this dataset's summary for aggregate reporting
            self.stage2_summaries[dataset_name] = summary

            # Compute aggregate mean/std across all processed datasets for numeric fields
            # We aggregate over keys present in any dataset summary and numeric in type
            all_keys = set()
            for s in self.stage2_summaries.values():
                all_keys.update(k for k, v in s.items() if isinstance(v, (int, float)))

            aggregate_metrics = {}
            for key in sorted(all_keys):
                values = [s[key] for s in self.stage2_summaries.values() if isinstance(s.get(key), (int, float))]
                if not values:
                    continue
                # Convert to numpy for mean/std; ignore NaNs if any slip in
                arr = np.array(values, dtype=float)
                arr = arr[~np.isnan(arr)] if np.isnan(arr).any() else arr
                if arr.size == 0:
                    continue
                aggregate_metrics[f"stage2/total/{key}/mean"] = float(np.mean(arr))
                aggregate_metrics[f"stage2/total/{key}/std"] = float(np.std(arr, ddof=0))

            # Merge per-dataset and aggregate metrics
            metrics.update(aggregate_metrics)

            self.log_metrics(metrics)
            print(f"📈 Logged Stage 2 results for {dataset_name}: {len(metrics)} metrics")
        except Exception as e:
            print(f"⚠️ Failed to log Stage 2 results for {dataset_name}: {e}")
    
    def finish(self) -> None:
        """Finish wandb run - alias for finish_run for compatibility."""
        self.finish_run()
    
    def finish_run(self) -> None:
        """Finish wandb run and cleanup."""
        if self.is_initialized and self.run:
            try:
                wandb.finish()
                print("✅ Wandb run finished successfully")
            except Exception as e:
                print(f"⚠️ Error finishing wandb run: {e}")
            finally:
                self.is_initialized = False
                self.run = None


def create_wandb_config_from_args(args, stage: str = "training") -> WandbConfig:
    """Create wandb config from command line arguments."""
    
    # Generate experiment name based on stage and key parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if stage == "stage1":
        exp_name = f"stage1_pretrain_{args.datasetA}_{timestamp}"
        tags = ["stage1", "pretraining", args.datasetA]
    elif stage == "stage2":
        exp_name = f"stage2_adaptation_{timestamp}"
        tags = ["stage2", "adaptation", "multi_dataset"]
    elif stage == "gnn_baseline":
        exp_name = f"gnn_baseline_{timestamp}"
        tags = ["baseline", "gnn"]
    else:
        exp_name = f"{stage}_{timestamp}"
        tags = [stage]
    
    # Add model architecture tags (simplified)
    tags.extend([
        f"max_depth_{args.max_depth}",
        f"adj_max_hop_{args.adj_max_hop}"
    ])
    
    return WandbConfig(
        project_name="graph_view_transformation",
        experiment_name=exp_name,
        tags=tags
    )


def extract_hyperparameters_from_args(args) -> Dict[str, Any]:
    """Extract hyperparameters from command line arguments."""
    hyperparams = {
        # Model architecture
        "max_depth": args.max_depth,
        "adj_max_hop": args.adj_max_hop,
        
        # Training settings
        "datasetA": args.datasetA,
        "checkpoint": args.checkpoint,
        "mode": args.mode,
    }
    
    return hyperparams
