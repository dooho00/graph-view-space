import torch
import os
import argparse
from utils.utils import set_seed
from stages.stage1 import pretrain
from stages.stage2 import evaluate_dataset_stage
from utils.summary import print_final_summary
from load_dataset import get_dataset_list
from utils.wandb_utils import WandbLogger, WandbConfig


def setup_wandb(args, device):
    """Initialize a minimal WandB logger with hyperparameters."""

    experiment_name = f"{args.mode}_{args.datasetA}_{args.max_depth}_{args.vt_depth}"
    # Include seed to distinguish repeats
    experiment_name += f"_seed_{args.seed}"
    
    # Minimal tags (keep user-provided too)
    tags = ['RGVT', args.mode, args.datasetA] + args.wandb_tags

    # Create wandb config
    wandb_config = WandbConfig(
        project_name=args.wandb_project,
        entity=args.wandb_entity,
        experiment_name=experiment_name,
        tags=tags,
        notes=args.wandb_notes
    )

    # Hyperparameters to report
    # Capture CUDA env and selected device for clarity in logs
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    try:
        cuda_ord = torch.cuda.current_device() if torch.cuda.is_available() else None
    except Exception:
        cuda_ord = None

    hyperparameters = {
        'mode': args.mode,
        'datasetA': args.datasetA,
        'learning_rate': args.learning_rate,
        'seed': args.seed,
        'max_depth': args.max_depth,
        'adj_max_hop': args.adj_max_hop,
        'vt_depth': args.vt_depth,
        'predictor_type': args.predictor_type,
        'device': str(device),
        'cuda_visible_devices': cuda_visible,
        'cuda_device_ordinal': int(cuda_ord) if cuda_ord is not None else None,
        'checkpoint': args.checkpoint
    }

    # Record hp_idx as a config value if provided (not a tag)
    if getattr(args, 'hp_idx', None) is not None:
        hyperparameters['hp_idx'] = args.hp_idx

    logger = WandbLogger(wandb_config, hyperparameters)
    return logger if logger.init_wandb() else None


def main():
    """Run the View Space / RGVT experiment pipeline."""
    
    # Configuration
    torch.set_num_threads(8)
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='View Space RGVT experiment pipeline')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--mode', choices=['pretrain', 'adaptation', 'both'], default='both', help='Which stage to run')
    parser.add_argument('--datasetA', default='27_ogbn_arxiv', help='Source graph for RGVT pretraining')
    parser.add_argument('--checkpoint', default='checkpoints/temp/RGVT.pth', help='Path to save/load the RGVT encoder checkpoint')
    
    # Hyperparameters related to RGVT
    parser.add_argument('--max_depth', type=int, default=8, help='Max recurrent depth for RGVT')
    parser.add_argument('--adj_max_hop', type=int, default=2, help='Maximum view-finder order K')
    parser.add_argument('--vt_depth', type=int, default=2, help='Depth of the GVT view-vector MLP')

    # Predictor Architecture
    parser.add_argument('--predictor_type', type=str, default='mlp', choices=['mlp', 'linear'], help='Lightweight predictor architecture')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for training')

    # Wandb options
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='RGVT', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity name')
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=[], help='WandB tags for the experiment')
    parser.add_argument('--wandb_notes', type=str, default=None, help='Notes for the experiment')
    parser.add_argument('--hp_idx', type=str, default=None, help='Stable hyperparameter index (logged as config, not a tag)')
    
    args = parser.parse_args()

    # Fix random seed using provided value
    set_seed(args.seed)
    
    # Initialize WandB logger if enabled (minimal)
    wandb_logger = setup_wandb(args, device) if args.wandb else None
    
    # Stage 1: Pretrain the RGVT encoder
    if args.mode in ['pretrain', 'both']:
        print("=" * 60)
        print(f"STAGE 1: Pretraining RGVT on {args.datasetA}")
        print("=" * 60)
        best_val, best_test = pretrain(args, device)
        
        # Report Stage 1 performance to wandb
        if wandb_logger:
            wandb_logger.log_metrics({
                'stage1/best_val_acc': best_val,
                'stage1/best_test_acc': best_test,
                'stage1/dataset': args.datasetA
            })
    
    # Stage 2: Freeze RGVT and adapt lightweight predictors
    if args.mode in ['adaptation', 'both']:
        print("\n" + "=" * 60)
        print("STAGE 2: Adapting frozen RGVT across datasets")
        print("=" * 60)
        
        # Create shared results directory with timestamp
        import datetime
        import os
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = f'results/{timestamp}'
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"Results will be saved to: {results_dir}")
        
        dataset_list = get_dataset_list()
        all_summaries = {}
        
        for dataset_name in dataset_list:
            summary = evaluate_dataset_stage(dataset_name, args, device, results_dir)
            all_summaries[dataset_name] = summary
            
            # Log Stage 2 results to wandb
            if wandb_logger and summary:
                wandb_logger.log_stage2_results(dataset_name, summary)
                
        
        # Print final summary to the same results directory
        if all_summaries:
            print_final_summary(all_summaries, dataset_list, results_dir)
        else:
            print("No datasets were successfully processed.")
    
    # Finish wandb run
    if wandb_logger:
        wandb_logger.finish()


if __name__ == '__main__':
    main()
            
