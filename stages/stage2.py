import torch
from model.rgvt import init_graph_view_transformation, recurrent_gvt, Predictor
from model.gvt import GraphViewTransformation
from utils.training import PredictorTrainer, TrainerConfig
from load_dataset import load_dataset

class HyperparameterTuner:

    def __init__(self, max_hops, config, device):
        self.max_hops = max_hops
        self.config = config
        self.device = device
        self.trainer = PredictorTrainer(config)

    def tune_recurrent_depth(self, features, edge_index, labels, masks, args, A_rw=None, A_sym=None):

        depth_results = {}
        pretrained_state = torch.load(args.checkpoint, map_location=self.device)

        num_classes = labels.max().item() + 1
        is_mlp = args.predictor_type == 'mlp'

        for depth in range(1, self.max_hops + 1):
            try:
                print(f"\n--- Processing recurrent depth L={depth} ---")
                
                # Load a fresh frozen RGVT encoder for each recurrent depth.
                rgvt_encoder = init_graph_view_transformation(args, device=self.device)
                rgvt_encoder.load_state_dict(pretrained_state)

                print(f"Predictor adaptation for L={depth}")
                rgvt_encoder.eval()
                # Freeze parameters
                for param in rgvt_encoder.parameters():
                    param.requires_grad = False
                
                # Generate RGVT representations using L recurrent steps.
                x_repr = recurrent_gvt(
                    rgvt_encoder, features, edge_index, args, manual_depth=depth,
                    training=False, A_rw=A_rw, A_sym=A_sym
                )

                # Train predictor only
                predictor = Predictor(x_repr.shape[1], num_classes, bias=True, is_mlp=is_mlp).to(self.device)

                best_val, best_test, _ = self.trainer.train(
                    predictor, x_repr, labels, masks, self.device,
                    desc=f"Predictor L={depth}"
                )

                depth_results[depth] = {'val': best_val, 'test': best_test}

                # Summary for this hop
                print(f"--- L={depth} Summary ---")
                print(f"  Predictor: Val {100 * best_val:.2f}%, Test {100 * best_test:.2f}%")

                # Clean up
                del rgvt_encoder, predictor, x_repr
                if torch.cuda.is_available():
                    print(f"Memory usage after L={depth}: {torch.cuda.memory_allocated(self.device) / 1e6:.2f} MB")
            
            except Exception as e:
                print(f"An error occurred during L={depth}: {e}")
                print("Setting results to 0.0 and continuing...")
                depth_results[depth] = {'val': 0.0, 'test': 0.0}
                continue

        return depth_results

def evaluate_dataset_stage(dataset_name, args, device='cpu', results_dir=None):
    
    print(f"\n=== Evaluating Dataset: {dataset_name} ===")

    max_hops = args.max_depth
    
    # Load dataset
    g = load_dataset(dataset_name, split_index=0)
    g = g.int().to(device)
    features = g.ndata["feat"] if "feat_norm" not in g.ndata else g.ndata["feat_norm"]
    labels = g.ndata["label"]
    masks = [g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]]
    edge_index = torch.stack(g.edges(), dim=0).long()
    
    # Visualization removed: skip feature style analysis
    
    if torch.cuda.is_available():
        print(f"Memory usage before tuning: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB")

    # Initialize hyperparameter tuner
    config = TrainerConfig()
    config.lr = args.learning_rate
    tuner = HyperparameterTuner(max_hops, config, device)

    # Pre-compute adjacency once and reuse
    A_rw, A_sym = GraphViewTransformation._build_adjacency_matrices(edge_index, features.size(0))

    depth_results = tuner.tune_recurrent_depth(
        features, edge_index, labels, masks, args, A_rw=A_rw, A_sym=A_sym
    )

    # Select the best recurrent depth by validation accuracy.
    if depth_results:
        best_depth = max(depth_results.keys(), key=lambda x: depth_results[x]['val'])
        best_val = depth_results[best_depth]['val']
        best_test = depth_results[best_depth]['test']
    else:
        best_depth = None
        best_val = float('nan')
        best_test = float('nan')
    
    print_results_summary(dataset_name, depth_results, best_depth, results_dir)
    
    return {
        'RGVT_best_val': best_val,
        'RGVT_best_test': best_test,
        'best_depth': best_depth,
    }


def print_results_summary(dataset_name, depth_results, best_depth, results_dir=None):

    import os
    import datetime
    
    # Use provided results_dir or create a new timestamped one
    if results_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = f'results/{timestamp}'
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate filename in the shared results directory
    filename = f'{results_dir}/{dataset_name}_detailed.txt'
    
    # Collect all output in a list
    output_lines = []
    
    # Header
    output_lines.append(f"{'='*80}")
    output_lines.append(f"DETAILED RESULTS SUMMARY FOR {dataset_name.upper()}")
    output_lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"{'='*80}")
    
    # Best results summary table
    output_lines.append("\nBEST RGVT ADAPTATION RESULT")
    output_lines.append(f"{'='*60}")
    
    if depth_results and best_depth is not None:
        best_val = depth_results[best_depth]['val']
        best_test = depth_results[best_depth]['test']

        output_lines.append("Frozen RGVT + predictor:")
        output_lines.append(f"  L={best_depth}: Val {best_val*100:6.2f}% | Test {best_test*100:6.2f}%")
        
    # Detailed hyperparameter tuning results
    output_lines.append("\n\nDETAILED RECURRENT DEPTH RESULTS")
    output_lines.append(f"{'='*60}")
    
    if depth_results:
        output_lines.append("\n[RGVT PREDICTOR ADAPTATION]")
        output_lines.append(f"{'L':<3} | {'Val%':<7} | {'Test%':<7}")
        output_lines.append(f"{'-'*3} | {'-'*7} | {'-'*7}")
        for depth, res in depth_results.items():
            marker = " *" if depth == best_depth else "  "
            output_lines.append(f"{depth:<3} | {100 * res['val']:<7.2f} | {100 * res['test']:<7.2f}{marker}")
    else:
        output_lines.append("\n[RGVT PREDICTOR ADAPTATION] No valid RGVT representations found!")
    
    output_lines.append(f"\n{'='*80}")
    
    # Write all output to file
    with open(filename, 'w') as f:
        f.write('\n'.join(output_lines))
    
    # Also print a concise summary to console
    print(f"\nQUICK SUMMARY FOR {dataset_name}:")
    if depth_results and best_depth is not None:
        best = depth_results[best_depth]
        print(f"   RGVT+predictor L={best_depth}: Val {best['val']*100:6.2f}% | Test {best['test']*100:6.2f}%")
    
    print(f"Detailed results saved to: {filename}")
