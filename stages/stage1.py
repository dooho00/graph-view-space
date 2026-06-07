import os
import torch
from model.rgvt import init_graph_view_transformation,  Predictor
from utils.training import ViewTransfTrainer, TrainerConfig
from load_dataset import load_dataset

def pretrain(args, device='cpu'):
    # Extract values from args
    dataset_name = args.datasetA
    checkpoint_path = args.checkpoint

    print(f"=== Training RGVT on {dataset_name} ===")

    # Load dataset
    g = load_dataset(dataset_name, split_index=0)
    g = g.int().to(device)
    features = g.ndata["feat"] if "feat_norm" not in g.ndata else g.ndata["feat_norm"]
    labels = g.ndata["label"]
    masks = [g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]]
    edge_index = torch.stack(g.edges(), dim=0).long()
    
    # Initialize models with args parameters
    graph_view_transformation = init_graph_view_transformation(args, device)
    is_mlp = args.predictor_type == 'mlp'
    predictor = Predictor(features.shape[1], labels.max().item() + 1, bias=True, is_mlp=is_mlp).to(device)

    # Reset parameters
    graph_view_transformation.reset_parameters()
    predictor.reset_parameters()
    
    # Train
    config = TrainerConfig()
    config.lr = args.learning_rate
    trainer = ViewTransfTrainer(config)
    best_val, best_test, best_state_dict = trainer.train(
        graph_view_transformation,
        predictor,
        features,
        edge_index,
        labels,
        masks,
        depth=args.max_depth,
        device=device,
        args=args,
        desc=f"Train RGVT: {dataset_name}"
    )
    
    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    # Save checkpoint
    if best_state_dict is not None:
        torch.save(best_state_dict, checkpoint_path)
        print(f'RGVT saved to {checkpoint_path}')

    print(f'RGVT training complete. Best Valid: {100 * best_val:.2f}%, Best Test: {100 * best_test:.2f}%')
    return best_val, best_test
