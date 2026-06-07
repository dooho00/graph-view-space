import logging
import os
import os.path as osp
import pickle
import ssl
import sys
import urllib
from typing import Optional, Tuple

import dgl
import numpy as np
import torch
import torch_geometric.datasets as pyg_datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_url(url: str, folder: str, log: bool = True, filename: Optional[str] = None) -> str:
    if filename is None:
        filename = url.rpartition("/")[2]
        filename = filename if filename[0] == "?" else filename.split("?")[0]

    path = osp.join(folder, filename)

    if osp.exists(path):
        if log and "pytest" not in sys.modules:
            logger.info(f"Using existing file {filename}")
        return path

    if log and "pytest" not in sys.modules:
        logger.info(f"Downloading {url}")

    os.makedirs(osp.expanduser(osp.normpath(folder)), exist_ok=True)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, "wb") as f:
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


def load_heterophilous_dataset(url: str, raw_dir: str) -> Tuple:
    download_path = download_url(url, raw_dir)
    data = np.load(download_path)
    
    node_features = torch.tensor(data["node_features"])
    labels = torch.tensor(data["node_labels"])
    edges = torch.tensor(data["edges"])

    graph = dgl.graph(
        (edges[:, 0], edges[:, 1]), 
        num_nodes=len(node_features), 
        idtype=torch.int
    )
    
    num_classes = len(labels.unique())
    train_masks = torch.tensor(data["train_masks"]).T
    val_masks = torch.tensor(data["val_masks"]).T
    test_masks = torch.tensor(data["test_masks"]).T

    return graph, labels, num_classes, node_features, train_masks, val_masks, test_masks


def get_data_split_masks(n_nodes: int, 
                        labels: torch.Tensor, 
                        num_train_nodes: int, 
                        test_ratio: float = 0.5,
                        seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    label_idx = np.arange(len(labels))
    test_rate_in_labeled_nodes = (len(labels) - num_train_nodes) / len(labels)
    
    train_idx, test_and_valid_idx = train_test_split(
        label_idx,
        test_size=test_rate_in_labeled_nodes,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    
    valid_idx, test_idx = train_test_split(
        test_and_valid_idx,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels[test_and_valid_idx],
    )
    
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[valid_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask

def handle_splits(graph: dgl.DGLGraph, 
                 data_name: str, 
                 split_index: int = 0,
                 cache_dir: str = "splits",
                 seed: int = 42) -> dgl.DGLGraph:
    seed = 42
    os.makedirs(cache_dir, exist_ok=True)
    split_path = osp.join(cache_dir, f'{data_name}_seed{seed}.splits')
    
    if osp.exists(split_path):
        logger.info(f"Loading splits for {data_name} from {split_path}")
        all_splits = pickle.load(open(split_path, 'rb'))
    else:
        logger.info(f"Generating a new split for {data_name}")
        all_splits = get_data_split_masks(graph.num_nodes(), graph.ndata['label'], 20 * (graph.ndata['label'].max().item() + 1), seed=seed)
        logger.info(f"Generated a split for {data_name}: {all_splits[0].shape}")
        pickle.dump(all_splits, open(split_path, 'wb'))

    # Access the correct split using the index
    train_masks, val_masks, test_masks = all_splits

    graph.ndata['train_mask'] = train_masks
    graph.ndata['val_mask'] = val_masks
    graph.ndata['test_mask'] = test_masks

    return graph


class DatasetConfig:
    """Configuration for dataset loading."""
    
    def __init__(self):
        # Data sources and their configurations
        self.data_sources = {
            "dgl": {
                #"26_full_cora": "CoraFullDataset",
            },
            "ogb": {
                "27_ogbn_arxiv": "ogbn-arxiv",
            },
            "heterophilous": {
                "21_texas": "texas_4_classes",
                "13_cornell": "cornell",
                "24_wisconsin": "wisconsin",
                # Wiki Traffic datasets
                "8_chameleon": "chameleon_filtered",
                "20_squirrel": "squirrel_filtered",
                # Heterophilous datasets
                "19_roman_empire": "roman_empire",
                "6_amazon_ratings": "amazon_ratings",
                "22_tolokers": "tolokers",
                "16_minesweeper": "minesweeper",
                "18_questions": "questions",
                # Actor dataset
                "0_actor": "actor",
            },
            "pyg": {
                # Document Topic Classification
                '12_cora': {"class": "Planetoid", "name": "Cora"},
                '9_citeseer': {"class": "Planetoid", "name": "CiteSeer"},
                '17_pubmed': {"class": "Planetoid", "name": "PubMed"},
                '25_wiki_cs': {"class": "WikiCS"},
                "26_full_cora": {"class": "CoraFull"},
                # Author Field Classification
                '10_co_cs': {"class": "Coauthor", "name": "CS"},
                '11_co_phy': {"class": "Coauthor", "name": "Physics"},
                # Ecommerce datasets
                '4_amz_computer': {"class": "Amazon", "name": "Computers"},
                '5_amz_photo': {"class": "Amazon", "name": "Photo"},
                # Document Topic Classification
                "14_dblp": {"class": "CitationFull", "name": "DBLP"},
                "23_wiki": {"class": "AttributedGraphDataset", "name": "Wiki"},
                # Airport Traffic datasets
                "1_air_brazil": {"class": "Airports", "name": "Brazil"},
                "3_air_usa": {"class": "Airports", "name": "USA"},
                "2_air_eu": {"class": "Airports", "name": "Europe"},
                # Social network communities
                "7_blogcatalog": {"class": "AttributedGraphDataset", "name": "BlogCatalog"},
                "15_deezer": {"class": "DeezerEurope"},
            }
        }
        
        # Datasets grouped by source
        self.dgl_datasets = set(self.data_sources["dgl"].keys())
        self.heterophilous_datasets = set(self.data_sources["heterophilous"].keys())
        self.ogb_datasets = set(self.data_sources["ogb"].keys())
        self.pyg_datasets = set(self.data_sources["pyg"].keys())

def get_dataset_list() -> list:
    """Get the list of available datasets based on the DatasetConfig."""
    config = DatasetConfig()
    dataset_list = []  

    dataset_list.extend(config.dgl_datasets)
    dataset_list.extend(config.heterophilous_datasets)
    dataset_list.extend(config.ogb_datasets)
    dataset_list.extend(config.pyg_datasets)

    # Sort by the numerical prefix, then alphabetically
    def sort_key(dataset_name):
        # Extract the number prefix if it exists
        parts = dataset_name.split('_', 1)
        if len(parts) == 2 and parts[0].isdigit():
            return (int(parts[0]), parts[1])  # Sort by number first, then by name
        else:
            return (float('inf'), dataset_name)  # Put datasets without numbers at the end
    
    dataset_list = sorted(dataset_list, key=sort_key)

    return dataset_list

def load_dataset(data_name: str, 
                split_index: int = 0,
                add_self_loop: bool = True,
                to_bidirected: bool = True,
                cache_dir: str = "dataset",
                seed: int = 42,
                splits_dir: str = "splits") -> dgl.DGLGraph:
    # Initialize configuration
    config = DatasetConfig()
    
    # Create cache directories
    os.makedirs(cache_dir, exist_ok=True)
    #splits_dir = split_cache_dir or os.path.join(cache_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    graph = None
    
    try:
        # Load dataset based on source
        if data_name in config.data_sources["dgl"]:
            graph = _load_dgl_dataset(data_name, config, split_index, seed, splits_dir, cache_dir)
            
        elif data_name in config.data_sources["ogb"]:
            graph = _load_ogb_dataset(data_name, config, cache_dir)
            
        elif data_name in config.data_sources["heterophilous"]:
            graph = _load_heterophilous_dataset(data_name, config, cache_dir, split_index, splits_dir)
            
        elif data_name in config.data_sources["pyg"]:
            graph = _load_pyg_dataset(data_name, config, split_index, seed, splits_dir, cache_dir)
            
        else:
            raise ValueError(f"Unknown dataset: {data_name}")
        
        # Apply graph preprocessing
        graph = _preprocess_graph(graph, add_self_loop, to_bidirected)

        # Log dataset statistics
        _dataset_statistics(graph, data_name)
        
        return graph
        
    except Exception as e:
        logger.error(f"Failed to load dataset {data_name}: {str(e)}")
        raise


def _load_dgl_dataset(data_name: str, 
                     config: DatasetConfig, 
                     split_index: int,
                     seed: int,
                     splits_dir: str,
                     cache_dir: str) -> dgl.DGLGraph:
    """Load DGL datasets."""
    dataset_class_name = config.data_sources["dgl"][data_name]
    
    # Import and instantiate dataset
    dataset_class = getattr(dgl.data, dataset_class_name)
    # Prefer per-dataset cache under <cache_dir>/<data_name>
    target_raw_dir = os.path.join(cache_dir, data_name)
    os.makedirs(target_raw_dir, exist_ok=True)
    try:
        dataset = dataset_class(raw_dir=target_raw_dir)
    except TypeError:
        # Fallback if signature does not support raw_dir
        dataset = dataset_class()
    graph = dataset[0]
    data = graph.ndata
    
    # Handle existing splits if available
    if 'train_mask' in data and data['train_mask'] is not None:
        _handle_existing_splits_dgl(graph, data_name, split_index)
    else:
        # Generate splits if not provided
        graph = handle_splits(graph, data_name, split_index, cache_dir=splits_dir, seed=seed)
    
    # Convert to bool if needed
    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
        if mask_name in graph.ndata and graph.ndata[mask_name].dtype != torch.bool:
            graph.ndata[mask_name] = graph.ndata[mask_name].bool()
        
    return graph


def _load_ogb_dataset(data_name: str, 
                     config: DatasetConfig, 
                     cache_dir: str) -> dgl.DGLGraph:
    """Load OGB datasets."""
    from ogb.nodeproppred import DglNodePropPredDataset
    
    ogb_name = config.data_sources["ogb"][data_name]
    dataset = DglNodePropPredDataset(
        name=ogb_name, 
        root=osp.join(cache_dir, "ogb")
    )
    
    graph, labels = dataset[0]
    graph.ndata['label'] = labels.squeeze()
    
    # Get official splits
    splits = dataset.get_idx_split()
    
    # Create masks
    for split_name, mask_name in [('train', 'train_mask'), ('valid', 'val_mask'), ('test', 'test_mask')]:
        mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
        mask[splits[split_name]] = True
        graph.ndata[mask_name] = mask
    
    return graph

def _load_heterophilous_dataset(data_name: str, 
                               config: DatasetConfig, 
                               cache_dir: str,
                               split_index: int,
                               splits_dir: str) -> dgl.DGLGraph:
    """Load heterophilous datasets."""
    dataset_filename = config.data_sources["heterophilous"][data_name]
    
    # Try local file first, then download
    local_path = osp.join(cache_dir, f"{dataset_filename}.npz")
    
    if osp.exists(local_path):
        dataset = np.load(local_path)
        node_features = torch.tensor(dataset['node_features'])
        edges = torch.tensor(dataset['edges'])
        
        graph = dgl.graph(
            (edges[:, 0], edges[:, 1]), 
            num_nodes=node_features.shape[0], 
            idtype=torch.int
        )
        
        graph.ndata['feat'] = node_features
        graph.ndata['label'] = torch.tensor(dataset['node_labels'])
        
        # Handle splits
        if 'train_masks' in dataset:
            num_splits = dataset['train_masks'].shape[0]
            split_index = split_index % num_splits
            
            graph.ndata['train_mask'] = torch.tensor(dataset['train_masks'][split_index])
            graph.ndata['val_mask'] = torch.tensor(dataset['val_masks'][split_index])
            graph.ndata['test_mask'] = torch.tensor(dataset['test_masks'][split_index])
        else:
            # Generate splits if not provided
            graph = handle_splits(graph, data_name, split_index, cache_dir=splits_dir)
            
    else:
        # Download from GitHub
        url = f"https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/{dataset_filename}.npz"
        raw_dir = osp.join(cache_dir, "heterophilous")
        
        graph, labels, num_classes, node_features, train_masks, val_masks, test_masks = load_heterophilous_dataset(url, raw_dir)
        
        graph.ndata['feat'] = node_features
        graph.ndata['label'] = labels
        
        # Select split
        num_splits = train_masks.shape[1]
        split_index = split_index % num_splits
        
        graph.ndata['train_mask'] = train_masks[:, split_index]
        graph.ndata['val_mask'] = val_masks[:, split_index]
        graph.ndata['test_mask'] = test_masks[:, split_index]
    
    return graph

def _load_pyg_dataset(data_name: str, 
                     config: DatasetConfig, 
                     split_index: int,
                     seed: int,
                     splits_dir: str,
                     cache_dir: str) -> dgl.DGLGraph:
    """Load PyTorch Geometric datasets."""
    
    # Get dataset configuration
    dataset_config = config.data_sources["pyg"][data_name]
    
    # Handle both string and dict configurations for backward compatibility
    if isinstance(dataset_config, str):
        dataset_class_name = dataset_config
        dataset_params = {}
    else:
        dataset_class_name = dataset_config["class"]
        dataset_params = {k: v for k, v in dataset_config.items() if k != "class"}
    
    dataset_class = getattr(pyg_datasets, dataset_class_name)
    
    # Create root directory
    # Prefer per-dataset cache under <cache_dir>/<data_name>
    root_dir = os.path.join(cache_dir, data_name)
    os.makedirs(root_dir, exist_ok=True)
    
    # Initialize dataset with parameters
    try:
        dataset = dataset_class(root=root_dir, **dataset_params)
    except Exception as e:
        raise e
    
    # Get the first (and usually only) graph
    data = dataset[0]
    
    # Convert PyG Data to DGL graph
    edge_index = data.edge_index
    graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=data.x.shape[0])
    
    # Add node features and labels
    graph.ndata['feat'] = data.x
    graph.ndata['label'] = data.y
    
    # Handle existing splits if available
    if hasattr(data, 'train_mask') and data.train_mask is not None:
        _handle_existing_splits(graph, data, data_name, split_index)
    else:
        # Generate splits if not provided
        graph = handle_splits(graph, data_name, split_index, cache_dir=splits_dir, seed=seed)
    
    return graph

def _handle_existing_splits_dgl(graph: dgl.DGLGraph, data_name: str, split_index: int):
    """Handle existing train/val/test splits from dgl data."""
    if graph.ndata['train_mask'].ndim > 1:
        # Multiple splits available
        num_splits = graph.ndata['train_mask'].shape[1]
        split_index = split_index % num_splits
        graph.ndata['train_mask'] = graph.ndata['train_mask'][:, split_index]
        graph.ndata['val_mask'] = graph.ndata['val_mask'][:, split_index]
        if data_name != '25_wiki_cs':
            # WikiCS has a single test mask for all splits
            graph.ndata['test_mask'] = graph.ndata['test_mask'][:, split_index]

def _handle_existing_splits(graph: dgl.DGLGraph, data, data_name: str, split_index: int):
    """Handle existing train/val/test splits from PyG data."""
    if data.train_mask.ndim > 1:
        # Multiple splits available
        num_splits = data.train_mask.shape[1]
        split_index = split_index % num_splits
        graph.ndata['train_mask'] = data.train_mask[:, split_index]
        graph.ndata['val_mask'] = data.val_mask[:, split_index]
        if data_name != '25_wiki_cs':
            # WikiCS has a single test mask for all splits
            graph.ndata['test_mask'] = data.test_mask[:, split_index]
        else:
            graph.ndata['test_mask'] = data.test_mask
    else:
        # Single split
        graph.ndata['train_mask'] = data.train_mask
        graph.ndata['val_mask'] = data.val_mask
        graph.ndata['test_mask'] = data.test_mask

def _preprocess_graph(graph: dgl.DGLGraph, 
                     add_self_loop: bool,
                     to_bidirected: bool) -> dgl.DGLGraph:
    """Apply graph preprocessing."""
    # Handle self loops
    if add_self_loop:
        graph = dgl.add_self_loop(graph)
    else:
        graph = dgl.remove_self_loop(graph)
    
    # Make bidirectional if requested
    if to_bidirected:
        graph = dgl.to_bidirected(graph, copy_ndata=True)
    
    # Remove duplicate edges
    graph = dgl.to_simple(graph)
    
    return graph


def _dataset_statistics(graph: dgl.DGLGraph, data_name: str):
    """Log and save dataset statistics."""
    import datetime
    import json
    
    num_nodes = graph.num_nodes()
    num_edges = graph.num_edges()
    num_feats = graph.ndata['feat'].shape[1] if 'feat' in graph.ndata else 0
    num_classes = len(torch.unique(graph.ndata['label']))
    
    train_samples = graph.ndata['train_mask'].sum().item()
    val_samples = graph.ndata['val_mask'].sum().item()
    test_samples = graph.ndata['test_mask'].sum().item()
    
    # Calculate homophily ratio
    edge_src, edge_dst = graph.edges()
    labels = graph.ndata['label']
    
    # For undirected graphs, count each edge once
    valid_edges = edge_src < edge_dst
    if valid_edges.sum() > 0:
        edge_src = edge_src[valid_edges]
        edge_dst = edge_dst[valid_edges]
        same_label_edges = (labels[edge_src] == labels[edge_dst]).sum().item()
        homophily_ratio = same_label_edges / valid_edges.sum().item()
    else:
        homophily_ratio = 0.0
    
    # Calculate additional statistics
    avg_degree = num_edges / num_nodes if num_nodes > 0 else 0.0
    
    # Format statistics text
    stats_text = f"""Dataset Statistics for {data_name}:
        Nodes: {num_nodes:,}
        Edges: {num_edges:,}
        Features: {num_feats}
        Classes: {num_classes}
        Train samples: {train_samples:,}
        Val samples: {val_samples:,}
        Test samples: {test_samples:,}
        Homophily ratio: {homophily_ratio:.4f}
        Average degree: {avg_degree:.2f}
        Train ratio: {train_samples/num_nodes:.4f}
        Val ratio: {val_samples/num_nodes:.4f}
        Test ratio: {test_samples/num_nodes:.4f}
        """
    
    # Save to file
    stats_dir = f'plots/{data_name}/'
    os.makedirs(stats_dir, exist_ok=True)
    stats_file = osp.join(stats_dir, f'dataset_statistics.txt')
    
    with open(stats_file, 'w') as f:
        f.write(stats_text)
        f.write(f"\nGenerated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def print_graph_statistics(graph: dgl.DGLGraph, data_name: Optional[str] = None):
    """Print detailed graph statistics (legacy function for backward compatibility)."""
    _dataset_statistics(graph, data_name or "Unknown")


def analyze_feature_style(graph: dgl.DGLGraph, data_name: str, save_plots: bool = True):
    """
    Analyze feature characteristics and save analysis results.
    
    Args:
        graph: DGL graph
        data_name: Dataset name
        save_plots: Whether to save analysis plots
    """
    feat = graph.ndata['feat'] if 'feat' in graph.ndata else graph.ndata.get('feat_norm')
    
    if feat is None:
        logger.warning(f"No features found for dataset {data_name}")
        return
    
    # Build analysis text directly
    analysis_lines = []
    analysis_lines.append(f"Feature Analysis for {data_name}")
    analysis_lines.append("=" * 50)
    analysis_lines.append(f"Feature shape: {feat.shape}")

    # Basic statistics
    analysis_lines.append(f"Min values: {feat.min(dim=0).values[:10].tolist()}")
    analysis_lines.append(f"Max values: {feat.max(dim=0).values[:10].tolist()}")
    analysis_lines.append(f"Mean values: {feat.mean(dim=0)[:10].tolist()}")
    analysis_lines.append(f"Std values: {feat.std(dim=0)[:10].tolist()}")

    # Feature characteristics
    is_binary = torch.all((feat == 0) | (feat == 1))
    analysis_lines.append(f"Is binary (0/1 only): {is_binary.item()}")

    is_onehot = torch.all(feat.sum(dim=1) == 1) and is_binary
    analysis_lines.append(f"Is one-hot encoded: {is_onehot.item()}")

    nonzero_ratio = (feat != 0).sum().item() / feat.numel()
    analysis_lines.append(f"Sparsity (non-zero ratio): {nonzero_ratio:.4f}")
    
    # Additional statistics
    analysis_lines.append(f"Feature range: [{feat.min().item():.4f}, {feat.max().item():.4f}]")
    analysis_lines.append(f"Feature mean: {feat.mean().item():.4f}")
    analysis_lines.append(f"Feature std: {feat.std().item():.4f}")
    
    analysis_text = "\n".join(analysis_lines)
    
    # Always save analysis results
    save_dir = f'plots/{data_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save text analysis
    with open(f'{save_dir}/feature_analysis.txt', 'w') as f:
        f.write(analysis_text)
    
    if save_plots:
        
        # Save histogram of first feature dimension
        try:
            plt.figure(figsize=(10, 6))
            plt.hist(feat[:, 0].cpu().numpy(), bins=50, alpha=0.7, edgecolor='black')
            plt.title(f'Feature Distribution - {data_name} (Dimension 0)')
            plt.xlabel('Feature Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{save_dir}/feature_dim0_hist.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save feature statistics visualization
            if feat.shape[1] > 1:
                plt.figure(figsize=(12, 8))
                
                plt.subplot(2, 2, 1)
                plt.plot(feat.mean(dim=0).cpu().numpy())
                plt.title('Mean per Feature Dimension')
                plt.xlabel('Feature Dimension')
                plt.ylabel('Mean Value')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 2, 2)
                plt.plot(feat.std(dim=0).cpu().numpy())
                plt.title('Std per Feature Dimension')
                plt.xlabel('Feature Dimension')
                plt.ylabel('Std Value')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 2, 3)
                plt.plot((feat != 0).sum(dim=0).cpu().numpy())
                plt.title('Non-zero Count per Feature Dimension')
                plt.xlabel('Feature Dimension')
                plt.ylabel('Non-zero Count')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 2, 4)
                plt.plot(feat.max(dim=0).values.cpu().numpy(), label='Max')
                plt.plot(feat.min(dim=0).values.cpu().numpy(), label='Min')
                plt.title('Min/Max per Feature Dimension')
                plt.xlabel('Feature Dimension')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f'{save_dir}/feature_statistics.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logger.warning(f"Failed to save plots for {data_name}: {str(e)}")


if __name__ == "__main__":
    # Check loading of all datasets
    config = DatasetConfig()
    for source, datasets in config.data_sources.items():
        for dataset_name in datasets.keys():
            try:
                logger.info(f"Loading dataset: {dataset_name} from {source}")
                graph = load_dataset(dataset_name, split_index=0)
                print_graph_statistics(graph, dataset_name)
                analyze_feature_style(graph, dataset_name)
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_name}: {str(e)}")
