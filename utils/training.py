import torch
import torch.nn.functional as F
from torch import nn
import copy
from tqdm import tqdm
from model.rgvt import recurrent_gvt
from model.gvt import GraphViewTransformation


class TrainerConfig:
    """Configuration for training."""
    def __init__(self):
        self.lr = 0.005  # Learning rate
        self.weight_decay = 0.0  # Weight decay
        self.max_epochs = 2500
        self.patience = 200
        self.log_interval = 10


class BaseTrainer:
    """Base trainer class with common training utilities."""
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.criterion = nn.NLLLoss()
    
    def evaluate_model(self, model, features, masks, labels, train_idx, device):
        """Evaluate model and return accuracies for train/val/test."""
        model.eval()
        with torch.no_grad():
            out = model(features)
            out = F.log_softmax(out, dim=1)
            pred = out.argmax(dim=1)
            
            accs = []
            for mask in masks:
                correct = pred[mask].eq(labels[mask]).sum().item()
                accs.append(correct / mask.sum().item())
        return accs
    
    def train_epoch(self, model, optimizer, features, train_idx, labels, device):
        """Train one epoch."""
        model.train()
        optimizer.zero_grad()
        out = model(features)
        out = F.log_softmax(out, dim=1)
        loss = self.criterion(out[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()
        return loss.item()

class PredictorTrainer(BaseTrainer):
    """Trainer for predictor-only models."""
    
    def train(self, predictor, features, labels, masks, device, desc="Training"):
        """Train predictor on given features."""
        train_idx = masks[0].nonzero(as_tuple=True)[0]
        optimizer = torch.optim.Adam(predictor.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        
        best_val = float('-inf')
        best_test = float('-inf')
        best_state_dict = None
        early_stopping_counter = 0
        
        for epoch in tqdm(range(self.config.max_epochs), desc=desc, leave=False):
            # Training step
            loss = self.train_epoch(predictor, optimizer, features, train_idx, labels, device)
            
            # Evaluation
            accs = self.evaluate_model(predictor, features, masks, labels, train_idx, device)
            
            # Early stopping check
            if accs[1] > best_val:
                best_val = accs[1]
                best_test = accs[2]
                early_stopping_counter = 0
                best_state_dict = copy.deepcopy(predictor.state_dict())
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= self.config.patience:
                break

        if best_state_dict is not None:
            predictor.load_state_dict(best_state_dict)

        return best_val, best_test, best_state_dict

class ViewTransfTrainer(BaseTrainer):
    def train(self, graph_view_transformation, predictor, features, edge_index, labels, masks, depth, device, args, desc="Training"):
        """Train the recurrent GVT encoder and predictor together."""
        train_idx = masks[0].nonzero(as_tuple=True)[0]
        
        # Combine parameters
        params = list(graph_view_transformation.parameters()) + list(predictor.parameters())
        optimizer = torch.optim.Adam(params, lr=self.config.lr, weight_decay=self.config.weight_decay)
        
        # Pre-compute adjacency matrices once

        A_rw, A_sym = GraphViewTransformation._build_adjacency_matrices(edge_index, features.size(0))
        
        best_val = float('-inf')
        best_test = float('-inf')
        best_state_dict = None
        early_stopping_counter = 0
        
        for epoch in tqdm(range(self.config.max_epochs), desc=desc, leave=False):
            
            # Training step
            graph_view_transformation.train()
            predictor.train()
            optimizer.zero_grad()
            
            x_agg = recurrent_gvt(
                graph_view_transformation, features, edge_index, args, manual_depth=depth,
                training=True, A_rw=A_rw, A_sym=A_sym
            )
            out = predictor(x_agg)
            out = F.log_softmax(out, dim=1)
            loss = self.criterion(out[train_idx], labels[train_idx])
            loss.backward()
            optimizer.step()
            
            # Evaluation
            graph_view_transformation.eval()
            predictor.eval()
            with torch.no_grad():
                x_agg = recurrent_gvt(
                    graph_view_transformation,
                    features,
                    edge_index,
                    args,
                    manual_depth=depth,
                    training=False,
                    A_rw=A_rw,
                    A_sym=A_sym,
                )
                accs = self.evaluate_model(predictor, x_agg, masks, labels, train_idx, device)
            
            # Early stopping check
            if accs[1] > best_val:
                best_val = accs[1]
                best_test = accs[2]
                early_stopping_counter = 0
                # Store best weights on CPU to avoid holding GPU memory across hops
                with torch.no_grad():
                    best_state_dict = {k: v.detach().cpu() for k, v in graph_view_transformation.state_dict().items()}
            else:
                early_stopping_counter += 1
            
            if epoch % self.config.log_interval == 0:
                print(
                    f'\nEpoch: {epoch:02d}, Loss: {loss:.4f}, '
                    f'Train: {100 * accs[0]:.2f}%, Valid: {100 * accs[1]:.2f}%, '
                    f'Test: {100 * accs[2]:.2f}%, Best Valid: {100 * best_val:.2f}%, '
                    f'Best Test: {100 * best_test:.2f}%'
                )
            
            if early_stopping_counter >= self.config.patience:
                print(f'Early stopping at epoch {epoch:02d}')
                break
        
        return best_val, best_test, best_state_dict


class GNNTrainer(BaseTrainer):
    """Trainer for GNN models that take x and edge_index as inputs."""
    
    def evaluate_gnn_model(self, model, features, edge_index, masks, labels, device):
        """Evaluate GNN model and return accuracies for train/val/test."""
        model.eval()
        with torch.no_grad():
            out = model(features, edge_index)
            out = F.log_softmax(out, dim=1)
            pred = out.argmax(dim=1)
            
            accs = []
            for mask in masks:
                correct = pred[mask].eq(labels[mask]).sum().item()
                accs.append(correct / mask.sum().item())
        return accs
    
    def train_gnn_epoch(self, model, optimizer, features, edge_index, train_idx, labels, device):
        """Train one epoch for GNN model."""
        model.train()
        optimizer.zero_grad()
        out = model(features, edge_index)
        out = F.log_softmax(out, dim=1)
        loss = self.criterion(out[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def train(self, model, features, edge_index, labels, masks, device, desc="Training GNN"):
        """Train GNN model on given features and edge_index."""
        train_idx = masks[0].nonzero(as_tuple=True)[0]
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        
        best_val = float('-inf')
        best_test = float('-inf')
        best_state_dict = None
        early_stopping_counter = 0
        
        for epoch in tqdm(range(self.config.max_epochs), desc=desc, leave=False):
            # Training step
            loss = self.train_gnn_epoch(model, optimizer, features, edge_index, train_idx, labels, device)
            
            # Evaluation
            accs = self.evaluate_gnn_model(model, features, edge_index, masks, labels, device)
            
            # Early stopping check
            if accs[1] > best_val:
                best_val = accs[1]
                best_test = accs[2]
                early_stopping_counter = 0
                best_state_dict = copy.deepcopy(model.state_dict())
            else:
                early_stopping_counter += 1
            
            # Logging
            if epoch % self.config.log_interval == 0:
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, '
                      f'Train: {100 * accs[0]:.2f}%, Valid: {100 * accs[1]:.2f}%, '
                      f'Test: {100 * accs[2]:.2f}%, Best Valid: {100 * best_val:.2f}%, '
                      f'Best Test: {100 * best_test:.2f}%')
            
            if early_stopping_counter >= self.config.patience:
                print(f'Early stopping at epoch {epoch:02d}')
                break
        
        return best_val, best_test, best_state_dict
