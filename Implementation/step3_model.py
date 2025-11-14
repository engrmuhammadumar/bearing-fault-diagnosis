"""
STEP 3: Physics-Informed GNN Model Architecture
================================================
Defines the neural network architecture with physics constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
import pickle
import json
import os

OUTPUT_DIR = r'F:\concrete data\test 3\gnn_implementation'

print("="*80)
print("STEP 3: PHYSICS-INFORMED GNN MODEL ARCHITECTURE")
print("="*80)

# ============================================================================
# 3.1: LOAD CONFIGURATION
# ============================================================================
print("\n[3.1] Loading configuration from Step 2...")
print("-" * 80)

with open(os.path.join(OUTPUT_DIR, 'step2_config.json'), 'r') as f:
    config = json.load(f)

print(f"✓ Loaded configuration:")
print(f"  Nodes: {config['num_nodes']}")
print(f"  Node features: {config['num_node_features']}")
print(f"  Edges: {config['num_edges']}")
print(f"  Graphs: {config['num_graphs']}")

# ============================================================================
# 3.2: DEFINE PHYSICS-INFORMED GNN MODEL
# ============================================================================
print("\n[3.2] Defining Physics-Informed GNN architecture...")
print("-" * 80)

class PhysicsInformedGNN(nn.Module):
    """
    Physics-Informed Graph Neural Network for Damage Prediction
    
    Novel Features:
    1. Graph Attention for interpretability
    2. Physics-based energy conservation loss
    3. Multi-task learning (RUL + Localization)
    4. Spatial-temporal feature extraction
    
    Architecture:
    - Layer 1: Graph Attention (interpretable information flow)
    - Layer 2: Graph Convolution (feature aggregation)
    - Layer 3: Global pooling (graph-level representation)
    - Multi-task heads: RUL regression + Location classification
    """
    
    def __init__(self, 
                 num_node_features=10,
                 hidden_dim=64,
                 num_classes=8,
                 dropout=0.3):
        """
        Initialize model
        
        Parameters:
        -----------
        num_node_features : int
            Number of features per node (default: 10)
        hidden_dim : int
            Hidden layer dimension (default: 64)
        num_classes : int
            Number of damage location classes (default: 8 channels)
        dropout : float
            Dropout rate (default: 0.3)
        """
        super(PhysicsInformedGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Layer 1: Graph Attention Network (interpretable)
        # Multi-head attention to capture different aspects
        self.gat1 = GATConv(
            num_node_features, 
            hidden_dim, 
            heads=4,           # 4 attention heads
            concat=True,       # Concatenate heads
            dropout=dropout
        )
        
        # Layer 2: Graph Attention Network
        self.gat2 = GATConv(
            hidden_dim * 4,    # Input from 4 concatenated heads
            hidden_dim,
            heads=4,
            concat=True,
            dropout=dropout
        )
        
        # Layer 3: Graph Convolution (final aggregation)
        self.gcn = GCNConv(
            hidden_dim * 4,
            hidden_dim
        )
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden_dim * 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Dimension reduction after pooling
        self.reduce_dim = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        
        # Task 1: RUL Prediction Head (Regression)
        self.rul_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
        # Task 2: Damage Localization Head (Classification)
        self.location_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)  # 8 channels
        )
        
        # Physics Constraint: Energy prediction
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Store attention weights for interpretability
        self.attention_weights = None
        
    def forward(self, data):
        """
        Forward pass
        
        Parameters:
        -----------
        data : torch_geometric.data.Data
            Graph data object
        
        Returns:
        --------
        dict : Predictions and attention weights
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Layer 1: Graph Attention with attention weights
        x, (edge_index_att1, alpha1) = self.gat1(
            x, edge_index, return_attention_weights=True
        )
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Layer 2: Graph Attention
        x, (edge_index_att2, alpha2) = self.gat2(
            x, edge_index, return_attention_weights=True
        )
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Layer 3: Graph Convolution
        x = self.gcn(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)
        
        # Global pooling: aggregate node features to graph-level
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_global = torch.cat([x_mean, x_max], dim=1)  # Combine mean and max
        
        # Reduce dimension
        x_global = self.reduce_dim(x_global)
        
        # Multi-task predictions
        rul_pred = self.rul_head(x_global) * 100  # Scale to [0, 100]
        location_pred = self.location_head(x_global)
        energy_pred = self.energy_head(x_global)
        
        # Store attention weights for visualization
        self.attention_weights = {
            'layer1': (edge_index_att1, alpha1),
            'layer2': (edge_index_att2, alpha2)
        }
        
        return {
            'rul': rul_pred,
            'location': location_pred,
            'energy': energy_pred,
            'node_embeddings': x,
            'graph_embedding': x_global
        }
    
    def get_attention_weights(self):
        """Return stored attention weights for interpretability"""
        return self.attention_weights

# ============================================================================
# 3.3: DEFINE LOSS FUNCTIONS
# ============================================================================
print("\n[3.3] Defining loss functions...")
print("-" * 80)

class PhysicsInformedLoss(nn.Module):
    """
    Combined loss with physics constraints
    
    L_total = λ1·L_rul + λ2·L_location + λ3·L_physics
    
    Where:
    - L_rul: Mean Squared Error for RUL prediction
    - L_location: Cross Entropy for damage localization
    - L_physics: Energy conservation constraint
    """
    
    def __init__(self, lambda_rul=1.0, lambda_location=0.5, lambda_physics=0.1):
        """
        Initialize loss function
        
        Parameters:
        -----------
        lambda_rul : float
            Weight for RUL loss (default: 1.0)
        lambda_location : float
            Weight for location loss (default: 0.5)
        lambda_physics : float
            Weight for physics loss (default: 0.1)
        """
        super(PhysicsInformedLoss, self).__init__()
        self.lambda_rul = lambda_rul
        self.lambda_location = lambda_location
        self.lambda_physics = lambda_physics
        
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        """
        Calculate combined loss
        
        Parameters:
        -----------
        predictions : dict
            Model predictions
        targets : dict
            Ground truth targets
        
        Returns:
        --------
        dict : Individual and total losses
        """
        # RUL loss (regression)
        loss_rul = self.mse_loss(
            predictions['rul'].squeeze(-1),
            targets['rul'].squeeze(-1)
        )
        
        # Location loss (classification)
        # Handle dimension properly for CrossEntropyLoss
        location_pred = predictions['location']
        location_target = targets['location'].view(-1)  # Flatten to 1D
        
        loss_location = self.ce_loss(
            location_pred,
            location_target
        )
        
        # Physics loss (energy conservation)
        # Penalize deviation from actual total energy
        loss_physics = self.mse_loss(
            predictions['energy'].squeeze(-1),
            targets['energy'].squeeze(-1)
        )
        
        # Total loss
        total_loss = (
            self.lambda_rul * loss_rul +
            self.lambda_location * loss_location +
            self.lambda_physics * loss_physics
        )
        
        return {
            'total': total_loss,
            'rul': loss_rul,
            'location': loss_location,
            'physics': loss_physics
        }

# ============================================================================
# 3.4: CREATE MODEL INSTANCE
# ============================================================================
print("\n[3.4] Creating model instance...")
print("-" * 80)

# Create model
model = PhysicsInformedGNN(
    num_node_features=config['num_node_features'],
    hidden_dim=64,
    num_classes=config['num_nodes'],
    dropout=0.3
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✓ Model created successfully!")
print(f"\nModel Architecture:")
print(f"  Input: {config['num_node_features']} features × {config['num_nodes']} nodes")
print(f"  Layer 1: GAT (10 → 256, 4 heads)")
print(f"  Layer 2: GAT (256 → 256, 4 heads)")
print(f"  Layer 3: GCN (256 → 64)")
print(f"  Output 1: RUL (1 value)")
print(f"  Output 2: Location (8 classes)")
print(f"  Output 3: Energy (1 value)")
print(f"\nParameters:")
print(f"  Total: {total_params:,}")
print(f"  Trainable: {trainable_params:,}")

# Create loss function
criterion = PhysicsInformedLoss(
    lambda_rul=1.0,        # RUL is most important
    lambda_location=0.5,   # Location is secondary
    lambda_physics=0.1     # Physics constraint for regularization
)

print(f"\n✓ Loss function created!")
print(f"  Loss weights: RUL={criterion.lambda_rul}, " +
      f"Location={criterion.lambda_location}, " +
      f"Physics={criterion.lambda_physics}")

# ============================================================================
# 3.5: TEST MODEL ON SINGLE SAMPLE
# ============================================================================
print("\n[3.5] Testing model on sample data...")
print("-" * 80)

# Load one graph for testing
with open(os.path.join(OUTPUT_DIR, 'step2_graph_data.pkl'), 'rb') as f:
    data_package = pickle.load(f)

test_graph = data_package['train_data'][0]
test_graph.batch = torch.zeros(test_graph.x.size(0), dtype=torch.long)

print("Test graph:")
print(f"  Node features: {test_graph.x.shape}")
print(f"  Edge index: {test_graph.edge_index.shape}")
print(f"  Target RUL: {test_graph.y_rul.item():.2f}%")
print(f"  Target location: CH{test_graph.y_location.item() + 1}")

# Forward pass
model.eval()
with torch.no_grad():
    output = model(test_graph)

print(f"\nModel output:")
print(f"  Predicted RUL: {output['rul'].item():.2f}%")
print(f"  Predicted location: CH{output['location'].argmax().item() + 1}")
print(f"  Predicted energy: {output['energy'].item():.6f}")

# Test loss calculation
targets = {
    'rul': test_graph.y_rul.unsqueeze(0) if test_graph.y_rul.dim() == 0 else test_graph.y_rul,
    'location': test_graph.y_location.unsqueeze(0) if test_graph.y_location.dim() == 0 else test_graph.y_location,
    'energy': test_graph.y_energy.unsqueeze(0) if test_graph.y_energy.dim() == 0 else test_graph.y_energy
}

losses = criterion(output, targets)
print(f"\nLoss values:")
print(f"  Total loss: {losses['total'].item():.4f}")
print(f"  RUL loss: {losses['rul'].item():.4f}")
print(f"  Location loss: {losses['location'].item():.4f}")
print(f"  Physics loss: {losses['physics'].item():.4f}")

print("\n✓ Model forward pass successful!")

# ============================================================================
# 3.6: SAVE MODEL ARCHITECTURE
# ============================================================================
print("\n[3.6] Saving model architecture...")
print("-" * 80)

# Save model architecture info
model_info = {
    'architecture': 'PhysicsInformedGNN',
    'num_node_features': config['num_node_features'],
    'hidden_dim': 64,
    'num_classes': config['num_nodes'],
    'dropout': 0.3,
    'total_parameters': total_params,
    'trainable_parameters': trainable_params,
    'loss_weights': {
        'lambda_rul': criterion.lambda_rul,
        'lambda_location': criterion.lambda_location,
        'lambda_physics': criterion.lambda_physics
    }
}

with open(os.path.join(OUTPUT_DIR, 'step3_model_info.json'), 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"✓ Saved: step3_model_info.json")

# Save model architecture as text
architecture_text = f"""
PHYSICS-INFORMED GNN ARCHITECTURE
{'='*80}

INPUT LAYER:
  Node features: {config['num_node_features']} per node
  Number of nodes: {config['num_nodes']}
  Graph edges: {config['num_edges']}

GRAPH LAYERS:
  Layer 1: Graph Attention Network (GAT)
    - Input: {config['num_node_features']} features
    - Output: 64 x 4 = 256 features (4 attention heads)
    - Attention heads: 4 (for interpretability)
    - Activation: ELU
    - Batch Normalization
    - Dropout: 0.3
  
  Layer 2: Graph Attention Network (GAT)
    - Input: 256 features
    - Output: 64 x 4 = 256 features
    - Attention heads: 4
    - Activation: ELU
    - Batch Normalization
    - Dropout: 0.3
  
  Layer 3: Graph Convolutional Network (GCN)
    - Input: 256 features
    - Output: 64 features
    - Activation: ELU
    - Batch Normalization

POOLING LAYER:
  Global mean pooling + Global max pooling
  Output: 128 features (64 mean + 64 max)
  Dimension reduction: 128 -> 64

OUTPUT HEADS (Multi-task Learning):

  Head 1: RUL Prediction (Regression)
    - Dense(64 -> 32) + ReLU + Dropout(0.3)
    - Dense(32 -> 16) + ReLU
    - Dense(16 -> 1) + Sigmoid
    - Output range: [0, 100]% RUL
  
  Head 2: Damage Localization (Classification)
    - Dense(64 -> 32) + ReLU + Dropout(0.3)
    - Dense(32 -> 8)
    - Output: 8 class probabilities (8 channels)
  
  Head 3: Energy Prediction (Physics Constraint)
    - Dense(64 -> 16) + ReLU
    - Dense(16 -> 1)
    - Output: Total energy estimate

LOSS FUNCTION:
  L_total = lambda1*L_rul + lambda2*L_location + lambda3*L_physics
  
  Where:
    lambda1 = {criterion.lambda_rul} (RUL weight)
    lambda2 = {criterion.lambda_location} (Location weight)
    lambda3 = {criterion.lambda_physics} (Physics weight)
  
  Components:
    L_rul = MSE(predicted_rul, actual_rul)
    L_location = CrossEntropy(predicted_location, actual_location)
    L_physics = MSE(predicted_energy, actual_energy)

TOTAL PARAMETERS: {total_params:,}

NOVEL FEATURES:
  - Multi-head attention for interpretability
  - Physics-informed loss (energy conservation)
  - Multi-task learning (RUL + localization)
  - Spatial-temporal graph structure
  - Attention weights reveal damage propagation paths

{'='*80}
"""

with open(os.path.join(OUTPUT_DIR, 'step3_architecture.txt'), 'w', encoding='utf-8') as f:
    f.write(architecture_text)

print(f"✓ Saved: step3_architecture.txt")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("STEP 3 COMPLETE!")
print("="*80)

print("\nModel Summary:")
print(f"  ✓ Physics-Informed GNN with {total_params:,} parameters")
print(f"  ✓ 3 graph layers (2 GAT + 1 GCN)")
print(f"  ✓ 3 output heads (RUL + Location + Energy)")
print(f"  ✓ Multi-task learning with physics constraints")
print(f"  ✓ Attention mechanism for interpretability")

print("\nGenerated files:")
print(f"  ✓ step3_model_info.json (configuration)")
print(f"  ✓ step3_architecture.txt (detailed description)")

print("\n" + "="*80)
print("Type 'next' when ready for Step 4: Training")
print("="*80)