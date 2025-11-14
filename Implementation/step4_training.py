"""
STEP 4: Model Training
======================
Trains the Physics-Informed GNN with proper validation and monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
import time
from tqdm import tqdm

# Import model from step 3
import sys
sys.path.append(os.path.dirname(__file__))

OUTPUT_DIR = r'F:\concrete data\test 3\gnn_implementation'

print("="*80)
print("STEP 4: MODEL TRAINING")
print("="*80)

# ============================================================================
# 4.1: LOAD DATA AND MODEL
# ============================================================================
print("\n[4.1] Loading data and model...")
print("-" * 80)

# Load graph data
with open(os.path.join(OUTPUT_DIR, 'step2_graph_data.pkl'), 'rb') as f:
    data_package = pickle.load(f)

train_data = data_package['train_data']
val_data = data_package['val_data']
test_data = data_package['test_data']

print(f"✓ Loaded datasets:")
print(f"  Train: {len(train_data)} graphs")
print(f"  Val:   {len(val_data)} graphs")
print(f"  Test:  {len(test_data)} graphs")

# Load configuration
with open(os.path.join(OUTPUT_DIR, 'step2_config.json'), 'r') as f:
    config = json.load(f)

# Recreate model (copy from step3_model.py)
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool

class PhysicsInformedGNN(nn.Module):
    def __init__(self, num_node_features=10, hidden_dim=64, num_classes=8, dropout=0.3):
        super(PhysicsInformedGNN, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.gat1 = GATConv(num_node_features, hidden_dim, heads=4, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True, dropout=dropout)
        self.gcn = GCNConv(hidden_dim * 4, hidden_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim * 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.reduce_dim = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.rul_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
        
        self.location_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        self.attention_weights = None
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x, (edge_index_att1, alpha1) = self.gat1(x, edge_index, return_attention_weights=True)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x, (edge_index_att2, alpha2) = self.gat2(x, edge_index, return_attention_weights=True)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.gcn(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_global = torch.cat([x_mean, x_max], dim=1)
        x_global = self.reduce_dim(x_global)
        
        rul_pred = self.rul_head(x_global) * 100
        location_pred = self.location_head(x_global)
        energy_pred = self.energy_head(x_global)
        
        self.attention_weights = {'layer1': (edge_index_att1, alpha1), 'layer2': (edge_index_att2, alpha2)}
        
        return {'rul': rul_pred, 'location': location_pred, 'energy': energy_pred,
                'node_embeddings': x, 'graph_embedding': x_global}

class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambda_rul=1.0, lambda_location=0.5, lambda_physics=0.1):
        super(PhysicsInformedLoss, self).__init__()
        self.lambda_rul = lambda_rul
        self.lambda_location = lambda_location
        self.lambda_physics = lambda_physics
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        loss_rul = self.mse_loss(predictions['rul'].squeeze(-1), targets['rul'].squeeze(-1))
        location_pred = predictions['location']
        location_target = targets['location'].view(-1)
        loss_location = self.ce_loss(location_pred, location_target)
        loss_physics = self.mse_loss(predictions['energy'].squeeze(-1), targets['energy'].squeeze(-1))
        
        total_loss = (self.lambda_rul * loss_rul + 
                     self.lambda_location * loss_location + 
                     self.lambda_physics * loss_physics)
        
        return {'total': total_loss, 'rul': loss_rul, 'location': loss_location, 'physics': loss_physics}

# Create model
model = PhysicsInformedGNN(
    num_node_features=config['num_node_features'],
    hidden_dim=64,
    num_classes=config['num_nodes'],
    dropout=0.3
)

criterion = PhysicsInformedLoss(lambda_rul=1.0, lambda_location=0.5, lambda_physics=0.1)

print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

# ============================================================================
# 4.2: CREATE DATA LOADERS
# ============================================================================
print("\n[4.2] Creating data loaders...")
print("-" * 80)

BATCH_SIZE = 32

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

print(f"✓ Data loaders created:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")

# ============================================================================
# 4.3: SETUP OPTIMIZER AND SCHEDULER
# ============================================================================
print("\n[4.3] Setting up optimizer and scheduler...")
print("-" * 80)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

print(f"✓ Optimizer: AdamW (lr=0.001, weight_decay=1e-5)")
print(f"✓ Scheduler: ReduceLROnPlateau (factor=0.5, patience=10)")

# ============================================================================
# 4.4: TRAINING FUNCTIONS
# ============================================================================
print("\n[4.4] Defining training functions...")
print("-" * 80)

def train_epoch(model, loader, criterion, optimizer):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    rul_loss = 0
    location_loss = 0
    physics_loss = 0
    
    for batch in loader:
        optimizer.zero_grad()
        
        output = model(batch)
        
        targets = {
            'rul': batch.y_rul,
            'location': batch.y_location,
            'energy': batch.y_energy
        }
        
        losses = criterion(output, targets)
        losses['total'].backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += losses['total'].item()
        rul_loss += losses['rul'].item()
        location_loss += losses['location'].item()
        physics_loss += losses['physics'].item()
    
    n_batches = len(loader)
    return {
        'total': total_loss / n_batches,
        'rul': rul_loss / n_batches,
        'location': location_loss / n_batches,
        'physics': physics_loss / n_batches
    }

def validate(model, loader, criterion):
    """Validate model"""
    model.eval()
    total_loss = 0
    rul_loss = 0
    location_loss = 0
    physics_loss = 0
    
    rul_preds = []
    rul_targets = []
    location_preds = []
    location_targets = []
    
    with torch.no_grad():
        for batch in loader:
            output = model(batch)
            
            targets = {
                'rul': batch.y_rul,
                'location': batch.y_location,
                'energy': batch.y_energy
            }
            
            losses = criterion(output, targets)
            
            total_loss += losses['total'].item()
            rul_loss += losses['rul'].item()
            location_loss += losses['location'].item()
            physics_loss += losses['physics'].item()
            
            rul_preds.extend(output['rul'].cpu().numpy())
            rul_targets.extend(batch.y_rul.cpu().numpy())
            location_preds.extend(output['location'].argmax(dim=1).cpu().numpy())
            location_targets.extend(batch.y_location.cpu().numpy())
    
    n_batches = len(loader)
    
    # Calculate metrics
    rul_preds = np.array(rul_preds)
    rul_targets = np.array(rul_targets)
    r2 = 1 - np.sum((rul_targets - rul_preds)**2) / np.sum((rul_targets - rul_targets.mean())**2)
    rmse = np.sqrt(np.mean((rul_targets - rul_preds)**2))
    mae = np.mean(np.abs(rul_targets - rul_preds))
    
    location_preds = np.array(location_preds)
    location_targets = np.array(location_targets)
    accuracy = np.mean(location_preds == location_targets)
    
    return {
        'loss': {
            'total': total_loss / n_batches,
            'rul': rul_loss / n_batches,
            'location': location_loss / n_batches,
            'physics': physics_loss / n_batches
        },
        'metrics': {
            'rul_r2': r2,
            'rul_rmse': rmse,
            'rul_mae': mae,
            'location_acc': accuracy
        }
    }

print("✓ Training functions defined")

# ============================================================================
# 4.5: TRAINING LOOP
# ============================================================================
print("\n[4.5] Starting training...")
print("="*80)

NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20

history = {
    'train_loss': [], 'val_loss': [],
    'train_rul_loss': [], 'val_rul_loss': [],
    'train_location_loss': [], 'val_location_loss': [],
    'train_physics_loss': [], 'val_physics_loss': [],
    'val_rul_r2': [], 'val_rul_rmse': [], 'val_rul_mae': [],
    'val_location_acc': [],
    'lr': []
}

best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

print(f"\nTraining for {NUM_EPOCHS} epochs...")
print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
print("-"*80)

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    
    # Train
    train_losses = train_epoch(model, train_loader, criterion, optimizer)
    
    # Validate
    val_results = validate(model, val_loader, criterion)
    val_losses = val_results['loss']
    val_metrics = val_results['metrics']
    
    # Update history
    history['train_loss'].append(train_losses['total'])
    history['val_loss'].append(val_losses['total'])
    history['train_rul_loss'].append(train_losses['rul'])
    history['val_rul_loss'].append(val_losses['rul'])
    history['train_location_loss'].append(train_losses['location'])
    history['val_location_loss'].append(val_losses['location'])
    history['train_physics_loss'].append(train_losses['physics'])
    history['val_physics_loss'].append(val_losses['physics'])
    history['val_rul_r2'].append(val_metrics['rul_r2'])
    history['val_rul_rmse'].append(val_metrics['rul_rmse'])
    history['val_rul_mae'].append(val_metrics['rul_mae'])
    history['val_location_acc'].append(val_metrics['location_acc'])
    history['lr'].append(optimizer.param_groups[0]['lr'])
    
    # Learning rate scheduling
    scheduler.step(val_losses['total'])
    
    # Early stopping
    if val_losses['total'] < best_val_loss:
        best_val_loss = val_losses['total']
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
    
    epoch_time = time.time() - epoch_start
    
    # Print progress
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_losses['total']:.4f} | "
              f"Val Loss: {val_losses['total']:.4f} | "
              f"Val R²: {val_metrics['rul_r2']:.4f} | "
              f"Val Acc: {100*val_metrics['location_acc']:.1f}% | "
              f"Time: {epoch_time:.1f}s")
    
    # Early stopping check
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        break

total_time = time.time() - start_time

print("-"*80)
print(f"\n✓ Training complete!")
print(f"  Total time: {total_time/60:.1f} minutes")
print(f"  Best validation loss: {best_val_loss:.4f}")
print(f"  Final validation R²: {history['val_rul_r2'][-1]:.4f}")
print(f"  Final validation Accuracy: {100*history['val_location_acc'][-1]:.1f}%")

# Load best model
model.load_state_dict(best_model_state)

# ============================================================================
# 4.6: SAVE RESULTS
# ============================================================================
print("\n[4.6] Saving training results...")
print("-" * 80)

# Save model
torch.save({
    'model_state_dict': best_model_state,
    'optimizer_state_dict': optimizer.state_dict(),
    'history': history,
    'config': config
}, os.path.join(OUTPUT_DIR, 'step4_trained_model.pth'))
print("✓ Saved: step4_trained_model.pth")

# Save history
with open(os.path.join(OUTPUT_DIR, 'step4_training_history.pkl'), 'wb') as f:
    pickle.dump(history, f)
print("✓ Saved: step4_training_history.pkl")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("STEP 4 COMPLETE!")
print("="*80)

print("\nTraining Summary:")
print(f"  ✓ Trained for {len(history['train_loss'])} epochs")
print(f"  ✓ Best val loss: {best_val_loss:.4f}")
print(f"  ✓ Final R²: {history['val_rul_r2'][-1]:.4f}")
print(f"  ✓ Final Accuracy: {100*history['val_location_acc'][-1]:.1f}%")
print(f"  ✓ Training time: {total_time/60:.1f} minutes")

print("\nGenerated files:")
print(f"  ✓ step4_trained_model.pth (trained weights)")
print(f"  ✓ step4_training_history.pkl (training curves)")

print("\n" + "="*80)
print("Type 'next' when ready for Step 5: Evaluation & Visualization")
print("="*80)