"""
STEP 5: Evaluation, Visualization & Diagnosis
==============================================
Comprehensive evaluation and publication-quality figures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os
from sklearn.metrics import confusion_matrix, classification_report, r2_score

OUTPUT_DIR = r'F:\concrete data\test 3\gnn_implementation'
os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)

print("="*80)
print("STEP 5: EVALUATION & VISUALIZATION")
print("="*80)

# ============================================================================
# 5.1: LOAD MODEL AND DATA
# ============================================================================
print("\n[5.1] Loading trained model and data...")
print("-" * 80)

# Recreate model architecture
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
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )
        self.location_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        self.energy_head = nn.Sequential(nn.Linear(hidden_dim, 16), nn.ReLU(), nn.Linear(16, 1))
        self.attention_weights = None
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x, _ = self.gat1(x, edge_index, return_attention_weights=True)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x, _ = self.gat2(x, edge_index, return_attention_weights=True)
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
        return {
            'rul': self.rul_head(x_global) * 100,
            'location': self.location_head(x_global),
            'energy': self.energy_head(x_global)
        }

# Load checkpoint
checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'step4_trained_model.pth'), 
                       weights_only=False)  # Safe since we created this file
config = checkpoint['config']
history = checkpoint['history']

model = PhysicsInformedGNN(
    num_node_features=config['num_node_features'],
    hidden_dim=64,
    num_classes=config['num_nodes'],
    dropout=0.3
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✓ Model loaded successfully")

# Load data
with open(os.path.join(OUTPUT_DIR, 'step2_graph_data.pkl'), 'rb') as f:
    data_package = pickle.load(f)

train_data = data_package['train_data']
val_data = data_package['val_data']
test_data = data_package['test_data']

train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"✓ Data loaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

# ============================================================================
# 5.2: COLLECT PREDICTIONS
# ============================================================================
print("\n[5.2] Collecting predictions on all splits...")
print("-" * 80)

def collect_predictions(model, loader):
    """Collect all predictions and targets"""
    model.eval()
    all_preds = {'rul': [], 'location': [], 'energy': []}
    all_targets = {'rul': [], 'location': [], 'energy': []}
    
    with torch.no_grad():
        for batch in loader:
            output = model(batch)
            all_preds['rul'].extend(output['rul'].cpu().numpy())
            all_preds['location'].extend(output['location'].argmax(dim=1).cpu().numpy())
            all_preds['energy'].extend(output['energy'].cpu().numpy())
            all_targets['rul'].extend(batch.y_rul.cpu().numpy())
            all_targets['location'].extend(batch.y_location.cpu().numpy())
            all_targets['energy'].extend(batch.y_energy.cpu().numpy())
    
    return {k: np.array(v) for k, v in all_preds.items()}, \
           {k: np.array(v) for k, v in all_targets.items()}

train_preds, train_targets = collect_predictions(model, train_loader)
val_preds, val_targets = collect_predictions(model, val_loader)
test_preds, test_targets = collect_predictions(model, test_loader)

print("✓ Predictions collected for all splits")

# ============================================================================
# 5.3: CALCULATE METRICS
# ============================================================================
print("\n[5.3] Calculating metrics...")
print("-" * 80)

def calculate_metrics(preds, targets):
    """Calculate comprehensive metrics"""
    # RUL metrics
    rul_r2 = r2_score(targets['rul'], preds['rul'])
    rul_rmse = np.sqrt(np.mean((targets['rul'] - preds['rul'])**2))
    rul_mae = np.mean(np.abs(targets['rul'] - preds['rul']))
    rul_mape = np.mean(np.abs((targets['rul'] - preds['rul']) / (targets['rul'] + 1e-8))) * 100
    
    # Location metrics
    loc_acc = np.mean(preds['location'] == targets['location'])
    
    return {
        'rul_r2': rul_r2,
        'rul_rmse': rul_rmse,
        'rul_mae': rul_mae,
        'rul_mape': rul_mape,
        'location_acc': loc_acc
    }

train_metrics = calculate_metrics(train_preds, train_targets)
val_metrics = calculate_metrics(val_preds, val_targets)
test_metrics = calculate_metrics(test_preds, test_targets)

print("\nTrain Metrics:")
print(f"  RUL R²:    {train_metrics['rul_r2']:.4f}")
print(f"  RUL RMSE:  {train_metrics['rul_rmse']:.2f}%")
print(f"  RUL MAE:   {train_metrics['rul_mae']:.2f}%")
print(f"  Location:  {100*train_metrics['location_acc']:.1f}%")

print("\nValidation Metrics:")
print(f"  RUL R²:    {val_metrics['rul_r2']:.4f}")
print(f"  RUL RMSE:  {val_metrics['rul_rmse']:.2f}%")
print(f"  RUL MAE:   {val_metrics['rul_mae']:.2f}%")
print(f"  Location:  {100*val_metrics['location_acc']:.1f}%")

print("\nTest Metrics:")
print(f"  RUL R²:    {test_metrics['rul_r2']:.4f}")
print(f"  RUL RMSE:  {test_metrics['rul_rmse']:.2f}%")
print(f"  RUL MAE:   {test_metrics['rul_mae']:.2f}%")
print(f"  Location:  {100*test_metrics['location_acc']:.1f}%")

# ============================================================================
# 5.4: DIAGNOSIS - WHAT WENT WRONG?
# ============================================================================
print("\n[5.4] Diagnosis - Analyzing issues...")
print("-" * 80)

print("\n🔍 DIAGNOSIS:")
print("-" * 80)

# Check 1: RUL prediction range
print(f"\n1. RUL Prediction Range:")
print(f"   Target range: [{train_targets['rul'].min():.1f}, {train_targets['rul'].max():.1f}]")
print(f"   Predicted range: [{train_preds['rul'].min():.1f}, {train_preds['rul'].max():.1f}]")
rul_std = train_preds['rul'].std()
print(f"   Prediction std: {rul_std:.2f}")
if rul_std < 5:
    print("   ⚠️ ISSUE: Model predictions have very low variance!")
    print("      → Model is predicting nearly constant values")

# Check 2: Location distribution
print(f"\n2. Location Prediction Distribution:")
unique_preds = np.unique(test_preds['location'])
print(f"   Unique predictions: {unique_preds}")
print(f"   Most predicted: CH{np.bincount(test_preds['location'].astype(int)).argmax() + 1}")
if len(unique_preds) < 3:
    print("   ⚠️ ISSUE: Model only predicts 1-2 classes!")
    print("      → Class imbalance problem")

# Check 3: Feature scale
print(f"\n3. Input Feature Statistics:")
sample_batch = next(iter(train_loader))
feature_means = sample_batch.x.mean(dim=0)
feature_stds = sample_batch.x.std(dim=0)
print(f"   Feature mean range: [{feature_means.min():.6f}, {feature_means.max():.6f}]")
print(f"   Feature std range: [{feature_stds.min():.6f}, {feature_stds.max():.6f}]")
if feature_means.max() > 1e3 or feature_means.min() < -1e3:
    print("   ⚠️ ISSUE: Features are not scaled!")
    print("      → Need feature normalization")

print("\n" + "="*80)
print("RECOMMENDATIONS:")
print("="*80)
print("1. ✓ Add feature standardization (StandardScaler)")
print("2. ✓ Balance class weights for location prediction")
print("3. ✓ Try different loss weights (increase RUL weight)")
print("4. ✓ Increase model capacity or reduce dropout")
print("5. ✓ Train longer with lower learning rate")

# ============================================================================
# 5.5: VISUALIZATIONS
# ============================================================================
print("\n[5.5] Creating visualizations...")
print("-" * 80)

# Figure 1: Training History
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Loss curves
ax = axes[0, 0]
ax.plot(history['train_loss'], label='Train', linewidth=2)
ax.plot(history['val_loss'], label='Validation', linewidth=2)
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('Total Loss', fontweight='bold')
ax.set_title('(a) Total Loss', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# RUL loss
ax = axes[0, 1]
ax.plot(history['train_rul_loss'], label='Train', linewidth=2)
ax.plot(history['val_rul_loss'], label='Validation', linewidth=2)
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('RUL Loss', fontweight='bold')
ax.set_title('(b) RUL Loss', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Location loss
ax = axes[0, 2]
ax.plot(history['train_location_loss'], label='Train', linewidth=2)
ax.plot(history['val_location_loss'], label='Validation', linewidth=2)
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('Location Loss', fontweight='bold')
ax.set_title('(c) Location Loss', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# R² over time
ax = axes[1, 0]
ax.plot(history['val_rul_r2'], linewidth=2, color='green')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('R² Score', fontweight='bold')
ax.set_title('(d) Validation R²', fontweight='bold')
ax.grid(True, alpha=0.3)

# Accuracy over time
ax = axes[1, 1]
ax.plot([100*x for x in history['val_location_acc']], linewidth=2, color='purple')
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontweight='bold')
ax.set_title('(e) Validation Location Accuracy', fontweight='bold')
ax.grid(True, alpha=0.3)

# Learning rate
ax = axes[1, 2]
ax.plot(history['lr'], linewidth=2, color='orange')
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('Learning Rate', fontweight='bold')
ax.set_title('(f) Learning Rate Schedule', fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'Fig1_Training_History.png'), 
           dpi=300, bbox_inches='tight')
print("✓ Saved: Fig1_Training_History.png")
plt.close()

# Figure 2: RUL Predictions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

splits_data = [
    (train_preds, train_targets, 'Train'),
    (val_preds, val_targets, 'Validation'),
    (test_preds, test_targets, 'Test')
]

for idx, (preds, targets, split_name) in enumerate(splits_data[:3]):
    ax = axes[idx // 2, idx % 2]
    
    # Ensure arrays are 1D
    pred_rul = preds['rul'].flatten()
    target_rul = targets['rul'].flatten()
    
    ax.scatter(target_rul, pred_rul, alpha=0.5, s=30, 
              edgecolors='black', linewidth=0.5)
    ax.plot([0, 100], [0, 100], 'r--', lw=2, label='Perfect')
    metrics = calculate_metrics(preds, targets)
    ax.set_xlabel('Actual RUL (%)', fontweight='bold')
    ax.set_ylabel('Predicted RUL (%)', fontweight='bold')
    ax.set_title(f'{split_name} Set\nR²={metrics["rul_r2"]:.3f}, MAE={metrics["rul_mae"]:.1f}%',
                fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Residuals
ax = axes[1, 1]
pred_rul = test_preds['rul'].flatten()
target_rul = test_targets['rul'].flatten()
residuals = target_rul - pred_rul

ax.scatter(pred_rul, residuals, alpha=0.5, s=30,
          edgecolors='black', linewidth=0.5)
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Predicted RUL (%)', fontweight='bold')
ax.set_ylabel('Residual (%)', fontweight='bold')
ax.set_title('Test Set Residuals', fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'Fig2_RUL_Predictions.png'),
           dpi=300, bbox_inches='tight')
print("✓ Saved: Fig2_RUL_Predictions.png")
plt.close()

# Figure 3: Confusion Matrix
cm = confusion_matrix(test_targets['location'], test_preds['location'])

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
           xticklabels=[f'CH{i}' for i in range(1, 9)],
           yticklabels=[f'CH{i}' for i in range(1, 9)])
ax.set_xlabel('Predicted Channel', fontweight='bold', fontsize=12)
ax.set_ylabel('Actual Channel', fontweight='bold', fontsize=12)
ax.set_title(f'Damage Location Confusion Matrix\nAccuracy: {100*test_metrics["location_acc"]:.1f}%',
            fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'Fig3_Confusion_Matrix.png'),
           dpi=300, bbox_inches='tight')
print("✓ Saved: Fig3_Confusion_Matrix.png")
plt.close()

# Figure 4: Timeline Comparison
fig, ax = plt.subplots(figsize=(16, 6))

# Combine all splits in order
all_targets = np.concatenate([train_targets['rul'], val_targets['rul'], test_targets['rul']])
all_preds = np.concatenate([train_preds['rul'], val_preds['rul'], test_preds['rul']])
segments = range(len(all_targets))

ax.plot(segments, all_targets, 'o-', label='Actual RUL', 
       linewidth=2, markersize=3, color='blue')
ax.plot(segments, all_preds, 's-', label='Predicted RUL',
       linewidth=2, markersize=3, alpha=0.7, color='red')

# Mark split points
train_end = len(train_data)
val_end = train_end + len(val_data)
ax.axvline(x=train_end, color='green', linestyle='--', linewidth=2, label='Train/Val Split')
ax.axvline(x=val_end, color='orange', linestyle='--', linewidth=2, label='Val/Test Split')

ax.set_xlabel('Segment Number', fontweight='bold', fontsize=12)
ax.set_ylabel('RUL (%)', fontweight='bold', fontsize=12)
ax.set_title('RUL Prediction Timeline', fontweight='bold', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'Fig4_Timeline.png'),
           dpi=300, bbox_inches='tight')
print("✓ Saved: Fig4_Timeline.png")
plt.close()

# ============================================================================
# 5.6: SAVE EVALUATION REPORT
# ============================================================================
print("\n[5.6] Generating evaluation report...")
print("-" * 80)

report = f"""
EVALUATION REPORT - PHYSICS-INFORMED GNN
{'='*80}

TRAINING SUMMARY:
  Epochs trained: {len(history['train_loss'])}
  Training time: {checkpoint.get('training_time', 'N/A')}
  Best validation loss: {min(history['val_loss']):.4f}

PERFORMANCE METRICS:

Train Set:
  RUL R²:        {train_metrics['rul_r2']:.4f}
  RUL RMSE:      {train_metrics['rul_rmse']:.2f}%
  RUL MAE:       {train_metrics['rul_mae']:.2f}%
  RUL MAPE:      {train_metrics['rul_mape']:.2f}%
  Location Acc:  {100*train_metrics['location_acc']:.1f}%

Validation Set:
  RUL R²:        {val_metrics['rul_r2']:.4f}
  RUL RMSE:      {val_metrics['rul_rmse']:.2f}%
  RUL MAE:       {val_metrics['rul_mae']:.2f}%
  RUL MAPE:      {val_metrics['rul_mape']:.2f}%
  Location Acc:  {100*val_metrics['location_acc']:.1f}%

Test Set:
  RUL R²:        {test_metrics['rul_r2']:.4f}
  RUL RMSE:      {test_metrics['rul_rmse']:.2f}%
  RUL MAE:       {test_metrics['rul_mae']:.2f}%
  RUL MAPE:      {test_metrics['rul_mape']:.2f}%
  Location Acc:  {100*test_metrics['location_acc']:.1f}%

DIAGNOSIS:
  Issue 1: Negative R² indicates poor RUL prediction
  Issue 2: Low prediction variance (model predicts similar values)
  Issue 3: Features may need standardization
  Issue 4: Class imbalance in location prediction

RECOMMENDATIONS FOR IMPROVEMENT:
  1. Add StandardScaler for feature normalization
  2. Use class weights for balanced location prediction
  3. Increase RUL loss weight (try lambda_rul=5.0)
  4. Reduce dropout rate (try 0.1 instead of 0.3)
  5. Train with smaller learning rate for longer
  6. Consider removing physics loss initially
  7. Add batch normalization before output heads

NEXT STEPS:
  → Implement feature scaling
  → Retrain with adjusted hyperparameters
  → Compare with baseline models (RF, XGBoost)
  → Publish results if R² > 0.85

{'='*80}
"""

with open(os.path.join(OUTPUT_DIR, 'step5_evaluation_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print("\n✓ Saved: step5_evaluation_report.txt")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("STEP 5 COMPLETE!")
print("="*80)

print("\nGenerated Figures:")
print("  ✓ Fig1_Training_History.png")
print("  ✓ Fig2_RUL_Predictions.png")
print("  ✓ Fig3_Confusion_Matrix.png")
print("  ✓ Fig4_Timeline.png")
print("  ✓ step5_evaluation_report.txt")

print("\nAll files in:", os.path.join(OUTPUT_DIR, 'figures'))

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nThe model needs improvements, but the pipeline is fully functional!")
print("See evaluation report for specific recommendations.")
print("\nTo improve: Re-run with feature scaling and adjusted hyperparameters")
print("="*80)