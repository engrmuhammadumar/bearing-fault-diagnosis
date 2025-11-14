"""
STEP 2: Data Loading & Graph Construction
==========================================
Loads concrete AE data and constructs sensor network graph
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import torch
from torch_geometric.data import Data
import pickle
import json
import os

# Configuration
FILE_PATH = r'F:\concrete data\test 3\per_file_features_800.csv'
OUTPUT_DIR = r'F:\concrete data\test 3\gnn_implementation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("STEP 2: DATA LOADING & GRAPH CONSTRUCTION")
print("="*80)

# ============================================================================
# 2.1: LOAD DATA
# ============================================================================
print("\n[2.1] Loading concrete failure data...")
print("-" * 80)

df = pd.read_csv(FILE_PATH)
print(f"✓ Loaded: {df.shape[0]} segments × {df.shape[1]} features")

# Display sample
print("\nData preview (first 5 rows, first 10 columns):")
print(df.iloc[:5, :10])

# Check for filename column
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"\n✓ Categorical columns: {categorical_cols}")

# ============================================================================
# 2.2: DEFINE SENSOR NETWORK TOPOLOGY
# ============================================================================
print("\n[2.2] Defining sensor network topology...")
print("-" * 80)

# Define 8-channel sensor positions (4×2 grid layout)
# Adjust these if you know actual positions
sensor_positions = {
    'ch1': np.array([0.00, 0.00]),  # Bottom-left
    'ch2': np.array([0.10, 0.00]),  # 10cm spacing
    'ch3': np.array([0.20, 0.00]),
    'ch4': np.array([0.30, 0.00]),  # Bottom-right
    'ch5': np.array([0.00, 0.10]),  # Top-left
    'ch6': np.array([0.10, 0.10]),
    'ch7': np.array([0.20, 0.10]),
    'ch8': np.array([0.30, 0.10])   # Top-right
}

print("Sensor Layout (4×2 grid, units: meters):")
print("\n  CH5(0.00,0.10)  CH6(0.10,0.10)  CH7(0.20,0.10)  CH8(0.30,0.10)")
print("       │               │               │               │")
print("  CH1(0.00,0.00)  CH2(0.10,0.00)  CH3(0.20,0.00)  CH4(0.30,0.00)")

channels = list(sensor_positions.keys())
positions = np.array(list(sensor_positions.values()))

print(f"\n✓ {len(channels)} sensors defined")

# ============================================================================
# 2.3: CREATE GRAPH STRUCTURE
# ============================================================================
print("\n[2.3] Creating graph adjacency matrix...")
print("-" * 80)

# Calculate distance matrix
distance_matrix = cdist(positions, positions)

print("Physical distance matrix (meters):")
print(pd.DataFrame(distance_matrix, 
                   index=[f'CH{i}' for i in range(1,9)],
                   columns=[f'CH{i}' for i in range(1,9)]).round(3))

# Create adjacency matrix (connect nearby sensors)
DISTANCE_THRESHOLD = 0.15  # 15cm - connect neighbors only
adjacency_matrix = (distance_matrix <= DISTANCE_THRESHOLD) & (distance_matrix > 0)
adjacency_matrix = adjacency_matrix.astype(float)

print(f"\n✓ Distance threshold: {DISTANCE_THRESHOLD}m")
print(f"✓ Graph edges: {int(adjacency_matrix.sum())} connections")
print(f"✓ Average node degree: {adjacency_matrix.sum() / 8:.1f}")

print("\nAdjacency matrix (1 = connected, 0 = not connected):")
print(pd.DataFrame(adjacency_matrix.astype(int),
                   index=[f'CH{i}' for i in range(1,9)],
                   columns=[f'CH{i}' for i in range(1,9)]))

# Convert to edge index for PyTorch Geometric
edge_list = []
for i in range(8):
    for j in range(8):
        if adjacency_matrix[i, j] > 0:
            edge_list.append([i, j])

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
print(f"\n✓ Edge index shape: {edge_index.shape}")
print(f"  Format: [2, {edge_index.shape[1]}] - PyTorch Geometric format")

# ============================================================================
# 2.4: SELECT FEATURES FOR NODES
# ============================================================================
print("\n[2.4] Selecting node features...")
print("-" * 80)

# Key features from each channel
feature_types = [
    'energy',           # Total energy
    'rms',             # Root mean square
    'peak_abs',        # Peak amplitude
    'line_length',     # Cumulative change
    'tkeo_energy',     # Teager-Kaiser energy
    'mean',            # Mean value
    'std',             # Standard deviation
    'dom_freq',        # Dominant frequency
    'spec_centroid',   # Spectral centroid
    'log_energy'       # Log energy
]

print(f"Selected {len(feature_types)} features per channel:")
for i, feat in enumerate(feature_types, 1):
    print(f"  {i}. {feat}")

# Check which features exist
available_features = []
for ch in channels:
    ch_features = []
    for feat in feature_types:
        col_name = f"{ch}_{feat}"
        if col_name in df.columns:
            ch_features.append(col_name)
        else:
            # Try without underscore
            alt_name = f"{ch}{feat}"
            if alt_name in df.columns:
                ch_features.append(alt_name)
    available_features.append(ch_features)

print(f"\n✓ Features found per channel: {len(available_features[0])}")

# ============================================================================
# 2.5: CREATE TARGETS
# ============================================================================
print("\n[2.5] Creating target variables...")
print("-" * 80)

# Target 1: RUL (Remaining Useful Life)
df['rul_percentage'] = 100 * (1 - df.index / (len(df) - 1))
print(f"✓ RUL range: [{df['rul_percentage'].min():.1f}%, {df['rul_percentage'].max():.1f}%]")

# Target 2: Damage Location (which channel has highest energy)
damage_locations = []
for idx in range(len(df)):
    energies = []
    for ch in channels:
        energy_col = f"{ch}_energy"
        if energy_col in df.columns:
            energies.append(df[energy_col].iloc[idx])
        else:
            energies.append(0.0)
    damage_locations.append(np.argmax(energies))

df['damage_location'] = damage_locations
print(f"✓ Damage locations: {df['damage_location'].nunique()} unique channels")
print(f"  Distribution:")
for loc in range(8):
    count = (df['damage_location'] == loc).sum()
    pct = 100 * count / len(df)
    print(f"    CH{loc+1}: {count} segments ({pct:.1f}%)")

# ============================================================================
# 2.6: BUILD GRAPH DATASET
# ============================================================================
print("\n[2.6] Building PyTorch Geometric dataset...")
print("-" * 80)

def extract_node_features(df_row, channels, feature_types):
    """Extract features for all 8 nodes from one time segment"""
    node_features = []
    for ch in channels:
        ch_features = []
        for feat in feature_types:
            col_name = f"{ch}_{feat}"
            if col_name in df.columns:
                value = df_row[col_name]
            else:
                value = 0.0
            ch_features.append(value)
        node_features.append(ch_features)
    return np.array(node_features, dtype=np.float32)

# Create graph for each time segment
graph_list = []
print("Creating graphs... ", end="", flush=True)

for idx in range(len(df)):
    # Node features (8 nodes × 10 features)
    x = extract_node_features(df.iloc[idx], channels, feature_types)
    x = torch.tensor(x, dtype=torch.float)
    
    # Targets
    rul = torch.tensor([df['rul_percentage'].iloc[idx]], dtype=torch.float)
    location = torch.tensor([df['damage_location'].iloc[idx]], dtype=torch.long)
    
    # Total energy (for physics constraint)
    total_energy = sum([df[f'ch{i}_energy'].iloc[idx] if f'ch{i}_energy' in df.columns 
                       else 0.0 for i in range(1, 9)])
    energy = torch.tensor([total_energy], dtype=torch.float)
    
    # Create PyG Data object
    graph_data = Data(
        x=x,                      # Node features [8, 10]
        edge_index=edge_index,    # Graph structure [2, num_edges]
        y_rul=rul,               # RUL target [1]
        y_location=location,      # Location target [1]
        y_energy=energy          # Total energy [1]
    )
    
    graph_list.append(graph_data)
    
    # Progress indicator
    if (idx + 1) % 100 == 0:
        print(f"{idx+1}...", end="", flush=True)

print(f"Done!")
print(f"\n✓ Created {len(graph_list)} graph samples")

# Show example graph
example_graph = graph_list[0]
print(f"\nExample graph structure:")
print(f"  Node features: {example_graph.x.shape}")
print(f"  Edge index: {example_graph.edge_index.shape}")
print(f"  RUL target: {example_graph.y_rul.item():.2f}%")
print(f"  Location target: CH{example_graph.y_location.item() + 1}")
print(f"  Total energy: {example_graph.y_energy.item():.6f}")

# ============================================================================
# 2.7: SPLIT DATA (TIME-SERIES SPLIT)
# ============================================================================
print("\n[2.7] Splitting dataset (time-series split)...")
print("-" * 80)

# 70% train, 15% validation, 15% test
train_size = int(0.70 * len(graph_list))
val_size = int(0.15 * len(graph_list))

train_data = graph_list[:train_size]
val_data = graph_list[train_size:train_size+val_size]
test_data = graph_list[train_size+val_size:]

print(f"Dataset split:")
print(f"  Train:      {len(train_data):4d} graphs (segments 0-{train_size-1})")
print(f"  Validation: {len(val_data):4d} graphs (segments {train_size}-{train_size+val_size-1})")
print(f"  Test:       {len(test_data):4d} graphs (segments {train_size+val_size}-{len(graph_list)-1})")
print(f"  Total:      {len(graph_list):4d} graphs")

# ============================================================================
# 2.8: SAVE PROCESSED DATA
# ============================================================================
print("\n[2.8] Saving processed data...")
print("-" * 80)

# Save graph data
data_package = {
    'train_data': train_data,
    'val_data': val_data,
    'test_data': test_data,
    'edge_index': edge_index,
    'sensor_positions': sensor_positions,
    'adjacency_matrix': adjacency_matrix,
    'feature_types': feature_types,
    'channels': channels
}

with open(os.path.join(OUTPUT_DIR, 'step2_graph_data.pkl'), 'wb') as f:
    pickle.dump(data_package, f)
print(f"✓ Saved: step2_graph_data.pkl ({len(graph_list)} graphs)")

# Save configuration
config = {
    'num_nodes': 8,
    'num_node_features': len(feature_types),
    'num_edges': edge_index.shape[1],
    'num_graphs': len(graph_list),
    'train_size': len(train_data),
    'val_size': len(val_data),
    'test_size': len(test_data),
    'distance_threshold': DISTANCE_THRESHOLD,
    'feature_types': feature_types,
    'sensor_positions': {k: v.tolist() for k, v in sensor_positions.items()}
}

with open(os.path.join(OUTPUT_DIR, 'step2_config.json'), 'w') as f:
    json.dump(config, f, indent=2)
print(f"✓ Saved: step2_config.json")

# ============================================================================
# 2.9: VISUALIZATION
# ============================================================================
print("\n[2.9] Creating visualizations...")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 2.9.1: Sensor Network Layout
ax = axes[0, 0]
for i, (ch, pos) in enumerate(sensor_positions.items()):
    ax.scatter(pos[0], pos[1], s=500, c='blue', alpha=0.7, 
              edgecolors='black', linewidth=2, zorder=3)
    ax.text(pos[0], pos[1], ch.upper(), ha='center', va='center',
           fontweight='bold', fontsize=11, color='white', zorder=4)

# Draw edges
for i in range(8):
    for j in range(i+1, 8):
        if adjacency_matrix[i, j] > 0:
            pos_i = list(sensor_positions.values())[i]
            pos_j = list(sensor_positions.values())[j]
            ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]],
                   'k-', alpha=0.4, linewidth=2, zorder=1)

ax.set_xlabel('X Position (m)', fontweight='bold', fontsize=11)
ax.set_ylabel('Y Position (m)', fontweight='bold', fontsize=11)
ax.set_title('(a) Sensor Network Topology', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# 2.9.2: RUL Distribution
ax = axes[0, 1]
ax.hist(df['rul_percentage'], bins=30, edgecolor='black', alpha=0.7, color='green')
ax.set_xlabel('RUL (%)', fontweight='bold', fontsize=11)
ax.set_ylabel('Frequency', fontweight='bold', fontsize=11)
ax.set_title('(b) RUL Distribution', fontweight='bold', fontsize=12)
ax.grid(axis='y', alpha=0.3)

# 2.9.3: Damage Location Over Time
ax = axes[1, 0]
colors = plt.cm.tab10(np.linspace(0, 1, 8))
for loc in range(8):
    mask = df['damage_location'] == loc
    if mask.sum() > 0:
        ax.scatter(df.index[mask], df['rul_percentage'][mask], 
                  label=f'CH{loc+1}', s=20, alpha=0.6, c=[colors[loc]])

ax.set_xlabel('Segment Number', fontweight='bold', fontsize=11)
ax.set_ylabel('RUL (%)', fontweight='bold', fontsize=11)
ax.set_title('(c) Damage Location Migration Over Time', fontweight='bold', fontsize=12)
ax.legend(ncol=2, fontsize=8)
ax.grid(True, alpha=0.3)

# 2.9.4: Total Energy Evolution
ax = axes[1, 1]
total_energies = [g.y_energy.item() for g in graph_list]
ax.plot(total_energies, linewidth=2, color='red')
ax.set_xlabel('Segment Number', fontweight='bold', fontsize=11)
ax.set_ylabel('Total Energy', fontweight='bold', fontsize=11)
ax.set_title('(d) Total Energy Evolution', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step2_data_analysis.png'), 
           dpi=300, bbox_inches='tight')
print("✓ Saved: step2_data_analysis.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("STEP 2 COMPLETE!")
print("="*80)

print("\nSummary:")
print(f"  ✓ Loaded {len(df)} temporal segments")
print(f"  ✓ Created graph with 8 nodes, {edge_index.shape[1]} edges")
print(f"  ✓ Each node has {len(feature_types)} features")
print(f"  ✓ Built {len(graph_list)} graph samples")
print(f"  ✓ Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

print("\nGenerated files:")
print(f"  ✓ {OUTPUT_DIR}/step2_graph_data.pkl")
print(f"  ✓ {OUTPUT_DIR}/step2_config.json")
print(f"  ✓ {OUTPUT_DIR}/step2_data_analysis.png")

print("\n" + "="*80)
print("Type 'next' when ready for Step 3: Model Architecture")
print("="*80)