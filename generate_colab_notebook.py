import json
import os

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "accelerator": "TPU",
            "colab": {
                "name": "F1_2026_TPU_Prediction_Agent.ipynb",
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_md(text):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": text.strip().splitlines(True)
        })

    def add_code(code):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code.strip().splitlines(True)
        })

    # Header
    add_md("""# 🏎️ 2026 Formula 1 Multi-Task ML Prediction Agent
### Trained on Google Cloud TPU (TensorFlow TPUStrategy) for 2026 FIA Formula 1 Race Predictions & F1-Score Metrics
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

---

### Overview
This notebook implements an end-to-end Machine Learning model and an autonomous **Prediction Agent** tailored for the **2026 Formula 1 World Championship**, executing exclusively on **Google Cloud TPU**:
1. **2026 Regulation Dynamics**: Incorporates the landmark 2026 regulations (100% sustainable fuels, active aerodynamics with Z-mode & X-mode, 350 kW MGU-K electric output, Audi factory team entry, Red Bull Ford Powertrains, and Hamilton at Ferrari).
2. **TPU Hardware Acceleration**: Built with TensorFlow `tf.distribute.TPUStrategy`, leveraging bfloat16 matrix multipliers and multi-core TPU pods (v2/v3/v4/v5e).
3. **Multi-Task Deep Neural Network**: Jointly predicts:
   - Championship Points Scored (Regression)
   - Top-10 Points Zone Finisher (Binary Classification evaluated via **F1-Score**)
   - Podium Finisher Top-3 (Binary Classification evaluated via **F1-Score**)
   - 20-Way Full Finishing Grid Probability Distribution (Multi-Class Softmax)
4. **Comprehensive F1-Score Optimization**: Evaluates Macro, Weighted, and Binary F1 scores alongside ROC-AUC, Precision, and Recall.
5. **Interactive Prediction Agent (`F1PredictionAgent`)**: Simulates races under dynamic weather/grid scenarios, forecasts the 2026 WDC & WCC titles, and answers natural language tactical queries.""")

    # Cell 1: Environment & Setup
    add_md("""## 1. TPU Initialization & Environment Verification
Ensure your Colab runtime is set to **TPU**:
*Menu: Runtime -> Change runtime type -> Hardware accelerator -> TPU*""")

    add_code("""# Verify Python environment and dependencies
import os
import sys
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score, classification_report,
    confusion_matrix, roc_auc_score, mean_absolute_error, mean_squared_error
)

import tensorflow as tf

print(f"TensorFlow Version: {tf.__version__}")

# Initialize TPU Cluster Resolver and TPUStrategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("=" * 60)
    print(f"✅ SUCCESS: Connected to TPU Cluster!")
    print(f"TPU Master Address: {tpu.master()}")
    print(f"Number of TPU Replicas (Cores): {strategy.num_replicas_in_sync}")
    print("=" * 60)
except Exception as e:
    print("⚠️ TPU not detected automatically. Falling back to default CPU/GPU strategy:")
    print(e)
    strategy = tf.distribute.get_strategy()
    print(f"Number of available devices: {strategy.num_replicas_in_sync}")
""")

    # Cell 2: Domain Data
    add_md("""## 2. 2026 Formula 1 Grid, Technical Specifications & Circuits
We configure the official projected 2026 grid incorporating team changes, power unit suppliers, and driver line-ups:
- **Red Bull Racing**: Max Verstappen & Liam Lawson (Red Bull Ford Powertrains)
- **Scuderia Ferrari**: Charles Leclerc & Lewis Hamilton (Ferrari Power Unit)
- **McLaren**: Lando Norris & Oscar Piastri (Mercedes Power Unit)
- **Mercedes-AMG**: George Russell & Andrea Kimi Antonelli (Mercedes Power Unit)
- **Aston Martin**: Fernando Alonso & Lance Stroll (Honda Power Unit)
- **Audi F1 Team**: Nico Hülkenberg & Gabriel Bortoleto (Audi Power Unit)
- **Williams**: Carlos Sainz & Alexander Albon (Mercedes Power Unit)
- **Alpine**: Pierre Gasly & Jack Doohan (Mercedes Power Unit)
- **Racing Bulls**: Yuki Tsunoda & Isack Hadjar (Red Bull Ford)
- **Haas F1**: Esteban Ocon & Oliver Bearman (Ferrari Power Unit)""")

    add_code("""# Define 2026 Driver and Constructor Database with Performance Indices
DRIVERS_2026 = [
    {"name": "Max Verstappen", "code": "VER", "team": "Red Bull Racing", "base_skill": 0.98, "wet_skill": 0.99, "racecraft": 0.97},
    {"name": "Lewis Hamilton", "code": "HAM", "team": "Ferrari", "base_skill": 0.96, "wet_skill": 0.98, "racecraft": 0.98},
    {"name": "Charles Leclerc", "code": "LEC", "team": "Ferrari", "base_skill": 0.96, "wet_skill": 0.91, "racecraft": 0.94},
    {"name": "Lando Norris", "code": "NOR", "team": "McLaren", "base_skill": 0.95, "wet_skill": 0.93, "racecraft": 0.93},
    {"name": "Oscar Piastri", "code": "PIA", "team": "McLaren", "base_skill": 0.93, "wet_skill": 0.90, "racecraft": 0.92},
    {"name": "George Russell", "code": "RUS", "team": "Mercedes", "base_skill": 0.93, "wet_skill": 0.92, "racecraft": 0.91},
    {"name": "Kimi Antonelli", "code": "ANT", "team": "Mercedes", "base_skill": 0.88, "wet_skill": 0.85, "racecraft": 0.86},
    {"name": "Fernando Alonso", "code": "ALO", "team": "Aston Martin", "base_skill": 0.94, "wet_skill": 0.95, "racecraft": 0.98},
    {"name": "Lance Stroll", "code": "STR", "team": "Aston Martin", "base_skill": 0.82, "wet_skill": 0.88, "racecraft": 0.83},
    {"name": "Carlos Sainz", "code": "SAI", "team": "Williams", "base_skill": 0.92, "wet_skill": 0.89, "racecraft": 0.92},
    {"name": "Alexander Albon", "code": "ALB", "team": "Williams", "base_skill": 0.88, "wet_skill": 0.87, "racecraft": 0.88},
    {"name": "Liam Lawson", "code": "LAW", "team": "Red Bull Racing", "base_skill": 0.87, "wet_skill": 0.86, "racecraft": 0.86},
    {"name": "Nico Hulkenberg", "code": "HUL", "team": "Audi", "base_skill": 0.88, "wet_skill": 0.89, "racecraft": 0.88},
    {"name": "Gabriel Bortoleto", "code": "BOR", "team": "Audi", "base_skill": 0.84, "wet_skill": 0.82, "racecraft": 0.83},
    {"name": "Pierre Gasly", "code": "GAS", "team": "Alpine", "base_skill": 0.87, "wet_skill": 0.88, "racecraft": 0.86},
    {"name": "Jack Doohan", "code": "DOO", "team": "Alpine", "base_skill": 0.82, "wet_skill": 0.81, "racecraft": 0.82},
    {"name": "Yuki Tsunoda", "code": "TSU", "team": "Racing Bulls", "base_skill": 0.87, "wet_skill": 0.86, "racecraft": 0.86},
    {"name": "Isack Hadjar", "code": "HAD", "team": "Racing Bulls", "base_skill": 0.83, "wet_skill": 0.81, "racecraft": 0.82},
    {"name": "Esteban Ocon", "code": "OCO", "team": "Haas", "base_skill": 0.87, "wet_skill": 0.90, "racecraft": 0.87},
    {"name": "Oliver Bearman", "code": "BEA", "team": "Haas", "base_skill": 0.85, "wet_skill": 0.84, "racecraft": 0.85}
]

# 2026 Engine & Aero Estimates for 2026 Regulations
TEAMS_2026 = {
    "Ferrari": {"engine_power": 0.96, "active_aero": 0.95, "mech_grip": 0.95, "pit_efficiency": 0.92, "pu_supplier": "Ferrari"},
    "McLaren": {"engine_power": 0.95, "active_aero": 0.97, "mech_grip": 0.96, "pit_efficiency": 0.97, "pu_supplier": "Mercedes"},
    "Red Bull Racing": {"engine_power": 0.93, "active_aero": 0.97, "mech_grip": 0.94, "pit_efficiency": 0.99, "pu_supplier": "Red Bull Ford"},
    "Mercedes": {"engine_power": 0.95, "active_aero": 0.93, "mech_grip": 0.93, "pit_efficiency": 0.94, "pu_supplier": "Mercedes"},
    "Aston Martin": {"engine_power": 0.94, "active_aero": 0.92, "mech_grip": 0.91, "pit_efficiency": 0.91, "pu_supplier": "Honda"},
    "Williams": {"engine_power": 0.95, "active_aero": 0.88, "mech_grip": 0.88, "pit_efficiency": 0.93, "pu_supplier": "Mercedes"},
    "Audi": {"engine_power": 0.90, "active_aero": 0.89, "mech_grip": 0.88, "pit_efficiency": 0.89, "pu_supplier": "Audi"},
    "Racing Bulls": {"engine_power": 0.93, "active_aero": 0.87, "mech_grip": 0.87, "pit_efficiency": 0.91, "pu_supplier": "Red Bull Ford"},
    "Alpine": {"engine_power": 0.95, "active_aero": 0.86, "mech_grip": 0.85, "pit_efficiency": 0.88, "pu_supplier": "Mercedes"},
    "Haas": {"engine_power": 0.96, "active_aero": 0.85, "mech_grip": 0.85, "pit_efficiency": 0.87, "pu_supplier": "Ferrari"}
}

# 24 Circuits with specific technical profiles
CIRCUITS_2026 = {
    "Bahrain (Sakhir)": {"power_bias": 0.70, "aero_bias": 0.65, "overtaking": 0.75, "street": 0, "sc_prob": 0.35},
    "Saudi Arabia (Jeddah)": {"power_bias": 0.85, "aero_bias": 0.70, "overtaking": 0.60, "street": 1, "sc_prob": 0.75},
    "Australia (Melbourne)": {"power_bias": 0.65, "aero_bias": 0.70, "overtaking": 0.55, "street": 1, "sc_prob": 0.65},
    "Japan (Suzuka)": {"power_bias": 0.60, "aero_bias": 0.95, "overtaking": 0.40, "street": 0, "sc_prob": 0.45},
    "China (Shanghai)": {"power_bias": 0.75, "aero_bias": 0.75, "overtaking": 0.70, "street": 0, "sc_prob": 0.40},
    "Miami": {"power_bias": 0.70, "aero_bias": 0.65, "overtaking": 0.55, "street": 1, "sc_prob": 0.60},
    "Monaco": {"power_bias": 0.15, "aero_bias": 0.99, "overtaking": 0.05, "street": 1, "sc_prob": 0.80},
    "Spain (Barcelona)": {"power_bias": 0.55, "aero_bias": 0.90, "overtaking": 0.45, "street": 0, "sc_prob": 0.30},
    "Canada (Montreal)": {"power_bias": 0.85, "aero_bias": 0.45, "overtaking": 0.70, "street": 1, "sc_prob": 0.70},
    "Austria (Spielberg)": {"power_bias": 0.80, "aero_bias": 0.60, "overtaking": 0.80, "street": 0, "sc_prob": 0.40},
    "Great Britain (Silverstone)": {"power_bias": 0.70, "aero_bias": 0.95, "overtaking": 0.70, "street": 0, "sc_prob": 0.50},
    "Hungary (Budapest)": {"power_bias": 0.30, "aero_bias": 0.90, "overtaking": 0.30, "street": 0, "sc_prob": 0.35},
    "Belgium (Spa-Francorchamps)": {"power_bias": 0.90, "aero_bias": 0.85, "overtaking": 0.80, "street": 0, "sc_prob": 0.60},
    "Netherlands (Zandvoort)": {"power_bias": 0.40, "aero_bias": 0.90, "overtaking": 0.35, "street": 0, "sc_prob": 0.55},
    "Italy (Monza)": {"power_bias": 0.99, "aero_bias": 0.20, "overtaking": 0.85, "street": 0, "sc_prob": 0.45},
    "Azerbaijan (Baku)": {"power_bias": 0.90, "aero_bias": 0.50, "overtaking": 0.80, "street": 1, "sc_prob": 0.80},
    "Singapore (Marina Bay)": {"power_bias": 0.30, "aero_bias": 0.95, "overtaking": 0.25, "street": 1, "sc_prob": 0.95},
    "United States (Austin)": {"power_bias": 0.70, "aero_bias": 0.85, "overtaking": 0.75, "street": 0, "sc_prob": 0.45},
    "Mexico (Mexico City)": {"power_bias": 0.75, "aero_bias": 0.80, "overtaking": 0.60, "street": 0, "sc_prob": 0.50},
    "Brazil (Interlagos)": {"power_bias": 0.70, "aero_bias": 0.80, "overtaking": 0.80, "street": 0, "sc_prob": 0.70},
    "Las Vegas": {"power_bias": 0.95, "aero_bias": 0.35, "overtaking": 0.75, "street": 1, "sc_prob": 0.65},
    "Qatar (Losail)": {"power_bias": 0.70, "aero_bias": 0.90, "overtaking": 0.50, "street": 0, "sc_prob": 0.40},
    "Abu Dhabi (Yas Marina)": {"power_bias": 0.70, "aero_bias": 0.75, "overtaking": 0.60, "street": 0, "sc_prob": 0.40}
}

POINTS_SYSTEM = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * 10
print(f"Loaded {len(DRIVERS_2026)} drivers across {len(TEAMS_2026)} teams and {len(CIRCUITS_2026)} circuits.")
""")

    # Cell 3: Realistic Dataset Generation
    add_md("""## 3. Dataset Synthesis & 2026 Physics-Informed Feature Engineering
We synthesize 25,000 multi-driver race instances modeling:
- Active Aero drag-reduction efficiency (X-mode on straights) and cornering downforce (Z-mode).
- 350kW MGU-K energy recovery and battery depletion curves.
- Grid position penalty vs race pace delta.
- Wet weather tire management & safety car disruptions.
- Multi-label targets: exact finish position, championship points, top-10 finish, and podium finish.""")

    add_code("""# Feature generation function
def generate_f1_dataset(n_races=1250, random_seed=42):
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    records = []
    circuit_names = list(CIRCUITS_2026.keys())
    
    for race_idx in range(n_races):
        circuit_name = random.choice(circuit_names)
        c_info = CIRCUITS_2026[circuit_name]
        
        # Weather generation
        rain_prob = 0.15 if c_info['street'] == 0 else 0.12
        is_wet = 1.0 if np.random.rand() < rain_prob else 0.0
        track_temp = np.random.uniform(18, 48) if is_wet == 0.0 else np.random.uniform(14, 24)
        sc_deployed = 1.0 if np.random.rand() < c_info['sc_prob'] else 0.0
        
        # Qualifying simulation to get starting grid
        quali_scores = []
        for d in DRIVERS_2026:
            t = TEAMS_2026[d['team']]
            driver_quali = d['base_skill'] * 0.4 + d['racecraft'] * 0.1
            car_quali = (t['engine_power'] * c_info['power_bias'] * 0.35 + 
                         t['active_aero'] * c_info['aero_bias'] * 0.35 + 
                         t['mech_grip'] * 0.30)
            noise = np.random.normal(0, 0.02)
            quali_score = driver_quali * 0.45 + car_quali * 0.55 + noise
            quali_scores.append((d['name'], quali_score))
        
        quali_scores.sort(key=lambda x: x[1], reverse=True)
        grid_positions = {name: rank + 1 for rank, (name, _) in enumerate(quali_scores)}
        
        # Race pace simulation
        race_scores = []
        for d in DRIVERS_2026:
            t = TEAMS_2026[d['team']]
            grid_pos = grid_positions[d['name']]
            
            # Active aero & engine contribution based on 2026 regulations
            eff_power = t['engine_power'] * (1.0 + (0.05 if is_wet == 0 else -0.04))
            eff_aero = t['active_aero'] * c_info['aero_bias']
            eff_mech = t['mech_grip'] * (1.1 if is_wet == 1 else 1.0)
            
            # Driver skill modulated by rain
            skill = d['wet_skill'] if is_wet == 1.0 else d['base_skill']
            
            # Pit efficiency and safety car interaction
            pit_impact = t['pit_efficiency'] * (1.2 if sc_deployed == 1.0 else 1.0)
            
            # Grid disadvantage: harder to overtake on Monaco/Hungary, easier on Monza/Spa
            grid_penalty = (grid_pos - 1) * (0.025 * (1.0 - c_info['overtaking']))
            
            # Random race events (DNFs, yellow flags, tire blistering)
            incident_penalty = 0.0
            if np.random.rand() < 0.04:  # 4% chance of collision or mechanical issue
                incident_penalty = np.random.uniform(0.15, 0.40)
                
            pace = (
                skill * 0.35 +
                (eff_power * c_info['power_bias'] + eff_aero * 0.5 + eff_mech * 0.3) * 0.40 +
                pit_impact * 0.10 -
                grid_penalty -
                incident_penalty +
                np.random.normal(0, 0.025)
            )
            race_scores.append((d, t, grid_pos, pace))
            
        race_scores.sort(key=lambda x: x[3], reverse=True)
        
        for finish_pos, (d, t, grid_pos, pace) in enumerate(race_scores, start=1):
            pts = POINTS_SYSTEM[finish_pos - 1]
            records.append({
                # Features
                'driver_base_skill': d['base_skill'],
                'driver_wet_skill': d['wet_skill'],
                'driver_racecraft': d['racecraft'],
                'team_engine_power': t['engine_power'],
                'team_active_aero': t['active_aero'],
                'team_mech_grip': t['mech_grip'],
                'team_pit_efficiency': t['pit_efficiency'],
                'grid_position': grid_pos,
                'circuit_power_bias': c_info['power_bias'],
                'circuit_aero_bias': c_info['aero_bias'],
                'circuit_overtaking': c_info['overtaking'],
                'circuit_is_street': c_info['street'],
                'track_temperature': track_temp,
                'is_wet': is_wet,
                'safety_car_deployed': sc_deployed,
                
                # Targets
                'finish_position': finish_pos,
                'points': pts,
                'is_top10': 1 if finish_pos <= 10 else 0,
                'is_podium': 1 if finish_pos <= 3 else 0,
                'position_class': finish_pos - 1  # 0 to 19 for multi-class classification
            })
            
    return pd.DataFrame(records)

df = generate_f1_dataset(n_races=1250)
print(f"Generated dataset shape: {df.shape}")
print(df[['grid_position', 'finish_position', 'points', 'is_top10', 'is_podium']].head())
""")

    # Cell 4: Train Test Split and TPU Data Pipeline
    add_md("""## 4. Scaled TPU Data Pipeline (`tf.data.Dataset`)
On TPU, high data throughput and batch sizing divisible by the number of TPU cores (typically 8 replicas) are essential to prevent step time stalls.""")

    add_code("""# Feature columns and target separation
FEATURE_COLS = [
    'driver_base_skill', 'driver_wet_skill', 'driver_racecraft',
    'team_engine_power', 'team_active_aero', 'team_mech_grip', 'team_pit_efficiency',
    'grid_position', 'circuit_power_bias', 'circuit_aero_bias', 'circuit_overtaking',
    'circuit_is_street', 'track_temperature', 'is_wet', 'safety_car_deployed'
]

X = df[FEATURE_COLS].values
y_points = df['points'].values.astype(np.float32)
y_top10 = df['is_top10'].values.astype(np.float32)
y_podium = df['is_podium'].values.astype(np.float32)
y_pos = df['position_class'].values.astype(np.int32)

# Train/Val/Test Split (80 / 10 / 10)
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=df['is_top10'])
val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42, stratify=df.loc[test_idx, 'is_top10'])

scaler = StandardScaler()
X_train = scaler.fit_transform(X[train_idx])
X_val = scaler.transform(X[val_idx])
X_test = scaler.transform(X[test_idx])

# Prepare target dictionaries for multi-task learning
def create_target_dict(indices):
    return {
        'points_out': y_points[indices],
        'top10_out': y_top10[indices],
        'podium_out': y_podium[indices],
        'position_out': y_pos[indices]
    }

y_train_dict = create_target_dict(train_idx)
y_val_dict = create_target_dict(val_idx)
y_test_dict = create_target_dict(test_idx)

# TPU-optimized batch sizing (scaled by number of replicas)
GLOBAL_BATCH_SIZE = 64 * strategy.num_replicas_in_sync
print(f"Using Global Batch Size for TPU: {GLOBAL_BATCH_SIZE}")

def make_tf_dataset(X_data, y_dict, is_training=True):
    ds = tf.data.Dataset.from_tensor_slices((X_data, y_dict))
    if is_training:
        ds = ds.shuffle(buffer_size=10000, seed=42)
    ds = ds.batch(GLOBAL_BATCH_SIZE, drop_remainder=is_training)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

train_ds = make_tf_dataset(X_train, y_train_dict, is_training=True)
val_ds = make_tf_dataset(X_val, y_val_dict, is_training=False)
test_ds = make_tf_dataset(X_test, y_test_dict, is_training=False)
print("✅ TPU tf.data pipelines successfully created and buffered.")
""")

    # Cell 5: Model Architecture
    add_md("""## 5. Multi-Task Deep Residual Network Architecture
Within `strategy.scope()`, we instantiate a deep multi-head architecture:
1. **Shared Representation**: Dense GELU layers with Residual skip connections and Layer Normalization.
2. **`points_out`**: Non-negative linear unit for expected Formula 1 points (0 to 26).
3. **`top10_out`**: Sigmoid binary classification for points-scoring finish (evaluated with **F1-Score**).
4. **`podium_out`**: Sigmoid binary classification for top-3 podium finish (evaluated with **F1-Score**).
5. **`position_out`**: 20-way Softmax distribution predicting finish rank.""")

    add_code("""# Define Model inside TPU Strategy Scope
def build_f1_multitask_model(input_dim=len(FEATURE_COLS)):
    inputs = tf.keras.Input(shape=(input_dim,), name='f1_input_features')
    
    # Dense Projection
    x = tf.keras.layers.Dense(256, activation='gelu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Residual Block 1
    res1 = x
    x = tf.keras.layers.Dense(256, activation='gelu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation='gelu')(x)
    x = tf.keras.layers.Add()([res1, x])
    x = tf.keras.layers.LayerNormalization()(x)
    
    # Residual Block 2
    res2 = tf.keras.layers.Dense(128, activation='gelu')(x)
    x = tf.keras.layers.Dense(128, activation='gelu')(res2)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation='gelu')(x)
    x = tf.keras.layers.Add()([res2, x])
    x = tf.keras.layers.LayerNormalization()(x)
    
    # Latent embedding
    latent = tf.keras.layers.Dense(64, activation='gelu', name='race_latent_embedding')(x)
    
    # Head 1: Championship Points (Regression)
    points_out = tf.keras.layers.Dense(32, activation='gelu')(latent)
    points_out = tf.keras.layers.Dense(1, activation='relu', name='points_out')(points_out)
    
    # Head 2: Top-10 Points Zone (Binary Classification - F1 Score Target)
    top10_out = tf.keras.layers.Dense(32, activation='gelu')(latent)
    top10_out = tf.keras.layers.Dense(1, activation='sigmoid', name='top10_out')(top10_out)
    
    # Head 3: Podium Finish (Binary Classification - F1 Score Target)
    podium_out = tf.keras.layers.Dense(32, activation='gelu')(latent)
    podium_out = tf.keras.layers.Dense(1, activation='sigmoid', name='podium_out')(podium_out)
    
    # Head 4: Exact Finishing Position Probability (20-Class Softmax)
    position_out = tf.keras.layers.Dense(64, activation='gelu')(latent)
    position_out = tf.keras.layers.Dense(20, activation='softmax', name='position_out')(position_out)
    
    model = tf.keras.Model(
        inputs=inputs, 
        outputs=[points_out, top10_out, podium_out, position_out],
        name="F1_2026_TPU_MultiTask_Model"
    )
    return model

with strategy.scope():
    model = build_f1_multitask_model()
    
    # Multi-task loss weights and optimizers
    losses = {
        'points_out': tf.keras.losses.Huber(delta=1.5),
        'top10_out': tf.keras.losses.BinaryCrossentropy(),
        'podium_out': tf.keras.losses.BinaryCrossentropy(),
        'position_out': tf.keras.losses.SparseCategoricalCrossentropy()
    }
    
    loss_weights = {
        'points_out': 0.35,
        'top10_out': 1.0,
        'podium_out': 1.5,
        'position_out': 0.8
    }
    
    metrics = {
        'points_out': ['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')],
        'top10_out': ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
        'podium_out': ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
        'position_out': ['accuracy']
    }
    
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

model.summary()
""")

    # Cell 6: Training
    add_md("""## 6. Training Execution on Google Cloud TPU""")

    add_code("""# Setup training callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_top10_out_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )
]

print("🚀 Starting TPU Training Loop...")
EPOCHS = 35

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)
print("🏁 TPU Training Complete!")
""")

    # Cell 7: Training History Visualization
    add_md("""## 7. Training Diagnostics & Learning Curves""")

    add_code("""# Plot training histories
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Total Loss
axes[0, 0].plot(history.history['loss'], label='Train Loss', color='dodgerblue', lw=2)
axes[0, 0].plot(history.history['val_loss'], label='Val Loss', color='darkorange', lw=2)
axes[0, 0].set_title('Total Multi-Task Loss', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# 2. Points MAE
axes[0, 1].plot(history.history['points_out_mae'], label='Train Points MAE', color='forestgreen', lw=2)
axes[0, 1].plot(history.history['val_points_out_mae'], label='Val Points MAE', color='crimson', lw=2)
axes[0, 1].set_title('Points Prediction MAE', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# 3. Top-10 Accuracy
axes[1, 0].plot(history.history['top10_out_accuracy'], label='Train Top-10 Acc', color='purple', lw=2)
axes[1, 0].plot(history.history['val_top10_out_accuracy'], label='Val Top-10 Acc', color='magenta', lw=2)
axes[1, 0].set_title('Top-10 Points Finisher Accuracy', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# 4. Podium Accuracy
axes[1, 1].plot(history.history['podium_out_accuracy'], label='Train Podium Acc', color='teal', lw=2)
axes[1, 1].plot(history.history['val_podium_out_accuracy'], label='Val Podium Acc', color='coral', lw=2)
axes[1, 1].set_title('Podium Finisher Accuracy', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
plt.show()
""")

    # Cell 8: F1 Score Evaluation
    add_md("""## 8. Comprehensive Evaluation & F1-Score Metrics
We evaluate the held-out test set specifically focusing on:
- **Binary F1-Score**: Detecting points-scorers (Top 10) and podium finishers (Top 3).
- **Macro & Weighted F1-Scores**: Robust across class imbalances.
- **Confusion Matrices & Precision-Recall trade-offs**.""")

    add_code("""# Run inference on test dataset
preds = model.predict(test_ds)
pred_points = preds[0].flatten()
pred_top10_prob = preds[1].flatten()
pred_podium_prob = preds[2].flatten()
pred_position_probs = preds[3]

# Extract ground truths aligned with test batches
y_test_true_top10 = y_test_dict['top10_out'][:len(pred_top10_prob)]
y_test_true_podium = y_test_dict['podium_out'][:len(pred_podium_prob)]
y_test_true_points = y_test_dict['points_out'][:len(pred_points)]

# Optimal threshold determination
pred_top10_binary = (pred_top10_prob >= 0.5).astype(int)
pred_podium_binary = (pred_podium_prob >= 0.35).astype(int)  # Balanced threshold for top-3

# Calculate F1 Scores
f1_top10_binary = f1_score(y_test_true_top10, pred_top10_binary)
f1_top10_macro = f1_score(y_test_true_top10, pred_top10_binary, average='macro')
f1_top10_weighted = f1_score(y_test_true_top10, pred_top10_binary, average='weighted')

f1_podium_binary = f1_score(y_test_true_podium, pred_podium_binary)
f1_podium_macro = f1_score(y_test_true_podium, pred_podium_binary, average='macro')
f1_podium_weighted = f1_score(y_test_true_podium, pred_podium_binary, average='weighted')

print("=" * 65)
print("📊 F1-SCORE PERFORMANCE REPORT ON HELD-OUT TEST DATA")
print("=" * 65)
print(f"Top-10 (Points Finisher) Binary F1-Score   : {f1_top10_binary:.4f}")
print(f"Top-10 (Points Finisher) Macro F1-Score    : {f1_top10_macro:.4f}")
print(f"Top-10 (Points Finisher) Weighted F1-Score : {f1_top10_weighted:.4f}")
print("-" * 65)
print(f"Podium (Top-3 Finish) Binary F1-Score      : {f1_podium_binary:.4f}")
print(f"Podium (Top-3 Finish) Macro F1-Score       : {f1_podium_macro:.4f}")
print(f"Podium (Top-3 Finish) Weighted F1-Score    : {f1_podium_weighted:.4f}")
print("=" * 65)

# Classification reports
print("\n--- Detailed Classification Report: Top 10 Finish ---")
print(classification_report(y_test_true_top10, pred_top10_binary, target_names=['Outside Points', 'In Points (P1-P10)']))

print("\n--- Detailed Classification Report: Podium Finish ---")
print(classification_report(y_test_true_podium, pred_podium_binary, target_names=['Off Podium', 'Podium (P1-P3)']))

# Confusion Matrix Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
cm1 = confusion_matrix(y_test_true_top10, pred_top10_binary)
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['No Points', 'Points'], yticklabels=['No Points', 'Points'])
ax1.set_title(f'Top-10 Finisher Confusion Matrix\n(F1 = {f1_top10_binary:.3f})', fontweight='bold')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

cm2 = confusion_matrix(y_test_true_podium, pred_podium_binary)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=ax2,
            xticklabels=['Off Podium', 'Podium'], yticklabels=['Off Podium', 'Podium'])
ax2.set_title(f'Podium Finisher Confusion Matrix\n(F1 = {f1_podium_binary:.3f})', fontweight='bold')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

plt.tight_layout()
plt.show()
""")

    # Cell 9: Prediction Agent Implementation
    add_md("""## 9. The 2026 Autonomous Formula 1 Prediction Agent
The `F1PredictionAgent` provides:
1. **Dynamic Grand Prix Simulation**: Predicts race finishing order, podium, points, and fast-lap probabilities given any 2026 circuit and weather.
2. **Season Monte Carlo Forecasting**: Simulates all 24 races of the 2026 calendar to forecast World Drivers' Championship (WDC) and Constructors' Championship (WCC).
3. **Strategic Race Briefing**: Produces analytical debriefs on active aero, tire degradation, and overtake difficulty.
4. **Natural Language Query Interface**: Handles custom queries.""")

    add_code("""class F1PredictionAgent:
    \"\"\"
    Autonomous 2026 Formula 1 Prediction & Strategy Agent powered by TPU Multi-Task Neural Network.
    \"\"\"
    def __init__(self, model, scaler, drivers, teams, circuits):
        self.model = model
        self.scaler = scaler
        self.drivers = drivers
        self.teams = teams
        self.circuits = circuits
        
    def _prepare_features(self, circuit_name, is_wet=0.0, track_temp=30.0, 
                          sc_prob=None, custom_grid=None):
        if circuit_name not in self.circuits:
            raise ValueError(f"Unknown circuit: {circuit_name}. Available: {list(self.circuits.keys())}")
        
        c_info = self.circuits[circuit_name]
        safety_car = 1.0 if (sc_prob if sc_prob is not None else c_info['sc_prob']) > 0.5 else 0.0
        
        # Determine starting grid
        if custom_grid is None:
            # Default grid by qualifying power
            quali_ranks = []
            for d in self.drivers:
                t = self.teams[d['team']]
                q_score = (d['base_skill'] * 0.45 + 
                           (t['engine_power'] * c_info['power_bias'] + t['active_aero'] * c_info['aero_bias']) * 0.55)
                quali_ranks.append((d['name'], q_score))
            quali_ranks.sort(key=lambda x: x[1], reverse=True)
            grid_map = {name: rank + 1 for rank, (name, _) in enumerate(quali_ranks)}
        else:
            grid_map = custom_grid

        features = []
        driver_meta = []
        for d in self.drivers:
            t = self.teams[d['team']]
            g_pos = grid_map.get(d['name'], 10)
            
            row = [
                d['base_skill'], d['wet_skill'], d['racecraft'],
                t['engine_power'], t['active_aero'], t['mech_grip'], t['pit_efficiency'],
                g_pos, c_info['power_bias'], c_info['aero_bias'], c_info['overtaking'],
                c_info['street'], track_temp, is_wet, safety_car
            ]
            features.append(row)
            driver_meta.append((d, t, g_pos))
            
        scaled_feat = self.scaler.transform(np.array(features))
        return scaled_feat, driver_meta, c_info

    def predict_grand_prix(self, circuit_name, is_wet=0.0, track_temp=30.0, custom_grid=None):
        scaled_feat, driver_meta, c_info = self._prepare_features(
            circuit_name, is_wet=is_wet, track_temp=track_temp, custom_grid=custom_grid
        )
        
        # Neural Model Inference
        pts_preds, top10_preds, podium_preds, pos_dist_preds = self.model.predict(scaled_feat, verbose=0)
        
        results = []
        for i, (d, t, g_pos) in enumerate(driver_meta):
            # Compute composite race performance score
            expected_pos = float(np.sum(pos_dist_preds[i] * np.arange(1, 21)))
            pred_pts = float(pts_preds[i][0])
            top10_prob = float(top10_preds[i][0])
            podium_prob = float(podium_preds[i][0])
            win_prob = float(pos_dist_preds[i][0])  # P1 probability
            
            results.append({
                'Driver': d['name'],
                'Team': d['team'],
                'PU': t['pu_supplier'],
                'Grid': g_pos,
                'Exp_Pos': round(expected_pos, 2),
                'Pred_Points': round(pred_pts, 2),
                'Top10_Prob': round(top10_prob * 100, 1),
                'Podium_Prob': round(podium_prob * 100, 1),
                'Win_Prob': round(win_prob * 100, 1)
            })
            
        # Rank by composite predicted points and expected position
        results.sort(key=lambda x: (x['Exp_Pos'], -x['Pred_Points']))
        for finish_idx, r in enumerate(results, start=1):
            r['Predicted_Finish'] = finish_idx
            
        res_df = pd.DataFrame(results)
        return res_df[['Predicted_Finish', 'Driver', 'Team', 'PU', 'Grid', 'Exp_Pos', 'Pred_Points', 'Podium_Prob', 'Win_Prob', 'Top10_Prob']]

    def simulate_2026_season(self, n_simulations=50):
        \"\"\"
        Runs Monte Carlo simulations across all 24 races of the 2026 calendar.
        \"\"\"
        driver_points = {d['name']: 0.0 for d in self.drivers}
        team_points = {t: 0.0 for t in self.teams}
        
        print(f"🏁 Simulating {n_simulations} complete 2026 F1 World Championships ({n_simulations * len(self.circuits)} total Grands Prix)...")
        
        for sim in range(n_simulations):
            for c_name in self.circuits.keys():
                is_wet = 1.0 if np.random.rand() < 0.15 else 0.0
                temp = np.random.uniform(20, 38)
                
                race_df = self.predict_grand_prix(c_name, is_wet=is_wet, track_temp=temp)
                
                # Allocate points to top 10
                for rank, row in race_df.head(10).iterrows():
                    allocated_pts = POINTS_SYSTEM[rank]
                    driver_points[row['Driver']] += allocated_pts / n_simulations
                    team_points[row['Team']] += allocated_pts / n_simulations
                    
        wdc_df = pd.DataFrame(list(driver_points.items()), columns=['Driver', 'Avg_Championship_Points'])
        wdc_df = wdc_df.sort_values(by='Avg_Championship_Points', ascending=False).reset_index(drop=True)
        wdc_df.index += 1
        
        wcc_df = pd.DataFrame(list(team_points.items()), columns=['Constructor', 'Avg_Championship_Points'])
        wcc_df = wcc_df.sort_values(by='Avg_Championship_Points', ascending=False).reset_index(drop=True)
        wcc_df.index += 1
        
        return wdc_df, wcc_df

    def query(self, question: str):
        \"\"\"
        Interactive agent query router to answer strategic 2026 F1 scenarios.
        \"\"\"
        q = question.lower()
        if "rain" in q or "wet" in q or "spa" in q:
            circuit = "Belgium (Spa-Francorchamps)"
            df_res = self.predict_grand_prix(circuit, is_wet=1.0, track_temp=16.0)
            summary = f"🌧️ Prediction for Wet Grand Prix at {circuit}:\\n"
            summary += f"Favorite: {df_res.iloc[0]['Driver']} ({df_res.iloc[0]['Team']}) with Win Prob {df_res.iloc[0]['Win_Prob']}%\\n"
            summary += f"Predicted Podium:\\n"
            for _, r in df_res.head(3).iterrows():
                summary += f"  P{r['Predicted_Finish']}: {r['Driver']} ({r['Team']}) - Podium Prob: {r['Podium_Prob']}%\\n"
            return summary, df_res.head(8)
            
        elif "audi" in q:
            circuit = "Bahrain (Sakhir)"
            df_res = self.predict_grand_prix(circuit, is_wet=0.0)
            audi_rows = df_res[df_res['Team'] == 'Audi']
            summary = f"🇩🇪 Audi F1 Team 2026 Evaluation at {circuit}:\\n"
            for _, r in audi_rows.iterrows():
                summary += f"  {r['Driver']}: Predicted Finish P{r['Predicted_Finish']} (Points Prob: {r['Top10_Prob']}%, Exp Points: {r['Pred_Points']})\\n"
            return summary, audi_rows
            
        elif "monza" in q or "power" in q:
            circuit = "Italy (Monza)"
            df_res = self.predict_grand_prix(circuit, is_wet=0.0, track_temp=32.0)
            summary = f"🇮🇹 Temple of Speed (Monza) 2026 Active Aero & Power Unit Predictions:\\n"
            summary += f"Pole/Winner Projection: {df_res.iloc[0]['Driver']} [{df_res.iloc[0]['PU']} Power]\\n"
            return summary, df_res.head(6)
            
        else:
            circuit = "Australia (Melbourne)"
            df_res = self.predict_grand_prix(circuit)
            summary = f"🇦🇺 2026 Season Opener Prediction at {circuit}:\\n"
            return summary, df_res.head(10)

agent = F1PredictionAgent(model, scaler, DRIVERS_2026, TEAMS_2026, CIRCUITS_2026)
print("🤖 F1PredictionAgent initialized and ready for tactical simulation!")
""")

    # Cell 10: Running Agent Demonstrations
    add_md("""## 10. Agent Scenarios & Grand Prix Predictions""")

    add_code("""# Scenario A: 2026 Season Opener at Albert Park, Melbourne (Dry)
print("=" * 70)
print("🇦🇺 SCENARIO 1: 2026 AUSTRALIAN GRAND PRIX (MELBOURNE - DRY)")
print("=" * 70)
melb_results = agent.predict_grand_prix("Australia (Melbourne)", is_wet=0.0, track_temp=28.0)
print(melb_results.to_string(index=False))
""")

    add_code("""# Scenario B: 2026 Belgian Grand Prix at Spa under torrential Ardennes rain
print("=" * 70)
print("🌧️ SCENARIO 2: 2026 BELGIAN GRAND PRIX (SPA - WET TRACK)")
print("=" * 70)
spa_results = agent.predict_grand_prix("Belgium (Spa-Francorchamps)", is_wet=1.0, track_temp=15.0)
print(spa_results.to_string(index=False))
""")

    add_code("""# Scenario C: 2026 Full Season Championship Forecast (Drivers & Constructors)
wdc_forecast, wcc_forecast = agent.simulate_2026_season(n_simulations=30)

print("\n" + "=" * 60)
print("🏆 2026 PROJECTED WORLD DRIVERS' CHAMPIONSHIP (WDC)")
print("=" * 60)
print(wdc_forecast.head(10).to_string())

print("\n" + "=" * 60)
print("🏆 2026 PROJECTED CONSTRUCTORS' CHAMPIONSHIP (WCC)")
print("=" * 60)
print(wcc_forecast.to_string())

# Plot WDC & WCC projections
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(data=wdc_forecast.head(10), x='Avg_Championship_Points', y='Driver', palette='rocket', ax=ax1)
ax1.set_title("2026 Projected Drivers' Championship Top 10", fontsize=14, fontweight='bold')
ax1.set_xlabel("Predicted Points Across 24 Rounds")

sns.barplot(data=wcc_forecast, x='Avg_Championship_Points', y='Constructor', palette='viridis', ax=ax2)
ax2.set_title("2026 Projected Constructors' Championship Standings", fontsize=14, fontweight='bold')
ax2.set_xlabel("Predicted Team Points")

plt.tight_layout()
plt.show()
""")

    add_code("""# Scenario D: Natural Language Strategic Queries to the Agent
print("=" * 70)
print("💬 INTERACTIVE AGENT QUERY 1: 'What are Audi\\'s point scoring chances in 2026?'")
print("=" * 70)
summary1, data1 = agent.query("What are Audi's point scoring chances in 2026?")
print(summary1)
print(data1.to_string(index=False))

print("\n" + "=" * 70)
print("💬 INTERACTIVE AGENT QUERY 2: 'Who will win at Spa in the rain?'")
print("=" * 70)
summary2, data2 = agent.query("Who will win at Spa in the rain?")
print(summary2)
print(data2.to_string(index=False))
""")

    # Save to file
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f1_2026_tpu_prediction_agent.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"✅ Notebook successfully written to {out_path}")

if __name__ == "__main__":
    create_notebook()
