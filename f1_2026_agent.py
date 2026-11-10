"""
================================================================================
🏎️ 2026 Formula 1 Multi-Task Machine Learning Model & Prediction Agent
Google Cloud TPU Optimized (TensorFlow TPUStrategy)
================================================================================
Evaluates Formula 1 Race Predictions & Championship Points using Classification F1-Score
and Multi-Output Deep Neural Networks.
"""

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

# -----------------------------------------------------------------------------
# 1. Hardware Strategy Initialization (TPU / GPU / CPU Fallback)
# -----------------------------------------------------------------------------
def initialize_tpu_strategy():
    """Initializes TPUClusterResolver and returns TPUStrategy if in Colab TPU runtime."""
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("=" * 60)
        print(f"✅ SUCCESS: Connected to TPU Cluster!")
        print(f"TPU Master Address: {tpu.master()}")
        print(f"Number of TPU Replicas (Cores): {strategy.num_replicas_in_sync}")
        print("=" * 60)
        return strategy
    except Exception as e:
        print("⚠️ TPU not detected automatically. Falling back to default CPU/GPU strategy:")
        print(e)
        strategy = tf.distribute.get_strategy()
        print(f"Number of available devices: {strategy.num_replicas_in_sync}")
        return strategy

# -----------------------------------------------------------------------------
# 2. 2026 Grid, Technical Regulations & Circuit Specifications
# -----------------------------------------------------------------------------
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
FEATURE_COLS = [
    'driver_base_skill', 'driver_wet_skill', 'driver_racecraft',
    'team_engine_power', 'team_active_aero', 'team_mech_grip', 'team_pit_efficiency',
    'grid_position', 'circuit_power_bias', 'circuit_aero_bias', 'circuit_overtaking',
    'circuit_is_street', 'track_temperature', 'is_wet', 'safety_car_deployed'
]

# -----------------------------------------------------------------------------
# 3. Dataset Generation Engine
# -----------------------------------------------------------------------------
def generate_f1_dataset(n_races=1250, random_seed=42):
    """Generates synthetic training dataset mimicking 2026 F1 active aero & hybrid regulations."""
    np.random.seed(random_seed)
    random.seed(random_seed)
    records = []
    circuit_names = list(CIRCUITS_2026.keys())
    
    for race_idx in range(n_races):
        circuit_name = random.choice(circuit_names)
        c_info = CIRCUITS_2026[circuit_name]
        
        rain_prob = 0.15 if c_info['street'] == 0 else 0.12
        is_wet = 1.0 if np.random.rand() < rain_prob else 0.0
        track_temp = np.random.uniform(18, 48) if is_wet == 0.0 else np.random.uniform(14, 24)
        sc_deployed = 1.0 if np.random.rand() < c_info['sc_prob'] else 0.0
        
        # Qualifying
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
        
        # Race
        race_scores = []
        for d in DRIVERS_2026:
            t = TEAMS_2026[d['team']]
            grid_pos = grid_positions[d['name']]
            eff_power = t['engine_power'] * (1.0 + (0.05 if is_wet == 0 else -0.04))
            eff_aero = t['active_aero'] * c_info['aero_bias']
            eff_mech = t['mech_grip'] * (1.1 if is_wet == 1 else 1.0)
            skill = d['wet_skill'] if is_wet == 1.0 else d['base_skill']
            pit_impact = t['pit_efficiency'] * (1.2 if sc_deployed == 1.0 else 1.0)
            grid_penalty = (grid_pos - 1) * (0.025 * (1.0 - c_info['overtaking']))
            
            incident_penalty = np.random.uniform(0.15, 0.40) if np.random.rand() < 0.04 else 0.0
            pace = (
                skill * 0.35 +
                (eff_power * c_info['power_bias'] + eff_aero * 0.5 + eff_mech * 0.3) * 0.40 +
                pit_impact * 0.10 - grid_penalty - incident_penalty + np.random.normal(0, 0.025)
            )
            race_scores.append((d, t, grid_pos, pace))
            
        race_scores.sort(key=lambda x: x[3], reverse=True)
        for finish_pos, (d, t, grid_pos, pace) in enumerate(race_scores, start=1):
            pts = POINTS_SYSTEM[finish_pos - 1]
            records.append({
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
                'finish_position': finish_pos,
                'points': pts,
                'is_top10': 1 if finish_pos <= 10 else 0,
                'is_podium': 1 if finish_pos <= 3 else 0,
                'position_class': finish_pos - 1
            })
            
    return pd.DataFrame(records)

# -----------------------------------------------------------------------------
# 4. Multi-Task Deep Neural Network for TPU
# -----------------------------------------------------------------------------
def build_f1_multitask_model(input_dim=len(FEATURE_COLS)):
    inputs = tf.keras.Input(shape=(input_dim,), name='f1_input_features')
    
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
    
    latent = tf.keras.layers.Dense(64, activation='gelu', name='race_latent_embedding')(x)
    
    # Multi-Task Heads
    points_out = tf.keras.layers.Dense(32, activation='gelu')(latent)
    points_out = tf.keras.layers.Dense(1, activation='relu', name='points_out')(points_out)
    
    top10_out = tf.keras.layers.Dense(32, activation='gelu')(latent)
    top10_out = tf.keras.layers.Dense(1, activation='sigmoid', name='top10_out')(top10_out)
    
    podium_out = tf.keras.layers.Dense(32, activation='gelu')(latent)
    podium_out = tf.keras.layers.Dense(1, activation='sigmoid', name='podium_out')(podium_out)
    
    position_out = tf.keras.layers.Dense(64, activation='gelu')(latent)
    position_out = tf.keras.layers.Dense(20, activation='softmax', name='position_out')(position_out)
    
    model = tf.keras.Model(
        inputs=inputs, 
        outputs=[points_out, top10_out, podium_out, position_out],
        name="F1_2026_TPU_MultiTask_Model"
    )
    return model

# -----------------------------------------------------------------------------
# 5. Autonomous 2026 F1 Prediction Agent
# -----------------------------------------------------------------------------
class F1PredictionAgent:
    """
    Autonomous 2026 Formula 1 Prediction & Strategy Agent powered by TPU Multi-Task Neural Network.
    """
    def __init__(self, model, scaler, drivers=DRIVERS_2026, teams=TEAMS_2026, circuits=CIRCUITS_2026):
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
        
        if custom_grid is None:
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
        
        pts_preds, top10_preds, podium_preds, pos_dist_preds = self.model.predict(scaled_feat, verbose=0)
        
        results = []
        for i, (d, t, g_pos) in enumerate(driver_meta):
            expected_pos = float(np.sum(pos_dist_preds[i] * np.arange(1, 21)))
            pred_pts = float(pts_preds[i][0])
            top10_prob = float(top10_preds[i][0])
            podium_prob = float(podium_preds[i][0])
            win_prob = float(pos_dist_preds[i][0])
            
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
            
        results.sort(key=lambda x: (x['Exp_Pos'], -x['Pred_Points']))
        for finish_idx, r in enumerate(results, start=1):
            r['Predicted_Finish'] = finish_idx
            
        res_df = pd.DataFrame(results)
        return res_df[['Predicted_Finish', 'Driver', 'Team', 'PU', 'Grid', 'Exp_Pos', 'Pred_Points', 'Podium_Prob', 'Win_Prob', 'Top10_Prob']]

    def simulate_2026_season(self, n_simulations=50):
        driver_points = {d['name']: 0.0 for d in self.drivers}
        team_points = {t: 0.0 for t in self.teams}
        
        print(f"🏁 Simulating {n_simulations} complete 2026 F1 seasons...")
        for sim in range(n_simulations):
            for c_name in self.circuits.keys():
                is_wet = 1.0 if np.random.rand() < 0.15 else 0.0
                temp = np.random.uniform(20, 38)
                race_df = self.predict_grand_prix(c_name, is_wet=is_wet, track_temp=temp)
                
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
        q = question.lower()
        if "rain" in q or "wet" in q or "spa" in q:
            circuit = "Belgium (Spa-Francorchamps)"
            df_res = self.predict_grand_prix(circuit, is_wet=1.0, track_temp=16.0)
            summary = f"🌧️ Prediction for Wet Grand Prix at {circuit}:\n"
            summary += f"Favorite: {df_res.iloc[0]['Driver']} ({df_res.iloc[0]['Team']}) with Win Prob {df_res.iloc[0]['Win_Prob']}%\n"
            summary += f"Predicted Podium:\n"
            for _, r in df_res.head(3).iterrows():
                summary += f"  P{r['Predicted_Finish']}: {r['Driver']} ({r['Team']}) - Podium Prob: {r['Podium_Prob']}%\n"
            return summary, df_res.head(8)
            
        elif "audi" in q:
            circuit = "Bahrain (Sakhir)"
            df_res = self.predict_grand_prix(circuit, is_wet=0.0)
            audi_rows = df_res[df_res['Team'] == 'Audi']
            summary = f"🇩🇪 Audi F1 Team 2026 Evaluation at {circuit}:\n"
            for _, r in audi_rows.iterrows():
                summary += f"  {r['Driver']}: Predicted Finish P{r['Predicted_Finish']} (Points Prob: {r['Top10_Prob']}%, Exp Points: {r['Pred_Points']})\n"
            return summary, audi_rows
            
        elif "monza" in q or "power" in q:
            circuit = "Italy (Monza)"
            df_res = self.predict_grand_prix(circuit, is_wet=0.0, track_temp=32.0)
            summary = f"🇮🇹 Temple of Speed (Monza) 2026 Predictions:\n"
            summary += f"Pole/Winner Projection: {df_res.iloc[0]['Driver']} [{df_res.iloc[0]['PU']} Power]\n"
            return summary, df_res.head(6)
            
        else:
            circuit = "Australia (Melbourne)"
            df_res = self.predict_grand_prix(circuit)
            summary = f"🇦🇺 2026 Season Opener Prediction at {circuit}:\n"
            return summary, df_res.head(10)
