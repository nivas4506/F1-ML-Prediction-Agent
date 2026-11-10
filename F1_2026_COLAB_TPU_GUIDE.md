# 🏎️ 2026 Formula 1 Multi-Task ML Model & Prediction Agent (Colab TPU)

An end-to-end Machine Learning pipeline and autonomous **Prediction Agent** built specifically for the **2026 FIA Formula One World Championship**, optimized for training and inference exclusively on **Google Cloud TPU** via Google Colab.

---

## ⚡ Quick Start: Run in Google Colab with TPU

1. Upload [`f1_2026_tpu_prediction_agent.ipynb`](file:///c:/Users/A/OneDrive/Documents/FeatherOS/F1/f1_2026_tpu_prediction_agent.ipynb) to [Google Colab](https://colab.research.google.com).
2. Select TPU Accelerator:
   - Go to **Runtime** > **Change runtime type**
   - Under **Hardware accelerator**, select **TPU v2 / TPU v3 / TPU v4 / TPU v5e**
   - Click **Save**
3. Run all cells (**Runtime** > **Run all**).

---

## 🛠️ Key Capabilities

### 1. 2026 F1 Technical Regulations Engine
- **Active Aerodynamics**: Models low-drag straight-line mode (X-mode) and high-downforce cornering mode (Z-mode).
- **Power Unit Architecture**: 1.6L V6 Turbo Hybrid with MGU-H removed and MGU-K electrical output tripled to 350 kW (~470 bhp, nearly 50% ICE / 50% electric).
- **100% Sustainable Fuels**: Adjusted thermal efficiency and combustion profiles.
- **2026 Grid Changes**:
  - **Audi F1 Team** factory entry with Audi Power Unit.
  - **Scuderia Ferrari**: Charles Leclerc & Lewis Hamilton.
  - **Red Bull Racing**: Max Verstappen & Liam Lawson (Red Bull Ford Powertrains).
  - **McLaren & Mercedes**: Powered by Mercedes High Performance Powertrains.
  - **Aston Martin**: Works partner with Honda Power Units.

### 2. Multi-Task Deep Neural Network for TPU
The neural architecture is compiled within `tf.distribute.TPUStrategy`:
- **Backbone**: Deep Residual blocks with GELU activations, Layer Normalization, and Dropout.
- **Points Output Head** (`points_out`): Huber Loss regression for expected championship points ($25, 18, 15, \dots, 0$).
- **Top-10 Points Zone Head** (`top10_out`): Binary Cross-Entropy with **F1-Score** optimization (Macro, Weighted, Binary F1).
- **Podium Head** (`podium_out`): Binary classification for Top-3 finishes with tuned decision thresholds.
- **Position Distribution Head** (`position_out`): 20-class Softmax across all finishing positions.

### 3. F1-Score Metric Focus
The model explicitly computes and reports:
- **Top-10 Finish F1-Score**: Evaluates whether a driver lands in the points.
- **Podium Finish F1-Score**: Evaluates whether a driver secures a Top-3 podium finish.
- **Macro & Weighted F1-Scores**: Accounts for class imbalances between podium finishers (3/20) and non-podium finishers (17/20).

### 4. Autonomous `F1PredictionAgent`
- `agent.predict_grand_prix(circuit_name, is_wet, track_temp, custom_grid)`: Produces detailed finishing order, podium likelihood, win probability, and points expectancy.
- `agent.simulate_2026_season(n_simulations=50)`: Executes Monte Carlo simulations across all 24 rounds of the 2026 calendar, delivering projected **World Drivers' Championship (WDC)** and **World Constructors' Championship (WCC)** standings.
- `agent.query(question)`: Natural language tactical query interface for wet weather scenarios, circuit comparisons, and manufacturer performance.

---

## 📂 Deliverables Included
- **Jupyter Notebook**: [`f1_2026_tpu_prediction_agent.ipynb`](file:///c:/Users/A/OneDrive/Documents/FeatherOS/F1/f1_2026_tpu_prediction_agent.ipynb) (24 cells, completely self-contained for Colab TPU).
- **Standalone Module**: [`f1_2026_agent.py`](file:///c:/Users/A/OneDrive/Documents/FeatherOS/F1/f1_2026_agent.py) (Reusable model and agent classes).
