# Continual Reinforcement Learning for Non-Stationary Cognitive Radio Networks 📡

A Continual Reinforcement Learning (CRL) framework for adaptive spectrum access in non-stationary Cognitive Radio Networks (CRNs). The project models changing Primary User (PU) activity across a full diurnal cycle and trains a Secondary User (SU) to adapt its channel-selection policy while reducing catastrophic forgetting.

## 📖 Overview

Spectrum occupancy in a CRN can change significantly throughout the day. A policy trained for one traffic regime may therefore perform poorly when PU activity changes, while sequential retraining can overwrite knowledge acquired from earlier environments.

This project addresses that problem using a **Deep Q-Network (DQN)** combined with continual-learning and drift-adaptation mechanisms:

1. **ADWIN Concept Drift Detection** — monitors the mean absolute Temporal Difference (TD) error and detects statistically significant changes in the learning stream.

2. **Adaptive Elastic Weight Consolidation (EWC)** — protects parameters important to previously learned policies using a Fisher Information Matrix (FIM). The EWC coefficient \(\lambda_t\) adapts according to recent TD-error variance to balance stability and plasticity.

3. **Episodic Memory Replay** — preserves high-information transitions selected using absolute TD error and mixes them with the active replay buffer during later training.

4. **Rollback-Based Recovery** — maintains lightweight policy checkpoints that can be restored when concept drift is detected.

5. **Asynchronous MEC Simulation** — offloads Fisher Information Matrix computation to a non-blocking worker thread, allowing the main spectrum-learning loop to continue while consolidation data is computed.

## 🧠 CRN Environment

The simulator contains **8 spectrum channels** and an additional idle action. Each observation contains recent channel occupancy and normalized signal information across a five-step history.

Seven sequential environments represent different PU traffic regimes:

| Task | Period              | PU Activity |
| ---- | ------------------- | ----------- |
| T1   | Early commute       | Low–Medium  |
| T2   | Peak commute        | High        |
| T3   | Midday traffic      | Medium      |
| T4   | Sparse IoT activity | Low         |
| T5   | Evening commute     | High        |
| T6   | Heavy streaming     | Very High   |
| T7   | Night activity      | Very Low    |

Each task is trained for **10,000 environment steps**, creating a 70,000-step non-stationary learning sequence.

## 📊 Evaluation

The notebook includes a complete evaluation pipeline for both spectrum-access performance and continual learning.

Reported metrics include:

* Final average performance and diagonal task performance
* Backward Transfer (BWT)
* Forward Transfer (FWT) against a random-policy reference
* Absolute and normalized forgetting
* Mean episode reward with standard deviation and 95% confidence intervals
* Collision / PU interference rate
* Successful transmission rate
* Throughput in successful transmissions per environment step
* Normalized access efficiency
* Channel utilization and idle rate
* Normalized channel-selection entropy
* Adaptation speed
* ADWIN detection statistics and detection delay
* DQN, EWC, and total training loss
* Adaptive EWC coefficient trajectory
* TD-error evolution

The final policy and the **Random** and **Greedy/Myopic** baselines are evaluated over **200 episodes per task**.

The notebook also includes IEEE-standard charts and comparison outputs for research-oriented reporting.

## 🧪 Ablation Study

A built-in ablation pipeline isolates the contribution of the major components using the same task sequence and training budget:

* Vanilla DQN
* DQN + Fixed EWC
* DQN + Adaptive EWC
* DQN + ADWIN
* DQN + Adaptive EWC + ADWIN
* **Full Model: Adaptive EWC + ADWIN + Episodic Replay**

The experiment is configured for **five independent training seeds**:

```python
[42, 123, 2024, 31415, 27182]
```

Results are aggregated across independent runs using mean, standard deviation, and 95% confidence intervals.

## 📈 Generated Outputs

Training and evaluation generate:

```text
checkpoints/
├── metrics.csv
├── training_log.csv
├── baseline_comparison.csv
├── experiment_metadata.csv
├── metric_definitions.csv
├── ieee_full_metric_report.csv
└── ablations/
    ├── ablation_all_seeds.csv
    ├── ablation_per_task.csv
    ├── ablation_summary.csv
    ├── ablation_effect_vs_full.csv
    └── fig_ablation_acc.pdf
```

Additional PDF figures include the continual-learning evaluation matrix, retention trace, normalized forgetting, throughput/access efficiency, PU interference, baseline comparisons, loss decomposition, adaptive EWC coefficient, and TD-error trajectory.

## 🗂️ Repository Structure

* **Training** — sequential CRN training, drift response, EWC updates, episodic replay, rollback, and MEC coordination.
* **Evaluation** — continual-learning and spectrum-access evaluation metrics.
* **Agents** — DQN architecture and Boltzmann exploration policy.
* **Environment** — 8-channel CRN simulator and seven diurnal PU-traffic regimes.
* **EWC** — adaptive Elastic Weight Consolidation and Fisher-based parameter protection.
* **Change-Point Detection** — ADWIN implementation operating on the TD-error stream.
* **Replay** — active experience replay and Top-K episodic memory.
* **MEC Server** — asynchronous Fisher Information Matrix computation.
* **Notebook** — end-to-end training, evaluation, chart generation, validation, and ablation experiments.
* **app.py** — interactive Streamlit simulation of the trained CRN agent.

## 🚀 Installation

```bash
git clone https://github.com/ClusterClassifier/nonstationary-continual-crn-rl.git
cd nonstationary-continual-crn-rl
pip install -r requirements.txt
```

CUDA is automatically used by PyTorch when an available compatible GPU is detected; otherwise the framework runs on CPU.

## ▶️ Running the Project

### Training

```bash
python train.py
```

This executes sequential training across all seven CRN environments and generates the model checkpoints and evaluation data.

### Generate Charts

```bash
python export_charts.py
```

### Interactive Simulation

A deployed version of the real-time simulation is available here:

**[Launch the CRN RL Streamlit Simulation](https://nonstationary-continual-crn-rl-app.streamlit.app/)**

Or run it locally:

```bash
streamlit run app.py
```

The dashboard visualizes the 8-channel spectrum, PU activity, SU decisions, task changes, and live performance statistics.

## 🔬 Reproducibility

The framework records experiment configuration and supports deterministic random seeds. Evaluation uncertainty is kept separate from variability across independently trained models, while the ablation pipeline performs dedicated multi-seed training for statistically meaningful comparisons.

## License

Distributed under the **CC-BY-ND License**. See `LICENSE` for more information.
