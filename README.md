# **Continual Reinforcement Learning for Non-Stationary Cognitive Radio Networks 📡**

This repository contains the official implementation of a Continual Reinforcement Learning (CRL) framework designed for autonomous Cognitive Radio Networks (CRNs). The system enables a secondary user agent to dynamically adapt to non-stationary, diurnal spectrum traffic patterns without suffering from catastrophic forgetting.

## **📖 Overview**

In real-world CRNs, Primary User (PU) traffic patterns shift significantly throughout the day (e.g., from sparse morning traffic to heavy evening video streaming). Traditional Reinforcement Learning (RL) agents overwrite their learned weights when adapting to new traffic regimes, completely forgetting how to navigate previous environments (Catastrophic Forgetting).

This framework solves this using a Deep Q-Network (DQN) integrated with three core Continual Learning mechanisms:

1. **Autonomous Concept Drift Detection (ADWIN):** Monitors the Temporal Difference (TD) error stream to detect task boundaries autonomously without explicit task labels.  
2. **Elastic Weight Consolidation (EWC):** Applies a dynamic, adaptive penalty (![][image1]) to protect critical neural network weights associated with past environments.  
3. **Episodic Memory Replay:** Curates and rehearses the "Top-K" most surprising transitions from previous tasks to anchor the EWC penalty and ensure robust backward transfer.

## **📊 Evaluation Results (IEEE Standard Metrics)**

The framework was evaluated across a 7-task diurnal cycle (T1 to T7), ranging from extremely sparse IoT traffic to heavily congested video streaming environments. Evaluated over 100 episodes per task, the agent achieved exceptional stability.

**Key Performance Metrics (After Training Task 7):**

* **Positive Backward Transfer (BWT):** \+7.39. The agent successfully utilized generalized knowledge from later tasks to retroactively improve its performance on earlier environments.  
* **Negative Forgetting Rates (FR):** Standard algorithms struggle to keep FR low. This agent achieved *negative* forgetting rates (e.g., FR\_T3 \= \-65.91), meaning performance actually improved on past tasks over time.  
* **High-Traffic Resilience:** In the most congested environment (T6: 95% PU activity), the agent maintained a high mean reward (725.56) while keeping the collision rate to a strictly cautious 10.8%. In low-to-medium traffic tasks (T1, T4, T7), collision rates remained at or near 0.0% with near-perfect rewards.

## **🗂️ Repository Structure**

* Training: The main entry point. Coordinates the sequential training loop, ADWIN drift signals, EWC updates, and MEC offloading.  
* Evaluation: Executes the IEEE evaluation protocol, generating the evaluation matrix, Forgetting Rate (FR), and Backward Transfer (BWT).  
* Agents: Contains the Feedforward DQN architecture and Boltzmann exploration policy.  
* Environment: Simulates the non-stationary 8-channel CRN environment across 7 distinct diurnal tasks.  
* Elastic Weight Consolidation: Implements the Elastic Weight Consolidation logic and adaptive ![][image1] penalty scheduler.  
* Change-Point Detection: Wraps the river ADWIN algorithm for autonomous concept drift detection based on TD-error.  
* Replay Classes: Manages the standard experience replay buffer and the Top-K Episodic Memory buffer.  
* MEC Server: Simulates an asynchronous Mobile Edge Computing (MEC) server to compute the Fisher Information Matrix (FIM) without blocking edge training.  
* app.py: A fully interactive, real-time web simulation dashboard built with Streamlit.  
* Exports: Generates publication-quality .pdf evaluation heatmaps and performance evolution charts.

## **🚀 Installation & Setup**

1. Clone this repository:  
   git clone https://github.com/ClusterClassifier/nonstationary-continual-crn-rl.git  
   cd nonstationary-continual-crn-rl

2. Install the required dependencies:  
   pip install \-r requirements.txt

   *(Note: The river library is required for ADWIN concept drift detection, and streamlit is required for the live simulation dashboard).*

## **🧠 Running the Code**

### **1\. Training the Agent**

To run the full sequential training process across all 7 tasks:

python train.py

This will automatically generate checkpoints, rollback files, and the final metrics.csv in the checkpoints/ directory.

### **2\. Generating Evaluation Charts**

After training completes, generate the IEEE-formatted evaluation matrix and forgetting charts:

python export\_charts.py

### **3\. Launching the Real-Time Simulation**

To visualize the trained agent's decision-making process in real-time, run the interactive Streamlit dashboard:

streamlit run app.py

The dashboard features an interactive 8-channel RF spectrum monitor, task switching, and real-time performance logging.

## License
Distributed under the CC-BY-ND License. See `LICENSE` for more information.

