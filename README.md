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

* train.py: The main entry point. Coordinates the sequential training loop, ADWIN drift signals, EWC updates, and MEC offloading.  
* evaluate.py: Executes the IEEE evaluation protocol, generating the ![][image2] matrix, Forgetting Rate (FR), and Backward Transfer (BWT).  
* agents.py: Contains the Feedforward DQN architecture and Boltzmann exploration policy.  
* env.py: Simulates the non-stationary 8-channel CRN environment across 7 distinct diurnal tasks.  
* ewc.py: Implements the Elastic Weight Consolidation logic and adaptive ![][image1] penalty scheduler.  
* cpd.py: Wraps the river ADWIN algorithm for autonomous concept drift detection based on TD-error.  
* replay.py: Manages the standard experience replay buffer and the Top-K Episodic Memory buffer.  
* mec\_server.py: Simulates an asynchronous Mobile Edge Computing (MEC) server to compute the Fisher Information Matrix (FIM) without blocking edge training.  
* app.py: A fully interactive, real-time web simulation dashboard built with Streamlit.  
* export\_charts.py: Generates publication-quality .pdf evaluation heatmaps and performance evolution charts.

## **🚀 Installation & Setup**

1. Clone this repository:  
   git clone https://github.com/YourUsername/continual-rl-crn.git  
   cd continual-rl-crn

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

## **📜 Citation & License**

**CUSTOM ACADEMIC USE LICENSE (NO MODIFICATIONS PERMITTED)**

Copyright (c) 2026 \[Your Name / Research Group\]. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software to execute, run, evaluate, and cite the Software for academic, non-commercial, and evaluation purposes, subject to the following conditions:

1. **ATTRIBUTION:** Any publication, presentation, paper, or derivative work that uses results produced by executing this Software must explicitly cite the original research paper:*\[Update this with your final IEEE paper title, authors, and DOI once published\]*  
2. **NO MODIFICATIONS:** The Software may NOT be modified, adapted, translated, reverse engineered, decompiled, or altered in any way. Redistribution or public hosting of altered or modified versions of this code is strictly prohibited.  
3. **RE-DISTRIBUTION:** You may redistribute verbatim, unmodified copies of the Software, provided that this copyright notice and permission notice appear in all copies.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAnCAYAAAA7HqkSAAACA0lEQVR4XpVUyyuEURSfv8VENBkixsKjpmGhPJIikigleaWURykbeSQLYuHVjAwWFBvyCMWMYmcjCztSlFgoZnF8534P537n3pHFr+9+557fPb/zuNezGf8GEwnYctYUCenfIxYx84cSGNnyMQnUydr4daIREuBhJ/0Bk2CdRCOwg7gkKzQj02Ik5BwY2QKNpiFYoAVwEyJn71Db0g/5hSEYnN6BjdjX3xGiF59Q09QLPn8uzGzdqgk0IVyPLhyB15sKnSOLkhRlBMTi/iMEDFmllU2wdv7himBXiXQUtde3DUOGLwum1m8cu0ygiKGsQyGrfWBWsuNXGj4by4fPECgqg2B5HYRP3/RJ22SUVdXQCWnpPpgIX5HDFJLQGaXkFpSANyUVWvsmpX1JUtRyDlU0wtzOvVGtMigOVcPKyatTGCcCntwxNA8FhvaFvQdha+4ZE7LGiSxB2IibzpnZeTAVuXY2UT8SmrvHfquEMrqMrmb4/DA8sytVbNWQgpKKghXG+sUkzG7fgT8nILQj2Xa2q4dJY09wZAQBta8ev0D08pPfaQNoXzp4Ev1wcnA70QjyjVMMH50rDkXj7A1mU82S7tWgD8L/3yW3IXkO9rTSKikI0o1jklg+rrIyAgN9EbWN00nS5qAHv6JsneTlE1AkqmmcnJxaqnaWkkwri8AI8voHiqL25aeYlmkAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAlCAYAAADIgFBEAAAEWElEQVR4Xr1WSU8UQRTmn3AgIgoKQXSQgAZBSURAMMoywQ0lgKMS4xAEQRSFsLiggqCBcUYW14QoLhCBqMO4HryoFy8SDxqNGhdcODz71Ux3V3e97pkJ6OFLd6pevfrqe9+r7pCLD/+AiGnifRouCXHBg8+hzxeiD1bgwedsEOAPZg6aDCPy/0GTMYCRSmbSa2GutELGLCgYqSlVvbl9OYh5GUEp86+hIaOqo1OCOI25kkQOAWLJZr9MJqD3UHNrlOkeeQ8bbYdh/Ra7hL3e52Z8t0OhrR6aXU+gf+InkTBAEArzIDwzDTuqz0BoaCiUHzqvjOOpSipPsXFraZ0Sy68TcwUGWbEQrcunYcA9BWs2FEP43PnQ6JjQLCqr7mBkUtcUgGv8i5BUAz8qULGCMt0j78CStIoB3+XgfvdPyMwvY2S27mkWE/qDb0OmggHREP1Eo8PNVMmy2mDA81sZR7/Mi4xmqjhGP/rG/ZSG2pQa80FQBn2Cpy+uOAFnb72Fs7ffwqZdDRAVs1jyzEm4cO+rkEQPqmuoMT00ZPrRL7nFEBY2B1avK1K6Kio6Dhp63MJi5/hnKNrTCtmFu6GuY0Td1PD0kielbuwefid15ZQwr2ttwi8SVmYWQtzSZOgcesMS8mu6hiYhflkatFx4KiSnUNc+DNGL4uHk1VfCnErGg36ZYH7BbkKV5PGsAhvzS0vvM0FuXJOYkqkhbwbMq3pOC40yZcr94tQEJa7IUMjoE+AamTxK77j7QWN8Hr33v7ESGc0rZJhf2P0SCQ26+2X+ghhYGGuB9sHXmnFsd1yDpse5gpJaSM2wslLoN2qVDoK3+1rrTiir6tDMcZee96Xr5iQssizj/KJ6Aw0duySJdReaNqewnHmlUwLGI5mK5ivQdvkFWBJXQpPrsUCmVLq98cDYmfnF+4V5RubIuXswJzyClYdHZl6p8h3CuwW9hJ8Da8kBsNWcY+ONPZLHIiIhSSojGnjA80vwlAwkgrc25qo+fh30jYDrhHuG+t5gIvTLvmODcGzguVJz2S/YZWjiytZrLLbvwQ/vev7WlYClTEheDW1XXgp7IAgyAYB9HqYgA/1Sf14y5ldIy97MTlxz6hYjLt9B9V1jCqlDnaOQkp4PzrFPSi5eSS8Z30dS3VD88dEDfZWUmgVNzkdMKTRv3vYq2G4/zoji70hCcrq33G5vuc38wpWJ2Jy6RXUfO/SAXDJ8Ylvz/zuusS+w+2APm0OlVqTnMeWEvD6wXwiVCPHh85j8jvoZR1PXnr4D2+xHWbmWp+WwLtTHyTkMPSMoRalngr4H36FKMjxehOX1TsgtqoTOG29McxiSmU14CYitrInzmJARgilQvjIFXU553JCMfrGRbwTSQmcGDj9k/i8CI0OcVm5vIXYGUMjwcgvSyyA2F2KJmEBhogzlF2LzoGHsJxMy3OIZnNYvuNzaMvGbGhII7vLj4W+dqozh5gTRGcG4TH8B11F3g9n+nn4AAAAASUVORK5CYII=>