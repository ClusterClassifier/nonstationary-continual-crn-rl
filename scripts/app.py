import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from collections import deque
import pandas as pd
import os

st.set_page_config(page_title="CRN Continual RL", layout="wide", page_icon="📡")

class CRNEnv:
    """Base Cognitive Radio Network Environment for Inference."""
    def __init__(self, num_channels=8, history_length=5):
        self.num_channels = num_channels
        self.history_length = history_length
        self.max_steps = 1000
        self.state_dim = self.num_channels * 2 * self.history_length
        self.action_space = self.num_channels + 1
        self.idle_action = self.num_channels
        self.R_success, self.R_collision, self.R_idle = 1.0, 1.0, 0.05
        self.pu_activity_probs = np.zeros(self.num_channels)
        self.snr_history = [deque(maxlen=self.history_length) for _ in range(self.num_channels)]
        self.occ_history = [deque(maxlen=self.history_length) for _ in range(self.num_channels)]
        self.current_step = 0

    def _set_probs(self, seed, base_low, base_high):
        rng = np.random.RandomState(seed)
        self.pu_activity_probs = rng.uniform(base_low, base_high, self.num_channels)
        self.pu_activity_probs[rng.randint(0, self.num_channels)] = max(0.01, base_low * 0.5)

    def reset(self):
        self.current_step = 0
        for c in range(self.num_channels):
            self.snr_history[c].clear()
            self.occ_history[c].clear()
        for _ in range(self.history_length):
            self._generate_observation()
        return self.get_state()

    def _generate_observation(self):
        for c in range(self.num_channels):
            is_busy = np.random.rand() < self.pu_activity_probs[c]
            snr = np.random.uniform(0.5, 1.0) if is_busy else np.random.uniform(0.0, 0.2)
            self.snr_history[c].append(snr)
            self.occ_history[c].append(1 if is_busy else 0)

    def step(self, action):
        current_occ = [self.occ_history[c][-1] for c in range(self.num_channels)]
        reward = 0.0
        if action == self.idle_action:
            reward = -self.R_idle
        elif 0 <= action < self.num_channels:
            reward = -self.R_collision if current_occ[action] == 1 else self.R_success
        
        self._generate_observation()
        self.current_step += 1
        return self.get_state(), reward, self.current_step >= self.max_steps, current_occ

    def get_state(self):
        state = []
        for c in range(self.num_channels):
            state.extend(self.snr_history[c])
            state.extend(self.occ_history[c])
        return np.array(state, dtype=np.float32)

class TaskEnv(CRNEnv):
    """Dynamic Task Environment wrapper."""
    def __init__(self, task_idx, **kwargs):
        super().__init__(**kwargs)
        # Match your 7 task seeds and probabilities
        configs = [
            (42, 0.20, 0.40), (43, 0.60, 0.90), (44, 0.40, 0.60), (45, 0.05, 0.20),
            (46, 0.65, 0.85), (47, 0.80, 0.95), (48, 0.01, 0.10)
        ]
        self._set_probs(*configs[task_idx])

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

class InferenceAgent:
    def __init__(self, input_dim, output_dim, checkpoint_path=None):
        self.device = torch.device("cpu") # Web deployments usually run on CPU
        self.policy_net = DQN(input_dim, output_dim).to(self.device)
        self.policy_net.eval()
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                # Load checkpoint trained during your simulation
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint['policy'] if 'policy' in checkpoint else checkpoint)
            except Exception as e:
                st.sidebar.error(f"Failed to load weights: {e}")

    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values, dim=1).item() # Greedy action for evaluation/deployment

def initialize_state():
    if 'env' not in st.session_state:
        st.session_state.current_task = 0
        st.session_state.env = TaskEnv(0)
        st.session_state.state = st.session_state.env.reset()
        st.session_state.agent = InferenceAgent(st.session_state.env.state_dim, st.session_state.env.action_space)
        st.session_state.history = deque(maxlen=100)
        st.session_state.total_reward = 0.0
        st.session_state.collisions = 0
        st.session_state.transmissions = 0
        st.session_state.is_running = False

initialize_state()

with st.sidebar:
    st.header("⚙️ Simulation Controls")
    
    # Task Switcher
    task_names = [
        "T1: Early Commute (Low)", "T2: Peak Commute (High)", "T3: Midday Office (Med)", 
        "T4: Sparse IoT (Low)", "T5: Evening Commute (High)", "T6: Heavy Streaming (Very High)", "T7: Night Low (Very Low)"
    ]
    selected_task = st.selectbox("Select Traffic Environment", range(7), format_func=lambda x: task_names[x])
    
    if selected_task != st.session_state.current_task:
        st.session_state.current_task = selected_task
        st.session_state.env = TaskEnv(selected_task)
        st.session_state.state = st.session_state.env.reset()
        st.session_state.total_reward = 0.0
        st.session_state.collisions = 0
        st.session_state.transmissions = 0
        st.session_state.history.clear()
        st.rerun()

    # Load Checkpoint Check
    ckpt_file = st.text_input("Load Checkpoint Path (.pt)", value="checkpoints/task_7_policy.pt")
    if st.button("Load Weights"):
        st.session_state.agent = InferenceAgent(st.session_state.env.state_dim, st.session_state.env.action_space, ckpt_file)
        st.success("Weights loaded successfully!")

    speed = st.slider("Simulation Speed (Delay in sec)", 0.0, 1.0, 0.1)
    
    if st.button("Start / Pause Simulation"):
        st.session_state.is_running = not st.session_state.is_running

st.title("📡 Non-Stationary Continual Cognitive Radio Network Simulation")
st.markdown("This dashboard simulates the **actual DQN-based algorithm** used in the project in real-time.")

col1, col2, col3, col4 = st.columns(4)
metric_reward = col1.empty()
metric_cr = col2.empty()
metric_action = col3.empty()
metric_step = col4.empty()

st.subheader("📻 Live 8-Channel Spectrum Occupancy")
channels_ui = st.columns(8)
channel_placeholders = [col.empty() for col in channels_ui]

st.subheader("📈 Real-Time Reward Trajectory")
chart_placeholder = st.empty()

if st.session_state.is_running:
    # Perform one step
    action = st.session_state.agent.get_action(st.session_state.state)
    next_state, reward, done, current_occ = st.session_state.env.step(action)
    
    # Update metrics
    st.session_state.total_reward += reward
    if action != 8:
        st.session_state.transmissions += 1
        if reward == -1.0:
            st.session_state.collisions += 1
            
    st.session_state.history.append(reward)
    st.session_state.state = env_state = next_state if not done else st.session_state.env.reset()

    # Render Metrics
    cr = (st.session_state.collisions / st.session_state.transmissions * 100) if st.session_state.transmissions > 0 else 0.0
    metric_reward.metric("Cumulative Reward", f"{st.session_state.total_reward:.2f}", f"{reward:+.2f}")
    metric_cr.metric("Collision Rate", f"{cr:.1f}%")
    
    action_text = f"Channel {action + 1}" if action < 8 else "Idle (Standby)"
    metric_action.metric("Agent Action", action_text)
    metric_step.metric("Step Count", st.session_state.env.current_step)

    # Render Channel Grid
    for i in range(8):
        is_busy = current_occ[i] == 1
        is_chosen = (action == i)
        
        bg_color = "#ffebee" if is_busy else "#e8f5e9"
        border = "2px solid #3f51b5" if is_chosen else "1px solid #ddd"
        status = "🔴 BUSY (PU)" if is_busy else "🟢 FREE"
        agent_marker = "⚡ Tx Active" if is_chosen else ""
        
        card_html = f"""
        <div style="background-color: {bg_color}; border: {border}; border-radius: 8px; padding: 10px; text-align: center; height: 100px;">
            <h4 style="margin:0; color: #333;">CH {i+1}</h4>
            <p style="margin: 5px 0; font-size: 12px; color: #555;">{status}</p>
            <p style="margin: 0; font-weight: bold; color: #3f51b5; font-size: 12px;">{agent_marker}</p>
        </div>
        """
        channel_placeholders[i].markdown(card_html, unsafe_allow_html=True)

    # Render Chart
    if len(st.session_state.history) > 0:
        chart_placeholder.line_chart(list(st.session_state.history), height=200)

    # Loop control
    time.sleep(speed)
    st.rerun()
