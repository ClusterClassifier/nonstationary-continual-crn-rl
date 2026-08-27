"""
Continual RL Cognitive Radio Network — Real-Time Model Dashboard

What is real in this dashboard:
- Loads the actual saved PyTorch policy + target networks from task_*_policy.pt checkpoints.
- Infers the channel count and state-history length from the checkpoint architecture.
- Runs real greedy DQN inference at every simulator step.
- Computes the real one-step TD target/error from the loaded policy and target networks.
- Feeds absolute TD error (not reward) to the same style of ADWIN drift detector used in training.
- Uses the same task occupancy ranges, state construction, action space, and reward structure as training.
- Supports manual task changes and an automatic T1→T7 diurnal non-stationary cycle.

Important scope note:
EWC, episodic replay, Fisher consolidation, rollback, and MEC are training/adaptation mechanisms.
A saved inference checkpoint does not execute those mechanisms by itself. The dashboard therefore
shows their saved artifact availability and the live signals that drove them during training, without
pretending that inference-only execution is performing Fisher/MEC updates.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import pickle
import re
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="CRN-CRL Live Model Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CONSTANTS — MATCH TRAINING NOTEBOOK
# =============================================================================

GAMMA = 0.97
DEFAULT_ADWIN_DELTA = 0.002
HISTORY_FALLBACK = 5
EPISODE_LENGTH = 1000
CHECKPOINT_DIR = Path("checkpoints")

TASK_META = {
    "T1 — Early Commute (06:00)": {
        "id": 1, "seed": 42, "low": 0.20, "high": 0.40, "desc": "Low–Medium PU load"
    },
    "T2 — Peak Commute (09:00)": {
        "id": 2, "seed": 43, "low": 0.60, "high": 0.90, "desc": "High PU load"
    },
    "T3 — Midday Office (12:00)": {
        "id": 3, "seed": 44, "low": 0.40, "high": 0.60, "desc": "Medium PU load"
    },
    "T4 — IoT Sensor Bursts (14:00)": {
        "id": 4, "seed": 45, "low": 0.05, "high": 0.20, "desc": "Very Low PU load"
    },
    "T5 — Evening Commute (17:00)": {
        "id": 5, "seed": 46, "low": 0.65, "high": 0.85, "desc": "High PU load"
    },
    "T6 — Video Streaming (20:00)": {
        "id": 6, "seed": 47, "low": 0.80, "high": 0.95, "desc": "Very High PU load"
    },
    "T7 — Night Low (23:00)": {
        "id": 7, "seed": 48, "low": 0.01, "high": 0.10, "desc": "Very Low PU load"
    },
}
TASK_KEYS = list(TASK_META.keys())


# =============================================================================
# THEME
# =============================================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0A0E1A;
    color: #C8D6E5;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #0D1220;
    border-right: 1px solid #1E2D45;
}
[data-testid="stSidebar"] * { color: #C8D6E5 !important; }
h1, h2, h3 { font-family: 'Share Tech Mono', monospace; letter-spacing: 0.04em; }
h1 { color: #0FF4C6; }
h2 { color: #D7E4F0; }
h3 { color: #4A90D9; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.1em; }
[data-testid="metric-container"] {
    background: #1A2235;
    border: 1px solid #1E2D45;
    border-radius: 8px;
    padding: 10px 14px;
}
[data-testid="metric-container"] label {
    color: #6B8299 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace;
    color: #0FF4C6 !important;
    font-size: 1.35rem !important;
}
.stButton > button {
    background: #1A2235;
    border: 1px solid #4A90D9;
    color: #4A90D9;
    border-radius: 6px;
    font-family: 'Share Tech Mono', monospace;
    width: 100%;
}
.stButton > button:hover { background: #4A90D9; color: #0A0E1A; }
.ch-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 9px;
    margin: 0 0 16px 0;
}
.ch-card {
    border-radius: 8px;
    padding: 12px 8px;
    text-align: center;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    border: 2px solid transparent;
}
.ch-clear  { background: #0D1F16; border-color: #1A4A30; color: #0FF4C6; }
.ch-busy   { background: #1F0D10; border-color: #4A1A20; color: #FF3B5C; }
.ch-idle   { background: #1A1800; border-color: #3A3000; color: #FFB800; }
.ch-selected-clear { background: #0A2E20; border-color: #0FF4C6; color: #0FF4C6; box-shadow: 0 0 14px #0FF4C630; }
.ch-selected-busy  { background: #2E0A0F; border-color: #FF3B5C; color: #FF3B5C; box-shadow: 0 0 14px #FF3B5C30; }
.ch-label  { font-size: 0.64rem; color: #6B8299; margin-bottom: 3px; }
.ch-prob   { font-size: 0.68rem; color: #6B8299; margin-top: 2px; }
.log-box {
    background: #0D1220;
    border: 1px solid #1E2D45;
    border-radius: 8px;
    padding: 12px 14px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.73rem;
    height: 235px;
    overflow-y: auto;
    color: #6B8299;
}
.log-ok  { color: #0FF4C6; }
.log-col { color: #FF3B5C; }
.log-idl { color: #FFB800; }
.log-det { color: #C084FC; }
.log-sys { color: #4A90D9; }
.divider { border: none; border-top: 1px solid #1E2D45; margin: 14px 0; }
.status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
}
.badge-run  { background: #0A2E20; color: #0FF4C6; border: 1px solid #0FF4C6; }
.badge-stop { background: #1A2235; color: #6B8299; border: 1px solid #2A3A55; }
.pipeline {
    display:grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap:8px;
    margin:6px 0 14px 0;
}
.pipe-card {
    background:#111827;
    border:1px solid #1E2D45;
    border-radius:8px;
    padding:10px;
    font-family:'Share Tech Mono', monospace;
    font-size:0.70rem;
    min-height:74px;
}
.pipe-title { color:#D7E4F0; font-size:0.72rem; margin-bottom:5px; }
.pipe-active { color:#0FF4C6; }
.pipe-training { color:#FFB800; }
.pipe-muted { color:#6B8299; }
.small-note { color:#6B8299; font-size:0.76rem; }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# MODEL
# =============================================================================

class DQN(nn.Module):
    def __init__(self, input_dim: int, hidden1: int, hidden2: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def safe_torch_load(source):
    """Load only tensors/state dictionaries when supported by installed PyTorch."""
    try:
        return torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(source, map_location="cpu")


def extract_checkpoint_state(checkpoint_obj):
    if not isinstance(checkpoint_obj, dict):
        raise ValueError("Checkpoint must be a dictionary/state_dict.")

    if "policy" in checkpoint_obj:
        policy_state = checkpoint_obj["policy"]
        target_state = checkpoint_obj.get("target")
    else:
        policy_state = checkpoint_obj
        target_state = None

    required = {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"}
    if not required.issubset(policy_state.keys()):
        missing = sorted(required.difference(policy_state.keys()))
        raise ValueError(f"Unsupported checkpoint architecture. Missing keys: {missing}")

    if target_state is None:
        target_state = deepcopy(policy_state)
        target_origin = "cloned from policy (checkpoint has no target network)"
    else:
        target_origin = "loaded from checkpoint"

    input_dim = int(policy_state["fc1.weight"].shape[1])
    hidden1 = int(policy_state["fc1.weight"].shape[0])
    hidden2 = int(policy_state["fc2.weight"].shape[0])
    output_dim = int(policy_state["fc3.weight"].shape[0])

    num_channels = output_dim - 1
    if num_channels <= 0:
        raise ValueError("Output layer does not contain channel actions + idle action.")

    denom = 2 * num_channels
    if input_dim % denom != 0:
        raise ValueError(
            f"Checkpoint dimensions are inconsistent with CRN state encoding: "
            f"input={input_dim}, output={output_dim}."
        )
    history_length = input_dim // denom

    return {
        "policy_state": policy_state,
        "target_state": target_state,
        "target_origin": target_origin,
        "input_dim": input_dim,
        "hidden1": hidden1,
        "hidden2": hidden2,
        "output_dim": output_dim,
        "num_channels": num_channels,
        "history_length": history_length,
    }


def build_networks(meta):
    policy = DQN(meta["input_dim"], meta["hidden1"], meta["hidden2"], meta["output_dim"])
    target = DQN(meta["input_dim"], meta["hidden1"], meta["hidden2"], meta["output_dim"])
    policy.load_state_dict(meta["policy_state"])
    target.load_state_dict(meta["target_state"])
    policy.eval()
    target.eval()
    return policy, target


# =============================================================================
# CHECKPOINT DISCOVERY / ASSOCIATED TRAINING ARTIFACTS
# =============================================================================

def discover_checkpoints(directory: Path):
    if not directory.exists():
        return []
    files = [p for p in directory.iterdir() if p.is_file() and (p.name.endswith("_policy.pt") or p.name == "rollback.pt")]

    def key(p: Path):
        m = re.search(r"task_(\d+)_policy\.pt$", p.name)
        if m:
            return (0, int(m.group(1)), p.name)
        return (1, 999, p.name)

    return sorted(files, key=key)


def artifact_paths_for_checkpoint(path: Path | None):
    if path is None:
        return {}
    m = re.search(r"task_(\d+)_policy\.pt$", path.name)
    if not m:
        return {}
    tid = int(m.group(1))
    base = path.parent
    return {
        "fisher": base / f"task_{tid}_fisher.pkl",
        "theta_star": base / f"task_{tid}_theta_star.pkl",
        "episodic": base / f"task_{tid}_episodic.pkl",
    }


def checkpoint_stage_label(name: str):
    m = re.search(r"task_(\d+)_policy\.pt$", name)
    return f"Task {m.group(1)} boundary checkpoint" if m else "Checkpoint"


# =============================================================================
# CRN ENVIRONMENT — MATCHES THE NOTEBOOK ABSTRACTION
# =============================================================================

class CRNEnvironment:
    def __init__(
        self,
        num_channels: int,
        history_length: int,
        seed: int,
        base_low: float,
        base_high: float,
        episode_length: int = EPISODE_LENGTH,
    ):
        self.num_channels = num_channels
        self.history_length = history_length
        self.seed = int(seed)
        self.base_low = float(base_low)
        self.base_high = float(base_high)
        self.episode_length = int(episode_length)
        self.rng = np.random.RandomState(self.seed)

        self.pu_probs = self.rng.uniform(self.base_low, self.base_high, self.num_channels)
        clearer = self.rng.randint(0, self.num_channels)
        self.pu_probs[clearer] = max(0.01, self.base_low * 0.5)

        self.snr_hist = [deque(maxlen=self.history_length) for _ in range(self.num_channels)]
        self.occ_hist = [deque(maxlen=self.history_length) for _ in range(self.num_channels)]

        self.R_success = 1.0
        self.R_collision = -1.0
        self.R_idle = -0.05
        self.idle_action = self.num_channels
        self.action_space = self.num_channels + 1
        self.state_dim = self.num_channels * 2 * self.history_length
        self.episode_step = 0
        self.reset()

    def _generate_channels(self):
        for c in range(self.num_channels):
            busy = self.rng.rand() < self.pu_probs[c]
            snr = self.rng.uniform(0.5, 1.0) if busy else self.rng.uniform(0.0, 0.2)
            self.snr_hist[c].append(float(snr))
            self.occ_hist[c].append(1 if busy else 0)

    def reset(self):
        for d in self.snr_hist:
            d.clear()
        for d in self.occ_hist:
            d.clear()
        self.episode_step = 0
        for _ in range(self.history_length):
            self._generate_channels()
        return self.get_state()

    def get_state(self) -> np.ndarray:
        state = []
        for c in range(self.num_channels):
            state.extend(self.snr_hist[c])
            state.extend(self.occ_hist[c])
        return np.asarray(state, dtype=np.float32)

    def current_occupancy(self):
        return [int(self.occ_hist[c][-1]) for c in range(self.num_channels)]

    def step(self, action: int):
        # Snapshot the state on which this decision is judged.
        decision_occupancy = self.current_occupancy()

        if action == self.idle_action:
            reward, outcome = self.R_idle, "idle"
        elif decision_occupancy[action] == 1:
            reward, outcome = self.R_collision, "collision"
        else:
            reward, outcome = self.R_success, "success"

        self._generate_channels()
        self.episode_step += 1
        done = self.episode_step >= self.episode_length
        next_state = self.get_state()
        return next_state, float(reward), outcome, done, decision_occupancy


# =============================================================================
# CUSTOM ADWIN — SAME SIGNAL TYPE AS TRAINING: ABSOLUTE TD ERROR
# =============================================================================

class ADWIN:
    _MAX_BUCKETS = 5

    class _Bucket:
        __slots__ = ("total", "variance", "size")

        def __init__(self):
            self.total = 0.0
            self.variance = 0.0
            self.size = 0

    def __init__(self, delta: float = DEFAULT_ADWIN_DELTA):
        self.delta = float(delta)
        self._total = 0.0
        self._variance = 0.0
        self._width = 0
        self._bucket_rows = [[]]
        self.drift_detected = False

    @property
    def width(self):
        return self._width

    def update(self, value: float):
        self.drift_detected = False
        self._insert(float(value))
        self._check_drift()
        return self.drift_detected

    def _insert(self, value):
        b = self._Bucket()
        b.total = value
        b.size = 1
        old_width = self._width
        old_mean = self._total / old_width if old_width > 0 else value
        self._width += 1
        new_mean = (self._total + value) / self._width
        self._variance += (value - old_mean) * (value - new_mean)
        self._total += value
        self._bucket_rows[0].insert(0, b)
        self._compress()

    def _compress(self):
        level = 0
        while level < len(self._bucket_rows):
            row = self._bucket_rows[level]
            if len(row) <= self._MAX_BUCKETS:
                break
            if level + 1 >= len(self._bucket_rows):
                self._bucket_rows.append([])
            b1, b2 = row[-2], row[-1]
            m = self._Bucket()
            m.size = b1.size + b2.size
            m.total = b1.total + b2.total
            d = (b2.total / b2.size - b1.total / b1.size) if b1.size and b2.size else 0.0
            m.variance = b1.variance + b2.variance + d * d * b1.size * b2.size / max(m.size, 1)
            self._bucket_rows[level] = row[:-2]
            self._bucket_rows[level + 1].insert(0, m)
            level += 1

    def _check_drift(self):
        if self._width < 4:
            return
        n0 = 0.0
        total0 = 0.0
        for row in reversed(self._bucket_rows):
            for bucket in reversed(row):
                n0 += bucket.size
                total0 += bucket.total
                n1 = self._width - n0
                if n1 <= 0 or n0 <= 0:
                    continue
                m0 = total0 / n0
                m1 = (self._total - total0) / n1
                dd = math.log(2.0 * math.log(self._width + 1) / self.delta)
                eps = math.sqrt(dd / (2.0 * n0)) + math.sqrt(dd / (2.0 * n1))
                if abs(m0 - m1) >= eps:
                    self._shrink(int(n0))
                    self.drift_detected = True
                    return

    def _shrink(self, drop):
        removed = 0.0
        rem = int(drop)
        for level in range(len(self._bucket_rows) - 1, -1, -1):
            row = self._bucket_rows[level]
            while row and rem > 0:
                b = row[-1]
                if b.size <= rem:
                    rem -= b.size
                    removed += b.total
                    row.pop()
                else:
                    fraction = rem / b.size
                    removed += b.total * fraction
                    b.total -= b.total * fraction
                    b.size -= rem
                    rem = 0
        self._width = max(0, self._width - int(drop))
        self._total -= removed
        self._variance = max(0.0, self._variance)


# =============================================================================
# SESSION HELPERS
# =============================================================================

def append_log(message: str, css: str = "log-sys"):
    st.session_state.log.appendleft(f"<span class='{css}'>{message}</span>")


def make_environment(task_key: str):
    meta = TASK_META[task_key]
    return CRNEnvironment(
        num_channels=st.session_state.model_meta["num_channels"],
        history_length=st.session_state.model_meta["history_length"],
        seed=meta["seed"],
        base_low=meta["low"],
        base_high=meta["high"],
    )


def reset_runtime(task_key: str, reset_detector: bool = True):
    ss = st.session_state
    ss.env = make_environment(task_key)
    ss.active_task = task_key
    ss.task_step = 0
    ss.global_step = 0
    ss.rewards = deque(maxlen=150)
    ss.td_errors = deque(maxlen=150)
    ss.collisions = 0
    ss.successes = 0
    ss.idles = 0
    ss.log = deque(maxlen=60)
    ss.last_action = None
    ss.last_outcome = None
    ss.last_reward = None
    ss.last_td_error = None
    ss.last_td_target = None
    ss.last_q_selected = None
    ss.last_q_values = None
    ss.last_decision_occupancy = ss.env.current_occupancy()
    ss.drift_count = 0
    ss.episode_count = 0
    ss.auto_task_index = TASK_KEYS.index(task_key)
    if reset_detector or "adwin" not in ss:
        ss.adwin = ADWIN(delta=ss.adwin_delta)
    append_log(f"SYSTEM → Environment initialized: {task_key}")


def switch_task(task_key: str, preserve_detector: bool = True):
    ss = st.session_state
    if task_key == ss.active_task:
        return
    previous = ss.active_task
    ss.env = make_environment(task_key)
    ss.active_task = task_key
    ss.task_step = 0
    ss.auto_task_index = TASK_KEYS.index(task_key)
    if not preserve_detector:
        ss.adwin = ADWIN(delta=ss.adwin_delta)
    append_log(f"TASK SHIFT → {previous} → {task_key} | ADWIN window preserved={preserve_detector}", "log-det")


def initialize_model(checkpoint_obj, source_id: str, source_label: str, checkpoint_path: Path | None):
    ss = st.session_state
    model_meta = extract_checkpoint_state(checkpoint_obj)
    policy, target = build_networks(model_meta)
    ss.policy_net = policy
    ss.target_net = target
    ss.model_meta = model_meta
    ss.model_source_id = source_id
    ss.model_source_label = source_label
    ss.model_checkpoint_path = str(checkpoint_path) if checkpoint_path else None
    ss.original_policy_state = deepcopy(model_meta["policy_state"])
    ss.original_target_state = deepcopy(model_meta["target_state"])
    reset_runtime(TASK_KEYS[0], reset_detector=True)


def run_one_step():
    ss = st.session_state
    env = ss.env
    policy = ss.policy_net
    target = ss.target_net

    state_np = env.get_state()
    state_t = torch.from_numpy(state_np).float().unsqueeze(0)

    with torch.no_grad():
        q_values_t = policy(state_t)
        action = int(torch.argmax(q_values_t, dim=1).item())
        q_selected = float(q_values_t[0, action].item())

    next_state_np, reward, outcome, done, decision_occ = env.step(action)
    next_state_t = torch.from_numpy(next_state_np).float().unsqueeze(0)

    with torch.no_grad():
        next_q = float(target(next_state_t).max(dim=1).values.item())
        td_target = reward if done else reward + GAMMA * next_q
        td_error = abs(q_selected - td_target)

    drift = ss.adwin.update(td_error)

    ss.global_step += 1
    ss.task_step += 1
    ss.rewards.append(reward)
    ss.td_errors.append(td_error)
    ss.last_action = action
    ss.last_outcome = outcome
    ss.last_reward = reward
    ss.last_td_error = td_error
    ss.last_td_target = td_target
    ss.last_q_selected = q_selected
    ss.last_q_values = q_values_t.squeeze(0).cpu().numpy()
    ss.last_decision_occupancy = decision_occ

    if outcome == "success":
        ss.successes += 1
    elif outcome == "collision":
        ss.collisions += 1
    else:
        ss.idles += 1

    step_str = f"[{ss.global_step:05d}]"
    if drift:
        ss.drift_count += 1
        append_log(
            f"{step_str} ⚡ ADWIN DRIFT | |TD|={td_error:.4f} | window={ss.adwin.width}",
            "log-det",
        )

    if outcome == "success":
        append_log(
            f"{step_str} CH{action + 1:02d} → TX OK | r={reward:+.2f} | |TD|={td_error:.4f}",
            "log-ok",
        )
    elif outcome == "collision":
        append_log(
            f"{step_str} CH{action + 1:02d} → COLLISION | r={reward:+.2f} | |TD|={td_error:.4f}",
            "log-col",
        )
    else:
        append_log(
            f"{step_str} IDLE → no TX | r={reward:+.2f} | |TD|={td_error:.4f}",
            "log-idl",
        )

    if done:
        ss.episode_count += 1
        env.reset()
        append_log(f"{step_str} EPISODE COMPLETE → environment history reset", "log-sys")

    # Automatic non-stationarity: switch tasks WITHOUT resetting ADWIN.
    if ss.simulation_mode == "Automatic diurnal cycle":
        if ss.task_step >= ss.steps_per_task:
            next_idx = (TASK_KEYS.index(ss.active_task) + 1) % len(TASK_KEYS)
            switch_task(TASK_KEYS[next_idx], preserve_detector=True)


# =============================================================================
# SIDEBAR — CHECKPOINT / SIMULATION CONTROLS
# =============================================================================

for k, default in {
    "running": False,
    "do_step": False,
    "model_source_id": None,
    "adwin_delta": DEFAULT_ADWIN_DELTA,
}.items():
    st.session_state.setdefault(k, default)

repo_checkpoints = discover_checkpoints(CHECKPOINT_DIR)

with st.sidebar:
    st.markdown("# 📡 CRN-CRL Model Hub")
    st.markdown(
        "<p style='color:#6B8299;font-size:0.78rem;margin-top:-8px'>Live inference + TD-error drift monitor</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown("### Model Checkpoint")
    uploaded = st.file_uploader("Upload task_*_policy.pt", type=["pt"], accept_multiple_files=False)

    selected_repo_path = None
    if repo_checkpoints:
        names = [p.name for p in repo_checkpoints]
        # Prefer the latest task checkpoint as a continual-policy demo.
        default_index = max(range(len(names)), key=lambda i: int(re.search(r"task_(\d+)", names[i]).group(1)) if re.search(r"task_(\d+)", names[i]) else -1)
        selected_name = st.selectbox("Repository checkpoint", names, index=default_index)
        selected_repo_path = CHECKPOINT_DIR / selected_name
    else:
        st.caption("No repository checkpoints discovered in checkpoints/.")

    if uploaded is not None:
        checkpoint_bytes = uploaded.getvalue()
        source_id = "upload:" + hashlib.sha256(checkpoint_bytes).hexdigest()
        source_label = uploaded.name
        checkpoint_path = None
        try:
            checkpoint_obj = safe_torch_load(io.BytesIO(checkpoint_bytes))
        except Exception as exc:
            st.error(f"Unable to read uploaded checkpoint: {exc}")
            st.stop()
    elif selected_repo_path is not None:
        stat = selected_repo_path.stat()
        source_id = f"path:{selected_repo_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
        source_label = selected_repo_path.name
        checkpoint_path = selected_repo_path
        try:
            checkpoint_obj = safe_torch_load(selected_repo_path)
        except Exception as exc:
            st.error(f"Unable to read checkpoint: {exc}")
            st.stop()
    else:
        st.error("No trained checkpoint is available. Add task_*_policy.pt to checkpoints/ or upload one above.")
        st.stop()

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### Simulation")
    simulation_mode = st.radio(
        "Mode",
        ["Manual task", "Automatic diurnal cycle"],
        horizontal=False,
    )
    st.session_state.simulation_mode = simulation_mode

    manual_task = st.selectbox("Task environment", TASK_KEYS, disabled=(simulation_mode != "Manual task"))

    steps_per_task = st.slider(
        "Auto-cycle steps per task",
        min_value=100,
        max_value=2000,
        value=400,
        step=100,
        disabled=(simulation_mode != "Automatic diurnal cycle"),
    )
    st.session_state.steps_per_task = steps_per_task

    delta = st.select_slider(
        "ADWIN sensitivity (δ)",
        options=[0.001, 0.002, 0.005, 0.01, 0.05, 0.1],
        value=DEFAULT_ADWIN_DELTA,
        help="Training configuration uses δ = 0.002.",
    )
    speed = st.slider("Simulation speed (steps/sec)", 1, 20, 5, 1)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        run_btn = st.button("▶ Run")
        reset_btn = st.button("↺ Reset")
    with c2:
        pause_btn = st.button("⏸ Pause")
        step_btn = st.button("⏭ Step")

    if run_btn:
        st.session_state.running = True
    if pause_btn:
        st.session_state.running = False
    if step_btn:
        st.session_state.running = False
        st.session_state.do_step = True


# =============================================================================
# INITIALIZE / RESPOND TO CHECKPOINT & CONTROL CHANGES
# =============================================================================

if st.session_state.model_source_id != source_id:
    try:
        initialize_model(checkpoint_obj, source_id, source_label, checkpoint_path)
    except Exception as exc:
        st.error(f"Checkpoint is incompatible with this dashboard: {exc}")
        st.stop()

ss = st.session_state

# ADWIN delta changes reset detector only; they do not alter the model.
if abs(float(ss.adwin_delta) - float(delta)) > 1e-15:
    ss.adwin_delta = float(delta)
    ss.adwin = ADWIN(delta=ss.adwin_delta)
    append_log(f"ADWIN CONFIG → δ changed to {ss.adwin_delta}; detector window reset", "log-det")

if reset_btn:
    ss.running = False
    ss.policy_net.load_state_dict(ss.original_policy_state)
    ss.target_net.load_state_dict(ss.original_target_state)
    task_to_reset = manual_task if simulation_mode == "Manual task" else TASK_KEYS[0]
    reset_runtime(task_to_reset, reset_detector=True)

# Manual task switching preserves ADWIN so the actual distribution change is visible.
if simulation_mode == "Manual task" and manual_task != ss.active_task:
    switch_task(manual_task, preserve_detector=True)

if ss.get("do_step"):
    run_one_step()
    ss.do_step = False


# =============================================================================
# HEADER + ARCHITECTURE VALIDATION
# =============================================================================

header_left, header_right = st.columns([5, 1])
with header_left:
    st.markdown(f"## {ss.active_task}")
    st.markdown(
        f"<div class='small-note'>{TASK_META[ss.active_task]['desc']} · Loaded: <b>{ss.model_source_label}</b></div>",
        unsafe_allow_html=True,
    )
with header_right:
    badge = "<span class='status-badge badge-run'>● LIVE</span>" if ss.running else "<span class='status-badge badge-stop'>● PAUSED</span>"
    st.markdown(f"<div style='margin-top:14px'>{badge}</div>", unsafe_allow_html=True)

meta = ss.model_meta
arch_cols = st.columns(5)
arch_cols[0].metric("State Dim", meta["input_dim"])
arch_cols[1].metric("Channels", meta["num_channels"])
arch_cols[2].metric("Actions", meta["output_dim"])
arch_cols[3].metric("History", meta["history_length"])
arch_cols[4].metric("Network", f"{meta['hidden1']}×{meta['hidden2']}")

st.caption(
    f"Architecture inferred directly from checkpoint tensors: "
    f"{meta['input_dim']} → {meta['hidden1']} → {meta['hidden2']} → {meta['output_dim']} | "
    f"Target network: {meta['target_origin']}."
)


# =============================================================================
# LIVE PIPELINE
# =============================================================================

checkpoint_artifacts = artifact_paths_for_checkpoint(Path(ss.model_checkpoint_path) if ss.model_checkpoint_path else None)
fisher_found = bool(checkpoint_artifacts.get("fisher") and checkpoint_artifacts["fisher"].exists())
theta_found = bool(checkpoint_artifacts.get("theta_star") and checkpoint_artifacts["theta_star"].exists())
episodic_found = bool(checkpoint_artifacts.get("episodic") and checkpoint_artifacts["episodic"].exists())

st.markdown("<h3>Live Algorithm Pipeline</h3>", unsafe_allow_html=True)
pipe_html = f"""
<div class='pipeline'>
  <div class='pipe-card'><div class='pipe-title'>1 · CRN STATE</div><div class='pipe-active'>● LIVE</div><div>{meta['input_dim']}-D observation</div></div>
  <div class='pipe-card'><div class='pipe-title'>2 · DQN POLICY</div><div class='pipe-active'>● REAL INFERENCE</div><div>argmax Q(s,a)</div></div>
  <div class='pipe-card'><div class='pipe-title'>3 · TD ERROR</div><div class='pipe-active'>● LIVE</div><div>policy + target nets</div></div>
  <div class='pipe-card'><div class='pipe-title'>4 · ADWIN</div><div class='pipe-active'>● LIVE</div><div>input = |TD error|</div></div>
  <div class='pipe-card'><div class='pipe-title'>5 · CRL TRAINING</div><div class='pipe-training'>TRAINING-TIME</div><div>EWC / replay / MEC</div></div>
</div>
"""
st.markdown(pipe_html, unsafe_allow_html=True)

st.caption(
    "EWC, episodic replay, rollback and MEC/Fisher consolidation are training/adaptation mechanisms; "
    "they are not fabricated during fixed-policy inference. Saved artifact availability for this checkpoint: "
    f"Fisher={'yes' if fisher_found else 'no'}, θ*={'yes' if theta_found else 'no'}, "
    f"episodic memory={'yes' if episodic_found else 'no'}."
)


# =============================================================================
# METRICS
# =============================================================================

total_attempts = ss.successes + ss.collisions
collision_rate = ss.collisions / total_attempts if total_attempts else 0.0
throughput = ss.successes / max(ss.global_step, 1)
access_efficiency = ss.successes / total_attempts if total_attempts else 0.0
rolling_reward = float(np.mean(ss.rewards)) if ss.rewards else 0.0
rolling_td = float(np.mean(ss.td_errors)) if ss.td_errors else 0.0

m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
m1.metric("Steps", f"{ss.global_step:,}")
m2.metric("Rolling Reward (150)", f"{rolling_reward:+.3f}")
m3.metric("Throughput", f"{throughput:.3f}")
m4.metric("Access Efficiency", f"{access_efficiency:.3f}")
m5.metric("Collision Rate", f"{collision_rate:.3f}")
m6.metric("Mean |TD| (150)", f"{rolling_td:.3f}")
m7.metric("Drift Events", ss.drift_count)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# =============================================================================
# CHANNEL GRID — DECISION-TIME OCCUPANCY, NOT NEXT-STATE OCCUPANCY
# =============================================================================

occ = ss.last_decision_occupancy if ss.last_action is not None else ss.env.current_occupancy()
action = ss.last_action

ch_html = "<div class='ch-grid'>"
for c in range(meta["num_channels"]):
    selected = action == c
    busy = occ[c] == 1
    pu_prob = ss.env.pu_probs[c]
    occ_mean = float(np.mean(ss.env.occ_hist[c])) if ss.env.occ_hist[c] else pu_prob

    if selected and busy:
        css, status = "ch-selected-busy", "COLLISION ✗"
    elif selected and not busy:
        css, status = "ch-selected-clear", "TX ✓"
    elif busy:
        css, status = "ch-busy", "PU ACTIVE"
    else:
        css, status = "ch-clear", "CLEAR"

    bar_fill = int(np.clip(round(occ_mean * 10), 0, 10))
    bar = "█" * bar_fill + "░" * (10 - bar_fill)

    ch_html += f"""
    <div class='ch-card {css}'>
      <div class='ch-label'>CH {c + 1:02d}</div>
      <div style='font-size:0.95rem;margin:4px 0'>{status}</div>
      <div class='ch-prob'>{bar}</div>
      <div class='ch-prob'>PU prior: {pu_prob:.2f}</div>
    </div>
    """

idle_css = "ch-idle" if action == meta["num_channels"] else "ch-clear"
ch_html += f"""
<div class='ch-card {idle_css}' style='grid-column:span 4'>
  <div class='ch-label'>IDLE ACTION</div>
  <div style='font-size:0.82rem'>No transmission · reward = -0.05</div>
</div>
</div>
"""

st.markdown("<h3>Decision-Time Spectrum State</h3>", unsafe_allow_html=True)
st.markdown(ch_html, unsafe_allow_html=True)
st.caption("The channel colors above are the exact occupancy snapshot used to judge the displayed DQN action; they are not the next-state occupancy.")


# =============================================================================
# REAL Q-VALUES + BELLMan / TD DETAILS
# =============================================================================

left, right = st.columns([3, 2])

with left:
    st.markdown("<h3>Real DQN Q-Values</h3>", unsafe_allow_html=True)
    if ss.last_q_values is not None:
        labels = [f"CH{i + 1:02d}" for i in range(meta["num_channels"])] + ["IDLE"]
        qdf = pd.DataFrame({"Action": labels, "Q(s,a)": ss.last_q_values})
        qdf["Selected"] = [i == ss.last_action for i in range(len(labels))]
        qdf["Decision PU Busy"] = [bool(x) for x in occ] + [False]
        st.dataframe(qdf, use_container_width=True, hide_index=True, height=290)
    else:
        st.info("Run or step the model to populate Q-values.")

with right:
    st.markdown("<h3>Bellman / Drift Telemetry</h3>", unsafe_allow_html=True)
    if ss.last_td_error is not None:
        selected_label = "IDLE" if ss.last_action == meta["num_channels"] else f"CH{ss.last_action + 1:02d}"
        td_table = pd.DataFrame(
            [
                ["Selected action", selected_label],
                ["Q_policy(s,a)", f"{ss.last_q_selected:.6f}"],
                ["Reward r", f"{ss.last_reward:+.3f}"],
                ["TD target", f"{ss.last_td_target:.6f}"],
                ["Absolute TD error", f"{ss.last_td_error:.6f}"],
                ["ADWIN δ", f"{ss.adwin.delta:.4f}"],
                ["ADWIN window width", str(ss.adwin.width)],
                ["Current task step", str(ss.task_step)],
            ],
            columns=["Signal", "Value"],
        )
        st.dataframe(td_table, use_container_width=True, hide_index=True, height=290)
    else:
        st.info("Run or step the model to populate TD-error telemetry.")


# =============================================================================
# SIGNAL CHARTS + EVENT LOG
# =============================================================================

chart1, chart2, log_col = st.columns([2, 2, 2])

with chart1:
    st.markdown("<h3>Reward Signal</h3>", unsafe_allow_html=True)
    vals = list(ss.rewards)
    if len(vals) > 1:
        rolling = [float(np.mean(vals[max(0, i - 19): i + 1])) for i in range(len(vals))]
        idx = range(max(0, ss.global_step - len(vals) + 1), ss.global_step + 1)
        st.line_chart(pd.DataFrame({"Reward": vals, "20-step mean": rolling}, index=idx), height=220)
    else:
        st.caption("No reward history yet.")

with chart2:
    st.markdown("<h3>ADWIN Input: |TD Error|</h3>", unsafe_allow_html=True)
    vals = list(ss.td_errors)
    if len(vals) > 1:
        rolling = [float(np.mean(vals[max(0, i - 19): i + 1])) for i in range(len(vals))]
        idx = range(max(0, ss.global_step - len(vals) + 1), ss.global_step + 1)
        st.line_chart(pd.DataFrame({"|TD error|": vals, "20-step mean": rolling}, index=idx), height=220)
    else:
        st.caption("No TD-error history yet.")

with log_col:
    st.markdown("<h3>Event Log</h3>", unsafe_allow_html=True)
    st.markdown("<div class='log-box'>" + "<br>".join(ss.log) + "</div>", unsafe_allow_html=True)


# =============================================================================
# MODEL / ARTIFACT DETAILS
# =============================================================================

with st.expander("Checkpoint and continual-learning details", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Checkpoint source**", ss.model_source_label)
        st.write("**Checkpoint stage**", checkpoint_stage_label(ss.model_source_label))
        st.write("**Policy architecture**", f"{meta['input_dim']} → {meta['hidden1']} → {meta['hidden2']} → {meta['output_dim']}")
        st.write("**Derived channels**", meta["num_channels"])
        st.write("**Derived history length**", meta["history_length"])
        st.write("**Target network**", meta["target_origin"])
    with c2:
        st.write("**Live inference**", "Active")
        st.write("**TD-error computation**", "Active — policy + target")
        st.write("**ADWIN**", f"Active — δ={ss.adwin.delta}")
        st.write("**Fisher artifact**", "Present" if fisher_found else "Not present for selected checkpoint")
        st.write("**θ* artifact**", "Present" if theta_found else "Not present for selected checkpoint")
        st.write("**Episodic artifact**", "Present" if episodic_found else "Not present for selected checkpoint")
        st.write("**MEC/Fisher worker**", "Training-time mechanism; not fabricated in inference dashboard")

    st.info(
        "The dashboard deliberately separates live inference/drift monitoring from training-time continual-learning operations. "
        "A .pt policy checkpoint contains the learned DQN weights; it does not by itself contain a running optimizer, replay buffer, "
        "or asynchronous MEC process. If the repository includes the Fisher/θ*/episodic artifacts, their availability is shown above."
    )


# =============================================================================
# AUTO-RUN
# =============================================================================

if ss.running:
    run_one_step()
    time.sleep(1.0 / speed)
    st.rerun()
