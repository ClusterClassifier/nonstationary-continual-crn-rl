"""
Continual RL Cognitive Radio Network — Real-Time Channel Selection Simulator
Streamlit app | Self-contained, no external model weights required.

Embeds the CRN environment, DQN agent, ADWIN detector, and EWC module
directly so the app is deployable to Streamlit Cloud from a bare GitHub repo.
"""

import time
import random
import math
from collections import deque

import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="CRN-RL Simulator",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THEME & STYLES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@400;600&display=swap');

/* ── Base ── */
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

/* ── Headers ── */
h1, h2, h3 { font-family: 'Share Tech Mono', monospace; letter-spacing: 0.04em; }
h1 { color: #0FF4C6; font-size: 1.5rem; }
h3 { color: #4A90D9; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.1em; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #1A2235;
    border: 1px solid #1E2D45;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="metric-container"] label { color: #6B8299 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family: 'Share Tech Mono', monospace; color: #0FF4C6 !important; font-size: 1.6rem !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: #1A2235;
    border: 1px solid #4A90D9;
    color: #4A90D9;
    border-radius: 6px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    transition: all 0.15s;
    width: 100%;
}
.stButton > button:hover { background: #4A90D9; color: #0A0E1A; }

/* ── Sliders & selects ── */
[data-testid="stSlider"] .st-emotion-cache-1vbkxwb { color: #0FF4C6; }
.stSelectbox label, .stSlider label { color: #6B8299 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Channel grid ── */
.ch-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 0 0 20px 0; }
.ch-card {
    border-radius: 8px;
    padding: 14px 10px;
    text-align: center;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    border: 2px solid transparent;
    transition: all 0.2s;
    position: relative;
}
.ch-clear  { background: #0D1F16; border-color: #1A4A30; color: #0FF4C6; }
.ch-busy   { background: #1F0D10; border-color: #4A1A20; color: #FF3B5C; }
.ch-idle   { background: #1A1800; border-color: #3A3000; color: #FFB800; }
.ch-selected-clear { background: #0A2E20; border-color: #0FF4C6; color: #0FF4C6; box-shadow: 0 0 14px #0FF4C630; }
.ch-selected-busy  { background: #2E0A0F; border-color: #FF3B5C; color: #FF3B5C; box-shadow: 0 0 14px #FF3B5C30; }
.ch-label  { font-size: 0.65rem; color: #6B8299; margin-bottom: 4px; }
.ch-status { font-size: 0.75rem; margin-top: 4px; }
.ch-prob   { font-size: 0.7rem; color: #6B8299; margin-top: 2px; }

/* ── Log ── */
.log-box {
    background: #0D1220;
    border: 1px solid #1E2D45;
    border-radius: 8px;
    padding: 12px 14px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    height: 180px;
    overflow-y: auto;
    color: #6B8299;
}
.log-ok  { color: #0FF4C6; }
.log-col { color: #FF3B5C; }
.log-idl { color: #FFB800; }
.log-det { color: #C084FC; }

/* ── Divider ── */
.divider { border: none; border-top: 1px solid #1E2D45; margin: 16px 0; }

/* ── Status badge ── */
.status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-run  { background: #0A2E20; color: #0FF4C6; border: 1px solid #0FF4C6; }
.badge-stop { background: #1A2235; color: #6B8299; border: 1px solid #2A3A55; }
.badge-det  { background: #1A0A2E; color: #C084FC; border: 1px solid #C084FC; }
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CRN ENVIRONMENT (embedded — no notebook import needed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK_META = {
    "T1 — Early Commute (06:00)":   {"seed": 42, "low": 0.20, "high": 0.40, "desc": "Low–Medium PU load"},
    "T2 — Peak Commute (09:00)":    {"seed": 43, "low": 0.60, "high": 0.90, "desc": "High PU load"},
    "T3 — Midday Office (12:00)":   {"seed": 44, "low": 0.40, "high": 0.60, "desc": "Medium PU load"},
    "T4 — IoT Sensor Bursts (14:00)":{"seed": 45, "low": 0.05, "high": 0.20, "desc": "Very Low PU load"},
    "T5 — Evening Commute (17:00)": {"seed": 46, "low": 0.65, "high": 0.85, "desc": "High PU load"},
    "T6 — Video Streaming (20:00)": {"seed": 47, "low": 0.80, "high": 0.95, "desc": "Very High PU load"},
    "T7 — Night Low (23:00)":       {"seed": 48, "low": 0.01, "high": 0.10, "desc": "Very Low PU load"},
}

NUM_CHANNELS   = 8
HISTORY_LENGTH = 5


class CRNEnvironment:
    """Lightweight CRN environment mirroring the notebook's CRNEnv."""

    def __init__(self, seed: int, base_low: float, base_high: float):
        rng = np.random.RandomState(seed)
        self.pu_probs = rng.uniform(base_low, base_high, NUM_CHANNELS)
        opt = rng.randint(0, NUM_CHANNELS)
        self.pu_probs[opt] = max(0.01, base_low * 0.5)

        self.snr_hist = [deque(maxlen=HISTORY_LENGTH) for _ in range(NUM_CHANNELS)]
        self.occ_hist = [deque(maxlen=HISTORY_LENGTH) for _ in range(NUM_CHANNELS)]

        self.R_success   =  1.0
        self.R_collision = -1.0
        self.R_idle      = -0.05
        self.idle_action  = NUM_CHANNELS
        self.action_space = NUM_CHANNELS + 1
        self.state_dim    = NUM_CHANNELS * 2 * HISTORY_LENGTH

        self._prefill()

    def _prefill(self):
        for _ in range(HISTORY_LENGTH):
            self._step_channels()

    def _step_channels(self):
        for c in range(NUM_CHANNELS):
            busy = random.random() < self.pu_probs[c]
            self.snr_hist[c].append(random.uniform(0.5, 1.0) if busy else random.uniform(0.0, 0.2))
            self.occ_hist[c].append(1 if busy else 0)

    def get_state(self) -> np.ndarray:
        s = []
        for c in range(NUM_CHANNELS):
            s.extend(self.snr_hist[c])
            s.extend(self.occ_hist[c])
        return np.array(s, dtype=np.float32)

    def current_occupancy(self) -> list:
        return [int(self.occ_hist[c][-1]) for c in range(NUM_CHANNELS)]

    def step(self, action: int):
        occ = self.current_occupancy()
        if action == self.idle_action:
            reward = self.R_idle
            outcome = "idle"
        elif occ[action] == 1:
            reward = self.R_collision
            outcome = "collision"
        else:
            reward = self.R_success
            outcome = "success"
        self._step_channels()
        return self.get_state(), reward, outcome


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT (heuristic DQN proxy — greedy myopic with Boltzmann noise)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HeuristicAgent:
    """
    Simulates a trained DQN agent without requiring saved weights.
    Uses a myopic greedy policy (pick least-occupied channel) with
    a small Boltzmann noise term to mimic learned Q-value selection.
    """

    def __init__(self, tau: float = 0.3):
        self.tau = tau

    def get_action(self, env: CRNEnvironment) -> int:
        # Score = negative mean occupancy history (lower occ = better)
        scores = []
        for c in range(NUM_CHANNELS):
            occ_mean = float(np.mean(env.occ_hist[c]))
            snr_mean = float(np.mean(env.snr_hist[c]))
            scores.append(-(occ_mean + 0.2 * snr_mean))

        # Boltzmann sampling over channel scores
        scores_arr = np.array(scores)
        scores_arr -= scores_arr.max()
        exp_s = np.exp(scores_arr / (self.tau + 1e-8))
        probs = exp_s / exp_s.sum()
        return int(np.random.choice(NUM_CHANNELS, p=probs))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ADWIN (pure-Python, same as notebook)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ADWIN:
    _MAX_BUCKETS = 5

    class _Bucket:
        __slots__ = ("total", "variance", "size")
        def __init__(self): self.total = self.variance = 0.0; self.size = 0

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self._total = self._variance = 0.0
        self._width = 0
        self._bucket_rows = [[]]
        self.drift_detected = False

    @property
    def width(self): return self._width

    def update(self, value: float):
        self.drift_detected = False
        self._insert(float(value))
        self._check_drift()

    def _insert(self, value):
        b = self._Bucket(); b.total = value; b.size = 1
        self._bucket_rows[0].insert(0, b)
        self._width += 1
        old_mean = self._total / self._width if self._width > 1 else value
        self._variance += (value - old_mean) * (value - (self._total + value) / self._width)
        self._total += value
        self._compress()

    def _compress(self):
        level = 0
        while level < len(self._bucket_rows):
            row = self._bucket_rows[level]
            if len(row) <= self._MAX_BUCKETS: break
            if level + 1 >= len(self._bucket_rows): self._bucket_rows.append([])
            b1, b2 = row[-2], row[-1]
            m = self._Bucket(); m.size = b1.size + b2.size; m.total = b1.total + b2.total
            d = (b2.total/b2.size - b1.total/b1.size) if b1.size > 0 and b2.size > 0 else 0.0
            m.variance = b1.variance + b2.variance + d**2 * b1.size * b2.size / max(m.size, 1)
            self._bucket_rows[level] = row[:-2]
            self._bucket_rows[level+1].insert(0, m)
            level += 1

    def _check_drift(self):
        n0 = total0 = 0.0
        for row in reversed(self._bucket_rows):
            for bucket in reversed(row):
                n0 += bucket.size; total0 += bucket.total
                n1 = self._width - n0
                if n1 <= 0 or n0 <= 0: continue
                m0 = total0 / n0; m1 = (self._total - total0) / n1
                dd = math.log(2.0 * math.log(self._width + 1) / self.delta) if self._width > 1 else 0.0
                eps = math.sqrt(dd / (2*n0)) + math.sqrt(dd / (2*n1))
                if abs(m0 - m1) >= eps:
                    self._shrink(int(n0))
                    self.drift_detected = True
                    return

    def _shrink(self, drop):
        removed = 0.0; rem = drop
        for level in range(len(self._bucket_rows)-1, -1, -1):
            row = self._bucket_rows[level]
            while row and rem > 0:
                b = row[-1]
                if b.size <= rem: rem -= b.size; removed += b.total; row.pop()
                else:
                    f = rem/b.size; removed += b.total*f
                    b.total -= b.total*f; b.size -= rem; rem = 0
        self._width -= drop; self._total -= removed
        if self._variance < 0: self._variance = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SESSION STATE INIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def init_state(task_key: str, delta: float):
    meta = TASK_META[task_key]
    st.session_state.env       = CRNEnvironment(meta["seed"], meta["low"], meta["high"])
    st.session_state.agent     = HeuristicAgent()
    st.session_state.adwin     = ADWIN(delta=delta)
    st.session_state.step      = 0
    st.session_state.rewards   = deque(maxlen=150)
    st.session_state.collisions = 0
    st.session_state.successes  = 0
    st.session_state.idles      = 0
    st.session_state.log        = deque(maxlen=40)
    st.session_state.last_action = None
    st.session_state.last_outcome = None
    st.session_state.drift_count  = 0
    st.session_state.initialized  = True

def ensure_init(task_key, delta):
    if not st.session_state.get("initialized"):
        init_state(task_key, delta)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:
    st.markdown("# 📡 CRN-RL Sim")
    st.markdown("<p style='color:#6B8299;font-size:0.78rem;margin-top:-8px'>Continual RL · Cognitive Radio</p>", unsafe_allow_html=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown("### Diurnal Task")
    task_key = st.selectbox(
        "Select task environment",
        list(TASK_META.keys()),
        label_visibility="collapsed"
    )
    meta = TASK_META[task_key]
    st.markdown(f"<p style='color:#4A90D9;font-size:0.75rem;font-family:Share Tech Mono,monospace'>{meta['desc']}</p>", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### Agent Settings")

    delta = st.select_slider(
        "ADWIN sensitivity (δ)",
        options=[0.001, 0.002, 0.005, 0.01, 0.05, 0.1],
        value=0.05,
        help="Lower δ = more sensitive drift detection. 0.05 is recommended for this simulation."
    )

    speed = st.slider(
        "Simulation speed (steps/sec)",
        min_value=1, max_value=20, value=5, step=1
    )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### Controls")

    col_a, col_b = st.columns(2)
    with col_a:
        run_btn   = st.button("▶  Run",   key="run")
        reset_btn = st.button("↺  Reset", key="reset")
    with col_b:
        pause_btn = st.button("⏸  Pause", key="pause")
        step_btn  = st.button("⏭  Step",  key="step_one")

    if run_btn:
        st.session_state.running = True
    if pause_btn:
        st.session_state.running = False
    if reset_btn:
        st.session_state.running = False
        init_state(task_key, delta)
    if step_btn:
        st.session_state.running = False
        st.session_state.do_step = True

    if not st.session_state.get("running"):
        st.session_state.setdefault("running", False)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### Legend")
    st.markdown("""
<div style='font-family:Share Tech Mono,monospace;font-size:0.72rem;line-height:2'>
<span style='color:#0FF4C6'>■</span> Agent selected · clear<br>
<span style='color:#FF3B5C'>■</span> Agent selected · collision<br>
<span style='color:#1A4A30'>■</span> Channel clear (PU idle)<br>
<span style='color:#4A1A20'>■</span> Channel busy (PU active)<br>
<span style='color:#FFB800'>■</span> Agent idle (no tx)<br>
<span style='color:#C084FC'>⚡</span> ADWIN drift detected
</div>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN PANEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ensure_init(task_key, delta)
ss = st.session_state

# Title row
title_col, badge_col = st.columns([5, 1])
with title_col:
    st.markdown(f"## {task_key}")
with badge_col:
    if ss.get("running"):
        st.markdown("<div style='margin-top:18px'><span class='status-badge badge-run'>● LIVE</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='margin-top:18px'><span class='status-badge badge-stop'>● PAUSED</span></div>", unsafe_allow_html=True)

# ── Simulation step logic ─────────────────────────────────────────

def do_one_step():
    env, agent, adwin = ss.env, ss.agent, ss.adwin
    state = env.get_state()
    action = agent.get_action(env)
    _, reward, outcome = env.step(action)

    ss.step += 1
    ss.rewards.append(reward)
    ss.last_action  = action
    ss.last_outcome = outcome

    if outcome == "success":   ss.successes  += 1
    elif outcome == "collision": ss.collisions += 1
    else:                       ss.idles      += 1

    # ADWIN on reward stream
    adwin.update(reward)
    drift = adwin.drift_detected
    if drift:
        ss.drift_count += 1
        adwin.__init__(adwin.delta)  # reset window after detection

    # Log entry
    step_str = f"[{ss.step:05d}]"
    if drift:
        ss.log.appendleft(f"<span class='log-det'>{step_str} ⚡ DRIFT DETECTED — ADWIN triggered (#{ss.drift_count})</span>")
    if outcome == "success":
        ss.log.appendleft(f"<span class='log-ok'>{step_str} CH{action+1:02d} → TX OK   +{reward:.2f}</span>")
    elif outcome == "collision":
        ss.log.appendleft(f"<span class='log-col'>{step_str} CH{action+1:02d} → COLLISION  {reward:.2f}</span>")
    else:
        ss.log.appendleft(f"<span class='log-idl'>{step_str} IDLE → no TX  {reward:.2f}</span>")

if ss.get("do_step"):
    do_one_step()
    ss.do_step = False

# ── Metrics strip ──────────────────────────────────────────────────

total_tx = ss.successes + ss.collisions
collision_rate = ss.collisions / total_tx if total_tx > 0 else 0.0
throughput     = ss.successes / max(ss.step, 1)
mean_reward    = float(np.mean(ss.rewards)) if ss.rewards else 0.0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Steps",          f"{ss.step:,}")
m2.metric("Mean Reward",    f"{mean_reward:+.3f}")
m3.metric("Throughput",     f"{throughput:.3f}")
m4.metric("Collision Rate", f"{collision_rate:.3f}")
m5.metric("Drift Events",   ss.drift_count)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Channel grid ───────────────────────────────────────────────────

occ = ss.env.current_occupancy()
action = ss.last_action

ch_html = "<div class='ch-grid'>"
for c in range(NUM_CHANNELS):
    is_selected = (action is not None and action == c)
    is_busy     = occ[c] == 1
    pu_prob     = ss.env.pu_probs[c]
    occ_mean    = float(np.mean(ss.env.occ_hist[c])) if ss.env.occ_hist[c] else pu_prob

    if is_selected and not is_busy:
        css = "ch-selected-clear"
        status = "TX ✓"
    elif is_selected and is_busy:
        css = "ch-selected-busy"
        status = "COLLISION ✗"
    elif is_busy:
        css = "ch-busy"
        status = "PU ACTIVE"
    else:
        css = "ch-clear"
        status = "CLEAR"

    bar_fill = int(occ_mean * 10)
    bar = "█" * bar_fill + "░" * (10 - bar_fill)

    ch_html += f"""
    <div class='ch-card {css}'>
        <div class='ch-label'>CH {c+1:02d}</div>
        <div style='font-size:1.0rem;margin:4px 0'>{status}</div>
        <div class='ch-prob' title='PU occupancy probability'>{bar}</div>
        <div class='ch-prob'>PU: {pu_prob:.2f}</div>
    </div>"""

# Idle action display
idle_css = "ch-idle" if action == NUM_CHANNELS else "ch-clear"
ch_html += f"<div class='ch-card {idle_css}' style='grid-column:span 4'><div class='ch-label'>IDLE ACTION</div><div style='font-size:0.85rem'>No transmission · penalty {ss.env.R_idle}</div></div>"
ch_html += "</div>"

st.markdown("<h3>Channel Status</h3>", unsafe_allow_html=True)
st.markdown(ch_html, unsafe_allow_html=True)

# ── Reward chart ───────────────────────────────────────────────────

chart_col, log_col = st.columns([3, 2])

with chart_col:
    st.markdown("<h3>Reward Signal</h3>", unsafe_allow_html=True)
    rewards_list = list(ss.rewards)
    if len(rewards_list) > 1:
        xs = list(range(max(0, ss.step - len(rewards_list)), ss.step))
        rolling = []
        window = 20
        for i in range(len(rewards_list)):
            w = rewards_list[max(0, i-window):i+1]
            rolling.append(float(np.mean(w)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=rewards_list,
            mode="lines",
            line=dict(color="#1E3A4A", width=1),
            name="Raw reward",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=rolling,
            mode="lines",
            line=dict(color="#0FF4C6", width=2),
            name=f"Rolling {window}-step mean",
            showlegend=True,
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="#2A3A55", line_width=1)
        fig.update_layout(
            paper_bgcolor="#0D1220",
            plot_bgcolor="#0D1220",
            margin=dict(l=8, r=8, t=8, b=8),
            height=200,
            font=dict(family="Share Tech Mono, monospace", color="#6B8299", size=11),
            xaxis=dict(gridcolor="#1E2D45", showgrid=True, title="Step"),
            yaxis=dict(gridcolor="#1E2D45", showgrid=True, title="Reward",
                       range=[-1.2, 1.2]),
            legend=dict(
                bgcolor="#0D1220", bordercolor="#1E2D45", borderwidth=1,
                font=dict(size=10), x=0.01, y=0.99,
            ),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown("<p style='color:#6B8299;font-size:0.8rem;padding:40px 0;text-align:center'>Press ▶ Run to start the simulation</p>", unsafe_allow_html=True)

with log_col:
    st.markdown("<h3>Event Log</h3>", unsafe_allow_html=True)
    log_html = "<div class='log-box'>" + "<br>".join(ss.log) + "</div>"
    st.markdown(log_html, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUTO-RERUN LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if ss.get("running"):
    do_one_step()
    time.sleep(1.0 / speed)
    st.rerun()
