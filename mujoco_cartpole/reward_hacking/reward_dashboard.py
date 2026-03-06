"""

reward weight sliders for cart-pole RL.

When sliders change, reward_functions weights are updated. You can then run
training or a short simulation to see how reward components and total reward
behave with the current weights.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
for _ in [REPO_ROOT, os.path.join(REPO_ROOT, "problem_setup")]:
    if _ not in sys.path:
        sys.path.insert(0, _)

import streamlit as st
import numpy as np
import reward_functions as rf

st.set_page_config(page_title="CartPole Reward Weights", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background-color: #ffffff !important; }
    .stApp * { color: #000000 !important; }
    h1, h2, h3, p, label, span { color: #000000 !important; }
    .stSlider label { color: #000000 !important; }
    div[data-testid="stVerticalBlock"] > div { padding: 0.2rem 0 !important; }
    .stMarkdown { padding-top: 0.1rem !important; padding-bottom: 0.1rem !important; }
    section[data-testid="stSidebar"] { background-color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

st.title("CartPole reward weight dashboard")

st.markdown("""
**What this is for:** Tune reward weights before or between training runs. The sliders update `reward_functions.py` weights in real time. Use the preview below to see how each component contributes to the total for a given state. Then run `python -m training.train_rl --train` or `python -m reward_hacking.reward_hacking_demo --play correct` to train or test with these weights.
""")

weights = rf.get_reward_weights()

col_sliders, col_preview = st.columns([1, 1])

with col_sliders:
    st.subheader("Reward weights")
    r1, r2 = st.columns(2)
    with r1:
        w_balance = st.slider("w_balance", 0.0, 3.0, float(weights["w_balance"]), 0.1)
        w_upright = st.slider("w_upright", 0.0, 3.0, float(weights["w_upright"]), 0.1)
        w_center = st.slider("w_center", 0.0, 2.0, float(weights["w_center"]), 0.1)
    with r2:
        w_velocity = st.slider("w_velocity", 0.0, 0.1, float(weights["w_velocity"]), 0.001)
        w_control = st.slider("w_control", 0.0, 0.01, float(weights["w_control"]), 0.0005)
        w_termination = st.slider("w_termination", 0.0, 2.0, float(weights["w_termination"]), 0.1)

# Update global weights so train_rl / env use them
rf.set_reward_weights({
    "w_balance": w_balance,
    "w_upright": w_upright,
    "w_center": w_center,
    "w_velocity": w_velocity,
    "w_control": w_control,
    "w_termination": w_termination,
})

with col_preview:
    st.subheader("Live preview (sample state)")
    p1, p2, p3 = st.columns(3)
    with p1:
        cart_pos = st.number_input("Cart pos", -2.0, 2.0, 0.0, 0.1, key="cp")
        pole_angle = st.number_input("Pole angle", -3.15, 3.15, 0.0, 0.1, key="pa")
    with p2:
        cart_vel = st.number_input("Cart vel", -5.0, 5.0, 0.0, 0.1, key="cv")
        pole_angular_vel = st.number_input("Pole ang vel", -10.0, 10.0, 0.0, 0.1, key="pav")
    with p3:
        action_val = st.number_input("Action", -10.0, 10.0, 0.0, 0.5, key="act")

    action = np.array([action_val], dtype=np.float32)
    components = rf.compute_reward_components(cart_pos, cart_vel, pole_angle, pole_angular_vel, action, terminated=False)
    total = rf.compute_total_reward(components)
    st.caption("Components: " + str({k: round(v, 3) for k, v in components.items()}) + "  |  Total: " + str(round(total, 4)))
