"""
runs three trained policies (bad, misleading, correct) side by side for comparison

Loads: model_bad.zip, model_misleading.zip, model_correct.zip (from reward_hacking_demo.py)

Layout: LEFT = Bad reward, CENTER = Misleading reward, RIGHT = Correct reward

Keyboard: R = reset all, P = pause, S = single step
"""

import os
import sys
import time
import threading
import warnings

warnings.filterwarnings("ignore", message=".*Wayland.*window position.*")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
REWARD_HACKING_RESULTS_DIR = os.path.join(REPO_ROOT, "results", "reward_hacking")
for _ in [REPO_ROOT, os.path.join(REPO_ROOT, "problem_setup")]:
    if _ not in sys.path:
        sys.path.insert(0, _)

import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

from env_cartpole_rl import CartPoleRLEnv
import reward_functions as rf

REWARD_TYPES = ("bad", "misleading", "correct")
MODEL_PATHS = {t: os.path.join(REWARD_HACKING_RESULTS_DIR, f"model_{t}.zip") for t in REWARD_TYPES}
CONTROL_DT = 0.02  # ~50 Hz control for display

# Shared state for keyboard (R=82, P=80, S=83 in ASCII)
reset_requested = False
paused = False
single_step_requested = False
viewers_running = [True, True, True]  # one per viewer thread


def key_callback(keycode):
    global reset_requested, paused, single_step_requested
    if keycode == 82:  # R
        reset_requested = True
    elif keycode == 80:  # P
        paused = not paused
    elif keycode == 83:  # S
        single_step_requested = True


def set_camera(viewer):
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.azimuth = 90.0
    viewer.cam.elevation = -15.0
    viewer.cam.distance = 2.8
    viewer.cam.lookat[:] = [0.0, 0.0, 0.4]


def run_viewer_thread(env: CartPoleRLEnv, label: str, viewer_index: int):
    # Run one MuJoCo viewer in a thread. Updates when env.data is stepped from main
    global viewers_running
    mj_model, mj_data = env.get_model_and_data()
    try:
        with mujoco.viewer.launch_passive(
            mj_model, mj_data, key_callback=key_callback, show_left_ui=False, show_right_ui=False
        ) as viewer:
            set_camera(viewer)
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.001)
    except Exception as e:
        print(f"[{label}] Viewer closed: {e}")
    viewers_running[viewer_index] = False


def compute_display_reward(reward_type: str, obs: np.ndarray, action: np.ndarray, terminated: bool):
    if reward_type == "bad":
        r, d = rf.compute_reward_bad(obs, action, terminated)
        return r, d
    if reward_type == "misleading":
        r, d = rf.compute_reward_misleading(obs, action, terminated)
        return r, d
    r, d = rf.compute_reward_correct(obs, action, terminated)
    return r, d


def main():
    global reset_requested, paused, single_step_requested, viewers_running
    for t in REWARD_TYPES:
        if not os.path.isfile(MODEL_PATHS[t]):
            print(f"Missing {MODEL_PATHS[t]}. Run: python reward_hacking_demo.py --train all")
            return
    # Load policies and envs (each env has its own model/data)
    policies = {}
    envs = {}
    obs = {}
    ep_step = {}
    ep_reward = {}
    for t in REWARD_TYPES:
        policies[t] = PPO.load(MODEL_PATHS[t])
        envs[t] = CartPoleRLEnv()
        obs[t], _ = envs[t].reset()
        ep_step[t] = 0
        ep_reward[t] = 0.0
    labels = ("BAD REWARD", "MISLEADING REWARD", "CORRECT REWARD")
    # Start three viewer threads (left, center, right)
    threads = []
    for i, t in enumerate(REWARD_TYPES):
        th = threading.Thread(target=run_viewer_thread, args=(envs[t], labels[i], i), daemon=True)
        th.start()
        threads.append(th)
    time.sleep(1.0)  # let viewers open
    print("Compare policies: LEFT=Bad, CENTER=Misleading, RIGHT=Correct. Keys: R=reset, P=pause, S=single step")
    last_control = time.perf_counter()
    while any(viewers_running):
        if reset_requested:
            for t in REWARD_TYPES:
                obs[t], _ = envs[t].reset()
                ep_step[t] = 0
                ep_reward[t] = 0.0
            reset_requested = False
            print("--- Reset ---")
        if not paused or single_step_requested:
            single_step_requested = False
            actions = {}
            for t in REWARD_TYPES:
                actions[t], _ = policies[t].predict(obs[t], deterministic=True)
            comp_correct = None
            for t in REWARD_TYPES:
                obs[t], _, terminated, truncated, _ = envs[t].step(actions[t])
                r, comp = compute_display_reward(t, obs[t], actions[t], terminated)
                if t == "correct":
                    comp_correct = comp
                ep_reward[t] += r
                ep_step[t] += 1
                if terminated or truncated:
                    obs[t], _ = envs[t].reset()
                    ep_step[t] = 0
                    ep_reward[t] = 0.0
            line_bad = f"[BAD]        step={ep_step['bad']:4d}  reward={ep_reward['bad']:.3f}"
            line_mis = f"[MISLEADING] step={ep_step['misleading']:4d}  reward={ep_reward['misleading']:.3f}"
            line_cor = f"[CORRECT]   step={ep_step['correct']:4d}  reward={ep_reward['correct']:.3f}"
            if comp_correct:
                parts = " ".join(f"{k[:6]}={v:.2f}" for k, v in list(comp_correct.items())[:6])
                line_cor += f"  |  {parts}"
            print(f"\r{line_bad}  |  {line_mis}  |  {line_cor}", end="", flush=True)
        now = time.perf_counter()
        elapsed = now - last_control
        if elapsed < CONTROL_DT:
            time.sleep(CONTROL_DT - elapsed)
        last_control = time.perf_counter()
    print("\nViewers closed.")


if __name__ == "__main__":
    main()
