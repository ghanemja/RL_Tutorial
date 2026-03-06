"""
trains and compares 3 reward functions.

toggle between:
  1. Bad reward:        reward = -|pole_angle|         -> unstable / fails to balance
  2. Misleading reward: reward = cart_velocity         -> cart slamming instead of balancing
  3. Correct reward:    upright + center - velocity - control -> stable balancing

Usage:
  python reward_hacking_demo.py --train bad
  python reward_hacking_demo.py --train misleading
  python reward_hacking_demo.py --train correct
  python reward_hacking_demo.py --train all     # train all three, then plot comparison
  python reward_hacking_demo.py --play bad      # run bad-reward policy in viewer
  python reward_hacking_demo.py --play misleading
  python reward_hacking_demo.py --play correct
  python reward_hacking_demo.py --plot          # generate side-by-side comparison plots
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*Wayland.*window position.*")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
REWARD_HACKING_RESULTS_DIR = os.path.join(RESULTS_DIR, "reward_hacking")
for _ in [REPO_ROOT, os.path.join(REPO_ROOT, "problem_setup")]:
    if _ not in sys.path:
        sys.path.insert(0, _)

import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env_cartpole_rl import CartPoleRLEnv, RewardWrapper
import reward_functions as rf
from policies import get_policy_kwargs


# Config
REWARD_TYPES = ("bad", "misleading", "correct")
TOTAL_TIMESTEPS = 80_000
MODEL_NAMES = {t: os.path.join(REWARD_HACKING_RESULTS_DIR, f"model_{t}.zip") for t in REWARD_TYPES}
HISTORY_DIR = REWARD_HACKING_RESULTS_DIR
PLOT_COMPARISON_PATH = os.path.join(REWARD_HACKING_RESULTS_DIR, "reward_hacking_comparison.png")


def _make_env(reward_type: str):
    # build env with the chosen reward function.
    rf.set_reward_type(reward_type)
    env = CartPoleRLEnv()
    env = RewardWrapper(env)
    env = Monitor(env)
    return env


class SaveHistoryCallback(BaseCallback):
    # saves to numpy files
    def __init__(self, reward_type: str, verbose: int = 0):
        super().__init__(verbose)
        self.reward_type = reward_type
        self._episode_rewards = []
        self._episode_lengths = []
        self._reward_components = []
        self._current_ep_reward = 0.0
        self._current_ep_length = 0
        self._current_ep_components = {}

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0.0])
        self._current_ep_reward += float(rewards[0] if hasattr(rewards, "__getitem__") else rewards)
        self._current_ep_length += 1
        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}
        if "reward_dict" in info:
            for k, v in info["reward_dict"].items():
                self._current_ep_components[k] = self._current_ep_components.get(k, 0) + v
        dones = self.locals.get("dones", [False])
        done = dones[0] if (hasattr(dones, "__getitem__") and len(dones)) else False
        if done:
            self._episode_rewards.append(self._current_ep_reward)
            self._episode_lengths.append(self._current_ep_length)
            self._reward_components.append(dict(self._current_ep_components))
            self._current_ep_reward = 0.0
            self._current_ep_length = 0
            self._current_ep_components = {}
        return True

    def _on_training_end(self):
        # save histories for comparison plots.
        path_r = os.path.join(HISTORY_DIR, f"rewards_{self.reward_type}.npy")
        path_l = os.path.join(HISTORY_DIR, f"lengths_{self.reward_type}.npy")
        np.save(path_r, np.array(self._episode_rewards, dtype=float))
        np.save(path_l, np.array(self._episode_lengths, dtype=float))
        # Save component keys and values for plotting (simplified: one key per run)
        if self._reward_components:
            keys = list(self._reward_components[0].keys())
            arr = np.array([[c.get(k, 0) for k in keys] for c in self._reward_components])
            np.save(os.path.join(HISTORY_DIR, f"components_{self.reward_type}.npy"), arr)
            with open(os.path.join(HISTORY_DIR, f"component_keys_{self.reward_type}.txt"), "w") as f:
                f.write("\n".join(keys))


def train_reward_type(reward_type: str, total_timesteps: int = TOTAL_TIMESTEPS):
    # train PPO with the selected reward function; save model and log curves
    if reward_type not in REWARD_TYPES:
        raise ValueError(f"reward_type must be one of {REWARD_TYPES}")
    os.makedirs(REWARD_HACKING_RESULTS_DIR, exist_ok=True)
    rf.set_reward_type(reward_type)
    env = DummyVecEnv([lambda: _make_env(reward_type)])
    policy_kwargs = get_policy_kwargs("small")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )
    callback = SaveHistoryCallback(reward_type=reward_type)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(MODEL_NAMES[reward_type])
    print(f"Saved {MODEL_NAMES[reward_type]}")
    rf.set_reward_type("default")
    return model


def play_policy(reward_type: str, max_steps: int = 10000):
    #run the trained policy for this reward type in the MuJoCo viewer
    import mujoco
    import mujoco.viewer

    if reward_type not in REWARD_TYPES:
        raise ValueError(f"reward_type must be one of {REWARD_TYPES}")
    path = MODEL_NAMES[reward_type]
    if not os.path.isfile(path):
        print(f"Model not found: {path}. Run: python reward_hacking_demo.py --train {reward_type}")
        return
    model = PPO.load(path)
    env = CartPoleRLEnv()
    mj_model, mj_data = env.get_model_and_data()

    def set_fixed_camera(viewer):
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -15.0
        viewer.cam.distance = 2.8
        viewer.cam.lookat[:] = [0.0, 0.0, 0.4]

    obs, _ = env.reset()
    step = 0
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        set_fixed_camera(viewer)
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            viewer.sync()
            set_fixed_camera(viewer)
            step += 1
            if terminated or truncated:
                obs, _ = env.reset()
    print(f"Playback finished after {step} steps.")


def plot_comparison():
    os.makedirs(REWARD_HACKING_RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    colors = {"bad": "C0", "misleading": "C1", "correct": "C2"}
    for i, rtype in enumerate(REWARD_TYPES):
        path_r = os.path.join(HISTORY_DIR, f"rewards_{rtype}.npy")
        path_l = os.path.join(HISTORY_DIR, f"lengths_{rtype}.npy")
        path_c = os.path.join(HISTORY_DIR, f"components_{rtype}.npy")
        if not os.path.isfile(path_r):
            axes[0, i].set_title(f"{rtype} (no data)")
            axes[0, i].text(0.5, 0.5, f"Run --train {rtype}\nfirst", ha="center", va="center")
            axes[1, i].set_visible(False)
            continue
        rewards = np.load(path_r)
        lengths = np.load(path_l)
        x = np.arange(1, len(rewards) + 1)
        axes[0, i].plot(x, rewards, color=colors[rtype], alpha=0.7, label="episode reward")
        axes[0, i].set_title(f"{rtype} reward")
        axes[0, i].set_xlabel("Episode")
        axes[0, i].legend()
        if len(rewards) >= 10:
            w = min(10, len(rewards) // 2)
            smooth = np.convolve(rewards, np.ones(w) / w, mode="valid")
            axes[0, i].plot(range(w, len(rewards) + 1), smooth, color="black", linewidth=2, label="smoothed")
        axes[1, i].plot(x, lengths, color=colors[rtype], alpha=0.7)
        axes[1, i].set_title(f"{rtype} episode length")
        axes[1, i].set_xlabel("Episode")
        if os.path.isfile(path_c):
            comp = np.load(path_c)
            keys_path = os.path.join(HISTORY_DIR, f"component_keys_{rtype}.txt")
            if os.path.isfile(keys_path):
                with open(keys_path) as f:
                    keys = [l.strip() for l in f if l.strip()]
                for j, k in enumerate(keys[:5]):  # max 5 components
                    if j < comp.shape[1]:
                        axes[1, i].plot(x, comp[:, j], alpha=0.5, label=k[:12])
            axes[1, i].legend(fontsize=7)
    plt.suptitle("Reward hacking: episode reward and length by reward type", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_COMPARISON_PATH, dpi=120)
    plt.close()
    print(f"Saved {PLOT_COMPARISON_PATH}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Reward hacking demo: train/play/plot")
    p.add_argument("--train", choices=list(REWARD_TYPES) + ["all"], help="Train with this reward (or 'all')")
    p.add_argument("--play", choices=list(REWARD_TYPES), help="Play trained policy in viewer")
    p.add_argument("--plot", action="store_true", help="Generate comparison plot from saved histories")
    p.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    args = p.parse_args()
    if args.train:
        if args.train == "all":
            for r in REWARD_TYPES:
                train_reward_type(r, total_timesteps=args.timesteps)
            plot_comparison()
        else:
            train_reward_type(args.train, total_timesteps=args.timesteps)
    if args.play:
        play_policy(args.play)
    if args.plot and not args.train:
        plot_comparison()
    if not (args.train or args.play or args.plot):
        p.print_help()
