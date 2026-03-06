"""
ppo training pipeline for the cart-pole RL environment.

Uses Stable-Baselines3 PPO and logs episode reward, reward components, and (where available) policy/value loss.
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message=".*Wayland.*window position.*")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
TRAINING_RESULTS_DIR = os.path.join(RESULTS_DIR, "training")
for _ in [REPO_ROOT, os.path.join(REPO_ROOT, "problem_setup")]:
    if _ not in sys.path:
        sys.path.insert(0, _)

import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")  # headless: no display needed; saves to file only
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env_cartpole_rl import CartPoleRLEnv, RewardWrapper
import reward_functions as rf
from policies import get_policy_kwargs

# hyperparameters
LEARNING_RATE = 3e-4
GAMMA = 0.99
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
TOTAL_TIMESTEPS = 100_000
POLICY_NAME = "small"   # "small" or "large"
SAVE_PATH = os.path.join(TRAINING_RESULTS_DIR, "ppo_cartpole.zip")
LOG_DIR = os.path.join(TRAINING_RESULTS_DIR, "logs")
PLOT_SAVE_PATH = os.path.join(TRAINING_RESULTS_DIR, "training_curves.png")
INFERENCE_VIDEO_PATH = os.path.join(TRAINING_RESULTS_DIR, "inference_demo.mp4")
INFERENCE_VIDEO_FPS = 30
INFERENCE_VIDEO_DURATION = 5.0  # seconds (short to keep render/encode fast)

# training data for live plots and diagnostics
episode_rewards: list = []
episode_lengths: list = []
episode_reward_components: list = []  # list of dicts
last_logger = None  # set by callback for diagnostics


def _make_env():
    env = CartPoleRLEnv()
    env = RewardWrapper(env)
    env = Monitor(env)
    return env


class TrainingLogCallback(BaseCallback):
    # logs episode reward, length, reward components; updates live plot periodically

    def __init__(self, plot_every: int = 5, verbose=0):
        super().__init__(verbose)
        self.plot_every = plot_every
        self._episode_rewards: list = []
        self._episode_lengths: list = []
        self._reward_components: list = []
        self._current_ep_reward = 0.0
        self._current_ep_length = 0
        self._current_ep_components: dict = {}

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
            global episode_rewards, episode_lengths, episode_reward_components
            episode_rewards = list(self._episode_rewards)
            episode_lengths = list(self._episode_lengths)
            episode_reward_components = list(self._reward_components)
            if len(episode_rewards) % self.plot_every == 0:
                self._update_plot()
        return True

    def _update_plot(self):
        if not episode_rewards:
            return
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        n = len(episode_rewards)
        x = range(1, n + 1)
        axes[0, 0].clear()
        axes[0, 0].plot(x, episode_rewards, color="C0", alpha=0.7)
        axes[0, 0].set_title("Episode reward")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 1].clear()
        axes[0, 1].plot(x, episode_lengths, color="C1", alpha=0.7)
        axes[0, 1].set_title("Episode length")
        axes[0, 1].set_xlabel("Episode")
        if episode_reward_components:
            keys = list(episode_reward_components[0].keys())
            axes[1, 0].clear()
            for i, k in enumerate(keys):
                vals = [c.get(k, 0) for c in episode_reward_components]
                axes[1, 0].plot(x, vals, label=k, alpha=0.7)
            axes[1, 0].set_title("Reward components (cumulative per episode)")
            axes[1, 0].legend(fontsize=8)
            axes[1, 0].set_xlabel("Episode")
        axes[1, 1].clear()
        if len(episode_rewards) >= 10:
            window = min(10, len(episode_rewards) // 2)
            smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
            axes[1, 1].plot(range(window, n + 1), smoothed, color="C2")
        axes[1, 1].set_title("Smoothed episode reward")
        axes[1, 1].set_xlabel("Episode")
        plt.tight_layout()
        plt.savefig(PLOT_SAVE_PATH, dpi=100)
        plt.close(fig)


def run_diagnostics():
    # run diagnostics on collected training data (call after training or from diagnostics.py)
    try:
        from training import diagnostics as diag
        diag.check_episode_rewards(episode_rewards)
        diag.check_reward_components(episode_reward_components)
        diag.check_episode_lengths(episode_lengths)  # infers max from data to avoid false warning
    except Exception as e:
        print("Diagnostics skipped:", e)


def train(
    total_timesteps: int = TOTAL_TIMESTEPS,
    learning_rate: float = LEARNING_RATE,
    gamma: float = GAMMA,
    n_steps: int = N_STEPS,
    batch_size: int = BATCH_SIZE,
    n_epochs: int = N_EPOCHS,
    policy_name: str = POLICY_NAME,
    save_path: str = SAVE_PATH,
    log_dir: str = LOG_DIR,
):
    os.makedirs(TRAINING_RESULTS_DIR, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    env = DummyVecEnv([_make_env])
    policy_kwargs = get_policy_kwargs(policy_name)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
    )
    callback = TrainingLogCallback(plot_every=5)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(save_path)
    print(f"Model saved to {save_path}")
    run_diagnostics()
    return model


def play_trained_agent(
    model_path: str = SAVE_PATH,
    max_steps: int = 2000,
):
    # Load the trained PPO model 
    import mujoco
    import mujoco.viewer

    model_ppo = PPO.load(model_path)
    # Single CartPoleRLEnv (no VecEnv) so we can use its model/data for the viewer
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
    # Throttle to ~real time (same control_dt as training) so motion is easy to watch
    control_dt = env._action_repeat * env._physics_dt
    last_sync = time.perf_counter()
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        set_fixed_camera(viewer)
        while viewer.is_running() and step < max_steps:
            action, _ = model_ppo.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            viewer.sync()
            set_fixed_camera(viewer)
            step += 1
            if terminated or truncated:
                obs, _ = env.reset()
            # Run at ~1x real time so playback isn't frantic
            elapsed = time.perf_counter() - last_sync
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
            last_sync = time.perf_counter()
    print(f"Playback finished after {step} steps.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run PPO training")
    parser.add_argument("--play", action="store_true", help="Play trained agent in viewer")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--model", type=str, default=SAVE_PATH)
    args = parser.parse_args()
    if args.train:
        train(total_timesteps=args.timesteps)
    elif args.play:
        play_trained_agent(model_path=args.model)
    elif not (args.train or args.play):
        parser.print_help()
        print("Use --train to train, --play to run the trained agent in the MuJoCo viewer.")
