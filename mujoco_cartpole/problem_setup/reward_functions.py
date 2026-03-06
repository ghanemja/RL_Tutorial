"""
Modular reward components for the cart-pole RL environment.

Each component is computed separately so you can:
- Understand what drives the policy
- Tune weights via reward_dashboard.py
- Debug reward shaping

Total reward = w_balance * balance + w_upright * upright + w_center * center
               - w_velocity * velocity_penalty - w_control * control_penalty
(+ optional termination_penalty when episode ends early)

Weights are adjustable dynamically (e.g. from sliders in reward_dashboard.py).
"""

import numpy as np
from typing import Dict, Any, Optional

DEFAULT_WEIGHTS = {
    "w_balance": 1.0,
    "w_upright": 1.0,
    "w_center": 0.5,
    "w_velocity": 0.01,
    "w_control": 0.001,
    "w_termination": 0.0,
}

# Current weights (can be updated live by reward_dashboard or train_rl)
_reward_weights: Dict[str, float] = dict(DEFAULT_WEIGHTS)

# Reward type for reward-hacking demo: "default" | "bad" | "misleading" | "correct"
_reward_type: str = "default"


def set_reward_type(reward_type: str) -> None:
    # Set reward function for reward-hacking demo. Use 'bad', 'misleading', 'correct', or 'default'
    global _reward_type
    _reward_type = reward_type


def get_reward_type() -> str:
    return _reward_type


def set_reward_weights(weights: Dict[str, float]) -> None:
    # updates from sliders
    global _reward_weights
    for k, v in weights.items():
        if k in _reward_weights:
            _reward_weights[k] = float(v)


def get_reward_weights() -> Dict[str, float]:
    return dict(_reward_weights)


def compute_reward_components(
    cart_position: float,
    cart_velocity: float,
    pole_angle: float,
    pole_angular_velocity: float,
    action: np.ndarray,
    terminated: bool = False,
) -> Dict[str, float]:
    #computes each reward component from state and action and returns a dictionary of component names to scalar values.
    
    # Balance: reward for pole being upright (cos(angle) near 1)
    balance_reward = np.cos(pole_angle)

    # Upright: same idea, often same as balance; can be scaled differently
    upright_reward = np.cos(pole_angle)

    # Cart near center (x=0)
    cart_center_reward = 1.0 - 0.5 * min(1.0, abs(cart_position) / 2.0)

    # Velocity penalty (discourage fast motion; optional)
    velocity_penalty = cart_velocity ** 2 + 0.1 * (pole_angular_velocity ** 2)

    # Control penalty (discourage large forces)
    control_penalty = float(action[0] ** 2) if action is not None else 0.0

    # Termination penalty (optional, when episode ends early)
    termination_penalty = 1.0 if terminated else 0.0

    return {
        "balance": float(balance_reward),
        "upright": float(upright_reward),
        "center": float(cart_center_reward),
        "velocity_penalty": float(velocity_penalty),
        "control_penalty": float(control_penalty),
        "termination_penalty": float(termination_penalty),
    }


def compute_total_reward(
    reward_dict: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    # combine components using configurable weights.
    # total = w_balance*balance + w_upright*upright + w_center*center
    #        - w_velocity*velocity_penalty - w_control*control_penalty
    #        - w_termination*termination_penalty
    w = weights if weights is not None else get_reward_weights()
    total = (
        w.get("w_balance", 1.0) * reward_dict["balance"]
        + w.get("w_upright", 1.0) * reward_dict["upright"]
        + w.get("w_center", 0.5) * reward_dict["center"]
        - w.get("w_velocity", 0.01) * reward_dict["velocity_penalty"]
        - w.get("w_control", 0.001) * reward_dict["control_penalty"]
        - w.get("w_termination", 0.0) * reward_dict["termination_penalty"]
    )
    return float(total)


def reward_from_obs_action(
    obs: np.ndarray,
    action: np.ndarray,
    terminated: bool = False,
    weights: Optional[Dict[str, float]] = None,
) -> tuple[float, Dict[str, float]]:
    """
    compute reward components and total from observation and action.
    obs = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    Returns (total_reward, reward_dict).
    """
    if _reward_type == "bad":
        return compute_reward_bad(obs, action, terminated)
    if _reward_type == "misleading":
        return compute_reward_misleading(obs, action, terminated)
    if _reward_type == "correct":
        return compute_reward_correct(obs, action, terminated)
    cart_pos, cart_vel, pole_angle, pole_angular_vel = obs[0], obs[1], obs[2], obs[3]
    reward_dict = compute_reward_components(
        cart_pos, cart_vel, pole_angle, pole_angular_vel, action, terminated
    )
    total = compute_total_reward(reward_dict, weights)
    return total, reward_dict


def compute_reward_bad(
    obs: np.ndarray,
    action: np.ndarray,
    terminated: bool = False,
) -> tuple[float, Dict[str, float]]:
    """
    Bad reward: only depends on pole angle.
    reward = -|pole_angle|
    Agent learns unstable behavior or fails to balance (no signal for cart/control).
    """
    pole_angle = float(obs[2])
    total = -abs(pole_angle)
    reward_dict = {
        "pole_angle_penalty": abs(pole_angle),
        "balance": 0.0,
        "upright": 0.0,
        "center": 0.0,
        "velocity_penalty": 0.0,
        "control_penalty": 0.0,
        "termination_penalty": 0.0,
    }
    return total, reward_dict


def compute_reward_misleading(
    obs: np.ndarray,
    action: np.ndarray,
    terminated: bool = False,
) -> tuple[float, Dict[str, float]]:
    """
    Misleading reward: reward forward cart velocity.
    reward = cart_velocity
    Agent learns to slam cart back and forth instead of balancing the pole.
    """
    cart_vel = float(obs[1])
    total = cart_vel
    reward_dict = {
        "cart_velocity": cart_vel,
        "balance": 0.0,
        "upright": 0.0,
        "center": 0.0,
        "velocity_penalty": 0.0,
        "control_penalty": 0.0,
        "termination_penalty": 0.0,
    }
    return total, reward_dict


def compute_reward_correct(
    obs: np.ndarray,
    action: np.ndarray,
    terminated: bool = False,
) -> tuple[float, Dict[str, float]]:
    """
    the reward focuses on having an upright pole, centered cart, and smooth control
    reward = upright_bonus + center_bonus - velocity_penalty - control_penalty
    to teach the agent stable pole balancing
    """
    
    cart_pos, cart_vel, pole_angle, pole_angular_vel = obs[0], obs[1], obs[2], obs[3]
    upright_bonus = np.cos(pole_angle)
    center_bonus = 1.0 - 0.5 * min(1.0, abs(cart_pos) / 2.0)
    velocity_penalty = cart_vel ** 2 + 0.1 * (pole_angular_vel ** 2)
    control_penalty = float(action[0] ** 2) if action is not None else 0.0
    total = upright_bonus + center_bonus - 0.01 * velocity_penalty - 0.001 * control_penalty

    reward_dict = {
        "upright_bonus": float(upright_bonus),
        "center_bonus": float(center_bonus),
        "velocity_penalty": float(velocity_penalty),
        "control_penalty": float(control_penalty),
        "balance": float(upright_bonus),
        "upright": float(upright_bonus),
        "center": float(center_bonus),
        "termination_penalty": 0.0,
    }
    return total, reward_dict
