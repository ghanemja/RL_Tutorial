"""
based on episode rewards and rewards components, prints warnings when:
1. Oscillating rewards (large up-down swings)
2. Reward collapse (reward drops and stays low)
3. Termination exploitation (episodes end very early consistently)
4. Unstable training (high variance, no improvement)
"""

from typing import List, Dict, Any, Optional
import numpy as np


def check_episode_rewards(rewards: List[float], window: int = 20) -> None:
    # Warn on oscillating rewards, collapse, or instability
    if len(rewards) < window * 2:
        return
    r = np.array(rewards, dtype=float)
    recent = r[-window:]
    earlier = r[-2 * window : -window]
    mean_recent = np.mean(recent)
    mean_earlier = np.mean(earlier)
    std_recent = np.std(recent)
    # Oscillation: high variance and sign changes in diffs
    diffs = np.diff(recent)
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    if std_recent > 0.5 * (np.abs(mean_recent) + 1e-6) and sign_changes >= window // 2:
        print("[Diagnostics] Warning: oscillating rewards (high variance and many sign changes).")
    # Collapse: reward dropped a lot and stayed low
    if mean_recent < mean_earlier - 0.5 and mean_recent < 0:
        print("[Diagnostics] Warning: possible reward collapse (reward dropped and stayed low).")
    # Unstable: very high variance
    if std_recent > 2.0 * np.abs(mean_recent) and len(rewards) > 50:
        print("[Diagnostics] Warning: unstable training (high variance in episode rewards).")


def check_reward_components(components: List[Dict[str, float]]) -> None:
    # Warn if one component dominates or termination penalty is high (exploitation)
    if len(components) < 10:
        return
    keys = list(components[0].keys()) if components[0] else []
    for k in keys:
        vals = [c.get(k, 0) for c in components]
        mean_val = np.mean(vals)
        if k == "termination_penalty" and mean_val > 0.5:
            print("[Diagnostics] Warning: high termination penalty (agent may be exploiting early termination).")
        if k == "control_penalty" and mean_val > 2.0 * np.mean([sum(c.values()) for c in components]):
            print("[Diagnostics] Info: control penalty is large relative to total (check w_control).")


def check_episode_lengths(lengths: List[int], max_steps: Optional[int] = None) -> None:
    # Warn if episodes are consistently very short (termination exploitation)
    if len(lengths) < 20:
        return
    recent = np.array(lengths[-20:])
    # Use actual max length seen if not provided, so we don't false-positive when env uses a small max_episode_steps
    effective_max = max_steps if max_steps is not None else int(np.max(lengths))
    if effective_max <= 0:
        effective_max = 500
    if np.mean(recent) < 0.2 * effective_max:
        print("[Diagnostics] Warning: episodes very short on average (possible termination exploitation).")
