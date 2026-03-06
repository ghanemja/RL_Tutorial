# gymnasium environment wrapping the same cart-pole model used in open_loop.py.

# Loads cartpole.xml and reuses the same stepping logic (mj_step). 

# Observation: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
# Action: continuous force on cart (same as data.ctrl[0] in open_loop.py).

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

XML_PATH = os.environ.get("MUJOCO_CARTPOLE_XML", os.path.join(os.path.dirname(__file__), "cartpole.xml"))

DEFAULT_MAX_EPISODE_STEPS = 500
DEFAULT_ACTION_REPEAT = 10
DEFAULT_INITIAL_STATE_NOISE = 0.02   # rad/s for angle, m/s for velocity, m for position
# Termination: pole angle (rad) beyond which episode ends
DEFAULT_POLE_ANGLE_LIMIT = np.pi / 2   # 90°
# Cart position bounds (m) ,  should match cartpole.xml slide range (-2, 2)
DEFAULT_CART_X_MIN = -2.0
DEFAULT_CART_X_MAX = 2.0


def _get_state_from_data(data: mujoco.MjData) -> tuple[float, float, float, float]:
    cart_pos = float(data.qpos[0])
    pole_angle = float(data.qpos[1])
    cart_vel = float(data.qvel[0])
    pole_angular_vel = float(data.qvel[1])
    return cart_pos, cart_vel, pole_angle, pole_angular_vel


class CartPoleRLEnv(gym.Env):
    # gym env with the same physics as open_loop.py; used for RL training and policy playback.

    metadata = {"render_modes": []}

    def __init__(
        self,
        xml_path: str = XML_PATH,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        action_repeat: int = DEFAULT_ACTION_REPEAT,
        initial_state_noise: float = DEFAULT_INITIAL_STATE_NOISE,
        pole_angle_limit: float = DEFAULT_POLE_ANGLE_LIMIT,
        cart_x_min: float = DEFAULT_CART_X_MIN,
        cart_x_max: float = DEFAULT_CART_X_MAX,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._xml_path = xml_path
        self._max_episode_steps = max_episode_steps
        self._action_repeat = action_repeat
        self._initial_state_noise = initial_state_noise
        self._pole_angle_limit = pole_angle_limit
        self._cart_x_min = cart_x_min
        self._cart_x_max = cart_x_max

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._physics_dt = float(self.model.opt.timestep)

        # Same action as open_loop: continuous force on cart (ctrl[0])
        self.action_space = spaces.Box(
            low=np.array([-10.0], dtype=np.float32),
            high=np.array([10.0], dtype=np.float32),
            dtype=np.float32,
        )
        # Observation: cart_position, cart_velocity, pole_angle, pole_angular_velocity
        self.observation_space = spaces.Box(
            low=np.array([cart_x_min, -np.inf, -np.pi, -np.inf], dtype=np.float32),
            high=np.array([cart_x_max, np.inf, np.pi, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

        self._step_count = 0

    def _get_obs(self) -> np.ndarray:
        cart_pos, cart_vel, pole_angle, pole_angular_vel = _get_state_from_data(self.data)
        return np.array([cart_pos, cart_vel, pole_angle, pole_angular_vel], dtype=np.float32)

    def _is_terminated(self) -> bool:
        _, _, pole_angle, _ = _get_state_from_data(self.data)
        cart_pos = float(self.data.qpos[0])
        if abs(pole_angle) > self._pole_angle_limit:
            return True
        if cart_pos < self._cart_x_min or cart_pos > self._cart_x_max:
            return True
        return False

    def _is_truncated(self) -> bool:
        return self._step_count >= self._max_episode_steps

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        if self._initial_state_noise > 0 and self.np_random is not None:
            self.data.qpos[0] += self.np_random.uniform(-self._initial_state_noise, self._initial_state_noise)
            self.data.qpos[1] += self.np_random.uniform(-self._initial_state_noise, self._initial_state_noise)
            self.data.qvel[0] += self.np_random.uniform(-self._initial_state_noise, self._initial_state_noise)
            self.data.qvel[1] += self.np_random.uniform(-self._initial_state_noise, self._initial_state_noise)
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Hold action for action_repeat steps (same idea as open_loop.py)
        for _ in range(self._action_repeat):
            self.data.ctrl[0] = float(action[0])
            mujoco.mj_step(self.model, self.data)
            self._step_count += 1
            if self._is_terminated() or self._is_truncated():
                break

        obs = self._get_obs()
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = {
            "cart_position": obs[0],
            "cart_velocity": obs[1],
            "pole_angle": obs[2],
            "pole_angular_velocity": obs[3],
        }
        # Base env returns 0; use RewardWrapper (below) to inject reward_functions.
        return obs, 0.0, terminated, truncated, info

    def get_model_and_data(self):
        return self.model, self.data


class RewardWrapper(gym.Wrapper):
    # Wraps CartPoleRLEnv and computes reward using reward_functions.py
    # Adds reward_dictionary to info so training can log component breakdown

    def __init__(self, env: gym.Env):
        super().__init__(env)
        import sys
        _dir = os.path.dirname(os.path.abspath(__file__))
        if _dir not in sys.path:
            sys.path.insert(0, _dir)
        import reward_functions as rf
        self._rf = rf

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        total_reward, reward_dict = self._rf.reward_from_obs_action(
            obs, action, terminated=terminated
        )
        info["reward_dict"] = reward_dict
        return obs, total_reward, terminated, truncated, info
