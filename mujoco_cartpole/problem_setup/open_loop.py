"""
open-loop control demo

force u(t) = amp*sin(2*pi*freq*t) depends only on time,
not on cart/pole state. 

What to expect: The cart will slide and the pole will swing, then the pole will
fall. That is expected because open-loop does not try to balance. Re-run to see it again,
or press R in the viewer to reset. Press S for script (sinusoid), G to try GUI
control (if the viewer shows actuator sliders in the right panel).

The point of this script is to check that:
The sim runs,
The cart motor responds to data.ctrl[0],
Motion looks reasonable (no explosion or jitter).
"""
import os
import time
import warnings

# On Linux Wayland, GLFW may warn about window position; viewer still works.
warnings.filterwarnings("ignore", message=".*Wayland.*window position.*")

import numpy as np
import mujoco
import mujoco.viewer

_XML_PATH = os.path.join(os.path.dirname(__file__), "cartpole.xml")
model = mujoco.MjModel.from_xml_path(_XML_PATH)
data = mujoco.MjData(model)

action_repeat = 10 # hold action for this many physics steps
amp = 12.0         # force amplitude (N)
freq = 0.5          # sinusoid frequency (Hz)
physics_dt = model.opt.timestep  # simulation step size (ie 0.002 s)
control_dt = physics_dt * action_repeat  # time between control updates

# Shared state for key callback: 'script' = sinusoid, 'gui' = don't set ctrl (try viewer sliders)
state = {"mode": "script"}


def key_callback(keycode):
    # 82 = R, 83 = S, 71 = G (ASCII)
    if keycode == 82:  # R: reset
        mujoco.mj_resetData(model, data)
    elif keycode == 83:  # S: script control (sinusoid)
        state["mode"] = "script"
    elif keycode == 71:  # G: GUI control
        state["mode"] = "gui"


def set_fixed_camera(viewer):
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.azimuth = 90.0   # side view so cart sliding (along x) is clearly visible
    viewer.cam.elevation = -15.0
    viewer.cam.distance = 2.8
    viewer.cam.lookat[:] = [0.0, 0.0, 0.4]


with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    set_fixed_camera(viewer)
    step = 0
    last_sync = time.perf_counter()
    while viewer.is_running():
        if state["mode"] == "script" and step % action_repeat == 0:
            u = amp * np.sin(2 * np.pi * freq * data.time)
            data.ctrl[0] = u
        # in gui mode don't set ctrl, instead use the viewer's right panel with actuator sliders
        mujoco.mj_step(model, data)
        viewer.sync()
        set_fixed_camera(viewer)
        step += 1
        if step % action_repeat == 0:
            now = time.perf_counter()
            elapsed = now - last_sync
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
            last_sync = time.perf_counter()
