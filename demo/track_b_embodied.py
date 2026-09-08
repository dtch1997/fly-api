"""Track B: embodied motor control + perception (flygym / NeuroMechFly v1.2.1).

B1  motor: hybrid CPG+rules walking controller, straight then left/right
    turns on flat terrain; saves video + trajectory metrics.
B2  perception: visual taxis -- fly follows a moving sphere using its
    retina (ommatidia) readings mapped to descending steering drives;
    saves video with retina insets + tracking metrics.

Run with MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa and LD_LIBRARY_PATH
pointing at the local OSMesa prefix (see setup_env.sh).
"""
import json
import numpy as np
from pathlib import Path
from tqdm import trange

OUT = Path('runs/track_b'); OUT.mkdir(parents=True, exist_ok=True)
RESULTS = []

# ---------------- B1: walking ----------------
def run_walking():
    from flygym import Fly
    from flygym.arena import FlatTerrain
    from flygym.examples.locomotion import HybridTurningController
    from flygym import YawOnlyCamera

    contact_sensor_placements = [
        f"{leg}{seg}" for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for seg in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]
    np.random.seed(0)
    fly = Fly(enable_adhesion=True, draw_adhesion=True,
              contact_sensor_placements=contact_sensor_placements)
    cam = YawOnlyCamera(attachment_point=fly.model.worldbody,
                        camera_name="camera_right",
                        targeted_fly_names=fly.name, play_speed=0.1)
    sim = HybridTurningController(fly=fly, cameras=[cam], timestep=1e-4,
                                  seed=0, arena=FlatTerrain())
    run_time = 1.5
    obs, info = sim.reset(0)
    traj = []
    for i in trange(int(run_time / sim.timestep), desc='B1 walking'):
        t = i * sim.timestep
        if t < 0.6:
            action = np.array([1.0, 1.0])      # straight
        elif t < 1.05:
            action = np.array([1.2, 0.4])      # turn left
        else:
            action = np.array([0.4, 1.2])      # turn right
        obs, reward, term, trunc, info = sim.step(action)
        traj.append(obs['fly'][0].copy())
        sim.render()
    traj = np.array(traj)
    cam.save_video(OUT / 'b1_walking.mp4', 0)
    np.save(OUT / 'b1_trajectory.npy', traj)
    disp = float(np.linalg.norm(traj[-1, :2] - traj[0, :2]))
    RESULTS.append({'track': 'B1', 'metric': 'displacement_mm', 'value': round(disp, 2)})
    RESULTS.append({'track': 'B1', 'metric': 'mean_speed_mm_s', 'value': round(disp / run_time, 2)})
    RESULTS.append({'track': 'B1', 'metric': 'video', 'value': str(OUT / 'b1_walking.mp4')})
    print('[B1] done, displacement', disp)

# ---------------- B2: visual taxis ----------------
def run_taxis():
    from flygym import Fly, Camera
    from flygym.vision import save_video_with_vision_insets
    from svt_arena import MovingObjArena
    import svt_taxis as svt

    contact_sensor_placements = [
        f"{leg}{seg}" for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for seg in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]
    arena = MovingObjArena()
    fly = Fly(contact_sensor_placements=contact_sensor_placements,
              enable_adhesion=True, enable_vision=True, neck_kp=1000)
    cam = Camera(attachment_point=arena.root_element.worldbody,
                 camera_name="birdeye_cam",
                 camera_parameters={"mode": "fixed", "pos": (15, 0, 35),
                                    "euler": (0, 0, 0), "fovy": 45},
                 play_speed=0.5, window_size=(800, 608))
    sim = svt.VisualTaxis(fly=fly, camera=cam, obj_threshold=0.15,
                          decision_interval=0.05, arena=arena,
                          intrinsic_freqs=np.ones(6) * 9)

    if hasattr(svt, 'calc_ipsilateral_speed'):
        calc = svt.calc_ipsilateral_speed
    else:  # fallback: turn toward the side that sees the object
        def calc(deviation, is_found):
            if not is_found:
                return 1.0
            return float(np.clip(0.4 + deviation * 0.8, 0.4, 1.2))

    obs, _ = sim.reset()
    deviations = []
    n_seen = 0
    n_steps = 120
    for i in trange(n_steps, desc='B2 visual taxis'):
        left_dev = 1 - obs[1]
        right_dev = obs[4]
        left_found = obs[2] > 0.01
        right_found = obs[5] > 0.01
        if not left_found:
            left_dev = np.nan
        if not right_found:
            right_dev = np.nan
        control = np.array([calc(left_dev, left_found), calc(right_dev, right_found)])
        obs, _, _, _, _ = sim.step(control)
        deviations.append([left_dev, right_dev])
        n_seen += int(left_found or right_found)
    cam.save_video(OUT / 'b2_taxis.mp4')
    save_video_with_vision_insets(sim, cam, OUT / 'b2_taxis_with_retina.mp4',
                                  sim.visual_inputs_hist)
    dev = np.array(deviations, dtype=float)
    RESULTS.append({'track': 'B2', 'metric': 'frac_steps_object_seen',
                    'value': round(n_seen / n_steps, 3)})
    RESULTS.append({'track': 'B2', 'metric': 'mean_abs_deviation',
                    'value': round(float(np.nanmean(np.abs(dev))), 3)})
    RESULTS.append({'track': 'B2', 'metric': 'video',
                    'value': str(OUT / 'b2_taxis_with_retina.mp4')})
    np.save(OUT / 'b2_deviations.npy', dev)
    print('[B2] done')

if __name__ == '__main__':
    if (OUT / 'b1_walking.mp4').exists():
        print('[B1] outputs exist, skipping')
    else:
        run_walking()
    run_taxis()
    with open('results.jsonl', 'a') as f:
        for r in RESULTS:
            f.write(json.dumps(r) + '\n')
    print(json.dumps(RESULTS, indent=2))
