"""Evaluate a trained ElbowDog policy in Isaac Gym.

  # scripted command sequence, follow-camera video, tracking plot, policy export
  python scripts/play.py --task elbowdog_flat --load_run <run> --mode demo
  # speed sweep: many robots with fixed forward commands, table of tracking / stability / energy
  python scripts/play.py --task elbowdog_flat --load_run <run> --mode speed
  # terrain sweep (rough task): success and travelled distance per terrain type and difficulty
  python scripts/play.py --task elbowdog_rough --load_run <run> --mode terrain

Results go to logs/<experiment>/<run>/eval/.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import isaacgym  # noqa: F401
from isaacgym import gymapi, gymutil
import legged_gym.envs  # noqa: F401
from legged_gym.utils import task_registry
import numpy as np
import torch

import elbowdog_gym  # noqa: F401
from elbowdog_gym.utils.export import export_policy
from elbowdog_gym.utils.terrain import TERRAIN_TYPES

DEMO_SCHEDULE = [  # (duration s, vx, vy, wz)
    (2.0, 0.0, 0.0, 0.0), (3.0, 0.5, 0.0, 0.0), (3.0, 1.0, 0.0, 0.0), (3.0, 1.5, 0.0, 0.0),
    (3.0, 2.0, 0.0, 0.0), (3.0, 2.5, 0.0, 0.0), (3.0, 3.0, 0.0, 0.0), (3.0, 1.0, 0.0, 1.0),
    (3.0, 0.0, 0.5, 0.0), (3.0, 0.0, -0.5, 0.0), (3.0, -0.8, 0.0, 0.0), (3.0, 0.0, 0.0, -1.2),
    (2.0, 0.0, 0.0, 0.0)]
SPEEDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def get_args():
    params = [
        {"name": "--task", "type": str, "default": "elbowdog_flat"},
        {"name": "--experiment_name", "type": str},
        {"name": "--load_run", "type": str, "default": "-1"},
        {"name": "--checkpoint", "type": int, "default": -1},
        {"name": "--mode", "type": str, "default": "demo", "help": "demo | speed | terrain"},
        {"name": "--num_envs", "type": int},
        {"name": "--no_video", "action": "store_true", "default": False},
        {"name": "--v_cmd", "type": float, "default": 0.8, "help": "forward command for terrain / tile modes"},
        {"name": "--terrain_type", "type": str, "default": "stairs_up", "help": "tile mode: terrain type"},
        {"name": "--level", "type": int, "default": 8, "help": "tile mode: difficulty row 0-9"},
        {"name": "--rl_device", "type": str, "default": "cuda:0"},
        {"name": "--seed", "type": int, "default": 1},
    ]
    args = gymutil.parse_arguments(description="ElbowDog play", custom_parameters=params)
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type + (f":{args.sim_device_id}" if args.sim_device_type == "cuda" else "")
    args.headless = True
    for k in ("resume", "max_iterations", "run_name", "horovod"):
        setattr(args, k, False if k in ("resume", "horovod") else None)
    return args


def resolve_run(log_root, load_run):
    runs = sorted(r for r in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, r)) and r != "exported")
    return runs[-1] if load_run in ("-1", None) else load_run


def make(args, num_envs, configure):
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = num_envs
    env_cfg.env.offscreen_render = not args.no_video
    env_cfg.env.episode_length_s = 1000.0
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_joint_friction = False
    env_cfg.domain_rand.randomize_delay = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 1e6
    env_cfg.commands.stand_prob = 0.0
    configure(env_cfg)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    exp = args.experiment_name or train_cfg.runner.experiment_name
    log_root = os.path.join(ROOT, "logs", exp)
    run = resolve_run(log_root, args.load_run)
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = run
    train_cfg.runner.checkpoint = args.checkpoint
    args.load_run, args.checkpoint = run, args.checkpoint
    runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=log_root)
    runner.log_dir = None
    policy = runner.get_inference_policy(device=env.device)
    out = os.path.join(log_root, run, "eval")
    os.makedirs(out, exist_ok=True)
    return env, env_cfg, runner, policy, out


class FollowCamera:
    def __init__(self, env, width=1280, height=720):
        self.env = env
        props = gymapi.CameraProperties()
        props.width, props.height = width, height
        props.horizontal_fov = 55.0
        self.handle = env.gym.create_camera_sensor(env.envs[0], props)
        self.frames = []
        self.offset = np.array([0.55, -1.05, 0.38])
        self.smoothed = None

    def capture(self, robot=0):
        gym, sim = self.env.gym, self.env.sim
        base = self.env.root_states[robot, :3].cpu().numpy()
        self.smoothed = base if self.smoothed is None else 0.6 * self.smoothed + 0.4 * base
        target = self.smoothed + np.array([0.0, 0.0, -0.05])
        gym.set_camera_location(self.handle, self.env.envs[0], gymapi.Vec3(*(target + self.offset)), gymapi.Vec3(*target))
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        img = gym.get_camera_image(sim, self.env.envs[0], self.handle, gymapi.IMAGE_COLOR)
        self.frames.append(img.reshape(img.shape[0], -1, 4)[..., :3].copy())

    def save(self, path, fps):
        import imageio
        imageio.mimsave(path, self.frames, fps=fps, quality=8, macro_block_size=8)
        return path


def run_demo(args):
    def configure(c):
        c.terrain.num_rows, c.terrain.num_cols = min(c.terrain.num_rows, 5), min(c.terrain.num_cols, 8)
        c.terrain.curriculum = False
        if c.terrain.mesh_type != "plane":
            c.terrain.terrain_proportions = {"flat": 1.0}
    env, env_cfg, runner, policy, out = make(args, 1, configure)
    export_dir = export_policy(runner, env, env_cfg, os.path.join(out, "..", "exported"))
    cam = None if args.no_video else FollowCamera(env)
    obs = env.get_observations()
    log = {k: [] for k in ("t", "cmd_vx", "cmd_vy", "cmd_wz", "vx", "vy", "wz", "z", "roll", "pitch", "power",
                           "contact", "tau", "foot_fz")}
    t = 0.0
    for dur, vx, vy, wz in DEMO_SCHEDULE:
        for _ in range(int(dur / env.dt)):
            env.commands[:, 0], env.commands[:, 1], env.commands[:, 2] = vx, vy, wz
            env.compute_observations()
            obs = env.get_observations()
            with torch.no_grad():
                actions = policy(obs.detach())
            obs, _, _, _, _ = env.step(actions.detach())
            t += env.dt
            g = env.projected_gravity[0].cpu().numpy()
            log["t"].append(t); log["cmd_vx"].append(vx); log["cmd_vy"].append(vy); log["cmd_wz"].append(wz)
            log["vx"].append(float(env.base_lin_vel[0, 0])); log["vy"].append(float(env.base_lin_vel[0, 1]))
            log["wz"].append(float(env.base_ang_vel[0, 2])); log["z"].append(float(env.root_states[0, 2]))
            log["roll"].append(float(np.degrees(np.arctan2(g[1], -g[2]))))
            log["pitch"].append(float(np.degrees(np.arctan2(-g[0], -g[2]))))
            log["power"].append(float(torch.sum(torch.abs(env.torques[0] * env.dof_vel[0]))))
            log["contact"].append((env.contact_forces[0, env.feet_indices, 2] > 1.0).int().tolist())
            log["tau"].append(env.torques[0].tolist())
            log["foot_fz"].append(env.contact_forces[0, env.feet_indices, 2].tolist())
            if cam is not None and len(log["t"]) % 2 == 0:
                cam.capture()
    with open(os.path.join(out, "demo_log.json"), "w") as f:
        json.dump(log, f)
    plot_demo(log, os.path.join(out, "demo_tracking.png"))
    plot_gait(log, os.path.join(out, "demo_gait.png"), list(env.dof_names))
    if cam is not None:
        print("video:", cam.save(os.path.join(out, "demo.mp4"), fps=int(round(0.5 / env.dt))))
    print("exported policy:", os.path.abspath(export_dir))
    print("plot:", os.path.join(out, "demo_tracking.png"))


def plot_demo(log, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array(log["t"])
    fig, axs = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axs[0].plot(t, log["cmd_vx"], "k--", label="command vx"); axs[0].plot(t, log["vx"], label="base vx")
    axs[0].plot(t, log["cmd_vy"], "k:", label="command vy"); axs[0].plot(t, log["vy"], label="base vy")
    axs[0].set_ylabel("m/s"); axs[0].legend(loc="upper right", ncol=4); axs[0].grid(alpha=0.3)
    axs[1].plot(t, log["cmd_wz"], "k--", label="command wz"); axs[1].plot(t, log["wz"], label="base wz")
    axs[1].set_ylabel("rad/s"); axs[1].legend(loc="upper right"); axs[1].grid(alpha=0.3)
    axs[2].plot(t, log["roll"], label="roll"); axs[2].plot(t, log["pitch"], label="pitch")
    ax2 = axs[2].twinx(); ax2.plot(t, log["z"], "g", alpha=0.5, label="base z"); ax2.set_ylabel("base z [m]")
    axs[2].set_ylabel("deg"); axs[2].legend(loc="upper left"); axs[2].grid(alpha=0.3)
    axs[3].plot(t, log["power"], color="tab:red"); axs[3].set_ylabel("mech. power |tau*w| [W]"); axs[3].grid(alpha=0.3)
    axs[3].set_xlabel("time [s]")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


def plot_gait(log, path, dof_names, windows=((4.0, 5.0), (10.0, 11.0), (16.5, 17.5))):
    """Foot-contact diagram (black = stance) and peak joint torques for a few 1 s windows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array(log["t"])
    c = np.array(log["contact"])
    tau = np.abs(np.array(log["tau"]))
    fz = np.array(log["foot_fz"])
    fig, axs = plt.subplots(1, len(windows), figsize=(5 * len(windows), 2.6))
    for ax, (a, b) in zip(axs, windows):
        sel = (t >= a) & (t < b)
        ax.imshow(c[sel].T, aspect="auto", cmap="Greys", interpolation="nearest", extent=[a, b, 3.5, -0.5])
        ax.set_yticks(range(4)); ax.set_yticklabels(["FL", "FR", "RL", "RR"])
        duty = c[sel].mean(axis=0)
        v = float(np.mean(np.array(log["cmd_vx"])[sel]))
        ax.set_title(f"cmd vx {v:.1f} m/s, duty factor {np.round(duty, 2).tolist()}", fontsize=8)
        ax.set_xlabel("time [s]")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)
    peaks = {n: float(tau[:, i].max()) for i, n in enumerate(dof_names)}
    with open(os.path.splitext(path)[0] + "_peaks.json", "w") as f:
        json.dump({"peak_abs_torque_Nm": peaks, "peak_foot_fz_N": float(fz.max()),
                   "p99_foot_fz_N": float(np.percentile(fz, 99))}, f, indent=2)


def run_speed(args):
    per = 32

    def configure(c):
        c.terrain.num_rows, c.terrain.num_cols = 5, 8
        c.terrain.curriculum = False
        if c.terrain.mesh_type != "plane":
            c.terrain.terrain_proportions = {"flat": 1.0}
    n = per * len(SPEEDS)
    args.no_video = True
    env, env_cfg, runner, policy, out = make(args, n, configure)
    target = torch.tensor(np.repeat(SPEEDS, per), device=env.device, dtype=torch.float)
    mass = float(sum(env.gym.get_actor_rigid_body_properties(env.envs[0], 0)[i].mass
                     for i in range(env.num_bodies)))
    steps_ramp, steps_meas = int(3.0 / env.dt), int(8.0 / env.dt)
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    acc = {k: torch.zeros(n, device=env.device) for k in ("vx", "vx2", "vy2", "wz2", "roll2", "pitch2", "power", "z", "z2", "az2")}
    last_vz = env.root_states[:, 9].clone()
    for k in range(steps_ramp + steps_meas):
        env.commands[:, 0] = target * min(1.0, k / (0.6 * steps_ramp))
        env.commands[:, 1:3] = 0.0
        env.compute_observations()
        with torch.no_grad():
            actions = policy(env.get_observations())
        env.step(actions)
        alive &= ~(env.reset_buf.bool() & ~env.time_out_buf)
        if k >= steps_ramp:
            g = env.projected_gravity
            vz = env.root_states[:, 9]
            acc["vx"] += env.base_lin_vel[:, 0]; acc["vx2"] += env.base_lin_vel[:, 0] ** 2
            acc["vy2"] += env.base_lin_vel[:, 1] ** 2; acc["wz2"] += env.base_ang_vel[:, 2] ** 2
            acc["roll2"] += torch.atan2(g[:, 1], -g[:, 2]) ** 2; acc["pitch2"] += torch.atan2(-g[:, 0], -g[:, 2]) ** 2
            acc["power"] += torch.sum(torch.abs(env.torques * env.dof_vel), dim=1)
            acc["z"] += env.root_states[:, 2]; acc["z2"] += env.root_states[:, 2] ** 2
            acc["az2"] += ((vz - last_vz) / env.dt) ** 2
        last_vz = env.root_states[:, 9].clone()
    rows = []
    for i, v in enumerate(SPEEDS):
        sl = slice(i * per, (i + 1) * per)
        ok = alive[sl]
        m = {k: (acc[k][sl] / steps_meas) for k in acc}
        vx = m["vx"]
        rows.append({
            "command_vx": v,
            "success_rate": float(ok.float().mean()),
            "mean_vx": float(vx[ok].mean()) if ok.any() else float("nan"),
            "tracking_error_rel": float((torch.abs(vx[ok] - v) / v).mean()) if ok.any() else float("nan"),
            "vx_std_in_time": float(torch.sqrt(torch.clamp(m["vx2"] - vx ** 2, min=0))[ok].mean()),
            "lateral_drift_rms": float(torch.sqrt(m["vy2"])[ok].mean()),
            "yaw_rate_rms": float(torch.sqrt(m["wz2"])[ok].mean()),
            "roll_rms_deg": float(np.degrees(torch.sqrt(m["roll2"])[ok].mean().item())),
            "pitch_rms_deg": float(np.degrees(torch.sqrt(m["pitch2"])[ok].mean().item())),
            "base_z_mean": float(m["z"][ok].mean()),
            "base_z_std": float(torch.sqrt(torch.clamp(m["z2"] - m["z"] ** 2, min=0))[ok].mean()),
            "vertical_acc_rms": float(torch.sqrt(m["az2"])[ok].mean()),
            "mech_power_W": float(m["power"][ok].mean()),
            "cost_of_transport": float((m["power"][ok] / (mass * 9.81 * torch.clamp(vx[ok], min=0.05))).mean()),
        })
    with open(os.path.join(out, "speed_sweep.json"), "w") as f:
        json.dump({"mass_kg": mass, "results": rows}, f, indent=2)
    hdr = f"{'cmd':>5} {'succ':>5} {'v_mean':>7} {'err%':>6} {'roll°':>6} {'pitch°':>7} {'z':>6} {'a_z rms':>8} {'P[W]':>7} {'CoT':>5}"
    print(hdr)
    for r in rows:
        print(f"{r['command_vx']:5.1f} {r['success_rate']:5.2f} {r['mean_vx']:7.3f} {100 * r['tracking_error_rel']:6.1f} "
              f"{r['roll_rms_deg']:6.2f} {r['pitch_rms_deg']:7.2f} {r['base_z_mean']:6.3f} {r['vertical_acc_rms']:8.2f} "
              f"{r['mech_power_W']:7.1f} {r['cost_of_transport']:5.2f}")
    print("saved:", os.path.join(out, "speed_sweep.json"))


def run_terrain(args):
    """Every robot starts at the centre of one tile (type, level), heading +x with a fixed forward command.
    Success = advanced 3.5 m along +x, i.e. crossed the whole obstacle zone (1-4 m from the centre)."""
    per = 8

    def configure(c):
        c.terrain.curriculum = True
        c.terrain.max_init_terrain_level = c.terrain.num_rows - 1
    args.no_video = True
    env_cfg_probe, _ = task_registry.get_cfgs(args.task)
    n_types = env_cfg_probe.terrain.num_cols
    n = per * env_cfg_probe.terrain.num_rows * n_types
    env, env_cfg, runner, policy, out = make(args, n, configure)
    rows, cols = env_cfg.terrain.num_rows, env_cfg.terrain.num_cols
    ids = torch.arange(n, device=env.device)
    env.terrain_levels[:] = (ids // per) % rows
    env.terrain_types[:] = ids // (per * rows)
    env.env_terrain_type[:] = torch.tensor(env.terrain.col_types, device=env.device)[env.terrain_types]
    env.env_origins[:] = env.terrain_origins[env.terrain_levels, env.terrain_types]
    env.cfg.terrain.curriculum = False
    env.reset_idx(ids)
    env.root_states[:, :2] = env.env_origins[:, :2]
    env.root_states[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
    env.root_states[:, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor(env.sim, isaacgym.gymtorch.unwrap_tensor(env.root_states))
    x0 = env.env_origins[:, 0].clone()
    v_cmd = torch.clamp(env.type_vx_cap[env.env_terrain_type], max=args.v_cmd)
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    done = torch.zeros(n, dtype=torch.bool, device=env.device)
    t_done = torch.full((n,), float("nan"), device=env.device)
    target = 3.5
    steps = int((target / args.v_cmd + 10.0) / env.dt)
    for k in range(steps):
        env.commands[:, 0] = v_cmd * min(1.0, k / 50)
        env.commands[:, 1:3] = 0.0
        env.compute_observations()
        with torch.no_grad():
            actions = policy(env.get_observations())
        env.step(actions)
        alive &= ~(env.reset_buf.bool() & ~env.time_out_buf & ~done)
        new = alive & ~done & (env.root_states[:, 0] - x0 > target)
        t_done[new] = k * env.dt
        done |= new
    table, fall = {}, {}
    for j in range(cols):
        name = TERRAIN_TYPES[env.terrain.col_types[j]]
        for i in range(rows):
            sl = (env.terrain_types == j) & (env.terrain_levels == i)
            table.setdefault(name, {}).setdefault(str(i), []).append(float(done[sl].float().mean()))
            fall.setdefault(name, {}).setdefault(str(i), []).append(float((~alive[sl]).float().mean()))
    table = {k: {i: float(np.mean(v)) for i, v in lv.items()} for k, lv in table.items()}
    fall = {k: {i: float(np.mean(v)) for i, v in lv.items()} for k, lv in fall.items()}
    c = env_cfg.terrain
    difficulty = {
        "slope": [round(c.max_slope_deg * i / rows, 1) for i in range(rows)],
        "stairs_cm": [round(100 * (c.step_height[0] + c.step_height[1] * i / rows), 1) for i in range(rows)],
        "discrete_cm": [round(100 * (c.discrete_height[0] + c.discrete_height[1] * i / rows), 1) for i in range(rows)],
        "rough_cm": [round(100 * (c.rough_noise[0] + c.rough_noise[1] * i / rows), 1) for i in range(rows)]}
    name = f"terrain_sweep_v{args.v_cmd:g}.json"
    with open(os.path.join(out, name), "w") as f:
        json.dump({"v_cmd": args.v_cmd, "success": table, "fall": fall, "difficulty": difficulty}, f, indent=2)
    print(f"success rate: advanced {target} m across the obstacle within {steps * env.dt:.0f} s at {args.v_cmd} m/s")
    print(f"{'level':>12} " + " ".join(f"{i:>5d}" for i in range(rows)))
    print(f"{'slope deg':>12} " + " ".join(f"{v:5.1f}" for v in difficulty["slope"]))
    print(f"{'step cm':>12} " + " ".join(f"{v:5.1f}" for v in difficulty["stairs_cm"]))
    for k, lv in table.items():
        print(f"{k:>12} " + " ".join(f"{lv[str(i)]:5.2f}" for i in range(rows)))
    print("saved:", os.path.join(out, name))


def run_tile(args):
    """One robot walks straight across a chosen terrain tile; follow-camera video and a state log."""
    def configure(c):
        c.terrain.curriculum = True
        c.terrain.max_init_terrain_level = c.terrain.num_rows - 1
    env, env_cfg, runner, policy, out = make(args, 1, configure)
    col = list(env.terrain.col_types).index(TERRAIN_TYPES.index(args.terrain_type))
    env.cfg.terrain.curriculum = False
    env.terrain_levels[:] = args.level
    env.terrain_types[:] = col
    env.env_terrain_type[:] = env.terrain.col_types[col]
    env.env_origins[:] = env.terrain_origins[env.terrain_levels, env.terrain_types]
    ids = torch.arange(1, device=env.device)
    env.reset_idx(ids)
    env.root_states[0, :2] = env.env_origins[0, :2]
    env.gym.set_actor_root_state_tensor(env.sim, isaacgym.gymtorch.unwrap_tensor(env.root_states))
    cam = None if args.no_video else FollowCamera(env)
    log = {k: [] for k in ("t", "x", "z", "pitch", "vx", "contact", "action")}
    for k in range(int(12.0 / env.dt)):
        env.commands[:, 0] = args.v_cmd * min(1.0, k / 50)
        env.commands[:, 1:3] = 0.0
        env.compute_observations()
        with torch.no_grad():
            actions = policy(env.get_observations())
        env.step(actions)
        g = env.projected_gravity[0].cpu().numpy()
        log["t"].append(k * env.dt); log["x"].append(float(env.root_states[0, 0] - env.env_origins[0, 0]))
        log["z"].append(float(env.root_states[0, 2])); log["vx"].append(float(env.base_lin_vel[0, 0]))
        log["pitch"].append(float(np.degrees(np.arctan2(-g[0], -g[2]))))
        log["contact"].append((env.contact_forces[0, env.feet_indices, 2] > 1.0).int().tolist())
        log["action"].append(actions[0].tolist())
        if cam is not None and k % 2 == 0:
            cam.capture()
        if env.reset_buf[0] and not env.time_out_buf[0]:
            print("fell at t=%.2f" % (k * env.dt))
            break
    name = f"tile_{args.terrain_type}_L{args.level}_v{args.v_cmd:g}"
    with open(os.path.join(out, name + ".json"), "w") as f:
        json.dump(log, f)
    print(f"travelled x = {log['x'][-1]:.2f} m in {log['t'][-1]:.1f} s, mean pitch {np.mean(log['pitch']):.1f} deg")
    if cam is not None:
        print("video:", cam.save(os.path.join(out, name + ".mp4"), fps=int(round(0.5 / env.dt))))


if __name__ == "__main__":
    a = get_args()
    {"demo": run_demo, "speed": run_speed, "terrain": run_terrain, "tile": run_tile}[a.mode](a)
