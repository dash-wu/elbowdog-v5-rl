"""Sim2sim: run a policy trained in Isaac Gym (PhysX) inside MuJoCo.

  # flat-ground speed sweep (table)
  python sim2sim/deploy_mujoco.py --policy <run>/exported --scene flat --mode speed
  # obstacle course with a follow-camera video and per-section results
  python sim2sim/deploy_mujoco.py --policy <run>/exported --scene course --mode course --video
  # interactive viewer:  Up/Down = vx,  Left/Right = yaw rate,  PageUp/PageDown = vy,  Home = stop
  python sim2sim/deploy_mujoco.py --policy <run>/exported --scene course --mode viewer

Differences to the training simulator, on purpose: MuJoCo contact model, 500 Hz PD instead of
200 Hz, Coulomb joint friction through frictionloss, exact ray-cast heights instead of the
heightfield lookup, no observation noise.
"""
import argparse
import json
import math
import os
import sys
import time

if "viewer" not in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")   # off-screen rendering for videos
import mujoco
import numpy as np
import torch

from scenes import build_scene

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOT_XML = os.path.join(ROOT, "assets", "elbowdog_v5", "mjcf", "elbowdog_v5.xml")
MESH_DIR = os.path.join(ROOT, "assets", "elbowdog_v5", "meshes")


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Sim2Sim:
    def __init__(self, policy_dir, scene="flat", sim_dt=0.002, seed=0, frictionloss=None, solref=None, delay_steps=0,
                 impratio=None):
        with open(os.path.join(policy_dir, "deploy.json")) as f:
            self.cfg = json.load(f)
        self.policy = torch.jit.load(os.path.join(policy_dir, "policy.pt"), map_location="cpu").eval()
        xml, hdata, self.sections = build_scene(ROBOT_XML, MESH_DIR, scene, seed)
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.m.opt.timestep = sim_dt
        if hdata is not None:
            hid = self.m.hfield("rough").id
            adr = self.m.hfield_adr[hid]
            self.m.hfield_data[adr:adr + hdata.size] = hdata.ravel()
        if frictionloss is not None:
            self.m.dof_frictionloss[6:] = frictionloss
        if solref is not None:
            self.m.geom_solref[:, 0] = solref      # contact time constant [s], MuJoCo default 0.02
        if impratio is not None:                   # elliptic friction cone, harder friction constraint
            self.m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
            self.m.opt.impratio = impratio
        self.delay_steps = delay_steps             # actuation delay in MuJoCo steps
        self.d = mujoco.MjData(self.m)
        c = self.cfg
        self.decimation = int(round(c["policy_dt"] / sim_dt))
        names = c["dof_names"]
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in names])
        self.vadr = np.array([self.m.jnt_dofadr[self.m.joint(n).id] for n in names])
        self.act = np.array([self.m.actuator(n).id for n in names])
        self.default = np.array(c["default_joint_angles"])
        self.kp, self.kd = np.array(c["kp"]), np.array(c["kd"])
        self.tau_max = np.array(c["torque_limits"])
        self.w_max = np.array(c["velocity_limits"])
        self.tau_sat = self.tau_max * c["motor_saturation_ratio"]
        sc = c["obs_scales"]
        self.cmd_scale = np.array([sc["lin_vel"], sc["lin_vel"], sc["ang_vel"]])
        gx, gy = np.meshgrid(np.array(c["measured_points_x"]), np.array(c["measured_points_y"]), indexing="ij")
        self.scan_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        self.geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self.base_id = self.m.body("base_link").id
        self.foot_geoms = [self.m.geom(f"{l}_foot_collision").id for l in ("FL", "FR", "RL", "RR")]
        self.feet_bodies = [self.m.body(f"{l}_foot").id for l in ("FL", "FR", "RL", "RR")]
        self.action = np.zeros(12)
        self.tau = np.zeros(12)

    # ---------------------------------------------------------------- state
    def reset(self, x=0.0, y=0.0, yaw=0.0):
        mujoco.mj_resetData(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)          # geom poses are needed by the ray cast below
        self.d.qpos[0:3] = [x, y, self.ground_height(x, y) + 0.30]
        self.d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.d.qpos[self.qadr] = self.default
        self.action[:] = 0
        mujoco.mj_forward(self.m, self.d)

    def ground_height(self, x, y):
        geomid = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(self.m, self.d, np.array([x, y, 5.0]), np.array([0, 0, -1.0]), self.geomgroup, 1, -1, geomid)
        return 5.0 - dist if dist >= 0 else 0.0

    def base_state(self):
        q = self.d.qpos[3:7]
        R = quat_to_mat(q)
        lin = R.T @ self.d.qvel[0:3]
        ang = self.d.qvel[3:6].copy()
        grav = R.T @ np.array([0, 0, -1.0])
        yaw = math.atan2(R[1, 0], R[0, 0])
        return R, lin, ang, grav, yaw

    def heights(self, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        base = self.d.qpos[0:3]
        out = np.empty(len(self.scan_pts))
        geomid = np.array([-1], dtype=np.int32)
        for i, (px, py) in enumerate(self.scan_pts):
            wx, wy = base[0] + c * px - s * py, base[1] + s * px + c * py
            dist = mujoco.mj_ray(self.m, self.d, np.array([wx, wy, base[2] + 1.0]), np.array([0, 0, -1.0]),
                                 self.geomgroup, 1, -1, geomid)
            out[i] = base[2] + 1.0 - dist if dist >= 0 else 0.0
        return out

    def observation(self, cmd):
        c, sc = self.cfg, self.cfg["obs_scales"]
        R, lin, ang, grav, yaw = self.base_state()
        q = self.d.qpos[self.qadr]
        dq = self.d.qvel[self.vadr]
        parts = []
        if c["actor_lin_vel"]:
            parts.append(lin * sc["lin_vel"])
        parts += [ang * sc["ang_vel"], grav, np.asarray(cmd) * self.cmd_scale,
                  (q - self.default) * sc["dof_pos"], dq * sc["dof_vel"], self.action]
        if c["measure_heights"]:
            h = self.heights(yaw)
            parts.append(np.clip(self.d.qpos[2] - c["height_obs_offset"] - h, -1, 1) * sc["height_measurements"])
        obs = np.concatenate(parts)
        return np.clip(obs, -c["clip_observations"], c["clip_observations"])

    # ---------------------------------------------------------------- control
    def motor_torque(self, target):
        q = self.d.qpos[self.qadr]
        dq = self.d.qvel[self.vadr]
        tau = self.kp * (target - q) - self.kd * dq
        ratio = dq / self.w_max
        hi = np.clip(self.tau_sat * (1 - ratio), 0, self.tau_max)
        lo = np.clip(self.tau_sat * (-1 - ratio), -self.tau_max, 0)
        return np.clip(tau, lo, hi)

    def step(self, cmd):
        obs = self.observation(cmd)
        with torch.no_grad():
            a = self.policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).numpy()[0]
        prev_target = self.action * self.cfg["action_scale"] + self.default
        self.action = np.clip(a, -self.cfg["clip_actions"], self.cfg["clip_actions"])
        target = self.action * self.cfg["action_scale"] + self.default
        power = 0.0
        for i in range(self.decimation):
            self.tau = self.motor_torque(prev_target if i < self.delay_steps else target)
            self.d.ctrl[self.act] = self.tau
            mujoco.mj_step(self.m, self.d)
            power += np.sum(np.abs(self.tau * self.d.qvel[self.vadr]))
        return power / self.decimation

    def fallen(self):
        _, _, _, grav, _ = self.base_state()
        if grav[2] > -0.3:
            return True
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            g1, g2 = con.geom1, con.geom2
            b1, b2 = self.m.geom_bodyid[g1], self.m.geom_bodyid[g2]
            if (b1 == self.base_id and self.m.geom_group[g2] == 0) or (b2 == self.base_id and self.m.geom_group[g1] == 0):
                return True
        return False


class Recorder:
    def __init__(self, sim, width=1280, height=720):
        self.sim = sim
        self.r = mujoco.Renderer(sim.m, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = sim.base_id
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.6, 125.0, -18.0
        self.opt = mujoco.MjvOption()
        self.frames = []

    def capture(self):
        self.r.update_scene(self.sim.d, camera=self.cam, scene_option=self.opt)
        self.frames.append(self.r.render().copy())

    def save(self, path, fps):
        import imageio
        imageio.mimsave(path, self.frames, fps=fps, quality=8, macro_block_size=8)
        return path


def heading_control(sim, yaw_target=0.0, y_target=0.0, k_yaw=1.0, k_y=1.0):
    _, _, _, _, yaw = sim.base_state()
    err = math.atan2(math.sin(yaw_target - yaw), math.cos(yaw_target - yaw))
    wz = float(np.clip(k_yaw * err, -0.8, 0.8))
    y_err = y_target - sim.d.qpos[1]
    c, s = math.cos(yaw), math.sin(yaw)
    vy = float(np.clip(k_y * (c * y_err), -0.3, 0.3))   # lateral correction in the body frame
    return vy, wz


def run_speed(sim, out_dir, speeds=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0), t_ramp=3.0, t_meas=8.0, tag=""):
    mass = float(np.sum(sim.m.body_mass))
    rows = []
    for v in speeds:
        sim.reset()
        dt = sim.cfg["policy_dt"]
        acc = {"vx": [], "vy": [], "roll": [], "pitch": [], "z": [], "p": []}
        ok = True
        for k in range(int((t_ramp + t_meas) / dt)):
            vy, wz = heading_control(sim)
            p = sim.step([v * min(1.0, k * dt / (0.6 * t_ramp)), vy, wz])
            if sim.fallen():
                ok = False
                break
            if k * dt >= t_ramp:
                R, lin, ang, grav, yaw = sim.base_state()
                acc["vx"].append(lin[0]); acc["vy"].append(lin[1]); acc["z"].append(sim.d.qpos[2]); acc["p"].append(p)
                acc["roll"].append(math.degrees(math.atan2(grav[1], -grav[2])))
                acc["pitch"].append(math.degrees(math.atan2(-grav[0], -grav[2])))
        r = {"command_vx": v, "success": ok}
        if ok:
            vx = np.array(acc["vx"])
            r.update({"mean_vx": float(vx.mean()), "tracking_error_rel": float(abs(vx.mean() - v) / v),
                      "vx_std": float(vx.std()), "roll_rms_deg": float(np.sqrt(np.mean(np.square(acc["roll"])))),
                      "pitch_rms_deg": float(np.sqrt(np.mean(np.square(acc["pitch"])))),
                      "base_z_mean": float(np.mean(acc["z"])), "mech_power_W": float(np.mean(acc["p"])),
                      "cost_of_transport": float(np.mean(acc["p"]) / (mass * 9.81 * max(vx.mean(), 0.05)))})
        rows.append(r)
        print(f"cmd {v:3.1f} m/s  ok={ok}  " + (f"v={r['mean_vx']:.3f}  err={100 * r['tracking_error_rel']:.1f}%  "
              f"roll={r['roll_rms_deg']:.2f}  pitch={r['pitch_rms_deg']:.2f}  P={r['mech_power_W']:.0f} W  "
              f"CoT={r['cost_of_transport']:.2f}" if ok else ""))
    path = os.path.join(out_dir, f"mujoco_speed_sweep{('_' + tag) if tag else ''}.json")
    with open(path, "w") as f:
        json.dump({"mass_kg": mass, "results": rows}, f, indent=2)
    print("saved:", path)


def run_course(sim, out_dir, video=False, v_obstacle=0.6, v_flat=1.0, v_sprint=2.0, timeout=120.0,
               scene_name="course", tag=""):
    rec = Recorder(sim) if video else None
    sim.reset(x=0.0)
    dt = sim.cfg["policy_dt"]
    secs = sim.sections
    reached = {name: None for name in secs}
    entered = {name: None for name in secs}
    t, log = 0.0, {"t": [], "x": [], "z": [], "vx": [], "cmd": []}
    sprint = secs.get("sprint")
    end_x = sprint[0] + 12.0 if sprint else max(b for _, b in secs.values()) + 1.5
    fell = False
    while t < timeout:
        x = sim.d.qpos[0]
        inside = [n for n, (a, b) in secs.items() if a - 0.6 <= x <= b and n != "sprint"]
        v = v_obstacle if inside else (v_sprint if sprint and x > sprint[0] else v_flat)
        vy, wz = heading_control(sim)
        sim.step([v, vy, wz])
        t += dt
        for n, (a, b) in secs.items():
            if entered[n] is None and x >= a:
                entered[n] = t
            if reached[n] is None and x >= b:
                reached[n] = t
        R, lin, *_ = sim.base_state()
        log["t"].append(t); log["x"].append(float(x)); log["z"].append(float(sim.d.qpos[2]))
        log["vx"].append(float(lin[0])); log["cmd"].append(v)
        if rec is not None and int(round(t / dt)) % 2 == 0:
            rec.capture()
        if sim.fallen():
            fell = True
            break
        if x > end_x:
            break
    result = {"fell": fell, "time_s": t, "final_x": float(sim.d.qpos[0]), "sections": {
        n: {"start_x": a, "end_x": b, "passed": reached[n] is not None,
            "time_s": None if reached[n] is None else reached[n] - entered[n]} for n, (a, b) in secs.items()}}
    name = f"mujoco_{scene_name}{('_' + tag) if tag else ''}"
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump({"result": result, "log": log}, f)
    for n, r in result["sections"].items():
        print(f"{n:16s} {'PASSED' if r['passed'] else 'failed'}" + (f"  {r['time_s']:.1f}s" if r["passed"] else ""))
    print(f"fell={fell}  time={t:.1f}s  final x={result['final_x']:.2f} m")
    if rec is not None:
        print("video:", rec.save(os.path.join(out_dir, name + ".mp4"), fps=int(round(0.5 / dt))))
    return result


def run_viewer(sim):
    import mujoco.viewer
    cmd = [0.0, 0.0, 0.0]
    keys = {265: (0, 0.25), 264: (0, -0.25), 263: (2, 0.4), 262: (2, -0.4), 266: (1, 0.2), 267: (1, -0.2)}

    def on_key(k):
        if k in keys:
            i, dv = keys[k]
            cmd[i] = float(np.clip(cmd[i] + dv, [-1.0, -0.6, -1.5][i], [3.0, 0.6, 1.5][i]))
        elif k == 268:
            cmd[:] = [0.0, 0.0, 0.0]
        print(f"\rcommand vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} wz={cmd[2]:+.2f}   ", end="", flush=True)

    sim.reset()
    print("Up/Down: vx   Left/Right: yaw rate   PageUp/PageDown: vy   Home: stop")
    with mujoco.viewer.launch_passive(sim.m, sim.d, key_callback=on_key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = sim.base_id
        viewer.cam.distance = 1.6
        dt = sim.cfg["policy_dt"]
        while viewer.is_running():
            t0 = time.perf_counter()
            sim.step(cmd)
            if sim.fallen():
                sim.reset(x=sim.d.qpos[0] - 0.5)
            viewer.sync()
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="directory with policy.pt and deploy.json")
    p.add_argument("--scene", default="flat", help="flat | course | ramp<deg> | stairs<cm>, e.g. ramp30, stairs12")
    p.add_argument("--mode", default="speed", choices=["speed", "course", "viewer"])
    p.add_argument("--video", action="store_true")
    p.add_argument("--sim_dt", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frictionloss", type=float, default=None, help="joint Coulomb friction [N*m]")
    p.add_argument("--solref", type=float, default=None, help="contact time constant [s]")
    p.add_argument("--delay_steps", type=int, default=0, help="actuation delay in MuJoCo steps")
    p.add_argument("--impratio", type=float, default=None, help="switch to elliptic cones with this impratio")
    p.add_argument("--tag", default="", help="suffix for result files")
    p.add_argument("--v_obstacle", type=float, default=0.6, help="forward command on obstacles [m/s]")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    sim = Sim2Sim(a.policy, a.scene, a.sim_dt, a.seed, a.frictionloss, a.solref, a.delay_steps, a.impratio)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.policy.rstrip("/"))), "sim2sim")
    os.makedirs(out, exist_ok=True)
    if a.mode == "speed":
        run_speed(sim, out, tag=a.tag)
    elif a.mode == "course":
        run_course(sim, out, video=a.video, v_obstacle=a.v_obstacle, timeout=a.timeout, scene_name=a.scene, tag=a.tag)
    else:
        run_viewer(sim)


if __name__ == "__main__":
    main()
