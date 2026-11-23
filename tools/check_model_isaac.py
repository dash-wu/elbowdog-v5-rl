"""Sanity checks of the RL model in Isaac Gym (and the same standing test in MuJoCo).

  python tools/check_model_isaac.py            # numbers + images in logs/model_check/
"""
import json
import os
import sys

import isaacgym  # noqa: F401  (must be imported before torch)
from isaacgym import gymapi, gymtorch
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_DIR = os.path.join(ROOT, "assets", "elbowdog_v5", "urdf")
OUT = os.path.join(ROOT, "logs", "model_check")
DEFAULT = {"hip": 0.0, "thigh": 0.65, "calf": -1.3}
KP, KD = 20.0, 0.5


def asset_options(fix_base):
    o = gymapi.AssetOptions()
    o.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
    o.collapse_fixed_joints = True
    o.replace_cylinder_with_capsule = False
    o.flip_visual_attachments = False
    o.fix_base_link = fix_base
    o.density = 0.001
    o.armature = 0.01
    o.thickness = 0.01
    return o


def make_sim(gym):
    sp = gymapi.SimParams()
    sp.dt = 0.005
    sp.substeps = 1
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0, 0, -9.81)
    sp.use_gpu_pipeline = False
    sp.physx.solver_type = 1
    sp.physx.num_position_iterations = 4
    sp.physx.num_velocity_iterations = 0
    sp.physx.contact_offset = 0.01
    sp.physx.rest_offset = 0.0
    sp.physx.use_gpu = True
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)
    return sim


def render(gym, sim, env, name, pos, target):
    props = gymapi.CameraProperties()
    props.width, props.height = 960, 720
    cam = gym.create_camera_sensor(env, props)
    gym.set_camera_location(cam, env, gymapi.Vec3(*pos), gymapi.Vec3(*target))
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    path = os.path.join(OUT, name)
    gym.write_camera_image_to_file(sim, env, cam, gymapi.IMAGE_COLOR, path)
    return path


def standing_test_isaac(gym, report):
    sim = make_sim(gym)
    asset = gym.load_asset(sim, URDF_DIR, "elbowdog_v5.urdf", asset_options(False))
    body_names = gym.get_asset_rigid_body_names(asset)
    dof_names = gym.get_asset_dof_names(asset)
    dof_props = gym.get_asset_dof_properties(asset)
    report["isaac_body_names"] = body_names
    report["isaac_dof_names"] = dof_names
    report["isaac_dof_limits"] = {n: [float(dof_props["lower"][i]), float(dof_props["upper"][i]),
                                      float(dof_props["effort"][i]), float(dof_props["velocity"][i])]
                                  for i, n in enumerate(dof_names)}
    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 0.32)
    actor = gym.create_actor(env, asset, pose, "elbowdog", 0, 0, 0)

    before = gym.get_actor_rigid_body_properties(env, actor)
    inertia_before = [[b.inertia.x.x, b.inertia.y.y, b.inertia.z.z] for b in before]
    gym.set_actor_rigid_body_properties(env, actor, before, recomputeInertia=True)
    after = gym.get_actor_rigid_body_properties(env, actor)
    inertia_after = [[b.inertia.x.x, b.inertia.y.y, b.inertia.z.z] for b in after]
    report["isaac_masses"] = {n: b.mass for n, b in zip(body_names, after)}
    report["isaac_total_mass"] = float(sum(b.mass for b in after))
    report["recompute_inertia_changes_values"] = bool(not np.allclose(inertia_before, inertia_after, rtol=1e-4))
    report["inertia_diag_before_after"] = {n: [i0, i1] for n, i0, i1 in zip(body_names, inertia_before, inertia_after)}
    gym.set_actor_rigid_body_properties(env, actor, before, recomputeInertia=False)

    default = np.array([DEFAULT[n.split("_")[1]] for n in dof_names], dtype=np.float32)
    dof_state = np.zeros(len(dof_names), dtype=gymapi.DofState.dtype)
    dof_state["pos"] = default
    gym.set_actor_dof_states(env, actor, dof_state, gymapi.STATE_ALL)
    gym.prepare_sim(sim)

    rb_tensor = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
    cf_tensor = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))
    feet = [i for i, n in enumerate(body_names) if n.endswith("_foot")]
    torques = np.zeros(len(dof_names), dtype=np.float32)
    for step in range(600):
        state = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
        torques = KP * (default - state["pos"]) - KD * state["vel"]
        gym.apply_actor_dof_efforts(env, actor, torques.astype(np.float32))
        gym.simulate(sim)
        gym.fetch_results(sim, True)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)
    state = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
    base_z = float(rb_tensor[0, 2])
    foot_fz = cf_tensor[feet, 2].numpy().tolist()
    report["isaac_stand"] = {"base_height": base_z, "foot_contact_fz": foot_fz,
                             "sum_foot_fz": float(sum(foot_fz)),
                             "weight": report["isaac_total_mass"] * 9.81,
                             "foot_z": rb_tensor[feet, 2].numpy().tolist(),
                             "joint_pos": state["pos"].tolist(), "joint_torque": torques.tolist()}
    os.makedirs(OUT, exist_ok=True)
    imgs = [render(gym, sim, env, "isaac_stand_iso.png", (0.75, -0.75, 0.45), (0, 0, 0.15)),
            render(gym, sim, env, "isaac_stand_front.png", (0.9, 0.0, 0.2), (0, 0, 0.15)),
            render(gym, sim, env, "isaac_stand_side.png", (0.0, -0.9, 0.2), (0, 0, 0.15))]
    report["images"] = imgs
    gym.destroy_sim(sim)


def joint_direction_test(gym, report):
    """Fixed base: +0.3 rad on each joint type, record how the FL / FR feet move."""
    sim = make_sim(gym)
    asset = gym.load_asset(sim, URDF_DIR, "elbowdog_v5.urdf", asset_options(True))
    body_names = gym.get_asset_rigid_body_names(asset)
    dof_names = gym.get_asset_dof_names(asset)
    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 0.6)
    actor = gym.create_actor(env, asset, pose, "elbowdog", 0, 1, 0)
    gym.prepare_sim(sim)
    rb = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
    default = np.array([DEFAULT[n.split("_")[1]] for n in dof_names], dtype=np.float32)

    def feet_at(q):
        s = np.zeros(len(dof_names), dtype=gymapi.DofState.dtype)
        s["pos"] = q
        gym.set_actor_dof_states(env, actor, s, gymapi.STATE_ALL)
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.set_actor_dof_states(env, actor, s, gymapi.STATE_ALL)
        gym.refresh_rigid_body_state_tensor(sim)
        return {n: (rb[i, :3] - rb[0, :3]).numpy().round(4).tolist()
                for i, n in enumerate(body_names) if n in ("FL_foot", "FR_foot")}

    result = {"default": feet_at(default)}
    for jt in ("hip", "thigh", "calf"):
        q = default.copy()
        for i, n in enumerate(dof_names):
            if n.split("_")[1] == jt:
                q[i] += 0.3
        result[f"{jt}+0.3"] = feet_at(q)
    report["joint_direction"] = result
    gym.destroy_sim(sim)


def standing_test_mujoco(report):
    import mujoco
    xml = open(os.path.join(ROOT, "assets", "elbowdog_v5", "mjcf", "elbowdog_v5.xml")).read()
    xml = xml.replace("<worldbody>", '<worldbody>\n    <geom name="floor" type="plane" size="5 5 0.1"/>')
    xml = xml.replace('meshdir="../meshes"', f'meshdir="{os.path.join(ROOT, "assets", "elbowdog_v5", "meshes")}"')
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    names = [m.joint(i).name for i in range(1, m.njnt)]
    default = np.array([DEFAULT[n.split("_")[1]] for n in names])
    d.qpos[7:] = default
    for _ in range(int(3.0 / m.opt.timestep)):
        tau = KP * (default - d.qpos[7:]) - KD * d.qvel[6:]
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
    feet = [m.geom(f"{leg}_foot_collision").id for leg in ("FL", "FR", "RL", "RR")]
    fz = np.zeros(4)
    for i in range(d.ncon):
        c = d.contact[i]
        for k, g in enumerate(feet):
            if g in (c.geom1, c.geom2):
                f6 = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, f6)
                fz[k] += (c.frame[:3] @ np.array([0, 0, 1])) * f6[0]
    report["mujoco_stand"] = {"base_height": float(d.qpos[2]), "foot_contact_fz": fz.tolist(),
                              "sum_foot_fz": float(fz.sum()), "weight": float(sum(m.body_mass) * 9.81),
                              "joint_pos": d.qpos[7:].tolist(), "joint_torque": d.ctrl.tolist()}


def main():
    gym = gymapi.acquire_gym()
    report = {}
    standing_test_isaac(gym, report)
    joint_direction_test(gym, report)
    standing_test_mujoco(report)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in ("inertia_diag_before_after",)}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
