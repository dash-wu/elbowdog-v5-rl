"""Export a trained actor and everything a deployment needs to reproduce its observations."""
import copy
import json
import os

import torch


def export_policy(runner, env, env_cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    actor = copy.deepcopy(runner.alg.actor_critic.actor).to("cpu").eval()
    torch.jit.script(actor).save(os.path.join(out_dir, "policy.pt"))

    c = env_cfg
    deploy = {
        "dof_names": list(env.dof_names),
        "default_joint_angles": [float(v) for v in env.default_dof_pos[0].cpu()],
        "kp": [float(v) for v in env.p_gains.cpu()],
        "kd": [float(v) for v in env.d_gains.cpu()],
        "torque_limits": [float(v) for v in env.torque_limits.cpu()],
        "velocity_limits": [float(v) for v in env.dof_vel_limits.cpu()],
        "motor_saturation_ratio": c.control.motor_saturation_ratio,
        "action_scale": c.control.action_scale,
        "decimation": c.control.decimation,
        "sim_dt": c.sim.dt,
        "policy_dt": c.sim.dt * c.control.decimation,
        "clip_actions": c.normalization.clip_actions,
        "clip_observations": c.normalization.clip_observations,
        "obs_scales": {k: getattr(c.normalization.obs_scales, k) for k in
                       ("lin_vel", "ang_vel", "dof_pos", "dof_vel", "height_measurements")},
        "actor_lin_vel": c.env.actor_lin_vel,
        "num_observations": c.env.num_observations,
        "measure_heights": c.terrain.measure_heights,
        "measured_points_x": list(c.terrain.measured_points_x),
        "measured_points_y": list(c.terrain.measured_points_y),
        "height_obs_offset": c.terrain.height_obs_offset,
        "base_height_nominal": c.rewards.base_height_target,
        "observation_layout": (["lin_vel(3)"] if c.env.actor_lin_vel else []) +
                              ["ang_vel(3)", "projected_gravity(3)", "command vx vy wz (3)",
                               "dof_pos - default (12)", "dof_vel (12)", "last_action (12)"] +
                              (["height scan, x-major (n_x * n_y)"] if c.terrain.measure_heights else []),
    }
    with open(os.path.join(out_dir, "deploy.json"), "w") as f:
        json.dump(deploy, f, indent=2)
    return out_dir
