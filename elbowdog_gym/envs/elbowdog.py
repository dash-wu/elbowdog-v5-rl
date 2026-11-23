"""ElbowDog V5 locomotion environment (Isaac Gym, extends legged_gym.LeggedRobot).

Additions over the legged_gym base class:
  * actuator model: PD -> DC-motor torque/speed envelope of GO-M8010-6 / IM6014 -> joint friction
  * domain randomisation: payload, COM offset, PD gains, motor strength, joint friction, control delay
  * terrain-dependent command ranges and a forward-speed curriculum
  * asymmetric actor-critic: the critic additionally sees clean states and the randomised parameters
  * reward terms for foot clearance, foot slip, power, hip posture, stand still
"""
import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate_inverse, torch_rand_float

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import quat_apply_yaw

from elbowdog_gym.utils.terrain import ElbowDogTerrain, TERRAIN_TYPES


class ElbowDog(LeggedRobot):

    # ------------------------------------------------------------------ simulation setup
    def create_sim(self):
        self.up_axis_idx = 2
        if getattr(self.cfg.env, "offscreen_render", False):
            self.graphics_device_id = self.sim_device_id   # camera sensors without a viewer window
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ["heightfield", "trimesh"]:
            self.terrain = ElbowDogTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        self._create_envs()

    def _process_dof_props(self, props, env_id):
        props = super()._process_dof_props(props, env_id)
        # the motor model limits speed through the torque envelope; keep PhysX from clamping velocities
        props["velocity"][:] = self.cfg.control.physx_max_joint_velocity
        return props

    def _process_rigid_body_props(self, props, env_id):
        dr = self.cfg.domain_rand
        if env_id == 0:
            self._payload = np.zeros(self.num_envs, dtype=np.float32)
            self._com_offset = np.zeros((self.num_envs, 3), dtype=np.float32)
            self._base_mass_nominal = props[0].mass
        if dr.randomize_base_mass:
            self._payload[env_id] = np.random.uniform(*dr.added_mass_range)
            props[0].mass = self._base_mass_nominal + self._payload[env_id]
        if dr.randomize_com:
            off = np.random.uniform(-1.0, 1.0, 3) * np.array(dr.com_range)
            self._com_offset[env_id] = off
            props[0].com = gymapi.Vec3(props[0].com.x + off[0], props[0].com.y + off[1], props[0].com.z + off[2])
        return props

    def _get_env_origins(self):
        super()._get_env_origins()
        if self.custom_origins:
            self.max_terrain_level = self.cfg.terrain.num_rows
            col_types = torch.tensor(self.terrain.col_types, device=self.device, dtype=torch.long)
            self.env_terrain_type = col_types[self.terrain_types]
        else:
            self.env_terrain_type = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _init_buffers(self):
        super()._init_buffers()
        n, d, dev = self.num_envs, self.num_dof, self.device
        dr, ctrl = self.cfg.domain_rand, self.cfg.control

        rb = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rb).view(n, self.num_bodies, 13)
        self.foot_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.foot_vel = self.rigid_body_states[:, self.feet_indices, 7:10]

        # actuator model
        self.sat_torque = self.torque_limits * ctrl.motor_saturation_ratio
        self.hip_indices = torch.tensor([i for i, nm in enumerate(self.dof_names) if "hip" in nm], device=dev)

        def rand(rng, shape):
            return torch_rand_float(rng[0], rng[1], shape, device=dev)
        self.kp_factor = rand(dr.kp_range, (n, d)) if dr.randomize_gains else torch.ones(n, d, device=dev)
        self.kd_factor = rand(dr.kd_range, (n, d)) if dr.randomize_gains else torch.ones(n, d, device=dev)
        self.motor_strength = rand(dr.motor_strength_range, (n, d)) if dr.randomize_motor_strength else torch.ones(n, d, device=dev)
        self.joint_friction = rand(dr.joint_friction_range, (n, d)) if dr.randomize_joint_friction else torch.zeros(n, d, device=dev)
        self.joint_damping = rand(dr.joint_damping_range, (n, d)) if dr.randomize_joint_friction else torch.zeros(n, d, device=dev)
        self.action_delay = (torch.randint(dr.delay_substeps[0], dr.delay_substeps[1] + 1, (n,), device=dev)
                             if dr.randomize_delay else torch.zeros(n, dtype=torch.long, device=dev))
        self.payload = torch.tensor(self._payload, device=dev)
        self.com_offset = torch.tensor(self._com_offset, device=dev)
        self.friction_value = (self.friction_coeffs.to(dev).view(n) if dr.randomize_friction
                               else torch.ones(n, device=dev) * self.cfg.terrain.static_friction)
        self.applied_torques = torch.zeros_like(self.torques)

        # commands
        self.cmd_vx_max = torch.full((n,), self.command_ranges["lin_vel_x"][1], device=dev)
        self.type_vx_cap = torch.tensor([self.cfg.commands.type_max_vx.get(t, 10.0) for t in TERRAIN_TYPES], device=dev)
        self.type_vy_cap = torch.tensor([self.cfg.commands.type_max_vy.get(t, 10.0) for t in TERRAIN_TYPES], device=dev)
        self.type_vx_min = torch.tensor([getattr(self.cfg.commands, "type_min_vx", {}).get(t, -10.0) for t in TERRAIN_TYPES],
                                        device=dev)
        self.foot_terrain_h = torch.zeros(n, len(self.feet_indices), device=dev)
        self.contact_filt = torch.zeros(n, len(self.feet_indices), dtype=torch.bool, device=dev)

    # ------------------------------------------------------------------ stepping
    def step(self, actions):
        clip = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip, clip).to(self.device)
        self.render()
        for i in range(self.cfg.control.decimation):
            act = torch.where((i < self.action_delay).unsqueeze(1), self.last_actions, self.actions)
            self.torques = self._compute_torques(act).view(self.torques.shape)
            v = self.dof_vel
            self.applied_torques = self.torques - self.joint_friction * torch.tanh(v / 0.05) - self.joint_damping * v
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.applied_torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _compute_torques(self, actions):
        target = actions * self.cfg.control.action_scale + self.default_dof_pos
        tau = self.kp_factor * self.p_gains * (target - self.dof_pos) - self.kd_factor * self.d_gains * self.dof_vel
        tau = tau * self.motor_strength
        # DC motor envelope: available torque falls linearly with speed in the motoring quadrant
        ratio = self.dof_vel / self.dof_vel_limits
        max_tau = torch.clamp(self.sat_torque * (1.0 - ratio), min=torch.zeros_like(ratio), max=self.torque_limits)
        min_tau = torch.clamp(self.sat_torque * (-1.0 - ratio), min=-self.torque_limits, max=torch.zeros_like(ratio))
        return torch.max(torch.min(tau, max_tau), min_tau)

    def post_physics_step(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.foot_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.foot_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        # PhysX contact reports on meshes flicker, so a foot counts as in contact for one extra step
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.foot_terrain_h = self._terrain_height_at(self.foot_pos[:, :, :2])
        super().post_physics_step()
        self.last_contacts = contact
        self.last_contacts[self.reset_buf.nonzero(as_tuple=False).flatten()] = False

    def check_termination(self):
        super().check_termination()
        tilt = self.projected_gravity[:, 2] > self.cfg.env.termination_gravity_z
        self.reset_buf |= tilt
        if self.custom_origins:
            b = self.cfg.terrain.border_size - 1.0
            x, y = self.root_states[:, 0], self.root_states[:, 1]
            out = (x < -b) | (y < -b) | (x > self.terrain.env_length * self.cfg.terrain.num_rows + b) | \
                  (y > self.terrain.env_width * self.cfg.terrain.num_cols + b)
            self.time_out_buf |= out
            self.reset_buf |= out

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        if self.custom_origins:
            for k, name in enumerate(TERRAIN_TYPES):
                mask = self.env_terrain_type == k
                if mask.any():
                    self.extras["episode"][f"level_{name}"] = torch.mean(self.terrain_levels[mask].float())
        self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.domain_rand.randomize_delay:
            lo, hi = self.cfg.domain_rand.delay_substeps
            self.action_delay[env_ids] = torch.randint(lo, hi + 1, (len(env_ids),), device=self.device)

    # ------------------------------------------------------------------ commands
    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        r = self.command_ranges
        n = len(env_ids)
        t = self.env_terrain_type[env_ids]
        vx_hi = torch.minimum(self.cmd_vx_max[env_ids], self.type_vx_cap[t])
        vx_lo = torch.maximum(torch.full_like(vx_hi, r["lin_vel_x"][0]), self.type_vx_min[t])
        u = torch.rand(n, device=self.device)
        self.commands[env_ids, 0] = vx_lo + (vx_hi - vx_lo) * u
        vy_cap = torch.minimum(torch.full_like(vx_hi, r["lin_vel_y"][1]), self.type_vy_cap[t])
        self.commands[env_ids, 1] = (2 * torch.rand(n, device=self.device) - 1) * vy_cap
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(r["heading"][0], r["heading"][1], (n, 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(r["ang_vel_yaw"][0], r["ang_vel_yaw"][1], (n, 1), device=self.device).squeeze(1)
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        stand = torch.rand(n, device=self.device) < self.cfg.commands.stand_prob
        self.commands[env_ids[stand], :3] = 0.0
        if self.cfg.commands.heading_command:
            # standing robots keep their current heading
            fwd = quat_apply_yaw(self.base_quat[env_ids[stand]], self.forward_vec[env_ids[stand]])
            self.commands[env_ids[stand], 3] = torch.atan2(fwd[:, 1], fwd[:, 0])

    def update_command_curriculum(self, env_ids):
        """Raise the forward-speed limit while the robots on fast terrain track it well."""
        c = self.cfg.commands
        fast = self.type_vx_cap[self.env_terrain_type[env_ids]] >= c.max_curriculum
        ids = env_ids[fast] if fast.any() else env_ids
        tracking = torch.mean(self.episode_sums["tracking_lin_vel"][ids]) / self.max_episode_length
        if tracking > c.curriculum_threshold * self.reward_scales["tracking_lin_vel"]:
            new_max = min(self.command_ranges["lin_vel_x"][1] + c.curriculum_step, c.max_curriculum)
            self.command_ranges["lin_vel_x"][1] = new_max
            self.cmd_vx_max[:] = new_max

    # ------------------------------------------------------------------ observations
    def compute_observations(self):
        cmd = self.commands[:, :3] * self.commands_scale
        dof_pos = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dof_vel = self.dof_vel * self.obs_scales.dof_vel
        parts = [self.base_ang_vel * self.obs_scales.ang_vel, self.projected_gravity, cmd, dof_pos, dof_vel, self.actions]
        if self.cfg.env.actor_lin_vel:
            parts.insert(0, self.base_lin_vel * self.obs_scales.lin_vel)
        heights = None
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - self.cfg.terrain.height_obs_offset
                                 - self.measured_heights, -1, 1.0) * self.obs_scales.height_measurements
            parts.append(heights)
        clean = torch.cat(parts, dim=-1)
        self.obs_buf = clean
        if self.add_noise:
            self.obs_buf = clean + (2 * torch.rand_like(clean) - 1) * self.noise_scale_vec

        if self.num_privileged_obs is not None:
            priv = [self.base_lin_vel * self.obs_scales.lin_vel, self.base_ang_vel * self.obs_scales.ang_vel,
                    self.projected_gravity, cmd, dof_pos, dof_vel, self.actions]
            if heights is not None:
                priv.append(heights)
            priv += [self.contact_filt.float(),
                     ((self.foot_pos[:, :, 2] - self.foot_terrain_h) * 10).clip(-1, 1),
                     (self.friction_value.unsqueeze(1) - 0.85) * 2,
                     self.payload.unsqueeze(1) * 0.4,
                     self.com_offset * 20,
                     (self.motor_strength - 1) * 5,
                     (self.kp_factor - 1) * 5,
                     self.joint_friction * 3,
                     self.action_delay.float().unsqueeze(1) * 0.5]
            self.privileged_obs_buf = torch.cat(priv, dim=-1)

    def _get_noise_scale_vec(self, cfg):
        ns, lvl, sc = self.cfg.noise.noise_scales, self.cfg.noise.noise_level, self.obs_scales
        vec = []
        if self.cfg.env.actor_lin_vel:
            vec += [ns.lin_vel * lvl * sc.lin_vel] * 3
        vec += [ns.ang_vel * lvl * sc.ang_vel] * 3 + [ns.gravity * lvl] * 3 + [0.0] * 3
        vec += [ns.dof_pos * lvl * sc.dof_pos] * 12 + [ns.dof_vel * lvl * sc.dof_vel] * 12 + [0.0] * 12
        if self.cfg.terrain.measure_heights:
            n_h = len(self.cfg.terrain.measured_points_x) * len(self.cfg.terrain.measured_points_y)
            vec += [ns.height_measurements * lvl * sc.height_measurements] * n_h
        self.add_noise = self.cfg.noise.add_noise
        return torch.tensor(vec, dtype=torch.float, device=self.device)

    def _terrain_height_at(self, xy):
        """Terrain height under arbitrary world xy points, shape (n, k, 2) -> (n, k)."""
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            return torch.zeros(xy.shape[:2], device=self.device)
        pts = torch.round((xy + self.terrain.cfg.border_size) / self.terrain.cfg.horizontal_scale).long()
        px = torch.clip(pts[..., 0].reshape(-1), 0, self.height_samples.shape[0] - 1)
        py = torch.clip(pts[..., 1].reshape(-1), 0, self.height_samples.shape[1] - 1)
        return self.height_samples[px, py].view(xy.shape[:2]).float() * self.terrain.cfg.vertical_scale

    # ------------------------------------------------------------------ rewards
    def _reward_base_height(self):
        under = self._terrain_height_at(self.root_states[:, :2].unsqueeze(1)).squeeze(1)
        return torch.square(self.root_states[:, 2] - under - self.cfg.rewards.base_height_target)

    def _reward_forward_progress(self):
        # fraction of the commanded forward speed actually achieved; 0 when stuck, unlike the exp tracking kernel
        cx = self.commands[:, 0]
        active = torch.abs(cx) > 0.1
        ratio = self.base_lin_vel[:, 0] / torch.where(active, cx, torch.ones_like(cx))
        return torch.clip(ratio, 0.0, 1.0) * active

    def _reward_feet_air_time(self):
        first_contact = (self.feet_air_time > 0.0) * self.contact_filt
        self.feet_air_time += self.dt
        air = torch.clip(self.feet_air_time, max=self.cfg.rewards.feet_air_time_max)
        rew = torch.sum((air - self.cfg.rewards.feet_air_time_target) * first_contact, dim=1)
        rew *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~self.contact_filt
        return rew

    def _reward_feet_clearance(self):
        height = self.foot_pos[:, :, 2] - self.foot_terrain_h - self.cfg.asset.foot_radius
        deficit = torch.clip(self.cfg.rewards.foot_clearance_target - height, min=0.0)
        speed = torch.norm(self.foot_vel[:, :, :2], dim=2)
        return torch.sum(torch.square(deficit / self.cfg.rewards.foot_clearance_target) * torch.tanh(2 * speed) * ~self.contact_filt, dim=1)

    def _reward_feet_slip(self):
        speed = torch.norm(self.foot_vel[:, :, :2], dim=2)
        return torch.sum(speed * self.contact_filt, dim=1)

    def _slope_grade(self):
        """Rise/run of the ground along the robot's heading. Zero on flat ground."""
        fwd = quat_apply_yaw(self.base_quat, self.forward_vec)
        xy = self.root_states[:, :2]
        step = 0.25
        ahead = self._terrain_height_at((xy + fwd[:, :2] * step).unsqueeze(1)).squeeze(1)
        behind = self._terrain_height_at((xy - fwd[:, :2] * step).unsqueeze(1)).squeeze(1)
        return torch.abs(ahead - behind) / (2 * step)

    def _reward_slope_slip(self):
        # Contact-foot speed times the local slope, so flat walking is left alone.
        return self._reward_feet_slip() * self._slope_grade()

    def _process_rigid_shape_props(self, props, env_id):
        props = super()._process_rigid_shape_props(props, env_id)
        dr = self.cfg.domain_rand
        if getattr(dr, "randomize_compliance", False):
            if env_id == 0:
                lo, hi = dr.compliance_range
                self.compliance = torch_rand_float(lo, hi, (self.num_envs, 1), device="cpu")
            c = float(self.compliance[env_id])
            for s in range(len(props)):
                props[s].compliance = c
        return props

    def _reward_dof_power(self):
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)

    def _reward_stand_still(self):
        still = torch.norm(self.commands[:, :3], dim=1) < 0.1
        return (torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) + 0.1 * torch.sum(torch.abs(self.dof_vel), dim=1)) * still

    def _reward_feet_stumble(self):
        horizontal = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
        return torch.any(horizontal > 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1).float()

    def _reward_stand_feet_contact(self):
        still = torch.norm(self.commands[:, :3], dim=1) < 0.1
        return torch.sum(~self.contact_filt, dim=1).float() * still
