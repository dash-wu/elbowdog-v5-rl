"""Training configurations for ElbowDog V5.

Stage 1  elbowdog_flat  : flat ground, proprioception only, forward-speed curriculum up to 3 m/s
Stage 2  elbowdog_rough : terrain curriculum (slopes, stairs, obstacles, rough ground) + height scan
"""
import os

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEGS = ["FL", "FR", "RL", "RR"]
STAND_POSE = {"hip": 0.0, "thigh": 0.65, "calf": -1.3}   # V5 design stance, feet below the hips
HEIGHT_X = [round(-0.6 + 0.1 * i, 2) for i in range(15)]  # -0.6 ... 0.8 m
HEIGHT_Y = [round(-0.4 + 0.1 * i, 2) for i in range(9)]   # -0.4 ... 0.4 m
N_PROPRIO = 48
N_PRIV_EXTRA = 50


class ElbowDogFlatCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        actor_lin_vel = True
        num_observations = N_PROPRIO
        num_privileged_obs = N_PROPRIO + N_PRIV_EXTRA
        num_actions = 12
        episode_length_s = 20
        termination_gravity_z = -0.3   # reset when the body tilts more than ~72 deg

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        height_obs_offset = 0.25        # nominal base height, centres the height scan around 0
        measured_points_x = HEIGHT_X
        measured_points_y = HEIGHT_Y

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 3.0            # forward speed limit reached by the curriculum [m/s]
        curriculum_step = 0.25
        curriculum_threshold = 0.75     # fraction of the max tracking reward needed to raise the limit
        num_commands = 4
        resampling_time = 8.0
        heading_command = True
        stand_prob = 0.1                # share of zero commands, teaches standing still
        type_max_vx = {"flat": 3.0}
        type_max_vy = {"flat": 0.6}

        class ranges:
            lin_vel_x = [-1.0, 1.0]     # upper bound grows with the curriculum
            lin_vel_y = [-0.6, 0.6]
            ang_vel_yaw = [-1.5, 1.5]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.30]
        default_joint_angles = {f"{leg}_{j}_joint": STAND_POSE[j] for leg in LEGS for j in STAND_POSE}

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"hip_joint": 25.0, "thigh_joint": 25.0, "calf_joint": 25.0}   # [N*m/rad]
        damping = {"hip_joint": 0.6, "thigh_joint": 0.6, "calf_joint": 0.6}        # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4                   # 200 Hz physics, 50 Hz policy
        motor_saturation_ratio = 1.0     # torque envelope: peak * (1 - w / w_max), conservative
        physx_max_joint_velocity = 100.0

    class asset(LeggedRobotCfg.asset):
        file = os.path.join(ROOT, "assets", "elbowdog_v5", "urdf", "elbowdog_v5.urdf")
        name = "elbowdog"
        foot_name = "foot"
        foot_radius = 0.016
        penalize_contacts_on = ["thigh", "calf", "hip"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0
        collapse_fixed_joints = True     # foot joints carry dont_collapse="true"
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False
        armature = 0.01

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.3, 1.25]
        randomize_base_mass = True
        added_mass_range = [-0.5, 3.0]
        randomize_com = True
        com_range = [0.03, 0.02, 0.02]
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.8
        randomize_gains = True
        kp_range = [0.85, 1.15]
        kd_range = [0.85, 1.15]
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        randomize_joint_friction = True
        joint_friction_range = [0.0, 0.3]     # Coulomb [N*m]
        joint_damping_range = [0.0, 0.03]     # viscous [N*m*s/rad]
        randomize_delay = True
        delay_substeps = [0, 2]               # 0-10 ms actuation delay

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.95
        base_height_target = 0.25
        max_contact_force = 160.0
        feet_air_time_target = 0.25
        feet_air_time_max = 0.5
        foot_clearance_target = 0.06

        class scales:
            termination = -0.0
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.75
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_power = -2e-4
            dof_acc = -2.5e-7
            action_rate = -0.01
            collision = -1.0
            feet_air_time = 1.0
            feet_clearance = -0.3
            feet_slip = -0.1
            hip_pos = -0.5
            dof_pos_limits = -10.0
            torque_limits = -0.01
            stand_still = -0.5
            stand_feet_contact = -0.3

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.03

    class viewer(LeggedRobotCfg.viewer):
        pos = [2.0, -2.0, 1.2]
        lookat = [0.0, 0.0, 0.2]


class ElbowDogFlatCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "elbowdog_flat"
        max_iterations = 1500
        save_interval = 100


N_HEIGHT = len(HEIGHT_X) * len(HEIGHT_Y)


class ElbowDogRoughCfg(ElbowDogFlatCfg):
    class env(ElbowDogFlatCfg.env):
        num_observations = N_PROPRIO + N_HEIGHT
        num_privileged_obs = N_PROPRIO + N_HEIGHT + N_PRIV_EXTRA

    class terrain(ElbowDogFlatCfg.terrain):
        mesh_type = "trimesh"
        measure_heights = True
        curriculum = True
        horizontal_scale = 0.05
        vertical_scale = 0.005
        border_size = 8.0
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10                    # difficulty levels 0.0 ... 0.9
        num_cols = 20
        max_init_terrain_level = 2
        terrain_proportions = {"flat": 0.15, "slope_up": 0.075, "slope_down": 0.075, "rough_slope": 0.15,
                               "stairs_up": 0.2, "stairs_down": 0.15, "discrete": 0.1, "rough": 0.1}
        slope_treshold = 0.75            # steeper cells become vertical walls (stair risers)
        platform_size = 2.0
        max_slope_deg = 38.0             # hardest row: 0.9 * 38 = 34 deg
        max_rough_slope_deg = 25.0
        rough_slope_noise = [0.01, 0.03]
        step_height = [0.03, 0.14]       # hardest row: 0.156 m
        step_width = [0.26, 0.34]
        discrete_height = [0.02, 0.10]
        discrete_size = [0.3, 1.0]
        discrete_count = 60
        rough_noise = [0.01, 0.05]
        rough_downsample = 0.1

    class commands(ElbowDogFlatCfg.commands):
        type_max_vx = {"flat": 3.0, "slope_up": 1.5, "slope_down": 1.5, "rough_slope": 1.5,
                       "stairs_up": 1.0, "stairs_down": 1.0, "discrete": 1.2, "rough": 2.0}
        type_max_vy = {"flat": 0.6, "slope_up": 0.5, "slope_down": 0.5, "rough_slope": 0.5,
                       "stairs_up": 0.4, "stairs_down": 0.4, "discrete": 0.4, "rough": 0.5}
        # v2: obstacles only get forward commands, so slow climbing is trained as well (v1 only climbed when fast)
        type_min_vx = {"slope_up": 0.3, "slope_down": 0.3, "rough_slope": 0.3,
                       "stairs_up": 0.3, "stairs_down": 0.3, "discrete": 0.3}

    class domain_rand(ElbowDogFlatCfg.domain_rand):
        friction_range = [0.4, 1.25]
        added_mass_range = [-0.5, 5.0]   # course requirement: >= 5 kg payload

    class rewards(ElbowDogFlatCfg.rewards):
        foot_clearance_target = 0.09

        class scales(ElbowDogFlatCfg.rewards.scales):
            orientation = -0.2
            base_height = -3.0
            feet_stumble = -0.5
            forward_progress = 0.5


class ElbowDogRoughCfgPPO(ElbowDogFlatCfgPPO):
    class runner(ElbowDogFlatCfgPPO.runner):
        experiment_name = "elbowdog_rough"
        max_iterations = 4000


class ElbowDogRampCfg(ElbowDogRoughCfg):
    """Approach-ramp terrain. v5 also penalises foot slip in proportion to the slope and pays more for actually matching the commanded speed."""

    class terrain(ElbowDogRoughCfg.terrain):
        terrain_proportions = {"flat": 0.10, "slope_up": 0.05, "slope_down": 0.05, "rough_slope": 0.10,
                               "approach_ramp": 0.20, "stairs_up": 0.20, "stairs_down": 0.15,
                               "discrete": 0.10, "rough": 0.05}

    class commands(ElbowDogRoughCfg.commands):
        type_max_vx = dict(ElbowDogRoughCfg.commands.type_max_vx, approach_ramp=1.2)
        type_max_vy = dict(ElbowDogRoughCfg.commands.type_max_vy, approach_ramp=0.4)
        type_min_vx = dict(ElbowDogRoughCfg.commands.type_min_vx, approach_ramp=0.3)

    class domain_rand(ElbowDogRoughCfg.domain_rand):
        # PhysX compliance is m/N. 1e-4 m/N sinks about 1 cm under a 100 N foot load,
        # covering the gap between a rigid Isaac contact and MuJoCo's softer default.
        randomize_compliance = True
        compliance_range = [0.0, 1.0e-4]

    class rewards(ElbowDogRoughCfg.rewards):
        class scales(ElbowDogRoughCfg.rewards.scales):
            # v6 raised this to -4 and squared the slip; 25 deg climbing got much slower and
            # the 12 cm stair started failing. Keep the v5 weight.
            slope_slip = -1.0
            forward_progress = 1.0


class ElbowDogRampCfgPPO(ElbowDogRoughCfgPPO):
    class runner(ElbowDogRoughCfgPPO.runner):
        experiment_name = "elbowdog_ramp"
        max_iterations = 1500
