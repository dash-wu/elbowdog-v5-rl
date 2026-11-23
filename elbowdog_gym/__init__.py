"""ElbowDog V5 reinforcement-learning tasks for legged_gym (import after isaacgym)."""
from legged_gym.utils.task_registry import task_registry

from .envs.elbowdog import ElbowDog
from .envs.elbowdog_config import (ElbowDogFlatCfg, ElbowDogFlatCfgPPO, ElbowDogRampCfg, ElbowDogRampCfgPPO,
                                   ElbowDogRoughCfg, ElbowDogRoughCfgPPO)

task_registry.register("elbowdog_flat", ElbowDog, ElbowDogFlatCfg(), ElbowDogFlatCfgPPO())
task_registry.register("elbowdog_rough", ElbowDog, ElbowDogRoughCfg(), ElbowDogRoughCfgPPO())
task_registry.register("elbowdog_ramp", ElbowDog, ElbowDogRampCfg(), ElbowDogRampCfgPPO())
