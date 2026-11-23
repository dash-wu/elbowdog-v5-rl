"""Train an ElbowDog policy.

  python scripts/train.py --task elbowdog_flat --headless
  python scripts/train.py --task elbowdog_rough --headless --run_name v1
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import isaacgym  # noqa: F401  (must come before torch)
import legged_gym.envs  # noqa: F401  (legged_gym needs envs imported before utils)
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict

import elbowdog_gym  # noqa: F401  (registers the tasks)


def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    _, train_cfg = task_registry.get_cfgs(args.task)
    exp = args.experiment_name or train_cfg.runner.experiment_name
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args,
                                                      log_root=os.path.join(ROOT, "logs", exp))
    os.makedirs(runner.log_dir, exist_ok=True)
    with open(os.path.join(runner.log_dir, "config.json"), "w") as f:
        json.dump({"task": args.task, "env": class_to_dict(env_cfg), "train": class_to_dict(train_cfg)},
                  f, indent=2, default=str)
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    train(get_args())
