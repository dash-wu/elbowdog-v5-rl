"""Plot training curves from the TensorBoard logs of one or more runs.

  python tools/plot_training.py logs/elbowdog_flat/<run> [more runs ...] --out docs/figures/flat_v1.png
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PANELS = [
    ("Train/mean_reward", "mean episode reward"),
    ("Train/mean_episode_length", "episode length [steps]"),
    ("Episode/rew_tracking_lin_vel", "lin. velocity tracking reward / s"),
    ("Episode/max_command_x", "curriculum forward-speed limit [m/s]"),
    ("Episode/rew_dof_power", "power penalty / s"),
    ("Episode/rew_collision", "collision penalty / s"),
]
LEVELS = ["flat", "slope_up", "slope_down", "rough_slope", "stairs_up", "stairs_down", "discrete", "rough",
          "approach_ramp"]


def load(run):
    files = glob.glob(os.path.join(run, "events.out.tfevents.*"))
    acc = EventAccumulator(run, size_guidance={"scalars": 0})
    acc.Reload()
    return {tag: ([e.step for e in acc.Scalars(tag)], [e.value for e in acc.Scalars(tag)]) for tag in acc.Tags()["scalars"]} if files else {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    data = {os.path.basename(r.rstrip("/")): load(r) for r in a.runs}
    has_levels = any(f"Episode/level_{LEVELS[0]}" in d for d in data.values())
    n = len(PANELS) + (1 if has_levels else 0)
    cols = 2
    rows = (n + 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(12, 3.0 * rows))
    axs = axs.ravel()
    for ax, (tag, label) in zip(axs, PANELS):
        for name, d in data.items():
            if tag in d:
                ax.plot(*d[tag], label=name, lw=1.2)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)
    if has_levels:
        ax = axs[len(PANELS)]
        for name, d in data.items():
            for lv in LEVELS:
                tag = f"Episode/level_{lv}"
                if tag in d:
                    ax.plot(*d[tag], label=f"{lv}" if len(data) == 1 else f"{name}:{lv}", lw=1.2)
        ax.set_title("mean terrain level (0-9) per terrain type", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    for ax in axs[n:]:
        ax.axis("off")
    if len(data) > 1:
        axs[0].legend(fontsize=8)
    for ax in axs[:n]:
        ax.set_xlabel("iteration", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    plt.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
