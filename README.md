# ElbowDog V5 强化学习运动控制

写报告和复现请先看 [`给工友.md`](给工友.md)。最优策略是 v12，文件在 [`best/v12/`](best/v12/)。环境清单和安装步骤在 [`environment/安装说明.md`](environment/安装说明.md)。

ElbowDog V5（13.2 kg，全肘式三自由度腿，8×GO-M8010-6 + 4×IM6014）的强化学习行走、爬坡、上下楼梯和不规则地形训练，
Isaac Gym 训练，MuJoCo sim2sim 验证。设计资料来源：`~/桌面/ElbowDog_V5_结构优化与三维仿真/v5`（只读，未修改）。

## 目录

| 路径 | 内容 |
|---|---|
| `tools/build_rl_model.py` | 由设计包 URDF 生成训练用 URDF 和 MuJoCo MJCF（两者参数同源） |
| `tools/check_model_isaac.py` | 模型检查：质量、惯量、关节方向、站立受力，Isaac 与 MuJoCo 对照 |
| `assets/elbowdog_v5/` | 生成的 URDF / MJCF / 网格 / `model_info.json` |
| `elbowdog_gym/envs/elbowdog.py` | 训练环境（继承 legged_gym `LeggedRobot`）：电机模型、域随机化、指令、观测、奖励 |
| `elbowdog_gym/envs/elbowdog_config.py` | 两个阶段的全部参数 |
| `elbowdog_gym/utils/terrain.py` | 地形课程：平地、坡道、带噪坡道、上/下台阶、离散障碍、碎石 |
| `scripts/train.py` / `scripts/play.py` | 训练 / 评估（演示视频、速度扫描、地形扫描、导出策略） |
| `sim2sim/deploy_mujoco.py`、`sim2sim/scenes.py` | MuJoCo 部署与测试场景（30° 坡、12 cm 台阶、碎石、障碍块、冲刺道） |
| `docs/训练迭代记录.md` | 每一轮的改动、理由、结果与反思 |
| `third_party/` | legged_gym、rsl_rl v1.0.2（开源库，未修改） |
| `logs/` | 训练日志、检查点、评估结果、导出的策略 |

## 环境

```bash
source ~/elbowdog_rl/env.sh      # conda 环境 elbowdog（由 go1_rl 克隆，Isaac Gym Preview 4，torch 2.0.1+cu118）
```

## 常用命令

```bash
# 0. 重新生成模型（改了设计包或电机参数后）
python tools/build_rl_model.py && python tools/check_model_isaac.py

# 1. 训练（--headless 不开窗口；TensorBoard: tensorboard --logdir logs）
python scripts/train.py --task elbowdog_flat  --headless --run_name v1
python scripts/train.py --task elbowdog_rough --headless --run_name v1

# 2. Isaac Gym 评估（结果在 logs/<实验>/<run>/eval/，策略导出到 logs/<实验>/<run>/exported/）
python scripts/play.py --task elbowdog_flat  --load_run <run> --mode demo     # 演示视频 + 跟踪曲线 + 导出
python scripts/play.py --task elbowdog_flat  --load_run <run> --mode speed    # 0.5~3 m/s 速度扫描
python scripts/play.py --task elbowdog_rough --load_run <run> --mode terrain  # 各地形各难度通过率

# 3. MuJoCo sim2sim
cd sim2sim
python deploy_mujoco.py --policy ../logs/elbowdog_rough/<run>/exported --scene flat   --mode speed
python deploy_mujoco.py --policy ../logs/elbowdog_rough/<run>/exported --scene course --mode course --video
python deploy_mujoco.py --policy ../logs/elbowdog_rough/<run>/exported --scene course --mode viewer
#   交互窗口按键：↑/↓ 前后速度，←/→ 转向，PageUp/PageDown 横移，Home 停止
```

## 关键建模假设

- 电机按数据手册：GO-M8010-6 峰值 23.7 N·m、30 rad/s（24 V）；IM6014 峰值 31.7 N·m、空载 54.2 rad/s@60 V，
  按 24 V 母线折算为 21.7 rad/s。转矩随转速线性下降（直流电机包络），比峰值恒定的假设保守。
- 训练时随机化：负载 −0.5~+5 kg、质心 ±3 cm、PD 增益 ±15%、电机强度 ±10%、关节摩擦 0~0.3 N·m、控制延迟 0~10 ms、
  摩擦系数、外推。
- 几何限位沿用 V5 干涉检查结果：髋 ±0.35，大腿 0.1~1.4，小腿 −2.8~−0.35 rad。
- 结构分析里的 12 N·m / 8 rad/s 是姿态演示的人为限幅，不用于训练；训练得到的关节力矩和足端力可回灌结构校核。
