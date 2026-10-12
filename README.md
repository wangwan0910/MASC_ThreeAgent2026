# MASC_ThreeAgent2026：三级供应链多智能体强化学习

一个串行三级供应链（工厂 → 仓库 → 零售商）的库存管理问题：每一级是一个独立的智能体，用 Ray RLlib
多智能体 PPO（每个智能体一个独立策略，不共享参数）训练。包括两类需求环境（随机需求 `sto/`、批量需求
`bat/`）上的单环境训练、多种子训练与制图，以及环境之间的连续学习（continual learning）。

---

## 目录

1. [目录结构](#1-目录结构)
2. [供应链环境](#2-供应链环境)
3. [多智能体封装（marl_env.py）](#3-多智能体封装marl_envpy)
4. [安装](#4-安装)
5. [单环境训练（main_ppo.py）](#5-单环境训练main_ppopy)
6. [多种子训练与制图](#6-多种子训练与制图)
7. [在 Iridis 集群上运行](#7-在-iridis-集群上运行)
8. [连续学习](#8-连续学习)
9. [输出文件](#9-输出文件)
10. [制图规范](#10-制图规范)
11. [设计决策与实验依据](#11-设计决策与实验依据)
12. [扩展：增加一个智能体](#12-扩展增加一个智能体)
13. [常见问题](#13-常见问题)

---

## 1. 目录结构

```
MASC_Code/
├── requirements.txt            依赖（Python 3.10，见第 4 节）
├── sto/                        随机需求环境：需求 ~ N(2, σ²)，σ = 0 / 0.1 / 1
│   ├── AssembySto.py           单规划者环境（Stocha_P0 / Stocha_P01 / Stocha_P1）
│   ├── demand.py               需求生成（每天独立抽样）
│   ├── supply_agents/          智能体定义：base_agent.py（模板）、factory.py、warehouse.py、retailer.py
│   ├── marl_env.py             多智能体封装 + PPO 配置（build_ppo_config）
│   ├── main_ppo.py             单环境多智能体 PPO 训练（训练监控、制图、CSV）
│   ├── main_rppo.py            同上，策略带 LSTM（Recurrent PPO）
│   ├── plot_tb.py              从 TensorBoard 日志画训练曲线（单种子 / 多种子均值±std）
│   ├── job_sto_ppo.sh          SLURM 作业脚本（数组作业，一个任务一个环境）
│   ├── job_sto_rppo.sh
│   └── continual_learning/     sto 环境之间的连续学习（continual_sto.py、plot_continual.py、作业脚本）
├── bat/                        批量需求环境：一个需求值重复 batch_size 天，batch = 10 / 7 / 3
│   └── （与 sto/ 结构相同：AssembyBat.py、main_ppo.py、job_bat_ppo.sh、continual_learning/ ...）
└── batsto/                     sto 与 bat 环境之间的连续学习（唯一同时使用两个目录的地方）
    ├── marl_env.py             从 ../sto 和 ../bat 分别导入环境
    ├── continual_batsto.py
    ├── plot_continual.py
    └── job_batsto_*.sh
```

`sto/` 与 `bat/` 是两份独立、可单独运行的代码（各自的 `supply_agents/`、`demand.py`、`marl_env.py`），
两者只在客户需求上不同。修改共享逻辑（例如 `marl_env.py`）时，三份 `marl_env.py` 要同步修改。

### 环境编号

| 环境 id | 类 | 简称 | 需求 |
|---|---|---|---|
| `StoSupplyEnv-v1` | `AssembySto:Stocha_P0` | Sto0 | 恒为 2（σ = 0） |
| `StoSupplyEnv-v2` | `AssembySto:Stocha_P01` | Sto01 | N(2, 0.1²)，取整、截断到 ≥ 0 |
| `StoSupplyEnv-v3` | `AssembySto:Stocha_P1` | Sto1 | N(2, 1²)，取整、截断到 ≥ 0 |
| `BatSupplyEnv-v1` | `AssembyBat:Batch_L10` | Bat10 | N(2, 0.1²) 抽一个值，重复 10 天 |
| `BatSupplyEnv-v2` | `AssembyBat:Batch_L7` | Bat7 | 同上，重复 7 天 |
| `BatSupplyEnv-v3` | `AssembyBat:Batch_L3` | Bat3 | 同上，重复 3 天 |

简称用于图名、图例和日志目录（`marl_env.ENV_SHORT_NAMES`）。

---

## 2. 供应链环境

### 2.1 结构与时间

- 串行三级：外部供应商（无限供货）→ **工厂** → **仓库** → **零售商** → 客户。
- 一个 episode 为 T = 100 天。每天：各级收货（上游只能发出自己的库存，缺的部分记为上游的缺货，**不补发**）
  → 零售商满足客户需求 → 库存截断到 [0, capacity − 1]（capacity = 30）。
- **事件驱动决策**：只有当零售商库存 ≤ 再订货点且没有在途订单时（补货日），三个智能体才同时下单；
  其余日子自动推进。一次 RLlib step = 一个补货日的决策，所以 γ 按决策折扣，而不是按天。

### 2.2 智能体参数（`supply_agents/`）

| 智能体 | 持有成本（/件/补货日） | 加工时间 | 对下游的服务时间 | 动作 |
|---|---|---|---|---|
| 工厂 factory | 1000 | 1 | 1（固定） | 订货量 0–19 |
| 仓库 warehouse | 5 | 3 | 4（固定） | 订货量 0–19 |
| 零售商 retailer | 1000 | 0 | 0 | 订货量 0–19，再订货点 0–9 |

- 缺货成本：每件 10000（三级相同），零售商的缺货指客户需求未满足，上游的缺货指下游订单未能发出。
- 交货期 = 上游的服务时间 + 本级加工时间（工厂 1 天，仓库 4 天，零售商 4 天）。
- 动作空间形式上是 `MultiDiscrete(服务时间, 订货量, 再订货点)`，但服务时间被固定值覆盖，
  再订货点只有零售商可选，所以实际决策是三级的订货量和零售商的再订货点。
- 零售商初始库存 10、初始再订货点 5；其余初始为 0。

### 2.3 成本（奖励）

成本只在补货日结算：每个智能体的成本 = 当前库存 × 持有成本 + 自上次下单以来的缺货数 × 10000。

### 2.4 缺货终止

原始环境在零售商累计缺货 12 天后终止 episode（`max_stockout_days = 12`）。
**训练中默认关闭**（作业脚本传 `--no-stockout-termination`，episode 总是跑满 100 天），原因见第 11 节。

---

## 3. 多智能体封装（marl_env.py）

`SupplyChainMARLEnv` 把单规划者环境（状态矩阵 9 × 3、动作矩阵 3 × 3）按列拆成 3 个 RLlib 智能体：

- **观测**：本级那一列状态（9 个字段：库存、再订货点、在途数量、到货天数、累计缺货、加工时间、当日缺货、
  容量、episode 累计缺货），按取值个数归一化到 [0, 1]。每个智能体**只看自己的状态**。
- **动作**：本级那一列动作。
- **策略**：每个智能体一个 RLModule（MLP 64×64，tanh，策略与价值网络不共享层）。

### env_config 选项

| 键 | 默认 | 含义 |
|---|---|---|
| `env_class` | — | 环境 id 或 `"模块:类"` |
| `reward_scale` | `1e-4` | 奖励缩放（成本为 1e4–1e6 量级，不缩放时价值网络学不动） |
| `reward_mode` | `"team"` | `team`：每个智能体得到 −团队总成本；`individual`：−自身成本；`mixed`：二者加权 |
| `team_weight` | 0.5 | `mixed` 模式中团队成本的权重 |
| `final_costs` | `True` | episode 结束时结算未结的成本（上次补货日之后的缺货、最后一天的持有成本） |
| `termination_penalty` | 0 | 缺货终止时每个智能体的额外成本 |
| `max_stockout_days` | `None` | `None`：沿用环境的 12；`0`：不因缺货终止 |
| `obs_rows` | 状态行数 | 用 0 行补齐观测，使不同环境的观测空间一致（连续学习用） |

### 团队回报 `team_return` 的统一定义

训练指标 `env_runners/team_return`、`evaluate()`、轨迹 CSV 的 `Cost` 列使用**同一个**定义：

> episode 总成本 = 每个补货日的成本（含第一个补货日，即由初始状态造成的成本）+ episode 结束时结算的成本

`team_return` 以原始成本单位记录，与 `reward_mode` 和 `reward_scale` 无关，不同奖励设置的运行可以直接比较。
`episode_return_mean` 则是三个智能体缩放后奖励之和，只用于训练监控。另记录 `initial_cost`（第一个补货日的成本）。

### PPO 超参数（`build_ppo_config`）

| 参数 | 值 | 说明 |
|---|---|---|
| lr | 命令行给出（推荐 1e-4） | |
| γ / λ | 0.99 / 0.95 | |
| clip_param | 0.2 | `clip_params={"factory": 0.1}` 可单独设置某个智能体 |
| entropy_coeff | 0.01 | 防止策略过早变成确定性 |
| vf_loss_coeff / vf_clip_param | 0.5 / 10 | 价值误差平方截断在 10 |
| grad_clip | 0.5 | |
| train_batch_size / minibatch / epochs | 2048 / 256 / 5 | 每次迭代 40 步 SGD |
| seed | `None` | 网络与需求的随机种子 |

---

## 4. 安装

Python 3.10，依赖固定在 `requirements.txt`（Ray 2.49.2、PyTorch 2.2.2 CPU 版、Gymnasium 1.1.1、
NumPy 1.26.4、TensorBoard 等）。

在 Iridis 6 上（计算节点没有外网，**在登录节点创建环境**）：

```bash
module load conda && eval "$(conda shell.bash hook)"
conda create -n MASC python=3.10 -y && conda activate MASC
pip install -r requirements.txt
python -c "import ray, torch, gymnasium, tensorboard; print('ray', ray.__version__, 'torch', torch.__version__)"
```

作业脚本默认激活名为 `MASC` 的 conda 环境（其他名字：`CONDA_ENV=xxx sbatch job_xxx.sh`）。
GPU 训练（`--num-gpus 1`）需要安装 CUDA 版 PyTorch，见 `requirements.txt` 开头的说明。

快速检查环境能否运行（随机动作跑完 episode）：

```bash
cd sto && python marl_env.py
```

---

## 5. 单环境训练（main_ppo.py）

在 `sto/` 或 `bat/` 目录下运行：

```bash
# 推荐设置：团队奖励、不因缺货终止、lr 1e-4、5 个种子、固定 2M 步
python main_ppo.py --env StoSupplyEnv-v3 --reward-mode team --no-stockout-termination \
    --lr 0.0001 --seeds 0 1 2 3 4 --no-early-stop --total-timesteps 2000000 --num-env-runners 2

# 单种子、自动判断收敛后停止
python main_ppo.py --env BatSupplyEnv-v1 --no-stockout-termination --lr 0.0001

# 学习率网格搜索（每个 lr 一个 Tune trial，并行运行）
python main_ppo.py --env BatSupplyEnv-v3 --no-stockout-termination --lr 0.0003 0.0001

tensorboard --logdir tu_logs      # 查看训练过程
```

`main_rppo.py` 参数相同，策略带 LSTM（图名前缀 `RPPO_`）。

### 主要参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--env` | sto: `StoSupplyEnv-v3`，bat: `BatSupplyEnv-v1` | 环境 id |
| `--lr` | 3e-4 | 一个或多个学习率（多个 → 网格搜索） |
| `--seeds` | 不固定 | 一个或多个种子（多个 → 每个种子一个 trial，图为均值 ± std） |
| `--reward-mode` | team | team / individual / mixed |
| `--reward-scale` | 1e-4 | |
| `--no-stockout-termination` | 关 | episode 总是跑满 100 天（作业脚本默认打开） |
| `--max-stockout-days N` | 环境值 12 | |
| `--no-final-costs` | — | 恢复旧行为：episode 结束时不结算 |
| `--termination-penalty` | 0 | |
| `--vf-clip` | 10 | 价值误差平方的截断值（`inf` 为不截断） |
| `--factory-clip` | 0.2 | 工厂策略单独的 PPO clip 参数 |
| `--total-timesteps` | 9e6 | 每个 trial 的最大步数（决策数） |
| `--no-early-stop` | 关 | 跑满 `--total-timesteps`，只记录监控结论 |
| `--time-budget-hours` | 25 | 所有 trial 的时间上限（低于 SLURM 时限） |
| `--num-env-runners` | 1 | 每个 trial 的采样进程数（每个 trial 占 1 + n 个 CPU） |
| `--dir` / `--name` | `tu` / 算法名 | 日志目录 `<dir>_logs/<环境简称>/<时间戳>/<name>/` |
| `--tag` | 空 | 图名和 CSV 名的后缀，变体与测试不会互相覆盖 |
| `--checkpoint-freq` | 50 | 每 n 次迭代保存一次 checkpoint |

### 训练监控（TrainingMonitor）

每次迭代检查 `episode_return_mean`（按 `--conv-window` 次迭代的窗口平均）：

| 结论 | 条件 | 动作 |
|---|---|---|
| converged | 相邻两个窗口变化 < `--conv-tol`（或在噪声范围内）且无显著上升趋势，持续 `--conv-patience` 次迭代，接近最好值，且比开始提高 ≥ `--min-gain` | 停止，画图 |
| diverged | loss 或回报非有限值 | 立即停止 |
| collapsed | 最后一个窗口比最好窗口差 `--collapse-tol` 以上，持续 `--collapse-patience` 次迭代 | 停止 |
| stalled / no_learning | `--stall-patience` 次迭代没有 `--conv-tol` 的提高 | 停止 |
| max_timesteps / time_budget | 达到步数或时间上限仍未收敛 | 停止 |
| completed | `--no-early-stop` 时跑满步数 | 画图；各结论第一次出现的步数记在 `would_stop` |

默认 `--conv-tol 0.01 --conv-window 50 --conv-patience 100 --min-timesteps 500000`。
在 24 条 3M 步完整曲线上回放校准：这组默认值平均在 1.14M 步停止，停止时的回报平均只比 3M 步的水平差 0.6%；
更严格的设置多花 60–80% 的步数，只改善约 0.4 个百分点，所以保留默认值。

只有 converged / completed 的 trial 会画正式图和写 CSV；否则只在日志目录写 `<名称>_NOT_CONVERGED.png`，
脚本退出码为 1（SLURM 显示 FAILED，这是预期行为，原因见 `training_status.json`）。

---

## 6. 多种子训练与制图

`--seeds 0 1 2 3 4` 在一个作业里并行训练 5 个 trial，每个种子同时固定网络初始化和需求序列
（RLlib `config.seed`）。5 个 trial × (1 + 2) 个 CPU = 15，所以用 `--num-env-runners 2` 正好放进 16 核的作业。

训练结束后，`main_ppo.py` 自动画出 `PPO_<环境>.png`：**5 个种子的均值为实线，均值 ± 标准差为浅色阴影**
（风格同 `PPO_demo.png`）。多个学习率时，选种子平均最终回报最好的那个学习率。曲线 CSV 保存
`team_return_mean`、`team_return_std`、`n_seeds`；轨迹 CSV 取最好的那个种子。

手动重画：

```bash
python plot_tb.py tu_seeds_logs/Bat10/<时间戳> --seeds --label "PPO in Bat10" --out PPO_Bat10.png
python plot_tb.py tu_seeds_logs/Bat10/<时间戳> --seeds --smoothing 0.6 ...   # 每个种子先平滑再平均
python plot_tb.py tu_team_logs/Bat10/<时间戳> --lr 0.0003 ...                # 单 trial：原始曲线 + 平滑曲线
```

---

## 7. 在 Iridis 集群上运行

每个目录的作业脚本是 SLURM 数组作业，一个任务一个环境（每个任务 16 CPU、20 GB、最长 26 小时）：

| 脚本 | 任务 0 | 任务 1 | 任务 2 |
|---|---|---|---|
| `bat/job_bat_ppo.sh` | Bat10 | Bat7 | Bat3 |
| `sto/job_sto_ppo.sh` | Sto1 | Sto01 | Sto0 |

脚本默认传入 `--reward-mode $REWARD_MODE --no-stockout-termination --dir tu_$REWARD_MODE`
（`REWARD_MODE` 默认 `team`）；`sbatch` 后面的参数原样传给 `main_ppo.py`，并且排在默认参数之后，可以覆盖它们。

```bash
cd bat
sbatch job_bat_ppo.sh                                    # 三个环境，团队奖励
sbatch --array=0 job_bat_ppo.sh                          # 只跑 Bat10
REWARD_MODE=individual sbatch job_bat_ppo.sh             # 个体奖励，日志在 tu_individual_logs/
sbatch job_bat_ppo.sh --seeds 0 1 2 3 4 --lr 0.0001 --no-early-stop \
    --total-timesteps 2000000 --num-env-runners 2 --dir tu_seeds   # 5 种子
squeue -u $USER                                          # 查看作业
```

- 输出：`<作业名>_<数组作业号>_<任务号>.out/.err`。作业结束会发邮件（脚本中的 `--mail-user`）。
- 多个作业可能被分到同一节点：`init_ray` 为每个作业单独启动 Ray（临时目录 `/tmp/ray_<作业号>`，对象存储上限 2 GB），
  脚本启动前随机等待 0–60 秒，Python 在 5 分钟内失败会自动重试一次。
- **不要在登录节点上运行训练**（Ray 在登录节点上会卡住）；本地测试用 `sbatch --time=00:30:00` 跑短作业。

---

## 8. 连续学习

智能体在一串环境上依次训练，每个阶段从上一阶段的策略继续，整个顺序重复 `--rounds` 次。
所有环境的观测补齐到 9 行（`obs_rows`），策略可以在环境间迁移。

| 脚本 | 任务（`--task`） | 默认 |
|---|---|---|
| `sto/continual_learning/continual_sto.py` | `1-01-0`（σ 1 → 0.1 → 0）、`0-01-1` | `1-01-0`，3 轮 |
| `bat/continual_learning/continual_bat.py` | `10-7-3`、`3-7-10` | `10-7-3`，3 轮 |
| `batsto/continual_batsto.py` | `sto1-bat10`、`bat10-sto1` | `sto1-bat10`，5 轮 |

```bash
cd sto && python continual_learning/continual_sto.py --task 0-01-1 --rounds 1
cd batsto && sbatch job_batsto_sto1-bat10.sh --lr 0.0003 0.0001
```

每阶段默认 200000 步（`--timesteps-per-phase`）。训练曲线写到 `<models_dir>/tb/`，结束时画
`PPOcon_<任务>_r<轮数>_lr<lr>_<时间>.png`（每个环境一种颜色）。`ContinualMonitor` 在发散、崩溃或第一阶段不学习时
提前结束该学习率的任务（`--no-monitor` 关闭），结论在 `<models_dir>/lr=<lr>/continual_status.json`。

> 注意：连续学习脚本尚未使用多种子和第 11 节的新默认设置（例如不因缺货终止），图的纵轴是缩放后的
> `episode_return_mean`，不是原始成本单位。

---

## 9. 输出文件

| 文件 | 位置 | 内容 |
|---|---|---|
| TensorBoard 日志、checkpoint | `<dir>_logs/<环境>/<时间戳>/<name>/<trial>/` | `progress.csv`、`params.json`、`checkpoint_00000k/`（每 50 次迭代） |
| 训练结论 | `<dir>_logs/<环境>/<时间戳>/training_status.json` | 每个 trial 的 lr、seed、status、steps、reason、would_stop、checkpoint |
| 训练曲线图 | `PPO_<环境>[_<tag>].png`（本目录，日志目录里有副本） | 单 trial 或多种子均值 ± std |
| 曲线 CSV | `csv_results/<tag>_curve.csv` | step, team_return(_smoothed) 或 team_return_mean/std/n_seeds |
| 轨迹 CSV | `csv_results/<tag>_seed<k>_trajectory.csv` | 确定性策略跑一个 episode：每天的状态、动作、需求、成本（`Cost` 列之和 = episode 总成本） |

CSV 文件名：`<算法>_<环境>[_<tag>]_<需求>_lr<lr>_<网络>_<步数>Msteps[_<n>seeds]_<日期时间>`，
例如 `PPO_Sto1_mu2-std1_lr0.0001_mlp64x64_2Msteps_5seeds_20261012-170501_curve.csv`，从不覆盖旧文件。

训练日志、checkpoint、SLURM 输出不进 git（见 `.gitignore`），需要时用 `rsync` 单独拷贝，例如：

```bash
rsync -av ww1a23@<iridis登录节点>:/iridisfs/scratch/ww1a23/MASC_Code/bat/tu_seeds_logs/ ./bat/tu_seeds_logs/
```

---

## 10. 制图规范

全项目统一：

| 环境 | Bat10 | Bat7 | Bat3 | Sto0 | Sto01 | Sto1 |
|---|---|---|---|---|---|---|
| 颜色 | darkgoldenrod | goldenrod | gold | dodgerblue | lightskyblue | skyblue |

- 横纵坐标都用科学计数法。
- 单环境训练图：图例在右下角；所有环境、所有算法使用相同的坐标范围（`plot_tb.XLIM = (0, 2e6)`，
  `YLIM = (-2e6, 0)`），超出范围时打印 WARNING，可用 `--xlim/--ylim` 覆盖。
- 颜色定义在 `plot_tb.ENV_COLORS` 和 `plot_continual.COLORS`（共 5 份，需同步修改）。

---

## 11. 设计决策与实验依据

最初的训练（个体奖励、不缩放奖励、lr 0.003、缺货 12 天终止）在全部 6 个环境上先上升后崩溃。逐项修复如下：

| 问题 | 现象 | 修复 |
|---|---|---|
| 奖励未缩放 | `vf_explained_var` 始终为 0，`vf_loss` ≈ 1e9–1e10 | `reward_scale = 1e-4` |
| 更新过猛、策略过早确定 | KL 尖峰（5.6、32）后崩溃，熵 → 0 | lr 3e-4 / 1e-4，entropy 0.01，minibatch 256，5 epochs |
| 提前终止的漏洞 | 学会"不订货、尽快终止"（成本 −19.7 万，好的策略约 −9 万） | 结算期末成本；**训练时不因缺货终止**（"不订货"变为 −138 万） |
| 个体奖励不一致 | 工厂为降低持有成本停止供货，下游级联缺货 | 默认团队奖励 |
| 价值网络在 KL 尖峰时爆炸 | `vf_loss` 达到 ~3500，梯度被价值项主导 | `vf_clip_param = 10`（最大值 < 10） |
| lr 3e-4 偶尔不稳定 | 6 个 3M 步 trial 中共 16 次 KL > 0.1 的尖峰，最大瞬时下降 195% | lr 1e-4：0 次尖峰，最大瞬时下降 < 11% |

**更长的训练无法缩小与最优固定策略的差距**：24 个 trial 在收敛点（平均 1.14M 步）与 3M 步时的确定性策略，
与最优固定策略的平均差距分别为 13.4% 和 12.9%；18/24 个 trial 在两个时刻的确定性策略完全相同。差距主要来自
不同种子落入的局部最优（同一环境内 0%–25%），因此改为多种子训练报告均值 ± std。

### 参考基线：最优固定策略

在所有"三级订同样数量 q（或上游订 19）、零售商再订货点 rp"的固定策略中网格搜索（10 个种子），
再在 50 个测试种子（100–149）上评估 episode 总成本（第 3 节的定义，不因缺货终止）：

| 环境 | Bat10 | Bat7 | Bat3 | Sto0 | Sto01 | Sto1 |
|---|---|---|---|---|---|---|
| 最优 (q, rp) | (18, 6) | (19, 7) | (18, 7) | (18, 6) | (19, 6) | (19, 7) |
| 总成本 | −71,840 | −75,260 | −77,800 | −94,000 | −63,640 | −84,680 |

---

## 12. 扩展：增加一个智能体

1. 在 `supply_agents/` 新建文件，从 `BaseAgent` 派生并设置参数（`holding_cost`、`processing_time`、
   `fixed_service_time`、`action_nvec` 等），示例见 `base_agent.py` 的说明。
2. 在 `supply_agents/__init__.py` 中导出。
3. 在环境的 `make_agents()` 中按从上游到下游的顺序加入，例如 `[Factory(), Warehouse(), Distributor(), Retailer()]`。

观测空间、动作空间、成本、CSV 列、多智能体封装都会自动跟随智能体列表；最下游必须是 `Retailer`。

---

## 13. 常见问题

- **作业显示 FAILED**：没有 trial 收敛时 `main_ppo.py` 退出码为 1，先看 `training_status.json` 和 `.out` 末尾；
  `.err` 里有 `Traceback` 才是程序错误。
- **`Failed to register worker to Raylet`**：多个作业同一时刻在同一节点启动 Ray，已由对象存储上限和自动重试处理；
  如仍发生，重新提交失败的数组任务（`sbatch --array=1,2 ...`）。
- **日志路径中的 `[64,64]`**：trial 目录名含方括号，Python `glob` 中要用 `glob.escape()`；
  `MultiRLModule.from_checkpoint()` 需要绝对路径。
- **只能在 Mac 上看图**：把 `PPO_*.png` 和需要的日志 `rsync` 到本地，再用 `plot_tb.py` 重画。
