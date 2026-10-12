# -*- coding: utf-8 -*-
"""
Multi-agent recurrent PPO (Ray RLlib PPO with an LSTM, replaces sb3_contrib RecurrentPPO)
on one supply chain environment. One LSTM policy per stage, see marl_env.py.
Same arguments, reward-curve checks, stopping rules and figures as masc-train (masc/train.py);
the figure is RPPO_<env>.png (e.g. RPPO_Bat10.png), the logs are in tu1_logs/ by default.

    masc-train-rppo --env BatSupplyEnv-v1 --no-stockout-termination
    masc-train-rppo --env StoSupplyEnv-v1 --num-env-runners 8 --num-gpus 1
    masc-train-rppo --env StoSupplyEnv-v3 --lr 0.0003 0.0001      # grid search over the learning rate
"""

import sys

from .marl_env import build_ppo_config
from .train import PLOTTED, env_config_from_args, fit_env_runners, lr_search_space, parse_args, ppo_kwargs, train


def main(argv=None):
    args = parse_args(default_name="RecurrentPPO_pi=64,64_vf=64,64_lstm=1", default_dir="tu1", argv=argv)
    args.num_env_runners = fit_env_runners(args.num_env_runners)
    config = build_ppo_config(env_config_from_args(args), lr=lr_search_space(args.lr), hiddens=(64, 64),
                              use_lstm=True, lstm_cell_size=64, max_seq_len=20,
                              num_env_runners=args.num_env_runners, num_gpus=args.num_gpus, **ppo_kwargs(args))
    _, status = train(args, config, algo="RPPO", net="mlp64x64-lstm64-seq20")
    return 0 if any(s["status"] in PLOTTED for s in status) else 1


if __name__ == '__main__':
    sys.exit(main())
