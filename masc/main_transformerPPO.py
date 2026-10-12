# -*- coding: utf-8 -*-
"""
Multi-agent Transformer + PPO (third algorithm, after PPO with MLP policies and recurrent PPO with LSTM policies)
on one supply chain environment: every stage has its own gated transformer policy (masc/models/transformer.py,
model "transformer" of the masc.models registry), in the manner of RLlib's attention net example
(ray-2.6.1/rllib/examples/attention_net.py: PPO + GTrXL, use_attention).

Same arguments, reward-curve checks, stopping rules, seeds, figures and CSV files as masc-train (masc/train.py),
plus the settings of the transformer (--memory, --num-layers, ...; default: masc.models.transformer.DEFAULTS).
The figure is TPPO_<env>.png (e.g. TPPO_Bat10.png), the logs are in tu2_logs/ by default.

    masc-train-transformer --env BatSupplyEnv-v1 --no-stockout-termination
    masc-train-transformer --env StoSupplyEnv-v3 --no-stockout-termination --memory 30 --num-layers 2 --d-model 128
    masc-train-transformer --env StoSupplyEnv-v3 --seeds 0 1 2 3 4 --lr 0.0001 --no-early-stop --num-env-runners 2

The algorithm is assembled from pluggable parts, so another architecture is added the same way: register its
RLModule in masc.models (register_model) and write an entry script like this one (model=<name>).
"""

import sys

from .marl_env import build_ppo_config
from .models import model_defaults
from .train import PLOTTED, env_config_from_args, finish_args, fit_env_runners, lr_search_space, ppo_kwargs, train
from .train import build_parser as build_train_parser

MODEL = "transformer"
ALGO = "TPPO"
# command-line option -> setting of the model (masc.models.transformer.DEFAULTS) and its help text
OPTIONS = {
    "--memory": ("memory", int, "past observations attended to (attention_memory_inference / _training)"),
    "--num-layers": ("num_layers", int, "gated transformer layers (attention_num_transformer_units)"),
    "--d-model": ("d_model", int, "width of the transformer (attention_dim)"),
    "--num-heads": ("num_heads", int, "attention heads (attention_num_heads)"),
    "--head-dim": ("head_dim", int, "width of an attention head (attention_head_dim)"),
    "--ff-dim": ("ff_dim", int, "hidden width of the position-wise MLP (attention_position_wise_mlp_dim)"),
    "--gate-bias": ("gate_bias", float, "initial bias of the GRU gates, > 0: close to the identity "
                                        "(attention_init_gru_gate_bias)"),
    "--max-seq-len": ("max_seq_len", int, "length of the training sequences (max_seq_len)"),
}


def build_parser():
    """Command line of masc-train + the settings of the transformer (also used by the web interface)."""
    parser = build_train_parser("TransformerPPO_gtrxl", "tu2", prog="masc-train-transformer")
    defaults = model_defaults(MODEL)
    group = parser.add_argument_group("transformer")
    for option, (key, kind, help_text) in OPTIONS.items():
        group.add_argument(option, dest=f"tf_{key}", type=kind, default=defaults[key],
                           help=f"{help_text}, default {defaults[key]}")
    group.add_argument("--head-hiddens", dest="tf_head_hiddens", type=int, nargs="+",
                       default=list(defaults["head_hiddens"]),
                       help=f"hidden layers of the policy and value heads, default {list(defaults['head_hiddens'])}")
    return parser


def model_config_from_args(args):
    """Settings of the transformer from the command line."""
    config = {key: getattr(args, f"tf_{key}") for key, _, _ in OPTIONS.values()}
    config["head_hiddens"] = tuple(args.tf_head_hiddens)
    return config


def network_tag(config):
    """Network part of the CSV file names, e.g. gtrxl1x64-h2-mem20."""
    return f"gtrxl{config['num_layers']}x{config['d_model']}-h{config['num_heads']}-mem{config['memory']}"


def main(argv=None):
    """masc-train-transformer: exit code 0 if a trial converged (or completed with --no-early-stop), else 1."""
    args = finish_args(build_parser().parse_args(argv))
    args.num_env_runners = fit_env_runners(args.num_env_runners)
    model_config = model_config_from_args(args)
    print("transformer:", model_config)
    config = build_ppo_config(env_config_from_args(args), lr=lr_search_space(args.lr),
                              num_env_runners=args.num_env_runners, num_gpus=args.num_gpus,
                              model=MODEL, model_config=model_config, **ppo_kwargs(args))
    _, status = train(args, config, algo=ALGO, net=network_tag(model_config))
    return 0 if any(s["status"] in PLOTTED for s in status) else 1


if __name__ == '__main__':
    sys.exit(main())
