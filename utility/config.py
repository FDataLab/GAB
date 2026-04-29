import argparse
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from static import *

parser = argparse.ArgumentParser(description="Adversarial attacks args parser")

# 1.dataset
parser.add_argument("--dataset", type=str, default="cora", help="baseline dataset")

# 2.experiment setup
parser.add_argument("--device", type=str, default="cpu", help="training device")
parser.add_argument("--device_id", type=str, default="0", help="device id for gpu")
parser.add_argument("--seed", type=int, default=1024, help="random seed")
parser.add_argument("--num_run", type=int, default=3, help="Number of time to run")
parser.add_argument(
    "--min_epoch", type=int, default=100, help="min epoch applied for early stopping"
)
parser.add_argument(
    "--log_interval", type=int, default=20, help="log interval, default: 20,[20,40,...]"
)
parser.add_argument(
    "--train_early_stopping", default=True, help="Apply early stopping for training"
)
parser.add_argument("--verbose", default=False, help="Log training and attack process")
parser.add_argument(
    "--split_state", type=int, default=1024, help="initial state for split"
)
parser.add_argument("--num_split", type=int, default=5, help="Number of split")
parser.add_argument("--split_idx", type=int, default=0, help="Split idx")
parser.add_argument(
    "--purification", type=str, default=None, help="name of purification"
)
parser.add_argument("--budget_start", type=int, default=1, help="starting budget")
parser.add_argument("--budget_end", type=int, default=5, help="endding budget")
parser.add_argument(
    "--config_setting",
    type=str,
    default="best_config",
    choices=["best_config", "default"],
    help="either best_config or default",
)
parser.add_argument(
    "--use-node-degree",
    dest="use_node_degree",
    action="store_true",
    help="Include degree criterion in target node selection",
)

parser.add_argument(
    "--no-use-node-degree",
    dest="use_node_degree",
    action="store_false",
    help="Exclude degree criterion from target node selection",
)

parser.set_defaults(use_node_degree=True)

parser.add_argument(
    "--evasion",
    dest="evasion",
    action="store_true",
    help="Enable evasion setting",
)

parser.add_argument(
    "--no-evasion",
    dest="evasion",
    action="store_false",
    help="Skip evasion setting",
)

parser.set_defaults(evasion=True)

parser.add_argument(
    "--poison",
    dest="poison",
    action="store_true",
    help="Enable poison setting",
)

parser.add_argument(
    "--no-poison",
    dest="poison",
    action="store_false",
    help="Skip poison setting",
)

parser.set_defaults(poison=True)

# 3.models
parser.add_argument("--model", type=str, default="GCN", help="models name")
parser.add_argument(
    "--in_channels", type=int, default=2, help="Size of each input sample"
)
parser.add_argument("--out_channels", type=int, default=None, help="Size of output")

# 4.adversarial
parser.add_argument(
    "--adversarial", type=str, default="nettack", help="adversarial methods"
)


args = parser.parse_args()

args.output_folder = "{}{}/{}/".format(PATH_LOG, args.dataset, args.model)


# set the running device
if int(args.device_id) >= 0 and torch.cuda.is_available():
    args.device = torch.device("cuda".format(args.device_id))
    print("INFO: using gpu:{} to train the models".format(args.device_id))
else:
    args.device = torch.device("cpu")
    print("INFO: using cpu to train the models")
