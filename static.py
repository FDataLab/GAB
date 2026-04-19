"""
Defines global constants used throughout the framework, organized into the following groups:

    - Filter: Purification method names and their associated hyperparameter keys
      (e.g., THRESHOLD, TOP_K, TOP_SINGULAR_R)

    - Model Setup: Keys for model architecture configuration
      (e.g., DEVICE, IN_CHANNEL, OUT_CHANNEL, MODEL)

    - Hyperparameters: Keys for training and model hyperparameters
      (e.g., NUM_LAYERS, LEARNING_RATE, HIDDEN_UNITS, EPOCHS, DROPOUT, OPTIMIZER_STATE)

    - Hyperparameter Options: Supported values for optimizers, schedulers, and aggregations
      (e.g., ADAM, SGD, STEP_LR, AVAILABLE_AGGRE)

    - Paths: Directory paths for datasets, configs, logs, results, and cached predictions
      (e.g., PATH_DATA, PATH_CONFIG_MODEL, PATH_RESULT, PATH_CACHED_PRED)

    - Adversarial: Names and identifiers for supported adversarial attack methods
      (e.g., NETTACK_NAME, FGA_NAME, SGA_NAME, PRBCD_NAME)
"""


#Filter
PURIFICATION = "purification"
THRESHOLD = "threshold"
TOP_K ="k"
TOP_SINGULAR_R ="r"
GARNET = "GARNET"
JACCARD = "Jaccard"
SVD = "SVD"

#Model set-up
DEVICE = "device"
IN_CHANNEL = "in_channel"
OUT_CHANNEL = "out_channel"
MODEL = "model"

#Hyper=parameters
NUM_LAYERS = "num_layers"
BATCH_SIZE = "batch_size"
LEARNING_RATE = "lr"
HIDDEN_UNITS ="hidden_unit"
EPOCHS = "num_epochs"
DROPOUT = "drop_out"
PATIENCE = "patience"
SCHEDULER_STATE = "scheduler_state"
AGGREGATION = "aggr"
OPTIMIZER_STATE = "optimizer_state"
SCHEDULER = "scheduler"
OPTIMIZER = "optimizer"
GAMMA_LIST = "gamma_list"
STEP_SIZE_LIST = "step_size_list"
STEP_SIZE = "step_size"
WEIGHT_DECAY = "weight_decay"
GAMMA_ATTENTION ="gamma_attention"
PROPA_STEP_K="propa_step_k"
LAMBDA1 ="lambda1"
LAMBDA2 ="lambda2"
DROP_NODE_RATE="drop_node_rate"
ORDER = "order"
TEMP = "temp"
LAMBDA ="lambda"
SAMPLE = "sample"
IDX_TRAIN = "idx_train"
IDX_VAL = "idx_val"
IDX_TEST = "idx_test"

#Hyper-parameters' options
SGD ="SGD"
ADAM = "ADAM"
STEP_LR = "StepLR"
GAMMA = "gamma"
SPLIT_IDX = "split_idx"
AVAILABLE_AGGRE = ['mean','sum','max','median','softmedian']

#PATH
BEST_CONFIG = "best_config_split_"
PATH_DATA = "data/"
PATH_CONFIG_MODEL = "{}configs/models/".format(PATH_DATA)
PATH_CONFIG_PURIFICATION = "{}configs/purifications/".format(PATH_DATA)
PATH_CONFIG_SPLIT = "{}configs/split/".format(PATH_DATA)
PATH_DATASET = "{}dataset".format(PATH_DATA)
PATH_OUTPUT = "output/"
PATH_LOG = "{}log/".format(PATH_OUTPUT)
PATH_RESULT = "{}results/".format(PATH_DATA)
PATH_CACHED_PRED = "{}pred_cache/".format(PATH_DATA)

#Adversarial
NETTACK_NAME = "nettack"
GOTTACK_NAME = "gottack"
FGA_NAME= "fga"
SGA_NAME= "sga"
RND_NAME = "rnd"
PGDATTACK_NAME = "pgdattack"
PRBCD_NAME = "prbcdattack"
EVASION = "evasion"
POSION = "poision"
GOTTACK ="gottack"
GRBCD_NAME = "GRBCD"
L1D_RND_ATTACK = "l1d_rnd_attack"



