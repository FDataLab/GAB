import os
import statistics as stats
import sys
import timeit

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluation import AdversarialAssessment
from static import *
from utility.config import args


def save_results(results: dict, result_path: str):

    if not os.path.exists(result_path):
        result_df = pd.DataFrame(columns=results.keys())
    else:
        result_df = pd.read_csv(result_path)

    result_df = result_df.append(results, ignore_index=True)
    result_df.to_csv(result_path, index=False)


def get_stat(results_list):
    return (stats.mean(results_list), stats.stdev(results_list))


def prepare_result_path(
    use_degree=False,
    config_setting="best_config",
    model="GCN",
    purification=None,
    dataset="cora",
):
    path = PATH_RESULT
    if not use_degree or config_setting == "default":
        path += "ablation"

    if not use_degree:
        path += "/no_degree"

    path += f"/{config_setting}/{model}"
    if purification is not None:
        path += f"_{purification}"

    path += f"/{dataset}"

    return path


def prepare_file_name(setting="evasion", adversarial="nettack", adaptive=False):
    file_name = f"{adversarial}_{setting}"
    if adaptive:
        file_name += f"_adaptive"
    return file_name


def flat(result_dict: dict):
    for key in result_dict.keys():
        result_dict[key] = result_dict[key][0]
    return result_dict


if __name__ == "__main__":
    path = prepare_result_path(
        use_degree=args.use_node_degree,
        config_setting=args.config_setting,
        model=args.model,
        purification=args.purification,
        dataset=args.dataset,
    )

    if not os.path.exists(path):
        os.makedirs(path)

    file_name_evasion = (
        prepare_file_name(setting="evasion", adversarial=args.adversarial)
        if args.evasion
        else None
    )

    file_name_poison = (
        prepare_file_name(setting="poison", adversarial=args.adversarial)
        if args.poison
        else None
    )

    evalution_pipeline = AdversarialAssessment(
        model=args.model,
        dataset=args.dataset,
        adversarial=args.adversarial,
        num_runs=args.num_run,
        num_split=args.num_split,
        base_seed=args.seed,
        purification=args.purification,  # None by default
        device=args.device,
        use_degree_node_selection=args.use_node_degree,
        use_tune_model=args.config_setting == "best_config",
    )

    for budget in [1, 2, 3, 4, 5]:
        evasion_df, poision_df = evalution_pipeline.evaluate(
            evasion=args.evasion, poision=args.poison, budgets_list=[budget]
        )

        save_results(
            flat(evasion_df.to_dict(orient="list")), f"{path}/{file_name_evasion}.csv"
        )
        save_results(
            flat(poision_df.to_dict(orient="list")), f"{path}/{file_name_poison}.csv"
        )
