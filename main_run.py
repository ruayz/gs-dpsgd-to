import argparse
import pprint
import random
import sys

import numpy as np
import torch
import copy
from opacus import PrivacyEngine

from config import get_config, parse_config_arg
from datasets import get_loaders_from_config
from evaluators import create_evaluator
from models import create_model
from privacy_engines.dpsgd_f_engine import DPSGDF_PrivacyEngine
from privacy_engines.dpsgd_global_adaptive_engine import DPSGDGlobalAdaptivePrivacyEngine
from privacy_engines.dpsgd_global_engine import DPSGDGlobalPrivacyEngine
from trainers import create_trainer
from utils import privacy_checker
from writer import Writer


def main(args):

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    cfg = get_config(
        dataset=args.dataset,
        method=args.method,
    )
    cfg = {**cfg, **dict(parse_config_arg(kv) for kv in args.config)}

    # Checks group_ratios is specified correctly
    if len(cfg["group_ratios"]) != cfg["num_groups"]:
        raise ValueError(
            "Number of group ratios, {}, not equal to number of groups of {}, {}"
                .format(len(cfg["group_ratios"]), cfg["protected_group"], cfg["num_groups"])
        )

    if any(x > 1 or (x < 0 and x != -1) for x in cfg["group_ratios"]):
        raise ValueError("All elements of group_ratios must be in [0,1]. Indicate no sampling with -1.")

    pprint.sorted = lambda x, key=None: x
    pp = pprint.PrettyPrinter(indent=4)
    print(10 * "-" + "-cfg--" + 10 * "-")
    pp.pprint(cfg)

    # Set random seeds based on config
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    train_loader, valid_loader, test_loader = get_loaders_from_config(
        cfg,
        device
    )

    writer = Writer(
        logdir=cfg.get("logdir_root", "runs"),
        make_subdir=True,
        tag_group=args.dataset,
        dir_name=cfg.get("logdir", "")
    )
    writer.write_json(tag="config", data=cfg)

    model, optimizer = create_model(cfg, device)

    # if cfg["method"] != "regular" and cfg["method"] != "separate" and cfg["method"] != "dpsgd-thresh":
    #     # separate has different len(train_loader), thus has different sample rate
    #     sample_rate = 1 / len(train_loader)
    #     privacy_checker(sample_rate, cfg)

    if cfg["method"] == "dpsgd":
        privacy_engine = PrivacyEngine(accountant=cfg["accountant"])
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            #noise_multiplier=cfg["noise_multiplier"],
            #clipping="adaptive",
            target_epsilon=cfg["target_epsilon"],
            target_delta=cfg["delta"],
            epochs=cfg["max_epochs"],
            max_grad_norm=cfg["l2_norm_clip"]  # C
        )
    elif cfg["method"] == "dpsgd-post":
        privacy_engine = PrivacyEngine(accountant=cfg["accountant"])
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            #noise_multiplier=cfg["noise_multiplier"],
            #clipping="adaptive",
            target_epsilon=cfg["target_epsilon"],
            target_delta=cfg["delta"],
            epochs=cfg["max_epochs"],
            max_grad_norm=cfg["l2_norm_clip"]  # C
        )
    elif cfg["method"] == "separate":
        privacy_engine0 = PrivacyEngine(accountant=cfg["accountant"])
        privacy_engine1 = PrivacyEngine(accountant=cfg["accountant"])
        privacy_engine = [privacy_engine0, privacy_engine1]
        # privacy_engine = PrivacyEngine(accountant=cfg["accountant"])
        # model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        #     module=model,
        #     optimizer=optimizer,
        #     data_loader=train_loader[2],
        #     # noise_multiplier=cfg["noise_multiplier"],
        #     # clipping="adaptive",
        #     target_epsilon=cfg["target_epsilon"],
        #     target_delta=cfg["delta"],
        #     epochs=cfg["max_epochs"],
        #     max_grad_norm=cfg["l2_norm_clip"]  # C
        # )
    elif cfg["method"] == "dpsgd-thresh":
        privacy_engine0 = PrivacyEngine(accountant=cfg["accountant"])
        privacy_engine1 = PrivacyEngine(accountant=cfg["accountant"])
        privacy_engine = [privacy_engine0, privacy_engine1]

    elif cfg["method"] == "dpsgd-global":
        privacy_engine = DPSGDGlobalPrivacyEngine(accountant=cfg["accountant"])
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            # noise_multiplier=cfg["noise_multiplier"],
            # clipping="adaptive",
            target_epsilon=cfg["target_epsilon"],
            target_delta=cfg["delta"],
            epochs=cfg["max_epochs"],
            max_grad_norm=cfg["l2_norm_clip"],  # C
        )
    elif cfg["method"] == "dpsgd-f":
        privacy_engine = DPSGDF_PrivacyEngine(accountant=cfg["accountant"])
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            # noise_multiplier=cfg["noise_multiplier"],
            # clipping="adaptive",
            target_epsilon=cfg["target_epsilon"],
            target_delta=cfg["delta"],
            epochs=cfg["max_epochs"],
            max_grad_norm=0  # this parameter is not applicable for DPSGD-F
        )

    elif cfg["method"] == "dpsgd-global-adapt":
        privacy_engine = DPSGDGlobalAdaptivePrivacyEngine(accountant=cfg["accountant"])
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            # noise_multiplier=cfg["noise_multiplier"],
            # clipping="adaptive",
            target_epsilon=cfg["target_epsilon"],
            target_delta=cfg["delta"],
            epochs=cfg["max_epochs"],
            max_grad_norm=cfg["l2_norm_clip"],  # C
        )
    else:
        # doing regular training
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=0,
            max_grad_norm=sys.float_info.max,
            poisson_sampling=False
        )

    evaluator = create_evaluator(
        model,
        valid_loader=valid_loader, test_loader=test_loader,
        valid_metrics=cfg["valid_metrics"],
        test_metrics=cfg["test_metrics"],
        num_groups=cfg["num_groups"],
        num_classes=2,  # binary task
        method=cfg["method"],
        dataset=cfg["dataset"],
        seed=cfg["seed"],
    )

    trainer = create_trainer(
        train_loader,
        valid_loader,
        test_loader,
        model,
        optimizer,
        privacy_engine,
        evaluator,
        writer,
        device,
        cfg
    )

    trainer.train()

def param(method, dataset, target_epsilon, parameter_norm_bound, seed):
    dir = "adult_$seed"
    angles = 'False'
    hessian = 'False'
    step = 50
    #parameter_norm_bound = 0.5  # [0.1,1]

    parser = argparse.ArgumentParser(description="Fairness for DP-SGD")

    parser.add_argument("--dataset", type=str, default="mnist",
                        help="Dataset to train on.")
    parser.add_argument("--method", type=str, default="regular",
                        choices=["regular", "dpsgd", "dpsgd-f", "fairness-lens", "dpsgd-global", "dpsgd-global-adapt"],
                        help="Method for training and clipping.")

    config_args = []
    if method == 'regular':
        config_args = [
            "group_ratios=-1,-1",
            "make_valid_loader=1",
            "net=mlp",
            "lr=0.01",
            "train_batch_size=256",
            "valid_batch_size=256",
            "test_batch_size=256",
            "max_epochs=20",
            f"evaluate_angles={angles}",
            f"evaluate_hessian={hessian}",
            f"angle_comp_step={step}",
            #f"logdir='noise_mul_{noise_multiplier}/{dataset}_{seed}/{dataset}_nonpriv'",
            f"logdir='target_epsilon/{dataset}_{seed}/{dataset}_nonpriv'",
            #f"logdir='separate_{parameter_norm_bound}/{dataset}_{seed}/{dataset}_nonpriv'",
            f"seed={seed}"
        ]
    elif method == 'dpsgd':
        config_args = [
            "group_ratios=-1,-1",
            "make_valid_loader=1",
            "net=mlp",
            "lr=0.01",
            "train_batch_size=256",
            "valid_batch_size=256",
            "test_batch_size=256",
            "max_epochs=20",
            "delta=1e-6",
            #f"noise_multiplier={noise_multiplier}",
            f"target_epsilon={target_epsilon}",
            "l2_norm_clip=0.5",  # default:0.5
            f"evaluate_angles={angles}",
            f"evaluate_hessian={hessian}",
            f"angle_comp_step={step}",
            #f"logdir='noise_mul_{noise_multiplier}/{dataset}_{seed}/{dataset}_dpsgd'",
            f"logdir='target_epsilon/{dataset}_{seed}/{dataset}_dpsgd'",
            #f"logdir='separate_{parameter_norm_bound}/{dataset}_{seed}/{dataset}_dpsgd'",
            f"seed={seed}"
        ]
    elif method == 'dpsgd-post':
        config_args = [
            "group_ratios=-1,-1",
            "make_valid_loader=1",
            "net=mlp",
            "lr=0.01",
            "train_batch_size=256",
            "valid_batch_size=256",
            "test_batch_size=256",
            "max_epochs=20",
            "delta=1e-6",
            # f"noise_multiplier={noise_multiplier}",
            f"target_epsilon={target_epsilon}",
            "l2_norm_clip=0.5",  # default:0.5
            f"evaluate_angles={angles}",
            f"evaluate_hessian={hessian}",
            f"angle_comp_step={step}",
            # f"logdir='noise_mul_{noise_multiplier}/{dataset}_{seed}/{dataset}_dpsgd'",
            f"logdir='target_epsilon/{dataset}_{seed}/{dataset}_dpsgdp'",
            # f"logdir='separate_{parameter_norm_bound}/{dataset}_{seed}/{dataset}_dpsgd'",
            f"seed={seed}"
        ]
    elif method == 'separate':  # each epoch diff group data,then agg
        config_args = [
            f"parameter_norm_bound={parameter_norm_bound}",
            "group_ratios=-1,-1",
            "make_valid_loader=1",
            "net=mlp",
            "lr=0.01",
            "last_lr=0.005",
            "train_batch_size=256",
            "valid_batch_size=256",
            "test_batch_size=256",
            "max_epochs=20",
            "delta=1e-6",
            #f"noise_multiplier={noise_multiplier}",
            f"target_epsilon={target_epsilon}",
            "l2_norm_clip=0.5",  # default:0.5
            f"evaluate_angles={angles}",
            f"evaluate_hessian={hessian}",
            f"angle_comp_step={step}",
            f"logdir='target_epsilon/{dataset}_{seed}/m_{parameter_norm_bound}/{dataset}_separate'",
            #f"logdir='separate_{parameter_norm_bound}/{dataset}_{seed}/{dataset}_separate'",
            #f"logdir='noise_mul_{noise_multiplier}/{dataset}_{seed}/{dataset}_separate'",
            f"seed={seed}"
        ]
    elif method == 'dpsgd-thresh':  # each epoch diff group data, then agg
        config_args = [
            "group_ratios=-1,-1",
            "make_valid_loader=1",
            "net=mlp",
            "lr=0.01",
            "train_batch_size=256",
            "valid_batch_size=256",
            "test_batch_size=256",
            "max_epochs=20",
            "delta=1e-6",
            #f"noise_multiplier={noise_multiplier}",
            f"target_epsilon={target_epsilon}",
            "l2_norm_clip=0.5",  # default:0.5
            f"evaluate_angles={angles}",
            f"evaluate_hessian={hessian}",
            f"angle_comp_step={step}",
            #f"logdir='separate_test/{dataset}_{seed}/{dataset}_separate'",
            f"logdir='target_epsilon/{dataset}_{seed}/{dataset}_dpsgdt'",
            #f"logdir='noise_mul_{noise_multiplier}/{dataset}_{seed}/{dataset}_dpsgdt'",
            f"seed={seed}"
        ]
    elif method == 'dpsgd-f':
        config_args = [
            "group_ratios=-1,-1",
            "make_valid_loader=1",
            "net=mlp",
            "lr=0.01",
            "train_batch_size=256",
            "valid_batch_size=256",
            "test_batch_size=256",
            "max_epochs=20",
            "delta=1e-6",
            #f"noise_multiplier={noise_multiplier}",
            f"target_epsilon={target_epsilon}",
            "base_max_grad_norm=0.5",
            "counts_noise_multiplier=10",
            f"evaluate_angles={angles}",
            f"evaluate_hessian={hessian}",
            f"angle_comp_step={step}",
            #f"logdir='noise_mul_{noise_multiplier}/{dataset}_{seed}/{dataset}_dpsgdf'",
            f"logdir='target_epsilon/{dataset}_{seed}/{dataset}_dpsgdf'",
            #f"logdir='separate_{parameter_norm_bound}/{dataset}_{seed}/{dataset}_dpsgdf'",
            f"seed={seed}"
        ]
    elif method == 'dpsgd-global':
        config_args = [
            "group_ratios=-1,-1",
            "make_valid_loader=1",
            "net=mlp",
            "train_batch_size=256",
            "valid_batch_size=256",
            "test_batch_size=256",
            "max_epochs=20",
            "delta=1e-6",
            #f"noise_multiplier={noise_multiplier}",
            f"target_epsilon={target_epsilon}",
            "l2_norm_clip=0.5",
            f"evaluate_angles={angles}",
            f"evaluate_hessian={hessian}",
            f"angle_comp_step={step}",
            "strict_max_grad_norm=50",
            "lr=0.2",
            #f"logdir='noise_mul_{noise_multiplier}/{dataset}_{seed}/{dataset}_dpsgdg'",
            f"logdir='target_epsilon/{dataset}_{seed}/{dataset}_dpsgdg'",
            #f"logdir='separate_{parameter_norm_bound}/{dataset}_{seed}/{dataset}_dpsgdg'",
            f"seed={seed}"
        ]
    elif method == 'dpsgd-global-adapt':
        config_args = [
            "group_ratios=-1,-1",
            "make_valid_loader=1",
            "net=mlp",
            "train_batch_size=256",
            "valid_batch_size=256",
            "test_batch_size=256",
            "max_epochs=20",
            "delta=1e-6",
            #f"noise_multiplier={noise_multiplier}",
            f"target_epsilon={target_epsilon}",
            "l2_norm_clip=0.5",
            f"evaluate_angles={angles}",
            f"evaluate_hessian={hessian}",
            f"angle_comp_step={step}",
            "strict_max_grad_norm=50",
            "lr=0.2",
            "bits_noise_multiplier=10",
            "lr_Z=0.1",
            "threshold=1",
            f"logdir='target_epsilon/{dataset}_{seed}/{dataset}_dpsgdga'",
            #f"logdir='noise_mul_{noise_multiplier}/{dataset}_{seed}/{dataset}_dpsgdga'",
            #f"logdir='separate_{parameter_norm_bound}/{dataset}_{seed}/{dataset}_dpsgdga'",
            f"seed={seed}"
        ]


    parser.add_argument("--config", default=config_args, action="append",
                            help="Override config entries. Specify as `key=value`.")

    args = parser.parse_args()
    args.dataset = dataset
    args.method = method

    return args


if __name__ == "__main__":
    #dataset = ['adult', 'dutch', 'bank', 'credit', 'compas', 'law']
    dataset = ['dutch', 'bank', 'credit', 'compas', 'law']
    # target_epsilon = [2.66, 2.27, 2.84, 3.37, 4.12, 3.68]
    target_epsilon = [2.27, 2.84, 3.37, 4.12, 3.68]
    protected_group = ['sex', 'sex', 'marital', 'SEX', 'race', 'race']
    #method = ['regular', 'dpsgd', 'separate', 'dpsgd-thresh', 'dpsgd-f', 'dpsgd-global', 'dpsgd-global-adapt', 'dpsgd-post']
    method = ['regular', 'dpsgd', 'dpsgd-global', 'dpsgd-global-adapt']
    #noise_multiplier = [0.3, 0.6, 1, 1.5]
    #parameter_norm_bound = [0.1, 0.2, 0.3, 0.4, 0.5, 1, None]
    #parameter_norm_bound = [None]
    #optimizer = ['sgd', 'adam', 'momentum', 'rmsprop']


    # for seed in range(4):
    #     args = param('regular', 'dutch', noise_multiplier[0], parameter_norm_bound[0], seed)
    #     main(args)

    for seed in [90]:
        for m in method:
            for i in range(len(dataset)):

                args = param(m, dataset[i], target_epsilon[i], p, seed)
                main(args)
