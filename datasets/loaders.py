import numpy as np
from torch.utils.data import DataLoader

from .image import get_image_datasets
from .tabular import get_tabular_datasets
from .FairDataset import get_tabular_fair_datasets
from .dataset import GroupLabelDataset


def get_loaders_from_config(cfg, device, **kwargs):
    if cfg["net"] == "cnn":
        flatten = False
    elif cfg["net"] == "mlp":
        flatten = True
    elif cfg["net"] == "logistic":
        flatten = True
    else:
        raise ValueError(f"Unknown net type {cfg['net']} for flattening")

    train_loader, valid_loader, test_loader = get_loaders(
        dataset=cfg["dataset"],
        device=device,
        data_root=cfg.get("data_root", "data/"),
        train_batch_size=cfg["train_batch_size"],
        valid_batch_size=cfg["valid_batch_size"],
        test_batch_size=cfg["test_batch_size"],
        group_ratios=cfg["group_ratios"],
        seed=cfg["seed"],
        protected_group=cfg["protected_group"],
        make_valid_loader=cfg["make_valid_loader"],
        flatten=flatten,
        method=cfg["method"],
    )

    if cfg["method"] == "separate" or cfg["method"] == "dpsgd-thresh":
        train_dataset_shape0 = train_loader[0].dataset.x.shape
        train_dataset_shape1 = train_loader[1].dataset.x.shape
        cfg["train_dataset0_size"] = train_dataset_shape0[0]
        cfg["train_dataset1_size"] = train_dataset_shape1[0]
        cfg["data0_shape"] = tuple(train_dataset_shape0[1:])
        cfg["data1_shape"] = tuple(train_dataset_shape1[1:])
        cfg["data_dim"] = int(np.prod(cfg["data0_shape"]))
    else:
        if cfg["dataset"] in ["celeba"]:
            train_dataset_shape = train_loader.dataset.shape
        else:
            train_dataset_shape = train_loader.dataset.x.shape
        cfg["train_dataset_size"] = train_dataset_shape[0]
        cfg["data_shape"] = tuple(train_dataset_shape[1:])
        cfg["data_dim"] = int(np.prod(cfg["data_shape"]))

    if not cfg["make_valid_loader"]:
        valid_loader = test_loader
        print("WARNING: Using test loader for validation")

    return train_loader, valid_loader, test_loader


def get_loaders(
        dataset,
        device,
        data_root,
        train_batch_size,
        valid_batch_size,
        test_batch_size,
        group_ratios,
        seed,
        protected_group,
        make_valid_loader,
        flatten,
        method
):
    # NOTE: only training and validation sets sampled according to group_ratios
    if dataset in ["mnist", "fashion-mnist", "cifar10", "svhn", "celeba"]:
        train_dset, valid_dset, test_dset = get_image_datasets(dataset, data_root, seed, group_ratios,
                                                               make_valid_loader, flatten)

    # NOTE: entire dataset sampled according to group_ratios
    #elif dataset in ["adult", "dutch"]:
        #train_dset, valid_dset, test_dset = get_tabular_datasets(dataset, data_root, seed, protected_group,
    # group_ratios, make_valid_loader)
    elif dataset in ['adult', 'dutch', 'bank', 'credit', 'compas', 'law']:
        train_dset, valid_dset, test_dset = get_tabular_fair_datasets(dataset, data_root, seed, protected_group,
                                                                      group_ratios, make_valid_loader)
        # new
        if method == "separate" or method == "dpsgd-thresh":
            train_dset0 = GroupLabelDataset("train",
                                            train_dset[train_dset.z == 0][0],
                                            train_dset[train_dset.z == 0][1],
                                            train_dset[train_dset.z == 0][2]
                                            )
            train_dset1 = GroupLabelDataset("train",
                                            train_dset[train_dset.z == 1][0],
                                            train_dset[train_dset.z == 1][1],
                                            train_dset[train_dset.z == 1][2]
                                            )
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    if method == "separate" or method == "dpsgd-thresh":
        train_loader0 = get_loader(train_dset0, device, train_batch_size, drop_last=False)
        train_loader1 = get_loader(train_dset1, device, train_batch_size, drop_last=False)
        train_loader_all = get_loader(train_dset, device, train_batch_size, drop_last=False)
        train_loader = [train_loader0, train_loader1, train_loader_all]
    else:
        train_loader = get_loader(train_dset, device, train_batch_size, drop_last=False)

    if make_valid_loader:
        valid_loader = get_loader(valid_dset, device, valid_batch_size, drop_last=False)
    else:
        valid_loader = None

    test_loader = get_loader(test_dset, device, test_batch_size, drop_last=False)

    return train_loader, valid_loader, test_loader


def get_loader(dset, device, batch_size, drop_last):
    return DataLoader(
        dset.to(device),
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=False
    )
