import os
import random as rand
import numpy as np

import pandas as pd
import regex as re
import torch

from .dataset import GroupLabelDataset
from .sample_weights import find_sample_weights


# normalize df columns
def normalize(df, columns):
    result = df.copy()
    for column in columns:
        mu = df[column].mean(axis=0)
        sigma = df[column].std(axis=0)
        assert sigma != 0
        result[column] = (df[column] - mu) / sigma
    return result


def make_tabular_train_valid_split(data, valid_frac):
    n_valid = int(valid_frac * data.shape[0])
    valid_data = data[:n_valid]
    train_data = data[n_valid:]
    return train_data, valid_data


def make_tabular_train_valid_test_split(data, valid_frac, test_frac, seed):
    # shuffle samples
    data = data.sample(frac=1, random_state=seed)

    n_test = int(test_frac * data.shape[0])
    test_data = data[:n_test]
    data = data[n_test:]

    train_data, valid_data = make_tabular_train_valid_split(data, valid_frac)

    return train_data, valid_data, test_data


# refer to sample_weights.py
def sample_by_group_ratios(group_ratios, df, seed):
    # Weighted sampling, based on group ratio and the number of samples in each group
    print("Number of samples by group (before sampling):")
    print(df.protected_group.value_counts())
    sample_weights = find_sample_weights(group_ratios, df.protected_group.value_counts().tolist())
    rand.seed(seed)
    idx = [rand.random() <= sample_weights[row.protected_group] for _, row in df.iterrows()]
    df = df.loc[idx]
    print("Number of samples by group (after sampling):")
    print(df.protected_group.value_counts())
    return df


def preprocess_adult(df, protected_group, target, group_ratios, seed):
    df = df.drop("fnlwgt", axis=1)

    numerical_columns = ["age", "educational-num", "capital-gain", "capital-loss",
                         "hours-per-week"]
    df = normalize(df, numerical_columns)

    df['Class-label'] = [1 if v == 1 else 0 for v in df['Class-label']]

    mapped_sex_values = df.sex.map({"Male": 0, "Female": 1})
    df.loc[:, "sex"] = mapped_sex_values

    # make race binary
    def race_map(value):
        if value != "White":
            return (1)
        return (0)

    mapped_race_values = df.race.map(race_map)
    df.loc[:, "race"] = mapped_race_values

    categorical = df.columns.tolist()
    for column in numerical_columns:
        categorical.remove(column)
    print("Possible protected groups are: {}".format(categorical))

    if protected_group == "labels":
        df.loc[:, "protected_group"] = df[target]
    elif protected_group not in categorical:
        raise ValueError(
            f"Invalid protected group {protected_group}. "
            + f"Valid choices are {categorical}."
        )
    else:
        df.loc[:, "protected_group"] = df[protected_group]

    df = sample_by_group_ratios(group_ratios, df, seed)

    # convert to one-hot vectors
    categorical_non_binary = ["workclass", "education", "marital-status", "occupation",
                              "relationship", "native-country"]
    df = pd.get_dummies(df, columns=categorical_non_binary)

    return df


def preprocess_dutch(df, protected_group, target, group_ratios, seed):
    mapped_sex_values = df.sex.map({"male": 0, "female": 1})
    df.loc[:, "sex"] = mapped_sex_values

    # note original dataset has values {0,1,9} for prev_res_place, but all samples with 9 are underage, hence get dropped
    mapped_prev_res_place_values = df.prev_residence_place.map({1: 0, 2: 1})
    df.loc[:, "prev_residence_place"] = mapped_prev_res_place_values

    categorical = df.columns.to_list()
    print("Possible protected groups are: {}".format(categorical))

    if protected_group == "labels":
        df.loc[:, "protected_group"] = df[target]
    elif protected_group not in categorical:
        raise ValueError(
            f"Invalid protected group {protected_group}. "
            + f"Valid choices are {categorical}."
        )
    else:
        df.loc[:, "protected_group"] = df[protected_group]

    # convert categorical unprotected features to one-hot vectors
    if target in categorical:
        categorical.remove(target)
    if "sex" in categorical:
        categorical.remove("sex")  # binary
    if "prev_res_place" in categorical:
        categorical.remove("prev_res_place")  # binary

    df = sample_by_group_ratios(group_ratios, df, seed)

    df = pd.get_dummies(df, columns=categorical)

    return df


def preprocess_bank(df, protected_group, target, group_ratios, seed):
    numerical_columns = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    if protected_group in numerical_columns:
        numerical_columns.remove(protected_group)
    df = normalize(df, numerical_columns)

    df['y'] = [1 if v == 'yes' else 0 for v in df['y']]

    df['marital'] = ['Married' if v == 'married' else 'Non-Married' for v in df['marital']]
    mapped_marital_values = df.marital.map({'Married': 1, 'Non-Married': 0})
    df.loc[:, "marital"] = mapped_marital_values

    df['default'] = [1 if v == 'yes' else 0 for v in df['default']]
    df['housing'] = [1 if v == 'yes' else 0 for v in df['housing']]
    df['loan'] = [1 if v == 'yes' else 0 for v in df['loan']]

    categorical = df.columns.to_list()
    for column in numerical_columns:
        categorical.remove(column)
    print("Possible protected groups are: {}".format(categorical))

    if protected_group == "labels":
        df.loc[:, "protected_group"] = df[target]
    elif protected_group not in categorical:
        raise ValueError(
            f"Invalid protected group {protected_group}. "
            + f"Valid choices are {categorical}."
        )
    else:
        df.loc[:, "protected_group"] = df[protected_group]

    df = sample_by_group_ratios(group_ratios, df, seed)

    # convert to one-hot vectors
    categorical_non_binary = ["job", "education", "contact", "month", "poutcome"]
    df = pd.get_dummies(df, columns=categorical_non_binary)

    return df


def preprocess_credit(df, protected_group, target, group_ratios, seed):
    numerical_columns = ["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4",
                         "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4",
                         "PAY_AMT5", "PAY_AMT6"]
    if protected_group in numerical_columns:
        numerical_columns.remove(protected_group)
    df = normalize(df, numerical_columns)

    df['SEX'] = ['Male' if v == 1 else 'Female' for v in df['SEX']]
    mapped_sex_values = df.SEX.map({"Male": 0, "Female": 1})
    df.loc[:, "SEX"] = mapped_sex_values

    categorical = df.columns.to_list()
    for column in numerical_columns:
        categorical.remove(column)
    print("Possible protected groups are: {}".format(categorical))

    if protected_group == "labels":
        df.loc[:, "protected_group"] = df[target]
    elif protected_group not in categorical:
        raise ValueError(
            f"Invalid protected group {protected_group}. "
            + f"Valid choices are {categorical}."
        )
    else:
        df.loc[:, "protected_group"] = df[protected_group]

    df = sample_by_group_ratios(group_ratios, df, seed)

    # convert to one-hot vectors
    categorical_non_binary = ["EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    df = pd.get_dummies(df, columns=categorical_non_binary)

    return df


def preprocess_compas(df, protected_group, target, group_ratios, seed):
    new_columns = ["age_cat", "race", "sex", "priors_count", "c_charge_degree", "score_text", "v_score_text",
                   "two_year_recid"]
    df = df[new_columns]

    # 让预测为1成为好事
    mapped_two_year_recid_values = df.two_year_recid.map({1: 0, 0: 1})
    df.loc[:, "two_year_recid"] = mapped_two_year_recid_values

    numerical_columns = ["priors_count"]
    if protected_group in numerical_columns:
        numerical_columns.remove(protected_group)
    df = normalize(df, numerical_columns)

    # 去掉了太多的数据
    df = df[(df['race'] == 'African-American') | (df['race'] == "Caucasian")]
    df['race'] = ['Black' if v == 'African-American' else "White" for v in df['race']]
    # 1是弱势群体
    mapped_race_values = df.race.map({"White": 0, "Black": 1})
    df.loc[:, "race"] = mapped_race_values

    df['sex'] = [1 if v == 'Female' else 0 for v in df['sex']]
    df['c_charge_degree'] = [1 if v == 'F' else 0 for v in df['c_charge_degree']]

    categorical = df.columns.to_list()
    for column in numerical_columns:
        categorical.remove(column)
    print("Possible protected groups are: {}".format(categorical))

    if protected_group == "labels":
        df.loc[:, "protected_group"] = df[target]
    elif protected_group not in categorical:
        raise ValueError(
            f"Invalid protected group {protected_group}. "
            + f"Valid choices are {categorical}."
        )
    else:
        df.loc[:, "protected_group"] = df[protected_group]

    df = sample_by_group_ratios(group_ratios, df, seed)

    # convert to one-hot vectors
    categorical_non_binary = ["age_cat", "score_text", "v_score_text"]
    df = pd.get_dummies(df, columns=categorical_non_binary)

    return df


def preprocess_law(df, protected_group, target, group_ratios, seed):
    numerical_columns = ["decile1b", "decile3", "lsat", "ugpa", "zfygpa", "zgpa"]
    if protected_group in numerical_columns:
        numerical_columns.remove(protected_group)
    df = normalize(df, numerical_columns)

    mapped_race_values = df.race.map({"White": 0, "Non-White": 1})
    df.loc[:, "race"] = mapped_race_values

    categorical = df.columns.to_list()
    for column in numerical_columns:
        categorical.remove(column)
    print("Possible protected groups are: {}".format(categorical))

    if protected_group == "labels":
        df.loc[:, "protected_group"] = df[target]
    elif protected_group not in categorical:
        raise ValueError(
            f"Invalid protected group {protected_group}. "
            + f"Valid choices are {categorical}."
        )
    else:
        df.loc[:, "protected_group"] = df[protected_group]

    df = sample_by_group_ratios(group_ratios, df, seed)

    # convert to one-hot vectors
    categorical_non_binary = ["fam_inc", "tier"]
    df = pd.get_dummies(df, columns=categorical_non_binary)

    return df



def get_data_raw(name, data_root, valid_frac, test_frac, seed, protected_group, target, group_ratios):
    if name == 'adult':
        df = pd.read_csv(os.path.join(data_root, 'adult-clean.csv'))
        df_preprocessed = preprocess_adult(df, protected_group, target, group_ratios, seed)
    elif name == 'dutch':
        df = pd.read_csv(os.path.join(data_root, 'dutch.csv'))
        df_preprocessed = preprocess_dutch(df, protected_group, target, group_ratios, seed)
    elif name == 'bank':
        df = pd.read_csv(os.path.join(data_root, 'bank-full.csv'))
        df_preprocessed = preprocess_bank(df, protected_group, target, group_ratios, seed)
    elif name == 'credit':
        df = pd.read_csv(os.path.join(data_root, 'credit-card-clients.csv'))
        df_preprocessed = preprocess_credit(df, protected_group, target, group_ratios, seed)
    elif name == 'compas':
        df = pd.read_csv(os.path.join(data_root, 'compas-scores-two-years_clean.csv'))
        df_preprocessed = preprocess_compas(df, protected_group, target, group_ratios, seed)
    elif name == 'law':
        df = pd.read_csv(os.path.join(data_root, 'law_school_clean.csv'))
        df_preprocessed = preprocess_law(df, protected_group, target, group_ratios, seed)

    train_raw, valid_raw, test_raw = make_tabular_train_valid_test_split(df_preprocessed, valid_frac, test_frac, seed)

    return train_raw, valid_raw, test_raw


def get_tabular_fair_datasets(name, data_root, seed, protected_group, group_ratios=None, make_valid_loader=False):
    if name == "adult":
        #data_fn = get_adult_raw
        target = "Class-label"
    elif name == "dutch":
        #data_fn = get_dutch_raw
        target = "occupation"
    elif name == "bank":
        target = "y"
    elif name == "credit":
        target = "default payment"
    elif name == "compas":
        target = "two_year_recid"
    elif name == "law":
        target = "pass_bar"
    else:
        raise ValueError(f"Unknown dataset {name}")

    valid_frac = 0
    if make_valid_loader:
        valid_frac = 0.1
    test_frac = 0.2
    train_raw, valid_raw, test_raw = get_data_raw(name, data_root, valid_frac, test_frac, seed, protected_group,
                                                  target, group_ratios)

    feature_columns = train_raw.columns.to_list()
    feature_columns.remove(target)
    feature_columns.remove("protected_group")

    train_dset = GroupLabelDataset("train",
                                   torch.tensor(train_raw[feature_columns].values.astype(np.float32), dtype=torch.get_default_dtype()),
                                   torch.tensor(train_raw[target].to_list(), dtype=torch.long),
                                   torch.tensor(train_raw["protected_group"].values.tolist(), dtype=torch.long)
                                   )
    valid_dset = GroupLabelDataset("valid",
                                   torch.tensor(valid_raw[feature_columns].values.astype(np.float32), dtype=torch.get_default_dtype()),
                                   torch.tensor(valid_raw[target].to_list(), dtype=torch.long),
                                   torch.tensor(valid_raw["protected_group"].values.tolist(), dtype=torch.long)
                                   )
    test_dset = GroupLabelDataset("test",
                                  torch.tensor(test_raw[feature_columns].values.astype(np.float32), dtype=torch.get_default_dtype()),
                                  torch.tensor(test_raw[target].to_list(), dtype=torch.long),
                                  torch.tensor(test_raw["protected_group"].values.tolist(), dtype=torch.long)
                                  )

    return train_dset, valid_dset, test_dset

