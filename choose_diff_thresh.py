import pandas as pd
import json
import numpy as np
import csv

def find_best_index(accs, demops, acps, values, hope_value=0.1):
    # 找到 demops 中所有小于 0.1 的索引
    valid_indices = np.where(demops < hope_value)[0]

    if len(valid_indices) == 0:
        return np.argmin(demops)  # 如果没有符合条件的索引，返回使demop最小的索引值

    # 提取这些索引对应的 values 值
    valid_values = values[valid_indices]
    # 找到最大的 values 值
    max_values_value = np.max(valid_values)
    # 找到最大的 values 值对应的所有索引
    max_values_index = valid_indices[valid_values == max_values_value]

    return max_values_index



def read_csv(path):
    df = pd.read_csv(path)

    # 将每一列的数据存储在各自的列名中
    accs = df['accs'].values
    demops = df['demops'].values
    acps = df['acps'].values
    aps0 = df['acps0'].values
    aps1 = df['acps1'].values
    values = df['values'].values


    return accs, demops, acps, aps0, aps1, values


def create_df():
    # 创建数据
    data = {
        'hope_value': [],
        'accuracy': [],
        'acps0': [],
        'acps1': [],
        'accuracy_parity': [],
        'demographic_parity': [],
    }

    df = pd.DataFrame(data)

    #df.to_csv("empty_data.csv", index=False)
    return df


def write_csv(hope_value, accs, demops, acps, aps0, aps1, path):
    data_accs = get_meanandstd(accs)
    data_acps = get_meanandstd(acps)
    data_demops = get_meanandstd(demops)
    data_aps0 = get_meanandstd(aps0)
    data_aps1 = get_meanandstd(aps1)

    df = create_df()
    for i in range(len(hope_value)):
        data = {
            'hope_value': hope_value[i],
            'accuracy': f"{data_accs[0][i]:.4f} ± {data_accs[1][i]:.4f}",
            'acps0': f"{data_aps0[0][i]:.4f} ± {data_aps0[1][i]:.4f}",
            'acps1': f"{data_aps1[0][i]:.4f} ± {data_aps1[1][i]:.4f}",
            'accuracy_parity': f"{data_acps[0][i]:.4f} ± {data_acps[1][i]:.4f}",
            'demographic_parity': f"{data_demops[0][i]:.4f} ± {data_demops[1][i]:.4f}",
        }
        # data = {
        #     'hope_value': hope_value[i],
        #     'accuracy': f"{data_accs[0][i]:.3f} $\pm$ {data_accs[1][i]:.3f}",
        #     # 'acps0': f"{data_aps0[0][i]:.3f} $\pm$ {data_aps0[1][i]:.3f}",
        #     # 'acps1': f"{data_aps1[0][i]:.3f} $\pm$ {data_aps1[1][i]:.3f}",
        #     'accuracy_parity': f"{data_acps[0][i]:.3f} $\pm$ {data_acps[1][i]:.3f}",
        #     'demographic_parity': f"{data_demops[0][i]:.3f} $\pm$ {data_demops[1][i]:.3f}",
        # }
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

    # 保存到CSV文件
    df.to_csv(path, index=False, encoding="utf-8-sig")


def single_res(dataset, seed, hope_value, base_path):
    w_accs, w_demops, w_acps, w_aps0, w_aps1 = [], [], [], [], []

    for hp in hope_value:
        path = base_path+f"{dataset}_valid_{seed}.csv"
        accs, demops, acps, _, _, values = read_csv(path)
        idx = find_best_index(accs, demops, acps, values, hope_value=hp)

        path = base_path + f"{dataset}_test_{seed}.csv"
        accs, demops, acps, aps0, aps1, _ = read_csv(path)
        w_accs.append(accs[idx].item())
        w_demops.append(demops[idx].item())
        w_acps.append(acps[idx].item())
        w_aps0.append(0)
        w_aps1.append(0)

    path = base_path+f"{dataset}_res_{seed}.csv"
    write_csv(hope_value, w_accs, w_demops, w_acps, w_aps0, w_aps1, path)


def get_meanandstd(data):
    data_mean = []
    data_std = []
    for key, values in data.items():
        mean = np.mean(values)
        std = np.std(values)
        data_mean.append(mean)
        data_std.append(std)

    return data_mean, data_std


def mean_res(dataset, seed, hope_value, base_path):
    w_accs, w_demops, w_acps, w_aps0, w_aps1 = ({i: [] for i in range(len(hope_value))}, {i: [] for i in range(len(hope_value))},
                                                {i: [] for i in range(len(hope_value))}, {i: [] for i in range(len(hope_value))},
                                                {i: [] for i in range(len(hope_value))})
    for s in seed:
        path = base_path + f"{dataset}_valid_{s}.csv"
        accs_valid, demops_valid, acps_valid, _, _, values_valid = read_csv(path)

        path = base_path + f"{dataset}_test_{s}.csv"
        accs, demops, acps, aps0, aps1, real_values = read_csv(path)

        for i in range(len(hope_value)):
            hp = hope_value[i]
            idx = find_best_index(accs_valid, demops_valid, acps_valid, values_valid, hope_value=hp)
            w_accs[i].append(accs[idx].item())
            w_demops[i].append(demops[idx].item())
            w_acps[i].append(acps[idx].item())
            w_aps0[i].append(aps0[idx].item())
            w_aps1[i].append(aps1[idx].item())


    path = base_path + f"{dataset}_res_mean.csv"
    write_csv(hope_value, w_accs, w_demops, w_acps, w_aps0, w_aps1, path)


if __name__ == "__main__":
    #dataset = ['adult', 'dutch', 'bank', 'credit']
    dataset = ['adult', 'dutch', 'bank', 'credit', 'compas', 'law']
    seed = [i for i in range(15)]
    base_path = 'runs/dpsgdp_res_min/'
    hope_value = [i for i in np.arange(0.00, 0.21, 0.01)]
    #hope_value = [0.05]

    for d in dataset:
        mean_res(d, seed, hope_value, base_path)
