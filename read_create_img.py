import pandas as pd
import json
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import pyplot


def change(metrics):
    res = []
    for metric in metrics:
        metric = metric.split()
        res.append([float(metric[0]), float(metric[2])])

    return np.array(res)

def read_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 创建数据
    accuracy = change(df["accuracy"])
    accuracy_parity = change(df["accuracy_parity"])
    demographic_parity = change(df["demographic_parity"])

    return accuracy, accuracy_parity, demographic_parity

def plot_figure(x, y, color, dataset_name, metric_name, ap_dpsgd=None, base_path=None):
    plt.figure(figsize=[8, 6])
    #plt.style.use('seaborn-whitegrid')
    #plt.grid(True)

    plt.plot(x[0][:, 0], y[0][:, 0], color=color[0], linewidth=2.5, label='GS-DP-SGD-TO')
    plt.fill_between(x[0][:, 0], y[0][:, 0] - y[0][:, 1], y[0][:, 0] + y[0][:, 1], color=color[0], alpha=0.2)

    plt.plot(x[1][:, 0], y[1][:, 0], color=color[1], linewidth=2.5, label='DP-SGD-P')
    plt.fill_between(x[1][:, 0], y[1][:, 0] - y[1][:, 1], y[1][:, 0] + y[1][:, 1], color=color[1], alpha=0.2)

    plt.plot(x[2][:, 0], y[2][:, 0], color=color[2], linewidth=2.5, label='FairDP')
    plt.fill_between(x[2][:, 0], y[2][:, 0] - y[2][:, 1], y[2][:, 0] + y[2][:, 1], color=color[2], alpha=0.2)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20) #刻度值大小
    plt.xlabel(r"DemParity$\downarrow$", fontsize=28)
    if metric_name == "accuracy parity":
        plt.ylabel(r"AccParity$\downarrow$", fontsize=28)
        plt.axhline(y=ap_dpsgd, color='gray', linestyle='--', linewidth=2, alpha=0.5, zorder=0)
        path = base_path + "ap.pdf"
    else:
        plt.ylabel(r"Acc$\uparrow$", fontsize=28)
        path = base_path + "acc.pdf"
    plt.axvline(x=0.1, color='gray', linestyle='--', linewidth=2, alpha=0.5, zorder=0)
    plt.legend(fontsize=18)
    plt.title(dataset_name.capitalize(), fontsize=28)
    plt.tight_layout()  # 自动调整布局以适应图形区域
    plt.savefig(path, dpi=400, bbox_inches='tight')
    plt.close()

def main(d, path_t, path_p, path_s, color, ap_dpsgd):
    acc0, ap0, dp0 = read_csv(path_t)
    acc1, ap1, dp1 = read_csv(path_p)
    acc2, ap2, dp2 = read_csv(path_s)

    acc = [acc0, acc1, acc2]
    ap = [ap0, ap1, ap2]
    dp = [dp0, dp1, dp2]

    base_path = "runs/image/" + d + "/"
    plot_figure(dp, ap, color, d, "accuracy parity", ap_dpsgd, base_path)
    plot_figure(dp, acc, color, d, "accuracy", base_path=base_path)


if __name__ == "__main__":
    dataset = ['adult', 'dutch', 'bank', 'credit', 'compas', 'law']
    #dataset = ['adult']
    #seed = [i for i in range(4)]
    method_name = ['dpsgdgt', 'dpsgdp', 'separate']
    #epsilon = [2.66, 2.27, 2.84, 3.37]
    #demops = [i for i in np.arange(0.00, 0.21, 0.01)]
    parameter_norm_bound = [0.1, 0.2, 0.3, 0.4, 0.5, 1, None]
    ap_dpsgd = [0.154, 0.111, 0.050, 0.031, 0.041, 0.202]

    #color = ['tomato', 'limegreen', 'skyblue']
    palette = pyplot.get_cmap('Set1')
    color = [palette(0), palette(1), palette(2)]

    for i in range(len(dataset)):
        d = dataset[i]
        ap = ap_dpsgd[i]
        path_p = f'runs/dpsgdp_res_min/{d}_res_mean.csv'
        path_t = f'runs/dpsgdt_res_min/{d}_res_mean.csv'
        path_s = f'runs/target_epsilon/data_separate_{d}.csv'

        main(d, path_t, path_p, path_s, color, ap)


