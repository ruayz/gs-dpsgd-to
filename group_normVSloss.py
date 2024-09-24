import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot


def read_csv_norm(path):
    df = pd.read_csv(path)
    # 将每一列的数据存储在各自的列名中
    grad_norm0 = df['ave_grads_0'].values
    grad_norm1 = df['ave_grads_1'].values
    return grad_norm0, grad_norm1


def read_csv_loss(path):
    df = pd.read_csv(path)
    # 将每一列的数据存储在各自的列名中
    train_loss = df['train_loss'].values
    train_loss0 = df['train_loss_0'].values
    train_loss1 = df['train_loss_1'].values
    return train_loss, train_loss0, train_loss1


def read_csv_acc(path):
    df = pd.read_csv(path)
    # 将每一列的数据存储在各自的列名中
    train_acc = df['train_acc'].values
    train_acc0 = df['train_acc_0'].values
    train_acc1 = df['train_acc_1'].values
    return train_acc, train_acc0, train_acc1


def base_plot(variable0, variable1, ylabel, method, base_path, acc=None):
    x = [i for i in range(len(variable0))]
    #color = ['limegreen', 'skyblue', 'tomato']
    palette = pyplot.get_cmap('Set1')
    color = [palette(0), palette(1), palette(2)]

    plt.figure()
    plt.plot(x, variable0, marker='o', linestyle='--', color=color[1], label='group0')
    plt.plot(x, variable1, marker='v', linestyle='--', color=color[2], label='group1')
    if acc is not None:
        plt.plot(x, acc, marker='s', linestyle='--', color=color[0], label='accuracy')
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.title(method)
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig(base_path+ylabel+".png", dpi=400, bbox_inches='tight')
    plt.close()


def plot_acc_parity(variable, base_path):
    x = [i for i in range(len(variable[0]))]
    palette = pyplot.get_cmap('Set1')
    color = [palette(0), palette(1), palette(2), palette(3), palette(4), palette(5)]

    plt.figure()
    plt.plot(x, variable[0], color=color[0], label='separate')
    plt.plot(x, variable[1], color=color[1], label='nonpriv')
    plt.plot(x, variable[2], color=color[2], label='dpsgd')
    plt.plot(x, variable[3], color=color[3], label='dpsgdf')
    plt.plot(x, variable[4], color=color[4], label='dpsgdga')
    #plt.plot(x, variable[5], color=color[5], label='single')

    plt.xlabel('Epochs')
    plt.ylabel("Accuracy parity")
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig(base_path+"acc_parity.png", dpi=400, bbox_inches='tight')
    plt.close()


def img_gn(dataset, seed, method):
    if method != 'single':
        if method == 'separate':
            path = f"runs/target_epsilon/{dataset}_{seed}/m_None/{dataset}_separate/avg_grad_norms_per_epochs.csv"
        else:
            path = f"runs/target_epsilon/{dataset}_{seed}/{dataset}_{method}/avg_grad_norms_per_epochs.csv"

        variable0, variable1 = read_csv_norm(path)
    else:
        path0 = f"runs/target_epsilon/{dataset}_{seed}/{dataset}_dpsgdt/0/avg_grad_norms_per_epochs.csv"
        path1 = f"runs/target_epsilon/{dataset}_{seed}/{dataset}_dpsgdt/1/avg_grad_norms_per_epochs.csv"
        df0 = pd.read_csv(path0)
        variable0 = df0['ave_grads_0'].values
        df1 = pd.read_csv(path1)
        variable1 = df1['ave_grads_0'].values

    base_path = "runs/group_specific/" + dataset + "/" + method + "/"
    base_plot(variable0, variable1, "Gradient norm", method, base_path)


def img_acc(dataset, seed, method):
    if method != 'single':
        if method == 'separate':
            path = f"runs/target_epsilon/{dataset}_{seed}/m_None/{dataset}_separate/train_acc_per_epochs.csv"
        else:
            path = f"runs/target_epsilon/{dataset}_{seed}/{dataset}_{method}/train_acc_per_epochs.csv"

        acc, variable0, variable1 = read_csv_acc(path)
    else:
        path0 = f"runs/target_epsilon/{dataset}_{seed}/{dataset}_dpsgdt/0/train_acc_per_epochs.csv"
        path1 = f"runs/target_epsilon/{dataset}_{seed}/{dataset}_dpsgdt/1/train_acc_per_epochs.csv"
        df0 = pd.read_csv(path0)
        variable0 = df0['train_acc'].values
        df1 = pd.read_csv(path1)
        variable1 = df1['train_acc'].values
        acc = None

    base_path = "runs/group_specific/" + dataset + "/" + method + "/"
    base_plot(variable0, variable1, "Accuracy", method, base_path, acc)

    acc_parity = [abs(variable0[i] - variable1[i]) for i in range(len(variable0))]
    #plot_loss_parity(loss_parity, method, base_path)
    return acc_parity


if __name__ == "__main__":
    #dataset = ['adult', 'dutch', 'bank', 'credit', 'compas', 'law']
    dataset = ['adult']
    #method = ['separate', 'nonpriv', 'dpsgd', 'dpsgdf', 'dpsgdga', 'single']
    method = ['separate', 'nonpriv', 'dpsgd', 'dpsgdf', 'dpsgdga']
    seed = 100

    for d in dataset:
        acc_parities = []
        for m in method:
            img_gn(d, seed, m)
            ap = img_acc(d, seed, m)
            acc_parities.append(ap)

        base_path = "runs/group_specific/" + d + "/"
        plot_acc_parity(acc_parities, base_path)


