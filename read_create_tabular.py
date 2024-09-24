import pandas as pd
import json
import numpy as np
import csv


def create_df():
    # 创建数据
    data = {
        "method": [],
        #"noise_mul": [],
        #"parameter_norm_bound": [],
        "epsilon": [],
        "accuracy": [],
        "accuracy_per_group_0": [],
        "accuracy_per_group_1": [],
        # "pai0": [],
        # "pai1": [],
        # "pai_all": [],
        # "final_loss": [],
        # "final_loss_per_group_0": [],
        # "final_loss_per_group_1": [],
        # "R0": [],
        # "R1": [],
        # "R_all": [],
        "accuracy_parity": [],
        "demographic_parity": [],
        # "equal_opportunity": [],
        # "equalized_odds": []
    }

    df = pd.DataFrame(data)

    #df.to_csv("empty_data.csv", index=False)
    return df

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    #print(data)
    return data

def get_meanandstd(metrics):
    metrics_mean = {}
    metrics_std = {}

    for key, values in metrics.items():
        metrics_mean[key] = {i: 0 for i in values[0]}
        metrics_std[key] = {i: 0 for i in values[0]}

        m_keys = metrics_mean[key].keys()
        for m in m_keys:
            all_values = [v[m] for v in values]

            metrics_std[key][m] = np.std(all_values)
            metrics_mean[key][m] = np.mean(all_values)

    #print(metrics_mean)
    #print(metrics_std)
    return metrics_mean, metrics_std


def get_pai_diff(pai, method_name):
    pai0, pai1, pai_all = {i: [] for i in method_name}, {i: [] for i in method_name}, {i: [] for i in method_name}
    for key, values in pai.items():
        for i, m in enumerate(method_name):
            if m == 'nonpriv':
                pai0[m].append(0)
                pai1[m].append(0)
                pai_all[m].append(0)
            pai0[m].append(values[i]["accuracy_per_group_0"] - values[0]["accuracy_per_group_0"])
            pai1[m].append(values[i]["accuracy_per_group_1"] - values[0]["accuracy_per_group_1"])
            pai_all[m].append(values[i]["accuracy"] - values[0]["accuracy"])

    return pai0, pai1, pai_all


def get_r_diff(r, method_name):
    r0, r1, r_all = {i: [] for i in method_name}, {i: [] for i in method_name}, {i: [] for i in method_name}
    for key, values in r.items():
        for i, m in enumerate(method_name):
            if m == 'nonpriv':
                r0[m].append(0)
                r1[m].append(0)
                r_all[m].append(0)
            r0[m].append(values[i]["final_loss_per_group_0"] - values[0]["final_loss_per_group_0"])
            r1[m].append(values[i]["final_loss_per_group_1"] - values[0]["final_loss_per_group_1"])
            r_all[m].append(values[i]["final_loss"] - values[0]["final_loss"])

    return r0, r1, r_all


def write_csv(dataset, method_name, parameter_norm_bound, seed):
    test_metrics = {i: [] for i in method_name}
    #final_loss_metrics = {i: [] for i in method_name}
    privacy = {i: [] for i in method_name}

    #pai, r = {i: [] for i in seed}, {i: [] for i in seed}

    for i in range(len(method_name)):
        m = method_name[i]
        p = parameter_norm_bound[i]
        for s in seed:
            if 'separate' in m:
                tms = read_json(f"runs/target_epsilon/{dataset}_{s}/m_{p}/{dataset}_separate/Test_metrics.json")
            else:
                tms = read_json(f"runs/target_epsilon/{dataset}_{s}/{dataset}_{m}/Test_metrics.json")
            test_metrics[m].append(tms)
            # flm = read_json(f"runs/target_epsilon/{dataset}_{s}/{dataset}_{m}/final_loss_metrics.json")
            # final_loss_metrics[m].append(flm)
            if m != 'nonpriv':
                if 'separate' in m:
                    privacy[m].append(read_json(f"runs/target_epsilon/{dataset}_{s}/m_{p}/{dataset}_separate/Privacy_metrics.json"))
                else:
                    privacy[m].append(read_json(f"runs/target_epsilon/{dataset}_{s}/{dataset}_{m}/Privacy_metrics.json"))
                # pai[s].append(tms)
                # r[s].append(flm)
            else:
                privacy[m].append({"epsilon": -1})
            # pai[s].append(tms)
            # r[s].append(flm)

    test_metrics_mean, test_metrics_std = get_meanandstd(test_metrics)
    #final_loss_metrics_mean, final_loss_metrics_std = get_meanandstd(final_loss_metrics)
    privacy_mean, privacy_std = get_meanandstd(privacy)

    # pai0, pai1, pai_all = get_pai_diff(pai, method_name)
    # r0, r1, r_all = get_r_diff(r, method_name)

    df = create_df()
    for method in method_name:
        new_data = {
            "method": method,
            #"noise_mul": noise_multiplier,
            #"parameter_norm_bound": parameter_norm_bound,
            "epsilon": f"{privacy_mean[method]['epsilon']:.3f}", # $\pm$ {privacy_std[method]['epsilon']:.3f}",
            "accuracy": f"{test_metrics_mean[method]['accuracy']:.3f} $\pm$ {test_metrics_std[method]['accuracy']:.3f}",
            "accuracy_per_group_0": f"{test_metrics_mean[method]['accuracy_per_group_0']:.3f} $\pm$ {test_metrics_std[method]['accuracy_per_group_0']:.3f}",
            "accuracy_per_group_1": f"{test_metrics_mean[method]['accuracy_per_group_1']:.3f} $\pm$ {test_metrics_std[method]['accuracy_per_group_1']:.3f}",
            # "final_loss": f"{final_loss_metrics_mean[method]['final_loss']:.4f} ± {final_loss_metrics_std[method]['final_loss']:.2f}",
            # "final_loss_per_group_0": f"{final_loss_metrics_mean[method]['final_loss_per_group_0']:.4f} ± {final_loss_metrics_std[method]['final_loss_per_group_0']:.4f}",
            # "final_loss_per_group_1": f"{final_loss_metrics_mean[method]['final_loss_per_group_1']:.4f} ± {final_loss_metrics_std[method]['final_loss_per_group_1']:.4f}",
            "accuracy_parity": f"{test_metrics_mean[method]['accuracy_parity']:.3f} $\pm$ {test_metrics_std[method]['accuracy_parity']:.3f}",
            #"accuracy_parity": f"{abs(test_metrics_mean[method]['accuracy_per_group_0'] - test_metrics_mean[method]['accuracy_per_group_1']):.4f}",
            "demographic_parity": f"{test_metrics_mean[method]['demographic_parity']:.3f} $\pm$ {test_metrics_std[method]['demographic_parity']:.3f}",
            # "equal_opportunity": f"{test_metrics_mean[method]['equal_opportunity']:.4f} ± {test_metrics_std[method]['equal_opportunity']:.4f}",
            # "equalized_odds": f"{test_metrics_mean[method]['equalized_odds']:.4f} ± {test_metrics_std[method]['equalized_odds']:.4f}",

            # "pai0": f"{np.mean(pai0[method]):.4f} ± {np.std(pai0[method]):.4f}",
            # "pai1": f"{np.mean(pai1[method]):.4f} ± {np.std(pai1[method]):.4f}",
            # "pai_all": f"{np.mean(pai_all[method]):.4f} ± {np.std(pai_all[method]):.4f}",
            # "R0": f"{np.mean(r0[method]):.4f} ± {np.std(r0[method]):.4f}",
            # "R1": f"{np.mean(r1[method]):.4f} ± {np.std(r1[method]):.4f}",
            # "R_all": f"{np.mean(r_all[method]):.4f} ± {np.std(r_all[method]):.4f}",
        }
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

    if method_name[0] == 'separate0':
        df.to_csv(f"runs/target_epsilon/data_separate_{dataset}.csv",
                  index=False, encoding="utf-8-sig")
    else:
        df.to_csv(f"runs/target_epsilon/data_analysis_{dataset}.txt",
                  index=False, sep='&', encoding="utf-8-sig")

def merge_csv(name, parameter_norm_bound):
    file_paths = []
    for p in parameter_norm_bound:
        file_paths.append(f'runs/separate_{p}/data_analysis_{name}_1.csv')

    # 输出文件路径
    output_file_path = f'runs/separate_{name}_1.csv'

    # 打开输出文件并准备写入
    with open(output_file_path, 'w', newline='') as output_file:
        writer = csv.writer(output_file)

        i = 0
        for fp in file_paths:
            # 打开每个输入文件
            with open(fp, 'r') as input_file:
                reader = csv.reader(input_file)

                # 读取第一行
                first_row = next(reader)
                second_row = next(reader, None)
                if i == 0:
                    writer.writerow(first_row)
                    writer.writerow(second_row)
                    for i in range(4):
                        row = next(reader, None)
                        writer.writerow(row)
                    i += 1

                last_row = next(reader, None)
                writer.writerow(last_row)


if __name__ == "__main__":
    dataset = ['adult', 'dutch', 'bank', 'credit', 'compas', 'law']
    seed = [i for i in range(15)]
    #method_name = ['nonpriv', 'dpsgd', 'dpsgdf', 'dpsgdg', 'dpsgdga', 'separate']
    method_name = ['nonpriv', 'dpsgd', 'dpsgdf', 'dpsgdga', 'separate', 'dpsgdt']
    target_epsilon = [2.66, 2.27, 2.84, 3.37, 4.12, 3.68]
    parameter_norm_bound = [None] * len(method_name)

    for d in dataset:
        write_csv(d, method_name, parameter_norm_bound, seed=seed)

    # dataset = ['adult', 'dutch', 'bank', 'credit', 'compas', 'law']
    # parameter_norm_bound = [0.1, 0.2, 0.3, 0.4, 0.5, 1]
    # method_name = ['separate0', 'separate1', 'separate2', 'separate3', 'separate4', 'separate5']
    # seed = [i for i in range(15)]
    # for d in dataset:
    #     write_csv(d, method_name, parameter_norm_bound, seed=seed)


