import torch
import torch.nn.functional as F
#import numpy as np
from opacus.grad_sample.grad_sample_module import GradSampleModule

from utils import split_by_group, save_thresh_res
from bayes_opt import BayesianOptimization

# unprivileged_group = 0.5
# privileged_group = 0.5


def find_best_index(accs, demops, acps, values, method, hope_value=0.1):
    # 找到 demops 中所有小于 0.1 的索引
    valid_indices = torch.where(demops < hope_value)[0]

    if len(valid_indices) == 0:
        return torch.argmin(demops)  # 如果没有符合条件的索引，返回使demop最小的索引值

    if method == 'dpsgdp':
        # 提取这些索引对应的 values 值
        valid_values = values[valid_indices]
        min_values_value = torch.min(valid_values)
        values_index = valid_indices[valid_values == min_values_value]
    else:
        # 提取这些索引对应的 values 值
        valid_values = values[valid_indices]
        # 找到最大的 values 值
        max_values_value = torch.max(valid_values)
        # 找到最大的 values 值对应的所有索引
        values_index = valid_indices[valid_values == max_values_value]

    return values_index


def choose_thresholds(model, dataloader, unprivileged_group=0.5, privileged_group=0.5, **kwargs):
    method = "dpsgdp" if kwargs["method"]=='dpsgd-post' else "dpsgdt"

    outputs = torch.tensor([])
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        for _batch_idx, (data, labels, group) in enumerate(dataloader[0]):
            data, labels = data.to(device), labels.to(device)
            if method == "dpsgdt":
                temp = torch.sigmoid(model(data)).view(-1)
                outputs = torch.cat((outputs, temp))
            else:
                temp = F.softmax(model(data), dim=1)
                outputs = torch.cat((outputs, temp))

    if method == "dpsgdt":
        # outputs = outputs[outputs <= 0.5]
        # values = torch.linspace(outputs.min().item(), 0.5, 100)
        # values = values[values != outputs.min()]
        values = torch.linspace(0.5, outputs.max().item(), 100)
    else:
        values = torch.linspace(0.5, torch.max(outputs), 100)

    accs = []
    demops = []
    acps = []
    aps0 = []
    aps1 = []
    for v in values:
        unprivileged_group = 1 - v
        accs.append(accuracy(model, dataloader[0], unprivileged_group, v, **kwargs))
        demops.append(demographic_parity(model, dataloader[0], unprivileged_group, v, **kwargs))
        #acps.append(accuracy_parity(model, dataloader[0], unprivileged_group, v, **kwargs))
        ap_list = accuracy_per_group(model, dataloader[0], unprivileged_group, v, **kwargs)
        acps.append(abs(ap_list[0] - ap_list[1]))
        aps0.append(ap_list[0])
        aps1.append(ap_list[1])
    # 画出dp和acc的曲线
    # print(accs)
    # print(demops)
    # print(acps)
    #draw_fairandper(accs, demops, acps)
    accs, demops, acps = torch.tensor(accs), torch.tensor(demops), torch.tensor(acps)
    idx = find_best_index(accs, demops, acps, values, method)
    save_thresh_res(accs, demops, acps, aps0, aps1, values,
                    path=f'runs/{method}_res_min/{kwargs["dataset"]}/{kwargs["dataset"]}_valid_{kwargs["seed"]}.csv')

    accs = []
    demops = []
    acps = []
    aps0 = []
    aps1 = []
    for v in values:
        unprivileged_group = 1 - v
        accs.append(accuracy(model, dataloader[1], unprivileged_group, v, **kwargs))
        demops.append(demographic_parity(model, dataloader[1], unprivileged_group, v, **kwargs))
        # acps.append(accuracy_parity(model, dataloader[1], unprivileged_group, v, **kwargs))
        ap_list = accuracy_per_group(model, dataloader[1], unprivileged_group, v, **kwargs)
        acps.append(abs(ap_list[0] - ap_list[1]))
        aps0.append(ap_list[0])
        aps1.append(ap_list[1])
    save_thresh_res(accs, demops, acps, aps0, aps1, values,
                    path=f'runs/{method}_res_min/{kwargs["dataset"]}/{kwargs["dataset"]}_test_{kwargs["seed"]}.csv')

    print(f"values[idx]:{values[idx].item()}")
    return 1-values[idx].item(), values[idx].item()


def bayes_thresholds(model, dataloader, unprivileged_group=0.5, privileged_group=0.5, **kwargs):
    method = "dpsgdp" if kwargs["method"]=='dpsgd-post' else "dpsgdt"

    def is_constraint_satisfied(threshold):
        return actual_function(threshold) <= 0.05

    accs = []
    demops = []
    acps = []
    aps0 = []
    aps1 = []
    thresholds = []
    def actual_function(threshold):
        privileged_group = threshold
        unprivileged_group = 1 - threshold
        thresholds.append(threshold)
        accs.append(accuracy(model, dataloader[0], unprivileged_group, privileged_group, **kwargs))
        demop = demographic_parity(model, dataloader[0], unprivileged_group, privileged_group, **kwargs)
        demops.append(demop)
        # acps.append(accuracy_parity(model, dataloader[0], unprivileged_group, privileged_group, **kwargs))
        ap_list = accuracy_per_group(model, dataloader[0], unprivileged_group, privileged_group, **kwargs)
        acps.append(abs(ap_list[0] - ap_list[1]))
        aps0.append(ap_list[0])
        aps1.append(ap_list[1])
        return demop

    # 缓存已评估的点
    evaluated_points = {}
    def objective(threshold):
        if threshold in evaluated_points:
            #print(f"Using cached value for a = {a}")
            return evaluated_points[threshold]

        # 如果 a 满足约束条件，返回 阈值（为了最小化）
        if is_constraint_satisfied(threshold):
            result = -threshold
        else:
            result = -actual_function(threshold)*100  # 为了最小化 demop

        evaluated_points[threshold] = result
        return result

    # 定义搜索空间
    pbounds = {'threshold': (0.5, 1)}
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=0,
        allow_duplicate_points=True  # 允许重复点
    )
    optimizer.maximize(
        init_points=50,  # 初始随机探索的点数
        n_iter=50  # 贝叶斯优化的迭代次数
    )
    # 检查优化结果中是否有满足约束条件的 a 值
    valid_points = [res for res in optimizer.res if is_constraint_satisfied(res['params']['threshold'])]

    if valid_points:
        # 存在满足约束条件的点，找到满足条件时使 阈值 最小的点
        optimal_result = min(valid_points, key=lambda x: -x['target'])
        optimal_a = optimal_result['params']['threshold']
        print(f"The optimal value of a that satisfies the constraint is: {optimal_a}")

        demop = demographic_parity(model, dataloader[0], 1-optimal_a, optimal_a, **kwargs)
        print(f"The corresponding objective function value is: {demop}")
    else:
        # 没有满足约束条件的点，返回使f最小的点
        optimal_result = min(optimizer.res, key=lambda x: -x['target'])
        optimal_a = optimal_result['params']['threshold']
        print(f"No a value satisfies the constraint. The optimal a minimizing f is: {optimal_a}")
        print(f"The corresponding objective function value is: {optimal_result['target']}")


    accs, demops, acps = torch.tensor(accs), torch.tensor(demops), torch.tensor(acps)
    save_thresh_res(accs, demops, acps, aps0, aps1, thresholds,
                    path=f'runs/{method}_res_min/{kwargs["dataset"]}/{kwargs["dataset"]}_valid_{kwargs["seed"]}.csv')

    return 1-optimal_a, optimal_a


def accuracy(model, dataloader, unprivileged_group=0.5, privileged_group=0.5, **kwargs):
    correct = 0
    total = 0
    #print(f"unprivileged_group:{unprivileged_group}")
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            if kwargs["method"] == "separate" or kwargs["method"] == "dpsgd-thresh":
                # separate method: sigmoid
                outputs = torch.sigmoid(model(data))
                thresholds = torch.where(group == 1, unprivileged_group, privileged_group).reshape(outputs.shape)
                predicted = (outputs > thresholds).long().view(-1)
            else:
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                if kwargs["method"] == "dpsgd-post":
                    outputs = F.softmax(outputs, dim=1)
                    # for idx in range(len(outputs)):
                    #     max_val = max(outputs[idx][1].item(), 1 - outputs[idx][1].item())
                    #     if max_val < unprivileged_group:
                    #         if group[idx].item() == 0:
                    #             predicted[idx] = 0
                    #         elif group[idx].item() == 1:
                    #             predicted[idx] = 1
                    thresholds = torch.where(group == 1, unprivileged_group, privileged_group).reshape(outputs[:,1].shape)
                    predicted = (outputs[:,1] > thresholds).long().view(-1)

            total += labels.size(0)
            correct += (predicted == labels).sum()
    return (correct / total).item()


def accuracy_per_group(model, dataloader, unprivileged_group=0.5, privileged_group=0.5, num_groups=None, method=None, **kwargs):
    correct_per_group = [0] * num_groups
    total_per_group = [0] * num_groups
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)

            per_group = split_by_group(data, labels, group, num_groups)
            for i, group in enumerate(per_group):
                data_group, labels_group = group
                if method == "separate" or method == "dpsgd-thresh":
                    # separate method: sigmoid
                    outputs = torch.sigmoid(model(data_group))
                    if i == 1:
                        threshold = unprivileged_group
                    else:
                        threshold = privileged_group
                    predicted = (outputs > threshold).long().view(-1)
                else:
                    outputs = model(data_group)
                    _, predicted = torch.max(outputs, 1)
                    if method == "dpsgd-post":
                        outputs = F.softmax(outputs, dim=1)
                        # for idx in range(len(outputs)):
                        #     max_val = max(outputs[idx][1].item(), 1 - outputs[idx][1].item())
                        #     if max_val < unprivileged_group:
                        #         if group[idx].item() == 0:
                        #             predicted[idx] = 0
                        #         elif group[idx].item() == 1:
                        #             predicted[idx] = 1
                        if i == 1:
                            threshold = unprivileged_group
                        else:
                            threshold = privileged_group
                        predicted = (outputs[:, 1] > threshold).long().view(-1)
                total_per_group[i] += labels_group.size(0)
                correct_per_group[i] += (predicted == labels_group).sum()
    return [float(correct_per_group[i] / total_per_group[i]) for i in range(num_groups)]


def accuracy_parity(model, dataloader, unprivileged_group=0.5, privileged_group=0.5, num_groups=None, method=None, **kwargs):
    ap_list = accuracy_per_group(model, dataloader, unprivileged_group, privileged_group, num_groups, method)
    ap = abs(ap_list[0] - ap_list[1])
    return ap


def demographic_parity(model, dataloader, unprivileged_group=0.5, privileged_group=0.5, num_groups=None, **kwargs):
    pre1_per_group = [0] * num_groups
    total_per_group = [0] * num_groups
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)

            per_group = split_by_group(data, labels, group, num_groups)
            for i, group in enumerate(per_group):
                data_group, labels_group = group
                labels_group_ones = torch.ones_like(labels_group)
                #print("labels_group_ones:", labels_group_ones)
                if kwargs["method"] == "separate" or kwargs["method"] == "dpsgd-thresh":
                    # separate method: sigmoid
                    outputs = torch.sigmoid(model(data_group))
                    if i == 1:
                        threshold = unprivileged_group
                    else:
                        threshold = privileged_group
                    predicted = (outputs > threshold).long().view(-1)
                else:
                    outputs = model(data_group)
                    _, predicted = torch.max(outputs, 1)
                    if kwargs["method"] == "dpsgd-post":
                        outputs = F.softmax(outputs, dim=1)
                        # for idx in range(len(outputs)):
                        #     max_val = max(outputs[idx][1].item(), 1 - outputs[idx][1].item())
                        #     if max_val < unprivileged_group:
                        #         if group[idx].item() == 0:
                        #             predicted[idx] = 0
                        #         elif group[idx].item() == 1:
                        #             predicted[idx] = 1
                        if i == 1:
                            threshold = unprivileged_group
                        else:
                            threshold = privileged_group
                        predicted = (outputs[:, 1] > threshold).long().view(-1)
                total_per_group[i] += labels_group.size(0)
                pre1_per_group[i] += (predicted == labels_group_ones).sum()

        # print(f"group0:{pre1_per_group[0]/ total_per_group[0]}")
        # print(f"group1:{pre1_per_group[1]/ total_per_group[1]}")
    return abs(float(pre1_per_group[0] / total_per_group[0]) - float(pre1_per_group[1] / total_per_group[1]))


def equal_opportunity(model, dataloader, unprivileged_group=0.5, privileged_group=0.5, num_groups=None, **kwargs):
    tpr_per_group = [0] * num_groups
    total_positive_per_group = [0] * num_groups
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)

            per_group = split_by_group(data, labels, group, num_groups)
            for i, group in enumerate(per_group):
                data_group, labels_group = group
                # Only consider the true positives (labels=1)
                positive_indices = (labels_group == 1)
                data_group = data_group[positive_indices]
                labels_group = labels_group[positive_indices]

                if data_group.size(0) == 0:  # No positive samples in this group for this batch
                    continue

                if kwargs["method"] == "separate" or kwargs["method"] == "dpsgd-thresh":
                    outputs = torch.sigmoid(model(data_group))
                    if i == 1:
                        threshold = unprivileged_group
                    else:
                        threshold = privileged_group
                    predicted = (outputs > threshold).long().view(-1)
                else:
                    outputs = model(data_group)
                    _, predicted = torch.max(outputs, 1)
                    if kwargs["method"] == "dpsgd-post":
                        outputs = F.softmax(outputs, dim=1)
                        # for idx in range(len(outputs)):
                        #     max_val = max(outputs[idx][1].item(), 1 - outputs[idx][1].item())
                        #     if max_val < unprivileged_group:
                        #         if group[idx].item() == 0:
                        #             predicted[idx] = 0
                        #         elif group[idx].item() == 1:
                        #             predicted[idx] = 1
                        if i == 1:
                            threshold = unprivileged_group
                        else:
                            threshold = privileged_group
                        predicted = (outputs[:, 1] > threshold).long().view(-1)

                total_positive_per_group[i] += labels_group.size(0)
                tpr_per_group[i] += (predicted == labels_group).sum().item()

    tpr_per_group = [tpr / total if total > 0 else 0 for tpr, total in zip(tpr_per_group, total_positive_per_group)]
    return abs(tpr_per_group[0] - tpr_per_group[1])


def equalized_odds(model, dataloader, unprivileged_group=0.5, privileged_group=0.5, num_groups=None, **kwargs):
    tpr_per_group = [0] * num_groups
    fpr_per_group = [0] * num_groups
    total_positive_per_group = [0] * num_groups
    total_negative_per_group = [0] * num_groups
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)

            per_group = split_by_group(data, labels, group, num_groups)
            for i, group in enumerate(per_group):
                data_group, labels_group = group

                if kwargs["method"] == "separate" or kwargs["method"] == "dpsgd-thresh":
                    outputs = torch.sigmoid(model(data_group))
                    if i == 1:
                        threshold = unprivileged_group
                    else:
                        threshold = privileged_group
                    predicted = (outputs > threshold).long().view(-1)
                else:
                    outputs = model(data_group)
                    _, predicted = torch.max(outputs, 1)
                    if kwargs["method"] == "dpsgd-post":
                        outputs = F.softmax(outputs, dim=1)
                        # for idx in range(len(outputs)):
                        #     max_val = max(outputs[idx][1].item(), 1 - outputs[idx][1].item())
                        #     if max_val < unprivileged_group:
                        #         if group[idx].item() == 0:
                        #             predicted[idx] = 0
                        #         elif group[idx].item() == 1:
                        #             predicted[idx] = 1
                        if i == 1:
                            threshold = unprivileged_group
                        else:
                            threshold = privileged_group
                        predicted = (outputs[:, 1] > threshold).long().view(-1)

                total_positive_per_group[i] += (labels_group == 1).sum().item()
                total_negative_per_group[i] += (labels_group == 0).sum().item()

                tpr_per_group[i] += ((predicted == 1) & (labels_group == 1)).sum().item()
                fpr_per_group[i] += ((predicted == 1) & (labels_group == 0)).sum().item()

    tpr_per_group = [tpr / total if total > 0 else 0 for tpr, total in zip(tpr_per_group, total_positive_per_group)]
    fpr_per_group = [fpr / total if total > 0 else 0 for fpr, total in zip(fpr_per_group, total_negative_per_group)]

    tpr_diff = abs(tpr_per_group[0] - tpr_per_group[1])
    fpr_diff = abs(fpr_per_group[0] - fpr_per_group[1])

    return tpr_diff+fpr_diff


def macro_accuracy(model, dataloader, unprivileged_group=0.5, privileged_group=0.5, num_classes=None, **kwargs):
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            if kwargs["method"] == "separate" or kwargs["method"] == "dpsgd-thresh":
                # separate method: sigmoid
                outputs = torch.sigmoid(model(data))
                # 根据group值设置阈值
                thresholds = torch.where(group == 1, unprivileged_group, privileged_group).reshape(outputs.shape)
                predicted = (outputs > thresholds).long().view(-1)
            else:
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                if kwargs["method"] == "dpsgd-post":
                    outputs = F.softmax(outputs, dim=1)
                    # for idx in range(len(outputs)):
                    #     max_val = max(outputs[idx][1].item(), 1 - outputs[idx][1].item())
                    #     if max_val < unprivileged_group:
                    #         if group[idx].item() == 0:
                    #             predicted[idx] = 0
                    #         elif group[idx].item() == 1:
                    #             predicted[idx] = 1
                    thresholds = torch.where(group == 1, unprivileged_group, privileged_group).reshape(outputs[:,1].shape)
                    predicted = (outputs[:, 1] > thresholds).long().view(-1)
            for true_p, all_p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[true_p.long(), all_p.long()] += 1

    accs = confusion_matrix.diag() / confusion_matrix.sum(1)
    return accs.mean().item()


