from bayes_opt import BayesianOptimization
import numpy as np
import random

def is_constraint_satisfied(a):
    # 这里定义你的约束条件
    return actual_function(a) < 100

# 实际的目标函数 f(a)
def actual_function(a):
    # 定义实际的函数 f(a)，例如一个简单的二次函数：
    return (a - 0.3)**2 + 1  # f(a) = (a - 0.3)^2 + 1


# 缓存已评估的点
evaluated_points = {}
# 包装目标函数用于贝叶斯优化
def objective(a):
    # 检查是否已经评估过这个 a 值
    if a in evaluated_points:
        # 对 a 施加一个小的随机扰动，避免完全重复
        a += random.uniform(-1e-1, 1e-1)

    # 如果 a 满足约束条件，返回 a（为了最大化），否则返回负的 f(a)
    if is_constraint_satisfied(a):
        result = -a  # 我们要最大化 a
    else:
        result = -actual_function(a)  # 为了最小化 f(a)

    # 缓存结果
    evaluated_points[a] = result
    return result


# 定义搜索空间，假设 a 的范围在 0 到 1 之间
pbounds = {'a': (0.5, 1)}

# 使用贝叶斯优化
optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=0,
    allow_duplicate_points=True  # 即使允许重复点，我们也会在函数中处理
)

# 开始优化过程
optimizer.maximize(
    init_points=5,  # 初始随机探索的点数
    n_iter=15       # 贝叶斯优化的迭代次数
)

# 检查优化结果中是否有满足约束条件的 a 值
valid_points = [res for res in optimizer.res if is_constraint_satisfied(res['params']['a'])]
#print(valid_points)

if valid_points:
    # 存在满足约束条件的点，找到满足条件时使 a 最大的点
    optimal_result = min(valid_points, key=lambda x: -x['target'])
    optimal_a = optimal_result['params']['a']
    print(f"The optimal value of a that satisfies the constraint is: {optimal_a}")
else:
    # 没有满足约束条件的点，返回使 f(a) 最小的点
    optimal_result = min(optimizer.res, key=lambda x: -x['target'])
    optimal_a = optimal_result['params']['a']
    print(f"No a value satisfies the constraint. The optimal a minimizing f(a) is: {optimal_a}")

# 输出结果
print(f"The corresponding objective function value is: {optimal_result['target']}")
