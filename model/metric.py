import torch

# 计算准确率的函数
def accuracy(output, target):
    with torch.no_grad():  # 禁用梯度计算，提高效率
        pred = torch.argmax(output, dim=1)  # 获取每个样本的预测类别（按最大值选择）
        assert pred.shape[0] == len(target)  # 确保预测结果与目标标签数量相等
        correct = 0
        correct += torch.sum(pred == target).item()  # 计算正确预测的个数
    return correct / len(target)  # 返回正确预测比例（即准确率）

# 计算 top-k 准确率的函数
def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]  # 获取每个样本的 top k 个预测类别（返回 top k 类别的索引）
        assert pred.shape[0] == len(target)  # 确保预测结果与目标标签数量相等
        correct = 0
        for i in range(k):  # 对每个 top k 预测类别进行检查
            correct += torch.sum(pred[:, i] == target).item()  # 计算目标标签是否出现在 top k 中
    return correct / len(target)  # 返回 top-k 准确率

# 计算精度、召回率和 F1 分数的函数
# 计算混淆矩阵
import torch


# 计算混淆矩阵


#
# 计算精度

def precision(output, target, num_classes=40):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)  # 获取每个样本的预测类别（按最大值选择）
        assert pred.shape[0] == len(target)  # 确保预测结果与目标标签数量相等

        # 初始化 TP 和 FP
        tp = torch.zeros(num_classes)
        fp = torch.zeros(num_classes)

        for i in range(len(target)):
            if pred[i] == target[i]:
                tp[target[i]] += 1  # 真阳性
            else:
                fp[pred[i]] += 1  # 假阳性

        precision_score = (tp / (tp + fp + 1e-8))  # 防止除以零的情况
        return precision_score.mean().item()  # 返回精确率的平均值


# 计算召回率
def recall(output, target, num_classes=40):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)  # 获取每个样本的预测类别（按最大值选择）
        assert pred.shape[0] == len(target)  # 确保预测结果与目标标签数量相等

        # 初始化 TP 和 FN
        tp = torch.zeros(num_classes)
        fn = torch.zeros(num_classes)

        for i in range(len(target)):
            if pred[i] == target[i]:
                tp[target[i]] += 1  # 真阳性
            else:
                fn[target[i]] += 1  # 假阴性

        recall_score = (tp / (tp + fn + 1e-8))  # 防止除以零的情况
        return recall_score.mean().item()  # 返回召回率的平均值


def f1_score(output, target, num_classes=40):
    precision_scores = precision(output, target, num_classes)
    recall_scores = recall(output, target, num_classes)

    # Calculate the F1 scores for each class
    f1_scores = 2 * (precision_scores * recall_scores) / (precision_scores + recall_scores + 1e-8)

    # If f1_scores is a tensor (array of values), return the mean, otherwise return the scalar F1 score directly
    if isinstance(f1_scores, float):
        return f1_scores  # Return as is if it's already a single scalar value
    else:
        return f1_scores.mean().item()  # Return the average if it's an array or tensor




