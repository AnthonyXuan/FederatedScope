import torch

# 创建一维张量
tensor = torch.tensor([4, 2, 9, 1, 7, 5])

# 指定要找到的最小值的个数
topk_count = 3

# 使用torch.topk()找到最小的前topk_count个值及其索引
topk_values, topk_indices = torch.topk(tensor, k=topk_count, largest=True, sorted=False)

# 打印最小的前topk_count个值及其索引
print("Top", topk_count, "values:", topk_values)
print("Top", topk_count, "indices:", topk_indices)