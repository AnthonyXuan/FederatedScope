import torch
import torch.nn as nn
import torch.optim as optim

# 定义模型
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleClassifier, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

# 数据准备
input_size = 10
output_size = 2
batch_size = 64

# 构造随机输入和标签
input_data = torch.randn(batch_size, input_size)
target_labels = torch.randint(output_size, (batch_size,))

# 创建模型实例
model = SimpleClassifier(input_size, output_size)

# 定义损失函数（criterion）
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 前向传播和损失计算
outputs = model(input_data)
loss = criterion(torch.empty(0), torch.empty(0))

# 反向传播和梯度更新
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 打印损失
print("Loss:", loss.item())