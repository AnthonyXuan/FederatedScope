# 这个脚本从输出的log中提取test poison acc，并求平均值。
import re

# 文件名
filename = "./poi-log.txt"

# 读取文件内容
with open(filename, "r") as file:
    content = file.readlines()

# 提取test poisoning accuracy
test_poisoning_accuracy = [float(re.search("the test poisoning accuracy: (.*)", line).group(1))
                        for line in content if "the test poisoning accuracy" in line]

# 计算平均值
average = sum(test_poisoning_accuracy) / len(test_poisoning_accuracy)

print(f"Test Poisoning Accuracy Vector: {test_poisoning_accuracy}")
print(f"Average Test Poisoning Accuracy: {average}")
print(f"len:{len(test_poisoning_accuracy)}")