import json
import matplotlib.pyplot as plt

# 提供了日志文件的路径
logfile = 'eval_results.log'

# 提供了你想要绘制的变量的路径
variable_path = ['Results_weighted_avg', 'test_acc']

# 读取日志文件
with open(logfile, 'r') as f:
    logs = [json.loads(line) for line in f]

# 提取训练轮数和你选择的变量
rounds = [log['Round'] for log in logs]
values = [log[variable_path[0]][variable_path[1]] for log in logs]

# 绘制图形
plt.plot(rounds, values)
plt.xlabel('Round')
plt.ylabel('.'.join(variable_path))
plt.savefig('nana.png')
plt.show()