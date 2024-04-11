import pandas as pd
import matplotlib.pyplot as plt

# 读取日志文件
with open('/root/autodl-tmp/shijieqi/GeneFacePlusPlus/checkpoints/motion2video_nerf/LiaoXueMin_torso/terminal_logs/log_20240407231144.txt', 'r') as file:
    lines = file.readlines()

# 处理数据
data = {}
for line in lines:
    if 'Validation results' in line:
        parts = line.split()
        step = int(parts[2].split('@')[1].strip(':'))  # 获取步数
        results = {}
        for i in range(3, len(parts), 2):  # 从第四个元素开始，每两个元素为一组
            key = parts[i].strip("':,{")  # 删除键的多余字符
            value = float(parts[i + 1].strip("',}"))  # 删除值的多余字符，并转换为浮点数
            results[key] = value
        data[step] = results

# 转换成DataFrame
df = pd.DataFrame.from_dict(data, orient='index')

# 绘制图表
for column in df.columns:
    plt.figure()
    plt.plot(df.index, df[column])
    plt.title(column)
    plt.xlabel('Steps')
    plt.ylabel(column)
    plt.grid(True)
    plt.show()