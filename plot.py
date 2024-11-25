import matplotlib.pyplot as plt
import os
import numpy as np
def myplot_single(file_path, label, color):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            numbers = line.split()
            if len(numbers) >= 3:
                y = float(numbers[1])
                x = float(numbers[2])
                data.append((x, y))
    y_data = [point[0] for point in data]
    x_data = [point[1] for point in data]
    plt.plot(x_data, y_data, marker='x', linestyle='dashdot', color=color, label=label)

# 直接赋值文件路径
size = -1
config_path = 'config.yml'
with open(config_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('n '):
                    _, size = line.split() 
                if line.startswith('plot_dir '):
                    _, plot_dir = line.split()
file_path = plot_dir + f'/result_{size}_10.res'

plt.figure(figsize=(10, 6))

myplot_single(file_path, 'Use_PQ=1;', 'blue')


plt.xlabel('Recall')
plt.ylabel('Average Time')
plt.title(f'Plot of Data(Size = {size})')
plt.legend()
plt.grid(True)

output_path = plot_dir + "/result.png"
plt.savefig(output_path)

# 显示图表
plt.show()