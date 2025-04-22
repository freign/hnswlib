import matplotlib.pyplot as plt
import os
import bisect

def extract_data(file_path):
    ef_list = []
    recall_list = []
    latency_list = []

    with open(file_path, 'r') as file:
        for line in file:
            numbers = line.split()
            if len(numbers) >= 3:
                ef = int(numbers[0])
                recall = float(numbers[1])
                latency_us = float(numbers[2])
                ef_list.append(ef)
                recall_list.append(recall)
                latency_list.append(latency_us)
    i = bisect.bisect_left(recall_list, 0.8)
    print(i)
    ef_list = ef_list[i:]
    recall_list = recall_list[i:]
    latency_list = latency_list[i:]

    qps_list = [1e6 / t for t in latency_list]
    return ef_list, recall_list, qps_list

# 读取 config
plot_dir = './'
with open('config_tiny0.yml', 'r') as file:
    for line in file:
        line = line.strip()
        if line.startswith('plot_dir '):
            _, plot_dir = line.split()

# 要画的文件和标签
inputs = [
    ('/root/tiny_opq/result_1000000_10oro.res', 'HNSW', 'red'),
    ('/root/tiny_opq/result_1000000_10_pq96_4.res', 'HNSWPQ96-4', 'blue')
]

# 创建横向 1x3 子图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = ['QPS vs Recall', 'Recall vs ef', 'QPS vs ef']
xlabels = ['Recall', 'ef', 'ef']
ylabels = ['QPS (Queries per Second)', 'Recall', 'QPS (Queries per Second)']

for file_path, label, color in inputs:
    ef, recall, qps = extract_data(file_path)

    # 子图 1: QPS vs Recall
    axes[0].plot(recall, qps, marker='x', linestyle='dashdot', label=label, color=color)

    # 子图 2: Recall vs ef
    axes[1].plot(ef, recall, marker='x', linestyle='dashdot', label=label, color=color)

    # 子图 3: QPS vs ef
    axes[2].plot(ef, qps, marker='x', linestyle='dashdot', label=label, color=color)

# 设置子图标题、坐标轴和图例
for i in range(3):
    axes[i].set_title(titles[i])
    axes[i].set_xlabel(xlabels[i])
    axes[i].set_ylabel(ylabels[i])
    axes[i].grid(True)
    axes[i].legend()

plt.tight_layout()
output_path = os.path.join(plot_dir, "combined_plot.png")
plt.savefig(output_path)
plt.show()
