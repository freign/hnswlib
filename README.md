### Usage
首先进入项目目录，按照需求修改项目目录下的config.yml
```
use_pq 1  # 1代表使用pq，0代表不使用
m 320  # m表示量化的子空间维度
nbits 4 # 每个子空间共有2^nbits个簇心
n 10000 # 输入向量个数
pq_dir /share/ann_benchmarks/gist/  # pq数据存放位置
train_dir /root/gist/train.fvecs   # train数据存放位置
test_dir /root/gist/test.fvecs     # test数据存放位置
plot_dir /root/gist   # 图片存放的目录
```

之后运行run.py即可

```
python3 run.py
```
config->use_dir_vector指示了是否使用方向向量

算法运行之后会在数据所在的目录创建.res文件，保存结果

如果有画图需求继续运行plot.py

```
python3 plot.py
```