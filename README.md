### Usage
```
./experiment [data path] [query data_path] [data name] [data point numbers]
e.g. ./experiment ~/datasets/bigann/learn.100M.u8bin ~/datasets/bigann/query.public.10K.u8bin bigann 10000
```
config->use_dir_vector指示了是否使用方向向量

算法运行之后会在数据所在的目录创建.res文件，保存结果