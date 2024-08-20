### Usage
```
./experiment [data path] [query data_path] [data name] [data point numbers] [pq cfg file]
e.g. 
./experiment /root/datasets/gist/train.fvecs /root/datasets/gist/test.fvecs  gist 10000 ../cfg/pq_config.txt
```
config->use_dir_vector指示了是否使用方向向量

算法运行之后会在数据所在的目录创建.res文件，保存结果