import numpy as np
import h5py
import os

def numpy_to_fvecs(data, fvecs_path):
    """
    将NumPy数组保存为.fvecs格式的文件。
    :param data: NumPy数组，假定为二维。
    :param fvecs_path: 输出.fvecs文件的路径。
    """
    # 确保数据是二维的
    if data.ndim != 2:
        raise ValueError("数据需要是二维数组。")
    
    # 打开文件准备写入
    with open(fvecs_path, 'wb') as f:
        for vector in data:
            # 写入向量维度，fvecs格式要求以4字节整数保存
            f.write(np.array(vector.shape, dtype='int32').tobytes())
            # 写入向量数据，以4字节浮点数保存
            f.write(vector.astype('float32').tobytes())

def save_datasets_as_fvecs(hdf5_path, output_folder):
    """
    遍历HDF5文件中的所有数据集，将它们分别保存为.fvecs格式的文件。
    :param hdf5_path: HDF5文件的路径。
    :param output_folder: 输出文件夹的路径。
    """
    with h5py.File(hdf5_path, 'r') as file:
        def visitor_func(name, node):
            if isinstance(node, h5py.Dataset):
                print(f"Processing dataset: {name}")
                data = node[:]
                # 检查数据是否为二维
                if data.ndim == 2:
                    output_path = os.path.join(output_folder, f"{name.replace('/', '_')}.fvecs")
                    numpy_to_fvecs(data, output_path)
                else:
                    print(f"跳过{name}，因为它不是二维数组。")
        
        file.visititems(visitor_func)

# 示例用法
save_datasets_as_fvecs('example.h5', 'output_folder')
