import numpy as np 
import os
import torch

if __name__ == "__main__":
    # depth_npy_dir = '/home/mhao/data/RHD_published_v2/processed/evaluation/depth'
    depth_npy_dir = '/home/mhao/data/RHD_published_v2/processed/training/depth'
    fi_li = os.listdir(depth_npy_dir)
    for f_name in fi_li:
        f_path = os.path.join(depth_npy_dir,f_name)
        if os.path.isfile(f_path) and f_name.endswith('.npy'):
            depthmap = np.load(f_path)
            depthmap = torch.from_numpy(depthmap).unsqueeze(0)
            if depthmap.max() == depthmap.min():
                print(f_name)
                print(depthmap)
                print('---------------------------------------------')