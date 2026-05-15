import os
import torch
import numpy as np
import scipy.io as sio

def NL2NetData(opt):
    
    # 取得目前 dataset.py 所在的資料夾
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 從目前程式目錄下的 ./data 讀取資料
    data_dir = os.path.join(BASE_DIR, "data")
    image_file = os.path.join(data_dir, opt.dataset + ".mat")

    print("讀取資料路徑:", image_file)

    if not os.path.exists(image_file):
        raise FileNotFoundError(
            f"找不到資料檔案：{image_file}\n"
            f"請確認資料是否放在：{data_dir}"
        )

    input_data = sio.loadmat(image_file)

    if "data" not in input_data:
        raise KeyError(
            f"{image_file} 裡面找不到 key: 'data'\n"
            f"目前可用 keys: {list(input_data.keys())}"
        )

    image = input_data["data"]
    image = image.astype(np.float32)

    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    band = image.shape[2]

    train_data = np.expand_dims(image, axis=0)
    loader_train = torch.from_numpy(train_data.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
        
    print("The training dataloader construction process is done")
    print("-" * 50)
    return loader_train, band