import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)  #列出这个文件夹下的所有内容

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("jpg", "_mask.gif")) #通过替换文件后缀，找到mask位置
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # 0.0, 255.0

        #掩码预处理
        mask[mask == 255.0] = 1.0  #最后一层要用sigmoid，转换为1表示

        if self.transform is not None:
            #使用数据增强，用albumentations库的函数,这个库怎么用，可以看他以前的视频
            argumentations = self.transform(image=image, mask=mask)
            image = argumentations["image"]
            mask = argumentations["mask"]

        return image, mask
