import logging  # 用于记录日志信息
import numpy as np  # 用于处理数组和数值计算
import torch  # 用于深度学习框架
from PIL import Image  # 用于图像处理
from functools import lru_cache  # 用于缓存函数结果
from functools import partial  # 用于偏函数
from itertools import repeat  # 用于重复迭代
from multiprocessing import Pool  # 用于多进程
from os import listdir  # 用于列出目录中的文件
from os.path import splitext, isfile, join  # 用于处理文件路径
from pathlib import Path  # 用于文件路径处理
from torch.utils.data import Dataset  # 用于构建数据集
from tqdm import tqdm  # 用于显示进度条

# 定义加载图像的函数
def load_image(filename):
    # 根据文件扩展名判断如何加载图像
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))  # 从.npy文件加载图像
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())  # 从.pt或.pth文件加载图像
    else:
        return Image.open(filename)  # 从其他文件类型加载图像

# 定义获取唯一掩码值的函数
def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]  # 获取掩码文件路径
    mask = np.asarray(load_image(mask_file))  # 加载掩码图像
    if mask.ndim == 2:
        return np.unique(mask)  # 返回2维掩码的唯一值
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])  # 将3维掩码展开
        return np.unique(mask, axis=0)  # 返回3维掩码的唯一值
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')  # 抛出异常

# 定义基本数据集类
class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)  # 图片目录路径
        self.mask_dir = Path(mask_dir)  # 掩码目录路径
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'  # 确保缩放比例在0和1之间
        self.scale = scale  # 缩放比例
        self.mask_suffix = mask_suffix  # 掩码后缀

        # 获取目录中所有图像文件的ID（文件名不含扩展名）
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')  # 没有找到文件时抛出异常

        logging.info(f'Creating dataset with {len(self.ids)} examples')  # 记录日志
        logging.info('Scanning mask files to determine unique values')  # 记录日志

        # 使用多进程扫描掩码文件以确定唯一值
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        # 计算并记录掩码的唯一值
        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)  # 返回数据集的大小

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size  # 获取图像尺寸
        newW, newH = int(scale * w), int(scale * h)  # 计算缩放后的尺寸
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'  # 确保缩放后尺寸有效
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)  # 缩放图像
        img = np.asarray(pil_img)  # 转为数组

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)  # 创建空掩码
            for i, v in enumerate(mask_values):  # 遍历唯一掩码值
                if img.ndim == 2:
                    mask[img == v] = i  # 2维掩码
                else:
                    mask[(img == v).all(-1)] = i  # 3维掩码

            return mask  # 返回预处理后的掩码

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]  # 灰度图像
            else:
                img = img.transpose((2, 0, 1))  # 彩色图像

            if (img > 1).any():
                img = img / 255.0  # 归一化

            return img  # 返回预处理后的图像

    def __getitem__(self, idx):
        name = self.ids[idx]  # 获取图像ID
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))  # 获取掩码文件
        img_file = list(self.images_dir.glob(name + '.*'))  # 获取图像文件

        # 确保每个ID只有一个对应的图像和掩码
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'

        mask = load_image(mask_file[0])  # 加载掩码图像
        img = load_image(img_file[0])  # 加载图像

        # 确保图像和掩码尺寸相同
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)  # 预处理图像
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)  # 预处理掩码

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),  # 返回图像张量
            'mask': torch.as_tensor(mask.copy()).long().contiguous()  # 返回掩码张量
        }

# 定义Carvana数据集类，继承自BasicDataset
class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')

