import os
import cv2
import albumentations as A
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, inp='input', target='target', img_options=None):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, inp)))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, target)))
        dep_files = sorted(os.listdir(os.path.join(rgb_dir, 'depth')))
        aop_files = sorted(os.listdir(os.path.join(rgb_dir, 'train_p/AOP')))
        dop_files = sorted(os.listdir(os.path.join(rgb_dir, 'train_p/DOP')))
        # mas_files = sorted(os.listdir(os.path.join(rgb_dir, 'mask')))

        self.inp_filenames = [os.path.join(rgb_dir, inp, x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, target, x) for x in tar_files if is_image_file(x)]
        self.dep_filenames = [os.path.join(rgb_dir, 'depth', x) for x in dep_files if is_image_file(x)]
        self.aop_filenames=[os.path.join(rgb_dir, 'train_p/AOP', x) for x in aop_files if is_image_file(x)]
        self.dop_filenames=[os.path.join(rgb_dir, 'train_p/DOP', x) for x in dop_files if is_image_file(x)]

        # self.mas_filenames = [os.path.join(rgb_dir, 'mask', x) for x in mas_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.transform = A.Compose([
            A.Resize(height=img_options['h'], width=img_options['w']),
            A.Transpose(p=0.3),
            A.HorizontalFlip(p=0.3),  # 水平翻转，概率 50%
            A.VerticalFlip(p=0.3),  # 垂直翻转，概率 50%
            # A.Flip(p=0.3),
            A.RandomRotate90(p=0.3),
            ],
            is_check_shapes=False,
            additional_targets={
                'target': 'image',
                'depth': 'image',
                'aop': 'image',  # 新增AOP目标
                'dop': 'image' , # 新增DOP目标
            }
        )

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        dep_path = self.dep_filenames[index_]
        aop_path = self.aop_filenames[index_]
        dop_path = self.dop_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')
        dep_img = Image.open(dep_path).convert('RGB')
        aop_img = Image.open(aop_path).convert('RGB')
        dop_img = Image.open(dop_path).convert('RGB')

        inp_img = np.array(inp_img)
        tar_img = np.array(tar_img)
        dep_img = np.array(dep_img)
        aop_img = np.array(aop_img)
        dop_img = np.array(dop_img)

        transformed = self.transform(image=inp_img, target=tar_img,
                                     depth=dep_img,
                                     aop=aop_img,
                                     dop=dop_img,
                                     )

        inp_img = F.to_tensor(transformed['image'])
        tar_img = F.to_tensor(transformed['target'])
        dep_img = F.to_tensor(transformed['depth'])
        aop_img = F.to_tensor(transformed['aop'])
        dop_img = F.to_tensor(transformed['dop'])

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return inp_img, dep_img, tar_img,aop_img, dop_img, filename


class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, inp='input', target='target', img_options=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, inp)))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, target)))
        dep_files = sorted(os.listdir(os.path.join(rgb_dir, 'depth')))
        aop_files = sorted(os.listdir(os.path.join(rgb_dir, 'test_p/AOP')))
        dop_files = sorted(os.listdir(os.path.join(rgb_dir, 'test_p/DOP')))

        self.inp_filenames = [os.path.join(rgb_dir, inp, x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, target, x) for x in tar_files if is_image_file(x)]
        self.dep_filenames = [os.path.join(rgb_dir, 'depth', x) for x in dep_files if is_image_file(x)]
        self.aop_filenames = [os.path.join(rgb_dir, 'test_p/AOP', x) for x in aop_files if is_image_file(x)]
        self.dop_filenames = [os.path.join(rgb_dir, 'test_p/DOP', x) for x in dop_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.transform = A.Compose([
            A.Resize(height=img_options['h'], width=img_options['w']), ],
            is_check_shapes=False,
            additional_targets={
                'target': 'image',
                'depth': 'image',
                'aop': 'image',  # 新增AOP目标
                'dop': 'image',  # 新增DOP目标
            }
        )

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        dep_path = self.dep_filenames[index_]
        aop_path = self.aop_filenames[index_]
        dop_path = self.dop_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')
        dep_img = Image.open(dep_path).convert('RGB')
        aop_img = Image.open(aop_path).convert('RGB')
        dop_img = Image.open(dop_path).convert('RGB')

        if not self.img_options['ori']:
            inp_img = np.array(inp_img)
            tar_img = np.array(tar_img)
            dep_img = np.array(dep_img)
            aop_img = np.array(aop_img)
            dop_img = np.array(dop_img)

            transformed = self.transform(image=inp_img, target=tar_img,
                                         depth=dep_img,
                                         aop=aop_img,
                                         dop=dop_img,
                                         )

            inp_img = transformed['image']
            tar_img = transformed['target']
            dep_img = transformed['depth']
            aop_img = transformed['aop']
            dop_img = transformed['dop']

        inp_img = F.to_tensor(inp_img)
        tar_img = F.to_tensor(tar_img)
        dep_img = F.to_tensor(dep_img)
        aop_img = F.to_tensor(aop_img)
        dop_img = F.to_tensor(dop_img)

        filename = os.path.split(tar_path)[-1]

        return inp_img, dep_img, tar_img,aop_img,dop_img, filename


class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir, inp='input', img_options=None):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, inp)))
        dep_files = sorted(os.listdir(os.path.join(rgb_dir, 'depth')))
        aop_files = sorted(os.listdir(os.path.join(rgb_dir, 'test_p/AOP')))
        dop_files = sorted(os.listdir(os.path.join(rgb_dir, 'test_p/DOP')))

        self.inp_filenames = [os.path.join(rgb_dir, inp, x) for x in inp_files if is_image_file(x)]
        self.dep_filenames = [os.path.join(rgb_dir, 'depth', x) for x in dep_files if is_image_file(x)]
        self.aop_filenames = [os.path.join(rgb_dir, 'test_p/AOP', x) for x in aop_files if is_image_file(x)]
        self.dop_filenames = [os.path.join(rgb_dir, 'test_p/DOP', x) for x in dop_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.inp_filenames)  # get the size of target

        self.transform = A.Compose([
            A.Resize(height=img_options['h'], width=img_options['w']), ],
            is_check_shapes=False,
            additional_targets={
                'depth': 'image',
                'aop': 'image',  # 新增AOP目标
                'dop': 'image',  # 新增DOP目标
            }
        )

    def _single_scale_retinex(self, img, sigma):
        """单尺度 Retinex (SSR)"""
        retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
        return retinex

    def _multi_scale_retinex(self, img, sigma_list=[15, 80, 250]):
        """多尺度 Retinex (MSR)"""
        retinex = np.zeros_like(img)
        for sigma in sigma_list:
            retinex += self._single_scale_retinex(img, sigma)
        retinex = retinex / len(sigma_list)
        return retinex

    def _msr_color_restoration(self, img, alpha=125, beta=46, G=192):
        """带色彩恢复的 MSR (MSRCR)"""
        img = img.astype(np.float32) + 1.0  # 避免 log(0)

        # 1. 计算 MSR
        msr = self._multi_scale_retinex(img)

        # 2. 色彩恢复
        color_restoration = beta * (np.log10(alpha * img) - np.log10(np.sum(img, axis=2, keepdims=True)))

        # 3. 合并 MSR 和色彩恢复
        msrcr = G * (msr * color_restoration)

        # 4. 归一化到 [0, 255]
        msrcr = cv2.normalize(msrcr, None, 0, 255, cv2.NORM_MINMAX)
        return msrcr.astype(np.uint8)

    def _apply_retinex(self, img):
        """应用 Retinex 预处理"""
        img = np.array(img)

        # 转换为 float32 并归一化
        img = img.astype(np.float32) / 255.0

        # 分离 RGB 通道
        channels = cv2.split(img)

        # 对每个通道应用 MSR
        retinex_channels = []
        for channel in channels:
            retinex = self._multi_scale_retinex(channel)
            retinex_channels.append(retinex)

        # 合并通道
        retinex_img = cv2.merge(retinex_channels)

        # 归一化到 [0, 1] 并转回 [0, 255]
        retinex_img = cv2.normalize(retinex_img, None, 0, 255, cv2.NORM_MINMAX)
        retinex_img = retinex_img.astype(np.uint8)

        # 可选：应用 MSRCR 增强颜色
        # retinex_img = self._msr_color_restoration(img * 255.0)

        return Image.fromarray(retinex_img)

    def __len__(self):
        return self.sizex


    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        dep_path = self.dep_filenames[index_]
        aop_path = self.aop_filenames[index_]
        dop_path = self.dop_filenames[index_]


        inp_img = Image.open(inp_path).convert('RGB')
        # 只对输入图像进行预处理
        inp_img = self._apply_retinex(inp_img)  # 应用 Retinex
        dep_img = Image.open(dep_path).convert('RGB')
        aop_img = Image.open(aop_path).convert('RGB')
        dop_img = Image.open(dop_path).convert('RGB')

        if not self.img_options['ori']:
            inp_img = np.array(inp_img)
            dep_img = np.array(dep_img)
            aop_img = np.array(aop_img)
            dop_img = np.array(dop_img)

            transformed = self.transform(image=inp_img, depth=dep_img,
                                         aop=aop_img,
                                         dop=dop_img,
                                         )

            inp_img = transformed['image']
            dep_img = transformed['depth']
            aop_img = transformed['aop']
            dop_img = transformed['dop']

        inp_img = F.to_tensor(inp_img)
        dep_img = F.to_tensor(dep_img)
        aop_img = F.to_tensor(aop_img)
        dop_img = F.to_tensor(dop_img)

        filename = os.path.split(inp_path)[-1]

        return inp_img, dep_img,aop_img,dop_img, filename
