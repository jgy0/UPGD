import os
import random
import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import numpy as np
import cv2
def apply_clahe_np(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced

def apply_gamma_np(img_rgb, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img_rgb, table)

def apply_retinex_np(img_rgb, sigma=30):
    img_float = img_rgb.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
    retinex = np.log10(img_float) - np.log10(blur)
    retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min()) * 255
    return np.uint8(retinex)

def preprocess_image(img_rgb, method='clahe'):
    if method == 'clahe':
        return apply_clahe_np(img_rgb)
    elif method == 'gamma':
        return apply_gamma_np(img_rgb, gamma=1.3)
    elif method == 'retinex':
        return apply_retinex_np(img_rgb)
    else:
        return img_rgb  # 不做预处理


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".bmp",'.JPG'])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img
    # img = Image.open(filepath).convert('RGB')
    # # 2. 转换为 NumPy 数组 (H x W x C, 0-255, uint8)
    # img_np = np.array(img)
    # # 3. 进行预处理（假设 preprocess_image 接受 NumPy 数组）
    # processed_img_np = preprocess_image(img_np, 'retinex')  # 确保 preprocess_image 返回 NumPy 数组
    # # 4. 转换回 PIL Image（如果 preprocess_image 返回 0-255 的 uint8）
    # processed_img = Image.fromarray(processed_img_np.astype('uint8'))
    # return processed_img


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


def get_patch(img_in, img_tar, img_aop, img_dop, patch_size, scale=1, ix=-1, iy=-1):
    (ih, iw) = img_in.size

    patch_mult = scale
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))
    img_aop_patch = img_aop.crop((iy, ix, iy + ip, ix + ip))  # 与 input 相同区域
    img_dop_patch = img_dop.crop((iy, ix, iy + ip, ix + ip))  # 与 input 相同区域
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar,img_aop_patch, img_dop_patch, info_patch


def augment(img_in, img_tar,aop,dop, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        aop = ImageOps.flip(aop)
        dop =ImageOps.flip(dop)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            aop = ImageOps.mirror(aop)
            dop = ImageOps.mirror(dop)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            aop = img_in.rotate(180)
            dop = img_in.rotate(180)
            info_aug['trans'] = True
            
    return img_in, img_tar,aop,dop, info_aug


class DatasetFromFolder(data.Dataset):
    def __init__(self, label_dir, data_dir, polar_dir,patch_size, data_augmentation, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.label_path = label_dir
        self.polar_path = polar_dir
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames
        self.patch_size = patch_size
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        _, file = os.path.split(self.data_filenames[index])

        k = random.randint(0,3)
        if k == 0:
            label_filenames = self.label_path + '/O/' + file
        if k == 1 :
            label_filenames = self.label_path + '/S/' + file
        if k == 2 :
            label_filenames = self.label_path + '/C/' + file
        if k == 3 :
            label_filenames = self.label_path + '/G/' + file

        # 加载polar数据
        aop_filename = os.path.join(self.polar_path, 'AOP', file)
        dop_filename = os.path.join(self.polar_path, 'DOP', file)

        target = load_img(label_filenames) # 256,256
        input = load_img(self.data_filenames[index]) # 1224 1024
        aop = load_img(aop_filename)# 1224 1024
        dop = load_img(dop_filename)# 1224 1024

        size = (512, 512)
        input = input.resize((512, 512), resample=Image.BICUBIC)
        target = target.resize((512, 512), resample=Image.BICUBIC)
        aop = aop.resize(size, resample=Image.BICUBIC)
        dop = dop.resize(size, resample=Image.BICUBIC)
        input, target,aop,dop ,_ = get_patch(input, target,aop,dop ,self.patch_size)
        
        if self.data_augmentation:
            input, target,aop,dop , _ = augment(input, target,aop,dop)
        
        if self.transform:
            input = self.transform(input)
            target = self.transform(target)
            aop= self.transform(aop)
            dop= self.transform(dop)

        return input, target,aop,dop, file

    def __len__(self):
        return len(self.data_filenames)


class DatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, label_dir,polar_dir, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames

        label_filenames = [join(label_dir, x) for x in listdir(label_dir) if is_image_file(x)]
        label_filenames.sort()
        self.label_filenames = label_filenames
        polar_aop_dir=polar_dir+'/AOP'
        polar_filenames = [join(polar_aop_dir, x) for x in listdir(polar_aop_dir) if is_image_file(x)]
        polar_filenames.sort()
        self.polar_filenames = polar_filenames

        self.polar_dir = polar_dir
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.data_filenames[index])
        label = load_img(self.label_filenames[index])
        _, file = os.path.split(self.data_filenames[index])
        _,polar_file = os.path.split(self.polar_filenames[index])

        aop_path = os.path.join(self.polar_dir, 'AOP', polar_file)
        dop_path = os.path.join(self.polar_dir, 'DOP', polar_file)
        aop = load_img(aop_path)
        dop = load_img(dop_path)

        (ih, iw) = input.size
        dh = ih % 8
        dw = iw % 8
        new_h, new_w = ih - dh, iw - dw

        input = input.resize((new_h, new_w))
        label = label.resize((new_h, new_w))
        aop = aop.resize((new_h, new_w))
        dop = dop.resize((new_h, new_w))

        if self.transform:
            input = self.transform(input)
            label = self.transform(label)
            aop = self.transform(aop)
            dop = self.transform(dop)
            
        return input, label,  aop, dop,file
      
    def __len__(self):
        return len(self.data_filenames)


