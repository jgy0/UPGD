import os
import cv2
import numpy as np
from tqdm import tqdm
import bm3d
from multiprocessing import Pool, cpu_count
from functools import partial


class BM3DDenoiser:
    def __init__(self, sigma=30.0, stage_arg=True, tile_size=512):
        """
        兼容版BM3D去噪器
        :param sigma: 噪声水平 (建议AOP:40, DOP:20)
        :param stage_arg: True=完整流程, False=仅硬阈值
        :param tile_size: 分块大小 (0=禁用分块)
        """
        self.sigma = sigma
        self.stage_arg = stage_arg
        self.tile_size = tile_size

    def _denoise_channel(self, channel):
        """单通道去噪核心方法"""
        channel_norm = channel.astype(np.float32) / 255.0
        denoised = bm3d.bm3d(
            channel_norm,
            sigma_psd=self.sigma,
            stage_arg=bm3d.BM3DStages.ALL_STAGES if self.stage_arg else bm3d.BM3DStages.HARD_THRESHOLDING
        )
        return np.clip(denoised * 255, 0, 255).astype(np.uint8)

    def denoise_rgb(self, rgb_img):
        """三通道RGB去噪"""
        if rgb_img.mean() < 5:  # 跳过全黑图像
            return rgb_img.copy()

        if self.tile_size > 0 and max(rgb_img.shape[:2]) > self.tile_size:
            return self._denoise_tiled(rgb_img)

        channels = [
            self._denoise_channel(rgb_img[:, :, i])
            for i in range(3)
        ]
        return np.stack(channels, axis=2)

    def _denoise_tiled(self, img):
        """大图像分块处理"""
        h, w = img.shape[:2]
        output = np.zeros_like(img)
        pad = 32  # 重叠区域防止边界伪影

        for y in range(0, h, self.tile_size - pad):
            for x in range(0, w, self.tile_size - pad):
                y1, y2 = max(0, y), min(h, y + self.tile_size)
                x1, x2 = max(0, x), min(w, x + self.tile_size)
                tile = img[y1:y2, x1:x2]
                output[y1:y2, x1:x2] = self.denoise_rgb(tile)

        return output


def process_image(img_path, output_dir, denoiser):
    """安全的图像处理函数"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")

        denoised = denoiser.denoise_rgb(img)
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, denoised)
        return True
    except Exception as e:
        print(f"\n处理 {img_path} 时出错: {str(e)}")
        return False


def preprocess_folder(input_dir, output_dir, denoiser, n_workers=8):
    """并行处理整个目录"""
    os.makedirs(output_dir, exist_ok=True)
    img_files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 过滤已处理文件
    existing = set(os.listdir(output_dir))
    todo = [os.path.join(input_dir, f) for f in img_files if f not in existing]

    if not todo:
        print(f"所有图像已存在于 {output_dir}")
        return

    worker = partial(process_image, output_dir=output_dir, denoiser=denoiser)

    with Pool(min(n_workers, cpu_count())) as pool:
        results = list(tqdm(
            pool.imap_unordered(worker, todo),
            total=len(todo),
            desc=f"处理 {os.path.basename(input_dir)}"
        ))

    success_rate = 100 * sum(results) / len(results)
    print(f"\n完成: 成功 {sum(results)}/{len(results)} ({success_rate:.1f}%)")


if __name__ == '__main__':
    # 配置参数
    config = {
        'base_path': '/home/guangyao/code/data/polar_split',
        'n_workers': 8,
        'aop_params': {'sigma': 0.1, 'stage_arg': True, 'tile_size': 512},
        'dop_params': {'sigma': 1, 'stage_arg': False, 'tile_size': 0}
    }

    # 初始化去噪器
    aop_denoiser = BM3DDenoiser(**config['aop_params'])
    dop_denoiser = BM3DDenoiser(**config['dop_params'])

    # 处理所有目录
    folders = [
        ('train_p/AOP', 'train_p/AOP_denoised', aop_denoiser),
        ('train_p/DOP', 'train_p/DOP_denoised', dop_denoiser),
        ('test_p/AOP', 'test_p/AOP_denoised', aop_denoiser),
        ('test_p/DOP', 'test_p/DOP_denoised', dop_denoiser)
    ]

    for input_suffix, output_suffix, denoiser in folders:
        input_dir = os.path.join(config['base_path'], input_suffix)
        output_dir = os.path.join(config['base_path'], output_suffix)

        print(f"\n{'=' * 40}")
        print(f"处理目录: {input_dir}")
        preprocess_folder(input_dir, output_dir, denoiser, config['n_workers'])