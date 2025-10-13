import os
import cv2
import numpy as np
from tqdm import tqdm
# 定义输入和输出文件夹路径
# input_folders = ['/home/guangyao/code/data/polar_raw/0', '/home/guangyao/code/data/polar_raw/45',
#                  '/home/guangyao/code/data/polar_raw/90', '/home/guangyao/code/data/polar_raw/135']
# output_folder = '/home/guangyao/code/data/polar/rgb'
input_folders = ['/home/guangyao/code/data/UCDP/0', '/home/guangyao/code/data/UCDP/45',
                 '/home/guangyao/code/data/UCDP/90']
output_folder = '/home/guangyao/code/data/UCDP/rgb'
# 设置伽马值
gamma = 1

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取其中一个文件夹中的文件名列表，假设每个文件夹中文件名相同
image_names = os.listdir(input_folders[0])
image_names.sort(key=lambda x: int(x.split('.')[0]))

# 伽马校正函数
def gamma_correction(image, gamma):
    # 将像素值归一化到 [0, 1] 范围
    normalized = image / 255.0
    # 应用伽马变换
    corrected = np.power(normalized, 1.0 / gamma)
    # 将结果还原到 [0, 255] 范围
    return (corrected * 255).astype(np.uint8)

# 逐个图像处理
print("开始处理图像...")
for image_name in tqdm(image_names, desc="Processing Images", unit="image"):
    # 初始化一个累加数组
    sum_image = None

    # 遍历每个文件夹
    for folder in input_folders:
        image_path = os.path.join(folder, image_name)

        # 读取图像并转换为浮点型以便进行计算
        image = cv2.imread(image_path).astype(np.float32)

        # 累加图像
        if sum_image is None:
            sum_image = image
        else:
            sum_image += image

    # 计算平均图像
    avg_image = sum_image / len(input_folders)

    # 将结果转换为uint8类型
    avg_image = avg_image.astype(np.uint8)

    # 应用伽马校正
    gamma_corrected_image = gamma_correction(avg_image, gamma)

    # 保存结果图像
    output_path = os.path.join(output_folder, image_name)
    if os.path.exists(output_path):
        print(f"警告：文件 '{output_path}' 已存在，将覆盖保存。")

    cv2.imwrite(output_path, gamma_corrected_image)


print("处理完成！")
print(f"输出图像已保存到文件夹: {output_folder}")
