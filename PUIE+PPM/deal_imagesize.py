import cv2
import os

# 输入和输出路径
gt_input_dir = "/home/guangyao/code/data/UCPD/DOP"  # 替换为你的GT图像目录
output_dir = "/home/guangyao/code/data/UCPD/polar/DOP"  # 替换为你想保存的目录

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 遍历GT图像目录
for filename in os.listdir(gt_input_dir):
    if filename.endswith((".jpg", ".jpeg", ".png", ".JPG")):  # 支持常见图像格式
        # 读取图像
        img_path = os.path.join(gt_input_dir, filename)
        img = cv2.imread(img_path)

        # 调整尺寸为256×256
        resized_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

        # 保存调整后的图像
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, resized_img)

print("GT图像尺寸调整完成！")