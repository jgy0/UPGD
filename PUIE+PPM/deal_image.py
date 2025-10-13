from PIL import Image
import os


def resize_images(input_folder, output_folder, target_size=(1224, 1024)):
    """
    调整输入文件夹中的所有图像到目标尺寸并保存到输出文件夹

    参数:
        input_folder: 包含原始图像的文件夹路径
        output_folder: 保存调整后图像的文件夹路径
        target_size: 目标尺寸 (width, height)
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 支持的图像文件扩展名
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.webp')

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            try:
                # 构建完整文件路径
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)

                # 打开图像文件
                with Image.open(input_path) as img:
                    # 调整图像尺寸
                    img_resized = img.resize(target_size, Image.LANCZOS)

                    # 保存调整后的图像
                    img_resized.save(output_path)

                    print(f"已调整并保存: {filename}")

            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")


# 使用示例
input_folder = "/home/guangyao/code/data/UCPD/DOP"  # 替换为你的输入文件夹路径
output_folder = "/home/guangyao/code/data/UCPD/polar_1224/DOP"  # 替换为你的输出文件夹路径

resize_images(input_folder, output_folder)