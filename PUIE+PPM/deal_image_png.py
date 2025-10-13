import os


def change_image_extensions_to_png(directory):
    """
    将指定目录下的所有图片文件后缀名改为.png
    :param directory: 目标目录路径
    """
    # 支持的图片扩展名列表
    image_extensions = ['.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp']

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 获取文件扩展名
        name, ext = os.path.splitext(filename)
        ext_lower = ext.lower()

        # 如果文件是图片且不是.png
        if ext_lower in image_extensions:
            # 构建新的文件名
            new_filename = name + '.png'
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)

            # 重命名文件
            try:
                os.rename(old_path, new_path)
                print(f'已重命名: {filename} -> {new_filename}')
            except Exception as e:
                print(f'重命名 {filename} 失败: {e}')


if __name__ == '__main__':
    # 使用当前目录
    target_directory = '/home/guangyao/code/data/UCPD/input_256'
    # 检查目录是否存在
    if not os.path.isdir(target_directory):
        print(f'错误: 目录 "{target_directory}" 不存在')
    else:
        print(f'正在处理目录: {target_directory}')
        change_image_extensions_to_png(target_directory)
        print('处理完成')