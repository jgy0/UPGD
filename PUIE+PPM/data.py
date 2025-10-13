from os.path import join
from torchvision.transforms import Compose, ToTensor, Resize
from dataset import DatasetFromFolderEval,  DatasetFromFolder


def transform():
    return Compose([
        # Resize((256, 256)),
        ToTensor(),
    ])


def get_training_set(path, label, data, polar,patch_size, data_augmentation):
    label_dir = join(path, label)
    data_dir = join(path, data)
    polar_dir= join(path, polar)

    return DatasetFromFolder(label_dir, data_dir, polar_dir,patch_size, data_augmentation, transform=transform())


def get_eval_set(data_dir, label_dir,polar):
    return DatasetFromFolderEval(data_dir, label_dir, polar,transform=transform())


