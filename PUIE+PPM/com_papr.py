from PUIENet_MC import mynet
import torch
import argparse


parser = argparse.ArgumentParser(description='PyTorch PUIE-Net')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batchSize', type=int, default=10, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=2, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--data_dir', type=str, default='/home/guangyao/code/comparative_ex/data/PUIE/train')
parser.add_argument('--label_train_dataset', type=str, default='label')
parser.add_argument('--data_train_dataset', type=str, default='image')
parser.add_argument('--polar_train_dataset', type=str, default='polar')
parser.add_argument('--patch_size', type=int, default=256, help='Size of cropped image')
parser.add_argument('--save_folder', default='weights/PPM/', help='Location to save checkpoint models')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--decay', type=int, default='10000', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--data_augmentation', type=bool, default=True)

opt = parser.parse_args()


model = mynet(opt)  # 替换为你的模型类
model.load_state_dict(torch.load('/home/guangyao/code/comparative_ex/PUIE/weights/256/epoch_49.pth'))
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params}")

