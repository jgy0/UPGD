import warnings
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1,0'
import torch
import socket
import time
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader
from PUIENet_MC import mynet
from torch.autograd import Variable
from data import get_training_set
# from torchsummary import summary
# from torchsummaryX import summary
from thop import profile
# Training settings
parser = argparse.ArgumentParser(description='PyTorch PUIE-Net')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=2, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--data_dir', type=str, default='/home/guangyao/code/comparative_ex/data/PUIE/train')
parser.add_argument('--label_train_dataset', type=str, default='label')
parser.add_argument('--data_train_dataset', type=str, default='image')
parser.add_argument('--polar_train_dataset', type=str, default='polar')
parser.add_argument('--patch_size', type=int, default=256, help='Size of cropped image')
parser.add_argument('--save_folder', default='weights/prompt/', help='Location to save checkpoint models')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--decay', type=int, default='10', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--data_augmentation', type=bool, default=True)
# 新增参数
parser.add_argument('--val_dir', type=str, default='/home/guangyao/code/comparative_ex/data/PUIE/test', help='Validation data directory')
parser.add_argument('--val_interval', type=int, default=1, help='Run validation every N epochs')
parser.add_argument('--model', default='/home/guangyao/code/comparative_ex/PUIE/weights/puie(1).pth', help='Pretrained base model')

opt = parser.parse_args()
device = torch.device(opt.device)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)


def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        aop,dop=Variable(batch[2]), Variable(batch[3])
        if cuda:
            input = input.to(device)
            target = target.to(device)
            aop = aop.to(device)
            dop = dop.to(device)

        t0 = time.time()        
        model.forward(input, target,aop,dop, training=True)
        loss = model.elbo(target)
        optimizer.zero_grad()
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        t1 = time.time()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Learning rate: lr={} || Timer: {:.4f} sec.".format(epoch, iteration, 
                          len(training_data_loader), loss.item(), optimizer.param_groups[0]['lr'], (t1 - t0)))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def validate():
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_data_loader:
            input, target = Variable(batch[0]), Variable(batch[1])
            aop, dop = Variable(batch[2]), Variable(batch[3])
            if cuda:
                input = input.to(device)
                target = target.to(device)
                aop = aop.to(device)
                dop = dop.to(device)

            out = model.decoder(input, target, aop, dop, training=False)  # 注意：测试时 training=False
            loss = model.criterion(out, target)
            val_loss += loss.item()

    avg_loss = val_loss / len(val_data_loader)
    print("===> Validation Loss: {:.4f}".format(avg_loss))
    return avg_loss

# def checkpoint(epoch):
#     model_out_path = opt.save_folder+"epoch_{}.pth".format(epoch)
#     torch.save(model.state_dict(), model_out_path)
#     print("Checkpoint saved to {}".format(model_out_path))
def checkpoint(epoch, is_best=False):
    if is_best:
        model_out_path = opt.save_folder + "model_best.pth"
    else:
        model_out_path = opt.save_folder + "epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')

train_set = get_training_set(opt.data_dir, opt.label_train_dataset, opt.data_train_dataset, opt.polar_train_dataset,opt.patch_size, opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

val_set = get_training_set(opt.val_dir, opt.label_train_dataset, opt.data_train_dataset, opt.polar_train_dataset, opt.patch_size, False)  # 验证集不需要数据增强
val_data_loader = DataLoader(dataset=val_set, batch_size=opt.batchSize, shuffle=False, num_workers=opt.threads)
model = mynet(opt)

# 加载预训练模型并自动冻结已有参数
if opt.model:
    print(f"===> Loading pretrained model from {opt.model}")
    pretrained_dict = torch.load(opt.model)
    model_dict = model.state_dict()

    # 1. 过滤掉预训练模型中不存在的键（这些就是新增的参数）
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. 更新当前模型中的参数（保留新增参数的随机初始化）
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # 自动冻结预训练模型中已有的参数，新参数保持可训练
    for name, param in model.named_parameters():
        if name in pretrained_dict:  # 如果参数在预训练模型中存在，则冻结
            param.requires_grad = False
            print(f"Freezing parameter: {name}")
        else:  # 否则是新参数，保持可训练
            param.requires_grad = True
            print(f"Training parameter: {name}")

# dummy_inputs = (
#     torch.randn(1, 3, 512, 512).to(device),  # Input
#     torch.randn(1, 3, 512, 512).to(device),  # label
#     torch.randn(1, 3, 512, 512).to(device),  # aop
#     torch.randn(1, 3, 512, 512).to(device)   # dop
# )
#
# flops, params = profile(model, inputs=(dummy_inputs[0],dummy_inputs[1],dummy_inputs[2],dummy_inputs[3]))
# print('flops:{}'.format(flops))
# print('params:{}'.format(params))

# print('---------- Networks architecture -------------')
# print_network(model)
# print('----------------------------------------------')
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(trainable_params, lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
# optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

milestones = []
for i in range(1, opt.nEpochs+1):
    if i % opt.decay == 0:
        milestones.append(i)

scheduler = lrs.MultiStepLR(optimizer, milestones, opt.gamma)


best_val_loss = float('inf')
for epoch in range(opt.start_iter, opt.nEpochs + 1):
    
    train(epoch)
    scheduler.step()

    # 每隔 val_interval 个 epoch 验证一次
    if epoch % opt.val_interval == 0:
        val_loss = validate()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint(epoch, is_best=True)  # 保存最佳模型

    if (epoch+1) % opt.snapshots == 0:
        checkpoint(epoch,is_best=False)
