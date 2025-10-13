import warnings
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch.optim as optim
from accelerate import Accelerator
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm import tqdm
from thop import profile
from config import Config
from data import get_training_data, get_validation_data
from models import *
from utils import *
import re
warnings.filterwarnings('ignore')

opt = Config('config.yml')

seed_everything(opt.OPTIM.SEED)

def train():
    # Accelerate
    accelerator = Accelerator(log_with='wandb') if opt.OPTIM.WANDB else Accelerator()
    device = accelerator.device
    config = {
        "dataset": opt.TRAINING.TRAIN_DIR
    }
    accelerator.init_trackers("shadow", config=config)

    if accelerator.is_local_main_process:
        os.makedirs(opt.TRAINING.SAVE_DIR, exist_ok=True)

    # Data Loader
    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir = opt.TRAINING.VAL_DIR

    train_dataset = get_training_data(train_dir, opt.MODEL.INPUT, opt.MODEL.TARGET, {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16,
                             drop_last=False, pin_memory=True)
    val_dataset = get_validation_data(val_dir, opt.MODEL.INPUT, opt.MODEL.TARGET, {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H, 'ori': opt.TRAINING.ORI})
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True)

    # Model & Loss
    model = Model()

    # # 1. 加载预训练模型（从TESTING.WEIGHT路径）
    # if opt.TESTING.WEIGHT:
    #     print(f"===> Loading pretrained model from {opt.TESTING.WEIGHT}")
    #     pretrained_dict = torch.load(opt.TESTING.WEIGHT, map_location='cpu')
    #     # 检查是否是嵌套的state_dict
    #     if 'state_dict' in pretrained_dict:
    #         true_state_dict = pretrained_dict['state_dict']  # 提取真正的参数
    #     else:
    #         true_state_dict = pretrained_dict  # 直接使用
    #
    #     model_dict = model.state_dict()
    #
    #     # 1.1 自动识别预训练和新增参数
    #     pretrained_keys = set(true_state_dict.keys())
    #     current_keys = set(model_dict.keys())
    #     new_params = current_keys - pretrained_keys
    #
    #     # 1.2 更新模型参数（保留新增参数的随机初始化）
    #     model.load_state_dict({**model_dict, **pretrained_dict}, strict=False)
    #
    #     # 2. 自动冻结策略
    #     for name, param in model.named_parameters():
    #         if re.search(r'aop_upsample ',name) or re.search(r'dop_upsample',name) or re.search(r'patch_embed*',name) or re.search(r'prompt_block',name) or re.search(r'attention_layer*',name):
    #             param.requires_grad = True
    #             if opt.VERBOSE and accelerator.is_local_main_process:
    #                 print(f"🔥 Training new param: {name}")
    #
    #         else:  # 保持新增参数可训练
    #             param.requires_grad = False
    #             if opt.VERBOSE and accelerator.is_local_main_process:
    #                 print(f"❄️ Freezing pretrained param: {name}")
    #
    #     # 3. 打印统计信息
    #     if opt.VERBOSE and accelerator.is_local_main_process:
    #         print(
    #             f"📊 Parameter Summary: Total={len(model_dict)} | Frozen={len(pretrained_keys)} | New={len(new_params)}")
    #
    # # 4. 优化器仅作用于可训练参数
    # optimizer_b = optim.AdamW(
    #     filter(lambda p: p.requires_grad, model.parameters()),  # 关键过滤
    #     lr=opt.OPTIM.LR_INITIAL,
    #     betas=(0.9, 0.999),
    #     eps=1e-8
    # )


    criterion_ssim = SSIM(data_range=1, size_average=True, channel=3).to(device)
    criterion_psnr = torch.nn.MSELoss()

    # Optimizer & Scheduler
    optimizer_b = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999), eps=1e-8)
    scheduler_b = optim.lr_scheduler.CosineAnnealingLR(optimizer_b, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    model = accelerator.prepare(model)
    optimizer_b, scheduler_b = accelerator.prepare(optimizer_b, scheduler_b)

    start_epoch = 1
    best_epoch = 1
    best_psnr = 0
    size = len(val_loader)
    # training
    # dummy_inputs = (
    #     torch.randn(4, 3, 256, 256).to(device),  # Input
    #     torch.randn(4, 3, 256, 256).to(device),  # label
    #     torch.randn(4, 3, 256, 256).to(device),  # aop
    #     torch.randn(4, 3, 256, 256).to(device)  # dop
    # )
    #
    # flops, params = profile(model, inputs=(dummy_inputs[0], dummy_inputs[1], dummy_inputs[2], dummy_inputs[3]))
    # print('flops:{}'.format(flops))
    # print('params:{}'.format(params))
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        model.train()

        for i, data in enumerate(tqdm(train_loader, disable=not accelerator.is_local_main_process)):
            # get the inputs; data is a list of [target, input, filename]
            inp = data[0].contiguous()
            dep = data[1].contiguous()
            tar = data[2]
            aop=data[3].contiguous()
            dop=data[4].contiguous()

            # forward
            optimizer_b.zero_grad()
            res = model(inp, dep,aop,dop)

            loss_psnr = criterion_psnr(res, tar)
            loss_ssim = 1 - criterion_ssim(res, tar)

            train_loss = loss_psnr + 0.4 * loss_ssim

            # backward
            accelerator.backward(train_loss)
            optimizer_b.step()

        scheduler_b.step()

        # testing
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()
            psnr = 0
            ssim = 0
            for idx, test_data in enumerate(tqdm(val_loader, disable=not accelerator.is_local_main_process)):
                # get the inputs; data is a list of [targets, inputs, filename]
                inp = test_data[0].contiguous()
                dep = test_data[1].contiguous()
                tar = test_data[2]
                aop = test_data[3].contiguous()
                dop = test_data[4].contiguous()

                with torch.no_grad():
                    res = model(inp, dep,aop,dop)

                res, tar = accelerator.gather((res, tar))

                psnr += peak_signal_noise_ratio(res, tar, data_range=1)
                ssim += structural_similarity_index_measure(res, tar, data_range=1)

            psnr /= size
            ssim /= size

            if psnr > best_psnr:
                # save model
                best_epoch = epoch
                best_psnr = psnr
                save_checkpoint({
                    'state_dict': model.state_dict(),
                }, epoch, opt.MODEL.SESSION, opt.TRAINING.SAVE_DIR)

            if accelerator.is_local_main_process:
                accelerator.log({
                    "PSNR": psnr,
                    "SSIM": ssim,
                }, step=epoch)

                print(
                    "epoch: {}, PSNR: {}, SSIM: {}, best PSNR: {}, best epoch: {}"
                    .format(epoch, psnr, ssim, best_psnr, best_epoch))

    accelerator.end_training()


if __name__ == '__main__':
    train()
