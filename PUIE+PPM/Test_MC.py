import warnings
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1,0'
import torch
import cv2
import numpy as np
import time
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PUIENet_MC import mynet
from data import get_eval_set

# settings
parser = argparse.ArgumentParser(description='PyTorch PUIE-Net')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use Default=123')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--input_dir', type=str, default='/home/guangyao/code/data/fog/intensity')
parser.add_argument('--polar', type=str, default='/home/guangyao/code/data/fog/test_p')
parser.add_argument('--output', default='/home/guangyao/code/comparative_ex/PUIE/results/dehaze_pretrain',
                    help='Location to save checkpoint models')
parser.add_argument('--sample_out', type=str, default='sample')
parser.add_argument('--reference_out', type=str, default='mc')
parser.add_argument('--model', default='/home/guangyao/code/comparative_ex/PUIE/weights/puie(1).pth',
                    help='Pretrained base model')

opt = parser.parse_args()
print(opt)
device = torch.device(opt.device)
cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

# torch.manual_seed(opt.seed)
# if cuda:
#     torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
test_set = get_eval_set(opt.input_dir, opt.input_dir, opt.polar)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')

model = mynet(opt)

model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained model is loaded.')

if cuda:
    model = model.cuda(device)


def eval():
    model.eval()
    torch.set_grad_enabled(False)
    for batch in testing_data_loader:
        with torch.no_grad():
            input, _, name = Variable(batch[0]), Variable(batch[1]), batch[4]
            aop, dop = Variable(batch[2]), Variable(batch[3])
        if cuda:
            input = input.cuda(device)
            aop = aop.cuda(device)
            dop = dop.cuda(device)

        with torch.no_grad():

            # Timing for forward pass
            t_forward_start = time.time()
            model.forward(input, input, aop, dop, training=False)
            t_forward_end = time.time()
            forward_time = t_forward_end - t_forward_start

            avg_pre = 0
            # Timing for sampling loop
            t_sampling_start = time.time()
            for i in range(20):
                t0 = time.time()
                prediction = model.sample(testing=True)
                t1 = time.time()
                avg_pre = avg_pre + prediction / 20
                save_img_1(prediction.cpu().data, name[0], i, opt.sample_out)

                print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
            t_sampling_end = time.time()
            sampling_time = t_sampling_end - t_sampling_start
            total_time = forward_time + sampling_time
            save_img_2(avg_pre.cpu().data, name[0], opt.reference_out)

            # Print per-image inference time summary
            print("===> Image: %s || Forward: %.4f sec || Sampling: %.4f sec || Total: %.4f sec" % (
            name[0], forward_time, sampling_time, total_time))


def save_img_1(img, img_name, i, out):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
    # save img
    save_dir = os.path.join(opt.output, out)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    name_list = img_name.split('.', 1)
    save_fn = save_dir + '/' + name_list[0] + '_' + str(i) + '.' + name_list[1]
    cv2.imwrite(save_fn, cv2.cvtColor(save_img * 255, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])


def save_img_2(img, img_name, out):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
    # save img
    save_dir = os.path.join(opt.output, out)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    name_list = img_name.split('.', 1)
    save_fn = save_dir + '/' + name_list[0] + '.' + name_list[1]
    cv2.imwrite(save_fn, cv2.cvtColor(save_img * 255, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])


eval()
