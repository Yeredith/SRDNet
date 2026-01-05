# coding:utf-8
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import time
import numpy as np
import torch.nn as nn

from loss import HybridLoss ,HFL
from data_utils import TrainsetFromFolder, ValsetFromFolder
from eval import PSNR
from option import opt

from torch.optim.lr_scheduler import MultiStepLR
from architecture.common import *
from architecture.SRDNet import SRDNet
import torch.nn.init as init
import scipy.io as scio


psnr = []
out_path = os.path.join('result', opt.datasetName)
os.makedirs(out_path, exist_ok=True)

log_interval = 50
per_epoch_iteration = 10

class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)
def main():
    global best_psnr
    global writer
    if opt.cuda:
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    train_set = TrainsetFromFolder('C:\\Users\\yered\\OneDrive\\Documentos\\REDES SR\\SRDNet\\CAVE\\datasets\\trains\\' + opt.datasetName + '/' + str(opt.upscale_factor) + '/')
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    val_set = ValsetFromFolder('C:\\Users\\yered\\OneDrive\\Documentos\\REDES SR\\SRDNet\\CAVE\\datasets\\tests\\' + opt.datasetName + '/' + str(opt.upscale_factor) + '/')
    val_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    
    model = SRDNet(opt)

    criterion = nn.L1Loss()
    HFL_loss=HFL()

    print('Parameters number is ', sum(param.numel() for param in model.parameters()))

    # loss functions to choose
    # mse_loss = torch.nn.MSELoss()
    # criterion = HybridLoss(spatial_tv=True, spectral_tv=True)
    # hylap_loss = HyLapLoss(spatial_tv=False, spectral_tv=True)

    if opt.cuda:

        model = model.cuda()
        criterion = criterion.cuda()
        HFL_loss= HFL_loss.cuda()

    else:
        model = model.cpu()
    print('# parameters:', sum(param.numel() for param in model.parameters()))

    # Setting Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08,weight_decay=1e-08)

    # optionally resuming from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Setting learning rate
    scheduler = MultiStepLR(optimizer, milestones=[35, 70, 105,140,175], gamma=0.5, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,opt.nEpochs,eta_min=2e-5)

    # Create a filesystem-safe timestamp for Windows (no spaces or colons)
    safe_ts = time.strftime("%Y%m%d_%H%M%S")
    logdir = os.path.join('logs', f"{opt.datasetName}{opt.method}_{safe_ts}")
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=logdir)
    # Training
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
        train(train_loader, optimizer, model, criterion, epoch,HFL_loss)
        # train(train_loader, optimizer, model, criterion, epoch, HFL_loss)
        val(val_loader, model, epoch)
        save_checkpoint(epoch, model, optimizer)
        scheduler.step()

    scio.savemat(out_path + 'LFF1GRL0.mat', {'psnr': psnr})  # , 'ssim':ssim, 'sam':sam})
    # scio.savemat(out_path + opt.datasetName + opt.method+'LFF1GRL0.txt', {'psnr': psnr})  # , 'ssim':ssim, 'sam':sam})
nearest = nn.Upsample(scale_factor=opt.upscale_factor, mode='nearest')

def train(train_loader, optimizer, model, criterion, epoch, HFL_loss):
    model.train()
    for iteration, batch in enumerate(train_loader, 1):
        input, label = batch[0], batch[1]
        # move to device
        if opt.cuda:
            input = input.cuda()
            label = label.cuda()

        lms = nearest(input)

        SR = model(input)

        hfl_val = HFL_loss(SR, label)
        loss = criterion(SR, label) + 0.1 * hfl_val

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % log_interval == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(train_loader), loss.item()))
        writer.add_scalar('train/loss', loss.item(), (epoch - 1) * len(train_loader) + iteration)
    
def val(val_loader, model, epoch):
    model.eval()
    val_psnr = 0.0
    with torch.no_grad():
        for iteration, batch in enumerate(val_loader, 1):
            input, HR = batch[0], batch[1]
            if opt.cuda:
                input = input.cuda()
                HR = HR.cuda()

            SR = model(input)

            # Convert to numpy arrays with shape (H, W, C)
            sr_np = SR.cpu().numpy()
            hr_np = HR.cpu().numpy()
            # remove batch dimension
            if sr_np.ndim == 4:
                sr_np = sr_np[0]
            if hr_np.ndim == 4:
                hr_np = hr_np[0]

            # currently sr_np is (C, H, W) -> transpose to (H, W, C)
            sr_np = np.transpose(sr_np, (1, 2, 0))
            hr_np = np.transpose(hr_np, (1, 2, 0))

            val_psnr += PSNR(sr_np, hr_np)

        val_psnr = val_psnr / len(val_loader)
        print("PSNR = {:.3f}".format(val_psnr))

    psnr.append(val_psnr)
    writer.add_scalar('val/psnr', val_psnr, epoch)
def save_checkpoint(epoch, model, optimizer):
    model_out_path = "checkpoint/" + "{}_model_{}_epoch_{}.pth".format(opt.datasetName, opt.upscale_factor, epoch)
    state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")
    torch.save(state, model_out_path)


if __name__ == "__main__":
    main()
