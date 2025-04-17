from networks.unet import UNet, UNet_2d
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from networks.vnet import VNet

def net_factory(net_type="unet", in_chns=1, class_num=2, mode = "train", tsne=0):
    if net_type == "unet" and mode == "train":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    if net_type == "VNet" and mode == "train" and tsne==0:
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    if net_type == "VNet" and mode == "test" and tsne==0:
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    return net

def BCP_net(in_chns=1, class_num=2, ema=False):
    net = UNet_2d(in_chns=in_chns, class_num=class_num).cuda()
    if ema:
        for param in net.parameters():
            param.detach_()
    return net


def BCP_net(in_chns=1, class_num=4, ema=False):
    net = UNet_2d(in_chns=in_chns, class_num=class_num).cuda()
    if ema:
        for param in net.parameters():
            param.detach_()
    return net

def Swinunet3d(in_channels, class_num):
    model = SwinUNETR(
        img_size=(96, 96, 64),
        in_channels=in_channels,
        out_channels=class_num,
        feature_size=48,
    )
    return model
