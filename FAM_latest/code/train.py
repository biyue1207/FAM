import math
import argparse
import logging
import os
import random
import shutil
import sys
import time
import numpy as np
import torch
import gc
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler, WeakStrongAugment)
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.unet import UNet
from networks.config import get_config
from utils import losses, ramps
from val_2D import test_single_volume, test_single_volume_two_model
from utils.displacement import ABD_I, ABD_R
from networks.net_factory import BCP_net
from utils.getTwoFreDomain import get_two_frequency_domain

gc.collect()
torch.cuda.empty_cache()
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/root/data/ABD/data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='train_ACDC_Cross_Teaching', help='experiment_name')
parser.add_argument('--model_1', type=str,
                    default='unet', help='model1_name')
parser.add_argument('--model_2', type=str,
                    default='swin_unet', help='model2_name')
parser.add_argument('--pre_iterations', type=int, 
                    default=15000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--image_size', type=list, default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4,
                    help='output channel of network')
parser.add_argument('--cfg', type=str,
                    default="./code/configs/swin_tiny_patch4_window7_224_lite.yaml",
                    help='path to config file', )
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ",
                    default=None, nargs='+', )
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, ''full: cache all data, ''part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
# patch size
parser.add_argument('--patch_size', type=int, default=56, help='patch_size')
parser.add_argument('--h_size', type=int, default=4, help='h_size')
parser.add_argument('--w_size', type=int, default=4, help='w_size')
# top num
parser.add_argument('--top_num', type=int, default=4, help='top_num')
args = parser.parse_args()  
config = get_config(args)

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def save_net_opt(net, optimizer, path):
    state = {
        'net':net.state_dict(),
        'opt':optimizer.state_dict(),
    }
    torch.save(state, str(path))

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)
    
def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch,args.consistency_rampup)  # args.consistency=0.1 # args.consistency_rampup=200

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

class Co_Training_LOSS(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ce = CrossEntropyLoss(reduction='none')
        self.ce_1 = CrossEntropyLoss()
        self.de = losses.DiceLoss(num_classes)

    def forward(self,model_out, label, mask=None):
        """""
        model_out:(bs,nun_classes,d1,d2)
        label:(bs,d1,d2)
        """""  
        model_out_soft = torch.softmax(model_out, dim=1)

        if mask is None:
            
            loss = 0.5 * (self.ce_1(model_out, label.squeeze(1).long()) + self.de(model_out_soft, label))
        
        else:
            ce_loss = self.ce(model_out, label.long())
            ce_loss = (ce_loss * mask).sum() / mask.sum()

            loss = 0.5 * (ce_loss + self.de(model_out_soft, label.unsqueeze(1), mask.unsqueeze(1)))
        
        return loss

def In_train(volume_batch, volume_batch_strong, model_high_frequency, model_low_frequency):
    HF_view_weak, LF_view_weak = get_two_frequency_domain(volume_batch, sep_kernel_range_HF=(9,25), sep_kernel_range_LF=(9,27))
    HF_view_strong, LF_view_strong = get_two_frequency_domain(volume_batch_strong, sep_kernel_range_HF=(9,25), sep_kernel_range_LF=(9,27))
    HF_view = torch.cat((HF_view_weak, HF_view_strong), dim=0)
    LF_view = torch.cat((LF_view_weak, LF_view_strong), dim=0)
    
    model_high_frequency_out = model_high_frequency(HF_view)
    model_low_frequency_out = model_low_frequency(LF_view)

    return model_high_frequency_out, model_low_frequency_out

def grid_dropout_regions(img, grid_width=16, cover_area_ratio=0.20, mode=0):
    b,c,h,w=img.shape

    hh = math.ceil(math.sqrt(h * h + w * w))
    l = math.ceil(grid_width * cover_area_ratio)

    mask = np.ones((hh, hh), np.float32)

    st_h = np.random.randint(grid_width)
    st_w = np.random.randint(grid_width)

    for i in range(-1, hh // grid_width + 1):
        s = grid_width * i + st_h
        t = s + l
        s = max(min(s, hh), 0)
        t = max(min(t, hh), 0)
        mask[s:t, :] *= 0

    for i in range(-1, hh // grid_width + 1):
        s = grid_width * i + st_w
        t = s + l
        s = max(min(s, hh), 0)
        t = max(min(t, hh), 0)
        mask[:, s:t] *= 0

    mask = mask[(hh - h) // 2: (hh - h) // 2 + h, (hh - w) // 2: (hh - w) // 2 + w]

    mask = torch.from_numpy(mask).float()

    if mode == 1:
        mask = 1 - mask

    mask=mask.unsqueeze(0).unsqueeze(0).cuda()
    mask = mask.expand_as(img)

    img = img * mask

    return img

def block_dropout_regions(img, mask_ratio=0.25):
    b,c,h,w=img.shape
    N = h * w  
    M = set()  
    while len(M) < mask_ratio * N and int(mask_ratio * N - len(M))>18:
        s = random.randint(16, int(mask_ratio * N - len(M)))
        r = random.uniform(0.3, 1 / 0.3)
        a = int(np.sqrt(s * r))  
        b = int(np.sqrt(s / r))  
        while a > h or b > w:
            r = random.uniform(0.3, 1 / 0.3)  
            a = int(np.sqrt(s * r))  
            b = int(np.sqrt(s / r))  
        t = random.randint(0, h - a)  
        l = random.randint(0, w - b)  
        for i in range(t, t + a):
            for j in range(l, l + b):
                M.add((i, j))
    mask = torch.ones((h, w), dtype=torch.float32)
    for i, j in M:
        mask[i, j] = 0
    
    mask=mask.unsqueeze(0).unsqueeze(0).cuda()
    mask = mask.expand_as(img)

    img = img * mask
    return img


def random_dropout_regions_1(batch, p=0.25):
    mask = (torch.rand(batch.shape[0], 1, batch.shape[2], batch.shape[3]) > p).float().to(batch.device)
    return batch * mask 
  
def random_dropout_regions(batch, grid_size=16, p=0.25):
    B, C, H, W = batch.shape
    mask = torch.ones(B, 1, H, W).to(batch.device)
 
    for i in range(0, H, grid_size):
        for j in range(0, W, grid_size):
            if torch.rand(1).item() < p:

                mask[:, :, i:i + grid_size, j:j + grid_size] = 0 
    return batch * mask

def Co_Training(L_t, Y_t, L_s, Y_s, U, teacher, ema_teacher, student, ema_student, teacher_optimizer, student_optimizer, loss, supervised=False, approx=False, previous_params=True):
    udd_loss = losses.UDDLoss()
    threshold = 0.7
    
    sub_batch_size = int(U.shape[0] / 2)
    U_t, U_s = get_two_frequency_domain(U)

    SPL_t = teacher(grid_dropout_regions(U_t[sub_batch_size:,:,:,:]))
    udd_out_t = teacher(U_t[:sub_batch_size,:,:,:])
    with torch.no_grad():
        ema_SPL_t = ema_teacher(U_t[:sub_batch_size,:,:,:])
        probabilities = torch.nn.Softmax(dim=1)(ema_SPL_t.detach())
        max_probs, PL_t = torch.max(probabilities, dim=1)
        PL_t_mask = max_probs.ge(threshold).float()

    if previous_params:
        self_loss_t = loss(SPL_t, PL_t, PL_t_mask)
        self_loss_t.backward() 
        self_loss_t_grads = [param.grad.detach().clone() if param.grad is not None else 0 for param in teacher.parameters()]
        teacher_optimizer.zero_grad()

    SPL_s = student(grid_dropout_regions(U_s[sub_batch_size:,:,:,:])) # compute the soft pseudo labels (student)
    udd_out_s = student(U_s[:sub_batch_size,:,:,:])
    with torch.no_grad():
        ema_SPL_s = ema_student(U_s[:sub_batch_size,:,:,:])
        probabilities = torch.nn.Softmax(dim=1)(ema_SPL_s.detach())
        max_probs, PL_s = torch.max(probabilities, dim=1)
        PL_s_mask = max_probs.ge(threshold).float()
    
    # probabilities = probabilities.permute(0, 2, 3, 1).reshape(-1, num_classes)
    # PL_s = torch.multinomial(probabilities, 1).reshape(b, d1, d2).unsqueeze(1).to(device)
    
    if previous_params:
        self_loss_s = loss(SPL_s, PL_s, PL_s_mask)
        self_loss_s.backward()
        
        self_loss_s_grads = [param.grad.detach().clone() for param in student.parameters()]
        student_optimizer.zero_grad()

    if approx:
        student_initial_output = student(L_s)
        student_loss_initial_l = loss(student_initial_output, Y_s).detach().clone()

        teacher_initial_output = teacher(L_t)
        teacher_loss_initial_l = loss(teacher_initial_output, Y_t).detach().clone()
    
    student_optimizer.zero_grad()
    teacher_optimizer.zero_grad()
    
    student_initial_output = student(grid_dropout_regions(U_s[sub_batch_size:,:,:,:]))
    student_out = student(L_s)
    Co_udd_loss_s = udd_loss(student_out.detach(), udd_out_s)
    student_loss_initial = loss(student_initial_output, PL_t, PL_t_mask) + Co_udd_loss_s
    
    student_optimizer.zero_grad()
    student_loss_initial.backward()

    if not approx:
        grads1_s = [param.grad.data.detach().clone() for param in student.parameters()]

    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    student_optimizer.step()
    student_optimizer.zero_grad()
    
    teacher_initial_output = teacher(grid_dropout_regions(U_t[sub_batch_size:,:,:,:]))
    teacher_out = teacher(L_t)
    Co_udd_loss_t = udd_loss(teacher_out.detach(), udd_out_t)
    teacher_loss_initial = loss(teacher_initial_output, PL_s, PL_s_mask) + Co_udd_loss_t

    teacher_optimizer.zero_grad()
    teacher_loss_initial.backward()
    if not approx:
        grads1_t = [param.grad.data.detach().clone() if param.grad is not None else 0 for param in teacher.parameters()]
    
    torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
    teacher_optimizer.step()
    teacher_optimizer.zero_grad()

    student_final_output = student(L_s)
    
    student_loss_final = loss(student_final_output, Y_s)

    if not approx:
        student_loss_final.backward()
        h_t = sum([(param.grad.data.detach() * grads).sum() for param, grads in zip(student.parameters(),  grads1_s)])

    student_optimizer.zero_grad()

    teacher_final_output = teacher(L_t)
    teacher_loss_final = loss(teacher_final_output, Y_t)

    if not approx:
        teacher_loss_final.backward()
        h_s = sum([(param.grad.data.detach() * grads).sum() for param, grads in zip(teacher.parameters(), grads1_t) if param.grad is not None])
        # h_s = sum([(param.grad.data.detach() * grads).sum() for param, grads in zip(teacher.parameters(),  grads1_t)])

    teacher_optimizer.zero_grad()

    # compute the teacher MPL loss
    if not previous_params:
        if approx:
            # https://github.com/google-research/google-research/issues/536
            # h is approximable by: student_loss_final - loss(student_initial(L), Y) where student_initial is before the gradient update for U
            h_approx_t = student_loss_initial_l - student_loss_final
            h_approx_s = teacher_loss_initial_l - teacher_loss_final
            # this is the first order taylor approximation of the above loss, and apparently has finite deviation from the true quantity.
            # for correctness, I include instead the theoretically correct computation of h
            student_loss_mpl = h_approx_s.detach() * loss(SPL_t, PL_t, PL_t_mask)
            teacher_loss_mpl = h_approx_t.detach() * loss(SPL_s, PL_s, PL_s_mask)
        else:
            SPL_t = teacher(grid_dropout_regions(U_t[sub_batch_size:,:,:,:])) # (re) compute the soft pseudo labels (teacher)
            SPL_s = student(grid_dropout_regions(U_s[sub_batch_size:,:,:,:])) # (re) compute the soft pseudo labels (student)
            
            teacher_loss_mpl = h_t.detach() * loss(SPL_t, PL_t, PL_t_mask)
            student_loss_mpl = h_s.detach() * loss(SPL_s, PL_s, PL_s_mask)
    else:
        if approx:
            # https://github.com/google-research/google-research/issues/536
            # h is approximable by: student_loss_final - loss(student_initial(L), Y) where student_initial is before the gradient update for U
            h_t = student_loss_initial_l - student_loss_final
            h_s = teacher_loss_initial_l - teacher_loss_final
            # this is the first order taylor approximation of the above loss, and apparently has finite deviation from the true quantity.
            # for correctness, I include instead the theoretically correct computation of h
        # it is already computed above
        teacher_loss_mpl = 0.0
        student_loss_mpl = 0.0
    
    if supervised:# optionally compute the supervised loss
        teacher_out = teacher(L_t)
        student_out = student(L_s)
        student_out_u = student(U_s)
        student_loss_sup = loss(student_out, Y_s)
        student_loss = student_loss_mpl + student_loss_sup
        
    else:
        student_loss = student_loss_mpl

    # update student based on teacher performance
    student_optimizer.zero_grad()
    
    if student_loss != 0.0:
        student_loss.backward()
    
    if previous_params: # compute the MPL update based on the original parameters
        for param, grad in zip(student.parameters(), self_loss_s_grads):
            if param.grad is not None:
                param.grad += h_s * grad
            else:
                param.grad = h_s * grad

    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    student_optimizer.step()
    student_optimizer.zero_grad()

    if supervised:# optionally compute the supervised loss
        # teacher_out = teacher(L_t)
        teacher_out_U = teacher(U_t)
        teacher_loss_sup = loss(teacher_out, Y_t)
        teacher_loss = teacher_loss_mpl + teacher_loss_sup
    else:
        teacher_loss = teacher_loss_mpl
    
    # update teacher based on student performance
    teacher_optimizer.zero_grad()
    if teacher_loss != 0.0:
        teacher_loss.backward()
    
    if previous_params: # compute the MPL update based on the original parameters
        for param, grad in zip(teacher.parameters(), self_loss_t_grads):
            if param.grad is not None:
                param.grad += h_t * grad
            else:
                try:
                    param.grad = h_t * grad
                except:
                    pass
    torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
    teacher_optimizer.step()
    teacher_optimizer.zero_grad()

    update_model_ema(teacher, ema_teacher, 0.99)
    update_model_ema(student, ema_student, 0.99)
    
    return student_loss_initial, student_loss_final, student_loss_mpl, student_loss_sup, teacher_loss_initial, teacher_loss_final, teacher_loss_mpl, teacher_loss_sup

def Independent_train(args, snapshot_path):
    base_lr = args.base_lr  
    num_classes = args.num_classes 
    batch_size = args.batch_size  
    max_iterations = args.pre_iterations 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # model1 = UNet(in_chns=1, class_num=num_classes).cuda()
    model1 = BCP_net(in_chns=1, class_num=num_classes) 
    
    model2 = ViT_seg(config, img_size=args.image_size, num_classes=args.num_classes).cuda()  
    model2.load_from(config)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([WeakStrongAugment(args.image_size)]))  # args.image_size=[224,224]
    db_val = BaseDataSets(base_dir=args.root_path, split="test")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)  # args.labeled_num=7
    print("Train labeled {} samples".format(labeled_slice))

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)  # args.labeled_bs=8
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    model1.train()
    model2.train()
    loader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    udd_loss = losses.UDDLoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start Independent_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1  

    best_performance_combining_1 = 0.0
    best_performance_combining_2 = 0.0
    best_performance_combining_3 = 0.0
    best_performance_combining = 0.0
    best_performance_1 = 0.0
    best_performance_2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)  

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'],  sampled_batch['label']
            volume_batch_strong, label_batch_strong = sampled_batch['image_strong'], sampled_batch['label_strong']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            volume_batch_strong, label_batch_strong = volume_batch_strong.cuda(), label_batch_strong.cuda()
            
            HF_view, LF_view = In_train(volume_batch[:args.labeled_bs], volume_batch_strong[:args.labeled_bs], model1, model2)
            In_udd_loss = udd_loss(HF_view, LF_view)
            HF_view_soft = torch.softmax(HF_view, dim=1)
            LF_view_soft = torch.softmax(LF_view, dim=1)
            loss1 = 0.5*(ce_loss(HF_view[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(HF_view_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2 = 0.5*(ce_loss(LF_view[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(LF_view_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss3 = 0.5*(ce_loss(HF_view[args.labeled_bs:], label_batch_strong[:args.labeled_bs].long()) + dice_loss(HF_view_soft[args.labeled_bs:], label_batch_strong[:args.labeled_bs].unsqueeze(1)))
            loss4 = 0.5*(ce_loss(LF_view[args.labeled_bs:], label_batch_strong[:args.labeled_bs].long()) + dice_loss(LF_view_soft[args.labeled_bs:], label_batch_strong[:args.labeled_bs].unsqueeze(1)))
            loss = loss1 + loss2 + loss3 + loss4
                
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9

            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss1', loss1, iter_num)
            writer.add_scalar('loss/loss2', loss2, iter_num)
            writer.add_scalar('loss/loss3', loss3, iter_num)
            writer.add_scalar('loss/loss4', loss4, iter_num)
            writer.add_scalar('loss/In_udd_loss', In_udd_loss, iter_num)

            logging.info('iteration %d : loss1 : %f loss2 : %f loss3 : %f loss4 : %f In_udd_loss : %f' % (iter_num, loss1.item(), loss2.item(), loss3.item(), loss4.item(), In_udd_loss.item()))

            if iter_num > 0 and iter_num % 200 == 0:
                model1.eval()
                model2.eval()
                combining_metric_list_1 = 0.0
                combining_metric_list_2 = 0.0
                combining_metric_list_3 = 0.0
                metric_list_1 = 0.0
                metric_list_2 = 0.0
                for i_batch, sampled_batch in enumerate(loader):
                    combining_metric_i_1, combining_metric_i_2, combining_metric_i_3, metric_i_1, metric_i_2 = test_single_volume_two_model(sampled_batch["image"], sampled_batch["label"], model1, model2, classes=num_classes, patch_size=args.image_size)
                    combining_metric_list_1 += np.array(combining_metric_i_1)
                    combining_metric_list_2 += np.array(combining_metric_i_2)
                    combining_metric_list_3 += np.array(combining_metric_i_3)
                    metric_list_1 += np.array(metric_i_1)
                    metric_list_2 += np.array(metric_i_2)

                combining_metric_list_1 = combining_metric_list_1 / len(db_val)
                combining_metric_list_2 = combining_metric_list_2 / len(db_val)
                combining_metric_list_3 = combining_metric_list_3 / len(db_val)
                metric_list_1 = metric_list_1 / len(db_val)
                metric_list_2 = metric_list_2 / len(db_val)

                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice_combining_1'.format(class_i+1), combining_metric_list_1[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95_combining_1'.format(class_i+1), combining_metric_list_1[class_i, 1], iter_num)

                    writer.add_scalar('info/val_{}_dice_combining_2'.format(class_i+1), combining_metric_list_2[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95_combining_2'.format(class_i+1), combining_metric_list_2[class_i, 1], iter_num)

                    writer.add_scalar('info/val_{}_dice_combining_3'.format(class_i+1), combining_metric_list_3[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95_combining_3'.format(class_i+1), combining_metric_list_3[class_i, 1], iter_num)

                    writer.add_scalar('info/val_{}_dice_1'.format(class_i+1), metric_list_1[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95_1'.format(class_i+1), metric_list_1[class_i, 1], iter_num)

                    writer.add_scalar('info/val_{}_dice_2'.format(class_i+1), metric_list_2[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95_2'.format(class_i+1), metric_list_2[class_i, 1], iter_num)

                performance_combining_1 = np.mean(combining_metric_list_1, axis=0)[0]
                performance_combining_2 = np.mean(combining_metric_list_2, axis=0)[0]
                performance_combining_3 = np.mean(combining_metric_list_3, axis=0)[0]
                performance_1 = np.mean(metric_list_1, axis=0)[0]
                performance_2 = np.mean(metric_list_2, axis=0)[0]

                mean_hd951_combining_1 = np.mean(combining_metric_list_1, axis=0)[1]
                mean_hd951_combining_2 = np.mean(combining_metric_list_2, axis=0)[1]
                mean_hd951_combining_3 = np.mean(combining_metric_list_3, axis=0)[1]
                mean_hd951_1 = np.mean(metric_list_1, axis=0)[1]
                mean_hd951_2 = np.mean(metric_list_2, axis=0)[1]

                writer.add_scalar('info/val_mean_dice_combining_1', performance_combining_1, iter_num)
                writer.add_scalar('info/val_mean_dice_combining_2', performance_combining_2, iter_num)
                writer.add_scalar('info/val_mean_dice_combining_3', performance_combining_3, iter_num)
                writer.add_scalar('info/val_mean_dice_1', performance_1, iter_num)
                writer.add_scalar('info/val_mean_dice_2', performance_2, iter_num)

                writer.add_scalar('info/model_val_mean_hd95_combining_1', mean_hd951_combining_1, iter_num)
                writer.add_scalar('info/model_val_mean_hd95_combining_2', mean_hd951_combining_2, iter_num)
                writer.add_scalar('info/model_val_mean_hd95_combining_3', mean_hd951_combining_3, iter_num)
                writer.add_scalar('info/model_val_mean_hd95_1', mean_hd951_1, iter_num)
                writer.add_scalar('info/model_val_mean_hd95_2', mean_hd951_2, iter_num)

                if performance_combining_1 > best_performance_combining_1:
                    best_performance_combining_1 = performance_combining_1
                if performance_combining_2 > best_performance_combining_2:
                    best_performance_combining_2 = performance_combining_2
                if performance_combining_3 > best_performance_combining_3:
                    best_performance_combining_3 = performance_combining_3
                
                iter_best_performance = max(performance_combining_1, performance_combining_2, performance_combining_3, performance_1, performance_2)

                if performance_1 > best_performance_1:                    
                    best_performance_1 = performance_1
                    # save_mode_path = os.path.join(snapshot_path, 'HF_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance_1, 4)))
                    save_best = os.path.join(snapshot_path,'{}_HF_best_model.pth'.format(args.model_1))
                    # save_net_opt(model1, optimizer1, save_mode_path)
                    save_net_opt(model1, optimizer1, save_best)
                    
                if performance_2 > best_performance_2:
                    best_performance_2 = performance_2
                    # save_mode_path = os.path.join(snapshot_path, 'LF_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance_2, 4)))
                    save_best = os.path.join(snapshot_path,'{}_LF_best_model.pth'.format(args.model_2))
                    # save_net_opt(model2, optimizer2, save_mode_path)
                    save_net_opt(model2, optimizer2, save_best)
                if iter_best_performance > best_performance_combining:
                    best_performance_combining = iter_best_performance
                    save_best_HF = os.path.join(snapshot_path,'{}_HF_combining_best_model.pth'.format(args.model_1))
                    save_best_LF = os.path.join(snapshot_path,'{}_LF_combining_best_model.pth'.format(args.model_2))
                    save_net_opt(model1, optimizer1, save_best_HF)
                    save_net_opt(model2, optimizer2, save_best_LF)
                
                logging.info('iteration %d : mean_dice_combining_1 : %f' % (iter_num, performance_combining_1))
                logging.info('iteration %d : mean_dice_combining_2 : %f' % (iter_num, performance_combining_2))
                logging.info('iteration %d : mean_dice_combining_3 : %f' % (iter_num, performance_combining_3))
                logging.info('iteration %d : mean_dice_1 : %f' % (iter_num, performance_1))
                logging.info('iteration %d : mean_dice_2 : %f' % (iter_num, performance_2))
                model1.train()
                model2.train()

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    logging.info('*******BEST!*******')
    logging.info('Best mean_dice_combining_1 : %f' % (best_performance_combining_1))
    logging.info('Best mean_dice_combining_2 : %f' % (best_performance_combining_2))
    logging.info('Best mean_dice_combining_3 : %f' % (best_performance_combining_3))
    logging.info('Best mean_dice_1 : %f' % (best_performance_1))
    logging.info('Best mean_dice_2 : %f' % (best_performance_2))
    writer.close()

def Collaborative_train(args, pre_snapshot_path, snapshot_path):
    base_lr = args.base_lr  
    num_classes = args.num_classes 
    batch_size = args.batch_size  
    max_iterations = args.max_iterations 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    HF_pre_trained_model = os.path.join(pre_snapshot_path,'{}_HF_best_model.pth'.format(args.model_1))
    LF_pre_trained_model = os.path.join(pre_snapshot_path,'{}_LF_best_model.pth'.format(args.model_2))

    # HF_pre_trained_model = "/root/data/ABD_official/model/conditionH/second_ACDC_train_ACDC_Cross_Teaching_7_labeled/HF_iter_21400_dice_0.9027.pth"
    # LF_pre_trained_model = "/root/data/ABD_official/model/conditionH/second_ACDC_train_ACDC_Cross_Teaching_7_labeled/LF_iter_20600_dice_0.8968.pth"

    # model1 = UNet(in_chns=1, class_num=num_classes).cuda() 
    # ema_model1 = UNet(in_chns=1, class_num=num_classes).cuda() 
    model1 = BCP_net(in_chns=1, class_num=num_classes) 
    ema_model1 = BCP_net(in_chns=1, class_num=num_classes, ema=True) 
    
    
    model2 = ViT_seg(config, img_size=args.image_size, num_classes=args.num_classes).cuda()  
    model2.load_from(config)
    ema_model2 = ViT_seg(config, img_size=args.image_size, num_classes=args.num_classes).cuda()  
    ema_model2.load_from(config)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([WeakStrongAugment(args.image_size)]))  # args.image_size=[224,224]
    db_val = BaseDataSets(base_dir=args.root_path, split="test")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)  # args.labeled_num=7
    print("Train labeled {} samples".format(labeled_slice))

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)  # args.labeled_bs=8
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    model1.train()
    ema_model1.train()
    model2.train()
    ema_model2.train()

    loader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    
    load_net_opt(model1, optimizer1, HF_pre_trained_model)
    load_net(ema_model1, HF_pre_trained_model)
    load_net_opt(model2, optimizer2, LF_pre_trained_model)
    load_net(ema_model2, LF_pre_trained_model)
    logging.info("Loaded from {}".format(HF_pre_trained_model))
    logging.info("Loaded from {}".format(HF_pre_trained_model))

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start Collaborative_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1  

    best_performance_combining_1 = 0.0
    best_performance_combining_2 = 0.0
    best_performance_combining_3 = 0.0
    best_performance_1 = 0.0
    best_performance_2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)  

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'],  sampled_batch['label']
            volume_batch_strong, label_batch_strong = sampled_batch['image_strong'], sampled_batch['label_strong']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            volume_batch_strong, label_batch_strong = volume_batch_strong.cuda(), label_batch_strong.cuda()
            
            HF_volume_batch, LF_volume_batch = get_two_frequency_domain(volume_batch)
            HF_volume_batch_strong, LF_volume_batch_strong = get_two_frequency_domain(volume_batch_strong)
            L_t = torch.cat([HF_volume_batch[:args.labeled_bs], HF_volume_batch_strong[:args.labeled_bs]], dim=0)
            Y_t = torch.cat([label_batch[:args.labeled_bs], label_batch_strong[:args.labeled_bs]], dim=0)
            L_s = torch.cat([LF_volume_batch[:args.labeled_bs], LF_volume_batch_strong[:args.labeled_bs]], dim=0)
            Y_s = torch.cat([label_batch[:args.labeled_bs], label_batch_strong[:args.labeled_bs]], dim=0)
            U_data = torch.cat([volume_batch[args.labeled_bs:], volume_batch_strong[args.labeled_bs:]], dim=0)
                
            student_loss_initial, student_loss_final, student_loss_mpl, student_loss_sup, teacher_loss_initial, teacher_loss_final, teacher_loss_mpl, teacher_loss_sup = Co_Training(L_t, Y_t.unsqueeze(1), L_s, Y_s.unsqueeze(1), U_data, model1, ema_model1, model2, ema_model2, optimizer1, optimizer2, Co_Training_LOSS(num_classes), supervised=True, approx=False, previous_params=True)

            iter_num = iter_num + 1

            lr_ = (base_lr * (1.0 - iter_num / max_iterations) ** 0.9)

            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('Co_Training_loss/student_loss_initial', student_loss_initial, iter_num)
            writer.add_scalar('Co_Training_loss/student_loss_final', student_loss_final, iter_num)
            writer.add_scalar('Co_Training_loss/student_loss_mpl', student_loss_mpl, iter_num)
            writer.add_scalar('Co_Training_loss/student_loss_sup', student_loss_sup, iter_num)
            writer.add_scalar('Co_Training_loss/teacher_loss_initial', teacher_loss_initial, iter_num)
            writer.add_scalar('Co_Training_loss/teacher_loss_final', teacher_loss_final, iter_num)
            writer.add_scalar('Co_Training_loss/teacher_loss_mpl', teacher_loss_mpl, iter_num)
            writer.add_scalar('Co_Training_loss/teacher_loss_sup', teacher_loss_sup, iter_num)
            logging.info('iteration %d : student_loss_initial : %f student_loss_final : %f student_loss_mpl : %f student_loss_sup : %f teacher_loss_initial : %f teacher_loss_final : %f teacher_loss_mpl : %f teacher_loss_sup : %f' % (iter_num, student_loss_initial, student_loss_final, student_loss_mpl, student_loss_sup, teacher_loss_initial, teacher_loss_final, teacher_loss_mpl, teacher_loss_sup))
            
            if iter_num > 0 and iter_num % 200 == 0:
                model1.eval()
                model2.eval()
                combining_metric_list_1 = 0.0
                combining_metric_list_2 = 0.0
                combining_metric_list_3 = 0.0
                metric_list_1 = 0.0
                metric_list_2 = 0.0
                for i_batch, sampled_batch in enumerate(loader):
                    combining_metric_i_1, combining_metric_i_2, combining_metric_i_3, metric_i_1, metric_i_2 = test_single_volume_two_model(sampled_batch["image"], sampled_batch["label"], model1, model2, classes=num_classes, patch_size=args.image_size)
                    combining_metric_list_1 += np.array(combining_metric_i_1)
                    combining_metric_list_2 += np.array(combining_metric_i_2)
                    combining_metric_list_3 += np.array(combining_metric_i_3)
                    metric_list_1 += np.array(metric_i_1)
                    metric_list_2 += np.array(metric_i_2)

                combining_metric_list_1 = combining_metric_list_1 / len(db_val)
                combining_metric_list_2 = combining_metric_list_2 / len(db_val)
                combining_metric_list_3 = combining_metric_list_3 / len(db_val)
                metric_list_1 = metric_list_1 / len(db_val)
                metric_list_2 = metric_list_2 / len(db_val)

                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice_combining_1'.format(class_i+1), combining_metric_list_1[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95_combining_1'.format(class_i+1), combining_metric_list_1[class_i, 1], iter_num)

                    writer.add_scalar('info/val_{}_dice_combining_2'.format(class_i+1), combining_metric_list_2[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95_combining_2'.format(class_i+1), combining_metric_list_2[class_i, 1], iter_num)

                    writer.add_scalar('info/val_{}_dice_combining_3'.format(class_i+1), combining_metric_list_3[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95_combining_3'.format(class_i+1), combining_metric_list_3[class_i, 1], iter_num)

                    writer.add_scalar('info/val_{}_dice_1'.format(class_i+1), metric_list_1[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95_1'.format(class_i+1), metric_list_1[class_i, 1], iter_num)

                    writer.add_scalar('info/val_{}_dice_2'.format(class_i+1), metric_list_2[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95_2'.format(class_i+1), metric_list_2[class_i, 1], iter_num)

                performance_combining_1 = np.mean(combining_metric_list_1, axis=0)[0]
                performance_combining_2 = np.mean(combining_metric_list_2, axis=0)[0]
                performance_combining_3 = np.mean(combining_metric_list_3, axis=0)[0]
                performance_1 = np.mean(metric_list_1, axis=0)[0]
                performance_2 = np.mean(metric_list_2, axis=0)[0]

                mean_hd951_combining_1 = np.mean(combining_metric_list_1, axis=0)[1]
                mean_hd951_combining_2 = np.mean(combining_metric_list_2, axis=0)[1]
                mean_hd951_combining_3 = np.mean(combining_metric_list_3, axis=0)[1]
                mean_hd951_1 = np.mean(metric_list_1, axis=0)[1]
                mean_hd951_2 = np.mean(metric_list_2, axis=0)[1]

                writer.add_scalar('info/val_mean_dice_combining_1', performance_combining_1, iter_num)
                writer.add_scalar('info/val_mean_dice_combining_2', performance_combining_2, iter_num)
                writer.add_scalar('info/val_mean_dice_combining_3', performance_combining_3, iter_num)
                writer.add_scalar('info/val_mean_dice_1', performance_1, iter_num)
                writer.add_scalar('info/val_mean_dice_2', performance_2, iter_num)

                writer.add_scalar('info/model_val_mean_hd95_combining_1', mean_hd951_combining_1, iter_num)
                writer.add_scalar('info/model_val_mean_hd95_combining_2', mean_hd951_combining_2, iter_num)
                writer.add_scalar('info/model_val_mean_hd95_combining_3', mean_hd951_combining_3, iter_num)
                writer.add_scalar('info/model_val_mean_hd95_1', mean_hd951_1, iter_num)
                writer.add_scalar('info/model_val_mean_hd95_2', mean_hd951_2, iter_num)

                if performance_combining_1 > best_performance_combining_1:
                    best_performance_combining_1 = performance_combining_1
                if performance_combining_2 > best_performance_combining_2:
                    best_performance_combining_2 = performance_combining_2
                if performance_combining_3 > best_performance_combining_3:
                    best_performance_combining_3 = performance_combining_3

                if performance_1 > best_performance_1:
                    best_performance_1 = performance_1
                    save_mode_path = os.path.join(snapshot_path, 'HF_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance_1, 4)))
                    print(f"save_path:{save_mode_path}")
                    save_best = os.path.join(snapshot_path,'{}_HF_best_model.pth'.format(args.model_1))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                if performance_2 > best_performance_2:
                    best_performance_2 = performance_2
                    save_mode_path = os.path.join(snapshot_path, 'LF_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance_2, 4)))
                    save_best = os.path.join(snapshot_path,'{}_LF_best_model.pth'.format(args.model_2))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)
                
                logging.info('iteration %d : mean_dice_combining_1 : %f' % (iter_num, performance_combining_1))
                logging.info('iteration %d : mean_dice_combining_2 : %f' % (iter_num, performance_combining_2))
                logging.info('iteration %d : mean_dice_combining_3 : %f' % (iter_num, performance_combining_3))
                logging.info('iteration %d : mean_dice_1 : %f' % (iter_num, performance_1))
                logging.info('iteration %d : mean_dice_2 : %f' % (iter_num, performance_2))
                model1.train()
                model2.train()

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    logging.info('*******BEST!*******')
    logging.info('Best mean_dice_combining_1 : %f' % (best_performance_combining_1))
    logging.info('Best mean_dice_combining_2 : %f' % (best_performance_combining_2))
    logging.info('Best mean_dice_combining_3 : %f' % (best_performance_combining_3))
    logging.info('Best mean_dice_1 : %f' % (best_performance_1))
    logging.info('Best mean_dice_2 : %f' % (best_performance_2))
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    first_snapshot_path = "./model/conditionH/first_ACDC_{}_{}_labeled".format(args.exp, args.labeled_num)
    second_snapshot_path = "./model/conditionH/second_ACDC_{}_{}_labeled".format(args.exp, args.labeled_num)
    for snapshot_path in [first_snapshot_path, second_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    shutil.copy(__file__, second_snapshot_path)

    # Independent-Learning
    logging.basicConfig(filename=first_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    Independent_train(args, first_snapshot_path)

    # Collaborative-Learning
    logging.basicConfig(filename=second_snapshot_path + "/log.txt", level=logging.INFO,format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    Collaborative_train(args, first_snapshot_path, second_snapshot_path)
