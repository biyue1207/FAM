import argparse
import os
import shutil
import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from networks.config import get_config
from networks.net_factory import net_factory,BCP_net
from networks.unet import UNet
from networks.vision_transformer import SwinUnet as ViT_seg
from utils.getTwoFreDomain import get_two_frequency_domain


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/root/data/ABD/data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='train_ACDC_Cross_Teaching', help='experiment_name')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--model_1', type=str,
                    default='unet', help='model_name')
parser.add_argument('--model_2', type=str,
                    default='swin_unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
parser.add_argument('--image_size', type=list, default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--cfg', type=str,
                    default="./code/configs/swin_tiny_patch4_window7_224_lite.yaml",
                    help='path to config file', )
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ",
                    default=None, nargs='+', )
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
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
args = parser.parse_args() 
config = get_config(args)

def load_net(net, path):
    state = torch.load(str(path))
    try:
        net.load_state_dict(state['net'])
    except:
        net.load_state_dict(state)

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd

def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (224 / x, 224 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 224, y / 224), order=0)
            prediction[ind] = pred
    if np.sum(prediction == 1)==0:
        first_metric = 0,0,0,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)
    if np.sum(prediction == 2)==0:
        second_metric = 0,0,0,0
    else:
        second_metric = calculate_metric_percase(prediction == 2, label == 2)
    if np.sum(prediction == 3)==0:
        third_metric = 0,0,0,0
    else:
        third_metric = calculate_metric_percase(prediction == 3, label == 3)
    return first_metric, second_metric, third_metric

def test_single_volume_two_model(case, net_1, net_2, test_save_path, FLAGS, separation=True):
    patch_size=[224, 224]
    classes=4
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction_1 = np.zeros_like(label)
    prediction_2 = np.zeros_like(label)
    combining_prediction_1 = np.zeros_like(label)
    combining_prediction_2 = np.zeros_like(label)
    combining_prediction_3 = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (224 / x, 224 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        if separation:
            HF_input, LF_input = get_two_frequency_domain(input, sep_kernel_range_HF=(13,13), sep_kernel_range_LF=(23,23))
            HF_input, LF_input = HF_input.cuda(), LF_input.cuda()
        else:
            HF_input, LF_input = input, input
        net_1.eval()
        net_2.eval()
        with torch.no_grad():
            if FLAGS.model_1 == "unet_urds":
                out_main, _, _, _ = net_1(input)
            else:
                out_1 = net_1(HF_input)
                out_2 = net_2(LF_input)

            combining_out_1 = torch.argmax(torch.softmax(out_1, dim=1) * torch.softmax(out_2, dim=1), dim=1).squeeze(0)
            combining_out_1 = combining_out_1.cpu().detach().numpy()

            combining_out_2 = torch.argmax(torch.softmax(out_1 + out_2, dim=1), dim=1).squeeze(0)
            combining_out_2 = combining_out_2.cpu().detach().numpy()  

            combining_out_3 = torch.argmax(((torch.softmax(out_1, dim=1) + torch.softmax(out_2, dim=1)) / 2.0), dim=1).squeeze(0)
            combining_out_3 = combining_out_3.cpu().detach().numpy()  

            out_1 = torch.argmax(torch.softmax(out_1, dim=1), dim=1).squeeze(0)
            out_1 = out_1.cpu().detach().numpy()

            out_2 = torch.argmax(torch.softmax(out_2, dim=1), dim=1).squeeze(0)
            out_2 = out_2.cpu().detach().numpy()
            
            pred = zoom(combining_out_1, (x / patch_size[0], y / patch_size[1]), order=0)
            combining_prediction_1[ind] = pred

            pred = zoom(combining_out_2, (x / patch_size[0], y / patch_size[1]), order=0)
            combining_prediction_2[ind] = pred
            
            pred = zoom(combining_out_3, (x / patch_size[0], y / patch_size[1]), order=0)
            combining_prediction_3[ind] = pred
            
            pred = zoom(out_1, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction_1[ind] = pred

            pred = zoom(out_2, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction_2[ind] = pred
        
    combining_metric_list_1 = []
    combining_metric_list_2 = []
    combining_metric_list_3 = []
    metric_list_1 = []
    metric_list_2 = []
    for i in range(1, classes):
        combining_metric_list_1.append(calculate_metric_percase(
            combining_prediction_1 == i, label == i))
        combining_metric_list_2.append(calculate_metric_percase(
            combining_prediction_2 == i, label == i))
        combining_metric_list_3.append(calculate_metric_percase(
            combining_prediction_3 == i, label == i))
        metric_list_1.append(calculate_metric_percase(
            prediction_1 == i, label == i))
        metric_list_2.append(calculate_metric_percase(
            prediction_2 == i, label == i))
    return combining_metric_list_1, combining_metric_list_2, combining_metric_list_3, metric_list_1, metric_list_2

def Inference_model1(FLAGS):
    print("——Starting the Model1 Prediction——")
    with open(FLAGS.root_path + '/test.list', 'r') as f:image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]for item in image_list])
    snapshot_path = "./model/Co_Training_ema_add_udd_v2_test/second_ACDC_{}_{}_labeled".format(FLAGS.exp, FLAGS.labeled_num)
    test_save_path = "./model/Co_Training_ema_add_udd_v2_test/second_ACDC_{}_{}/{}_predictions_model/".format(FLAGS.exp, FLAGS.labeled_num, FLAGS.model_1)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    HF_pre_trained_model = os.path.join(snapshot_path,'{}_HF_best_model.pth'.format(FLAGS.model_1))
    LF_pre_trained_model = os.path.join(snapshot_path,'{}_LF_best_model.pth'.format(FLAGS.model_2))

    HF_pre_trained_model = "/root/data/ABD_official/model/Main_AS_woUDD/second_ACDC_train_ACDC_Cross_Teaching_7_labeled/HF_iter_10000_dice_0.8906.pth"
    LF_pre_trained_model = "/root/data/ABD_official/model/Main_AS_woUDD/second_ACDC_train_ACDC_Cross_Teaching_7_labeled/LF_iter_9600_dice_0.8861.pth"
    # model1 = net_factory(net_type=FLAGS.model_1, in_chns=1,class_num=FLAGS.num_classes)
    model1 = BCP_net(in_chns=1, class_num=FLAGS.num_classes) 
    # model1 = UNet(in_chns=1, class_num=FLAGS.num_classes).cuda() 


    model2 = ViT_seg(config, img_size=FLAGS.image_size, num_classes=FLAGS.num_classes).cuda()  
    model2.load_from(config)

    load_net(model1, HF_pre_trained_model)
    load_net(model2, LF_pre_trained_model)
    print("HF model init weight from {}".format(HF_pre_trained_model))
    print("LF model init weight from {}".format(LF_pre_trained_model))
    model1.eval()
    model2.eval()

    combining_metric_list_1 = 0.0
    combining_metric_list_2 = 0.0
    combining_metric_list_3 = 0.0
    metric_list_1 = 0.0
    metric_list_2 = 0.0

    for case in tqdm(image_list):
        combining_metric_i_1, combining_metric_i_2, combining_metric_i_3, metric_i_1, metric_i_2 = test_single_volume_two_model(case, model1, model2, test_save_path, FLAGS, separation=True)
        combining_metric_list_1 += np.array(combining_metric_i_1)
        combining_metric_list_2 += np.array(combining_metric_i_2)
        combining_metric_list_3 += np.array(combining_metric_i_3)
        metric_list_1 += np.array(metric_i_1)
        metric_list_2 += np.array(metric_i_2)
    
    combining_metric_list_1 = combining_metric_list_1 / len(image_list)
    combining_metric_list_2 = combining_metric_list_2 / len(image_list)
    combining_metric_list_3 = combining_metric_list_3 / len(image_list)
    metric_list_1 = metric_list_1 / len(image_list)
    metric_list_2 = metric_list_2 / len(image_list)
    
    for class_i in range(FLAGS.num_classes - 1):
        print('******************************')

        print('test_{}_dice_combining_1:'.format(class_i+1), combining_metric_list_1[class_i, 0])
        print('test_{}_jc_combining_1:'.format(class_i+1), combining_metric_list_1[class_i, 1])
        print('test_{}_hd95_combining_1:'.format(class_i+1), combining_metric_list_1[class_i, 2])
        print('test_{}_asd_combining_1:'.format(class_i+1), combining_metric_list_1[class_i, 3])

        print('test_{}_dice_combining_2:'.format(class_i+1), combining_metric_list_2[class_i, 0])
        print('test_{}_jc_combining_2:'.format(class_i+1), combining_metric_list_2[class_i, 1])
        print('test_{}_hd95_combining_2:'.format(class_i+1), combining_metric_list_2[class_i, 2])
        print('test_{}_asd_combining_2:'.format(class_i+1), combining_metric_list_2[class_i, 3])

        print('test_{}_dice_combining_3:'.format(class_i+1), combining_metric_list_3[class_i, 0])
        print('test_{}_jc_combining_3:'.format(class_i+1), combining_metric_list_3[class_i, 1])
        print('test_{}_hd95_combining_3:'.format(class_i+1), combining_metric_list_3[class_i, 2])
        print('test_{}_asd_combining_3:'.format(class_i+1), combining_metric_list_3[class_i, 3])

        print('test_{}_dice_1:'.format(class_i+1), metric_list_1[class_i, 0])
        print('test_{}_jc_1:'.format(class_i+1), metric_list_1[class_i, 1])
        print('test_{}_hd95_1:'.format(class_i+1), metric_list_1[class_i, 2])
        print('test_{}_asd_1:'.format(class_i+1), metric_list_1[class_i, 3])

        print('test_{}_dice_2:'.format(class_i+1), metric_list_2[class_i, 0])
        print('test_{}_jc_2:'.format(class_i+1), metric_list_2[class_i, 1])
        print('test_{}_hd95_2:'.format(class_i+1), metric_list_2[class_i, 2])
        print('test_{}_asd_2:'.format(class_i+1), metric_list_2[class_i, 3])

        print('******************************Total********************************')
        performance_combining_1 = np.mean(combining_metric_list_1, axis=0)[0]
        performance_combining_2 = np.mean(combining_metric_list_2, axis=0)[0]
        performance_combining_3 = np.mean(combining_metric_list_3, axis=0)[0]
        performance_1 = np.mean(metric_list_1, axis=0)[0]
        performance_2 = np.mean(metric_list_2, axis=0)[0]
        print('*******DSC*******')
        print('performance_combining_1:',performance_combining_1)
        print('performance_combining_2:',performance_combining_2)
        print('performance_combining_3:',performance_combining_3)
        print('performance_1:',performance_1)
        print('performance_2:',performance_2)

        mean_jc_combining_1 = np.mean(combining_metric_list_1, axis=0)[1]
        mean_jc_combining_2 = np.mean(combining_metric_list_2, axis=0)[1]
        mean_jc_combining_3 = np.mean(combining_metric_list_3, axis=0)[1]
        mean_jc_1 = np.mean(metric_list_1, axis=0)[1]
        mean_jc_2 = np.mean(metric_list_2, axis=0)[1]
        print('*******Jaccard*******')
        print('mean_jc_combining_1:',mean_jc_combining_1)
        print('mean_jc_combining_2:',mean_jc_combining_2)
        print('mean_jc_combining_3:',mean_jc_combining_3)
        print('mean_jc_1:',mean_jc_1)
        print('mean_jc_2:',mean_jc_2)

        mean_hd951_combining_1 = np.mean(combining_metric_list_1, axis=0)[2]
        mean_hd951_combining_2 = np.mean(combining_metric_list_2, axis=0)[2]
        mean_hd951_combining_3 = np.mean(combining_metric_list_3, axis=0)[2]
        mean_hd951_1 = np.mean(metric_list_1, axis=0)[2]
        mean_hd951_2 = np.mean(metric_list_2, axis=0)[2]
        print('*******95HD*******')
        print('mean_hd951_combining_1:',mean_hd951_combining_1)
        print('mean_hd951_combining_2:',mean_hd951_combining_2)
        print('mean_hd951_combining_3:',mean_hd951_combining_3)
        print('mean_hd951_1:',mean_hd951_1)
        print('mean_hd951_2:',mean_hd951_2)

        mean_asd_combining_1 = np.mean(combining_metric_list_1, axis=0)[3]
        mean_asd_combining_2 = np.mean(combining_metric_list_2, axis=0)[3]
        mean_asd_combining_3 = np.mean(combining_metric_list_3, axis=0)[3]
        mean_asd_1 = np.mean(metric_list_1, axis=0)[3]
        mean_asd_2 = np.mean(metric_list_2, axis=0)[3]
        print('*******ASD*******')
        print('mean_asd_combining_1:',mean_asd_combining_1)
        print('mean_asd_combining_2:',mean_asd_combining_2)
        print('mean_asd_combining_3:',mean_asd_combining_3)
        print('mean_asd_1:',mean_asd_1)
        print('mean_asd_2:',mean_asd_2)

    return 0
    # avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]
    # average = (avg_metric[0]+avg_metric[1]+avg_metric[2])/3
    # print(avg_metric)
    # print(average)
    # with open(os.path.join(test_save_path, 'performance.txt'), 'w') as file:
    #     file.write(str(avg_metric) + '\n')
    #     file.write(str(average) + '\n')
    # return avg_metric

def Inference_model2(FLAGS):
    print("——Starting the Model2 Prediction——")
    with open(FLAGS.root_path + '/test.list', 'r') as f:image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]for item in image_list])
    snapshot_path = "/data/chy_data/ABD-main/model/Cross_Teaching/ACDC_{}_{}".format(FLAGS.exp, FLAGS.labeled_num)
    test_save_path = "/data/chy_data/ABD-main/model/Cross_Teaching/ACDC_{}_{}/{}_predictions_model/".format(FLAGS.exp, FLAGS.labeled_num, FLAGS.model_2)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    net = ViT_seg(config, img_size=FLAGS.image_size, num_classes=FLAGS.num_classes).cuda()
    net.load_from(config)
    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model_2))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]
    average = (avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3
    print(avg_metric)
    print(average)
    with open(os.path.join(test_save_path, 'performance.txt'), 'w') as file:
        file.write(str(avg_metric) + '\n')
        file.write(str(average) + '\n')
    return avg_metric

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    FLAGS = parser.parse_args()
    metric_model1 = Inference_model1(FLAGS)
    # metric_model2 = Inference_model2(FLAGS)