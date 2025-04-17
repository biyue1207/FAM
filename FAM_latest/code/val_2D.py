import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from utils.getTwoFreDomain import get_two_frequency_domain

def calculate_metric_percase_promise(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        return dice
    else:
        return 0

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0
    
def test_single_volume_promise(image, label, net, classes):
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            prediction[ind] = out
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase_promise(prediction == i, label == i))
    return metric_list

def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = net(input)
            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_two_model(image, label, net_1, net_2, classes, patch_size=[256, 256], separation=True):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction_1 = np.zeros_like(label)
    prediction_2 = np.zeros_like(label)
    combining_prediction_1 = np.zeros_like(label)
    combining_prediction_2 = np.zeros_like(label)
    combining_prediction_3 = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        if separation:
            HF_input, LF_input = get_two_frequency_domain(input)
            HF_input, LF_input = HF_input.cuda(), LF_input.cuda()
        else:
            HF_input, LF_input = input, input
        net_1.eval()
        net_2.eval()
        with torch.no_grad():
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



def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
