import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import time
import scipy.ndimage
import random

def gaussian_kernel(kernel_size=21, sigma=5):
    x = torch.arange(kernel_size).float() - kernel_size // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    return gauss

def apply_gaussian_filter(img, kernel_size=21, sigma=5):
    kernel_1d = gaussian_kernel(kernel_size, sigma)
    kernel_3d = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
    kernel_3d = kernel_3d.to(img.device)
    
    padding = kernel_size // 2
    img = img.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    low_freq = F.conv3d(img, kernel_3d.unsqueeze(0).unsqueeze(0), padding=padding)
    low_freq = low_freq.squeeze()  # Remove batch and channel dimensions
    return low_freq

def separate_frequencies_3d(image, kernel_size=5, sigma=None):

    if sigma is None:
        sigma = (kernel_size - 1) / 6  # 经验公式来推导sigma

    device = image.device

    # 高斯模糊（低频信息）
    # low_freq = apply_gaussian_filter(image, kernel_size, sigma)
    low_freq = torch.tensor(scipy.ndimage.gaussian_filter(image.to("cpu").numpy(), sigma=sigma), dtype=torch.float, device=device)

    # 高频信息
    high_freq = (image - low_freq).float()
    
    return image, low_freq, high_freq

def add_gaussian_noise_3d(image, mean=0, sigma=25):
    noise = torch.randn(image.shape).to(image.device) * sigma + mean
    noisy_image = image + noise
    return torch.clamp(noisy_image, image.min()-1, image.max()+1)

def add_gaussian_noise_3d_2(data_sample: np.ndarray, noise_variance= (0, 1.5),
                           p_per_channel: float = 0.5, per_channel: bool = True) -> np.ndarray:
    if not per_channel:
        variance = noise_variance[0] if noise_variance[0] == noise_variance[1] else \
            random.uniform(noise_variance[0], noise_variance[1])
    else:
        variance = None
    for c in range(data_sample.shape[0]):
        if np.random.uniform() < p_per_channel:
            variance_here = variance if variance is not None else \
                noise_variance[0] if noise_variance[0] == noise_variance[1] else \
                    random.uniform(noise_variance[0], noise_variance[1])
            data_sample[c] = data_sample[c] + np.random.normal(0.0, variance_here, size=data_sample[c].shape)
    return data_sample

def gaussian_blur_3d(image, kernel_size=5, sigma=None):
    if sigma is None:
        sigma = (kernel_size - 1) / 6
    return torch.tensor(scipy.ndimage.gaussian_filter(image.cpu().numpy(), sigma=sigma), dtype=torch.float, device=image.device)

def augment_gaussian_blur(data_sample: np.ndarray, sigma_range=(0.5,1), per_channel: bool = True,
                          p_per_channel: float = 1, different_sigma_per_axis: bool = False,
                          p_isotropic: float = 0) -> np.ndarray:
    if not per_channel:
        sigma = get_range_val(sigma_range) if ((not different_sigma_per_axis) or
                                               ((np.random.uniform() < p_isotropic) and
                                                different_sigma_per_axis)) \
            else [get_range_val(sigma_range) for _ in data_sample.shape[1:]]
    else:
        sigma = None
    for c in range(data_sample.shape[0]):
        if np.random.uniform() <= p_per_channel:
            if per_channel:
                sigma = get_range_val(sigma_range) if ((not different_sigma_per_axis) or
                                                       ((np.random.uniform() < p_isotropic) and
                                                        different_sigma_per_axis)) \
                    else [get_range_val(sigma_range) for _ in data_sample.shape[1:]]
            data_sample[c] = scipy.ndimage.gaussian_filter(data_sample[c], sigma, order=0)
    return data_sample

def get_range_val(value, rnd_type="uniform"):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 2:
            if value[0] == value[1]:
                n_val = value[0]
            else:
                orig_type = type(value[0])
                if rnd_type == "uniform":
                    n_val = random.uniform(value[0], value[1])
                elif rnd_type == "normal":
                    n_val = random.normalvariate(value[0], value[1])
                n_val = orig_type(n_val)
        elif len(value) == 1:
            n_val = value[0]
        else:
            raise RuntimeError("value must be either a single value or a list/tuple of len 2")
        return n_val
    else:
        return value

def get_two_frequency_domain(images, sep_kernel_range_HF=(13,25), sep_kernel_range_LF=(9,23), need_visualization=False, save_path=None):
    B, C, _, _ =  images.shape
    HF_view = torch.zeros_like(images)
    LF_view = torch.zeros_like(images)
    for i in range(B):
        # start_time = time.time()
        image = images[i, 0, :, :]

        if need_visualization:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            slice_index = original_img.shape[2] // 2
            plt.imsave(os.path.join(save_path, f'original_img_slice_{i}.png'), original_img[:, :, slice_index].cpu().numpy(), cmap='gray')
            plt.imsave(os.path.join(save_path, f'low_freq_slice_{i}.png'), low_freq[:, :, slice_index].cpu().numpy(), cmap='gray')
            plt.imsave(os.path.join(save_path, f'high_freq_slice_{i}.png'), high_freq[:, :, slice_index].cpu().numpy(), cmap='gray')

            plt.imsave(os.path.join(save_path, f'high_freq_noise_img{i}.png'), high_freq_noise_img[:, :, slice_index].cpu().numpy(), cmap='gray')
            plt.imsave(os.path.join(save_path, f'low_freq_blur_img{i}.png'), low_freq_blur_img[:, :, slice_index].cpu().numpy(), cmap='gray')
            breakpoint()

        view_selection = random.choice([1, 1])


        if view_selection == 0:
            HF_view[i, 0, :, :] = image
            LF_view[i, 0, :, :] = image

        elif view_selection == 1:
            kernel_size_HF = sep_kernel_range_HF[0] if sep_kernel_range_HF[0]==sep_kernel_range_HF[1] else random.uniform(sep_kernel_range_HF[0],sep_kernel_range_HF[1])
            kernel_size_LF = sep_kernel_range_LF[0] if sep_kernel_range_LF[0]==sep_kernel_range_LF[1] else random.uniform(sep_kernel_range_LF[0],sep_kernel_range_LF[1])
            
            original_img, _, high_freq = separate_frequencies_3d(image, kernel_size=kernel_size_HF)
            original_img, low_freq, _ = separate_frequencies_3d(image, kernel_size=kernel_size_LF)
            
            HF_view[i, 0, :, :] = 3*high_freq + original_img
            LF_view[i, 0, :, :] = 3*low_freq + original_img

            # save_path = './save_image'
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # plt.imsave(os.path.join(save_path, f'original_img.png'), image.cpu().numpy(), cmap='gray')
            # plt.imsave(os.path.join(save_path, f'high_freq_img.png'), high_freq.cpu().numpy(), cmap='gray')
            # plt.imsave(os.path.join(save_path, f'low_freq_img.png'), low_freq.cpu().numpy(), cmap='gray')
            # breakpoint()

        elif view_selection == 2:
            high_freq_noise_img = add_gaussian_noise_3d_2(image, noise_variance= (0, 1.5),p_per_channel= 0.5, per_channel=False)
            low_freq_blur_img = gaussian_blur_3d(image, kernel_size=random.choice([1, 3, 5]))
            HF_view[i, 0, :, :] = high_freq_noise_img
            LF_view[i, 0, :, :] = low_freq_blur_img

        elif view_selection == 3:
            filtered_image_high,filtered_image_low= fourier_transformation(image)
            k1=1.5
            k2=1

            HF_view[i, 0, :, :] = k1*filtered_image_high + image
            LF_view[i, 0, :, :] = k1*filtered_image_low + image

    return HF_view.float(), LF_view.float()

# 只用增强，从弱到强

def fourier_transformation(image_array):
    pass_r = 28

    # 执行傅里叶变换并移到中心
    fft_image = torch.fft.fft2(image_array)
    fft_image_shifted = torch.fft.fftshift(fft_image)
    
    # 获取图像中心点位置
    rows, cols = image_array.shape
    crow, ccol = rows // 2, cols // 2  # Center position

    # 创建低通滤波器
    mask_low = torch.zeros((rows, cols), dtype=image_array.dtype, device=image_array.device)
    x = torch.arange(rows, device=image_array.device)
    y = torch.arange(cols, device=image_array.device)
    x, y = torch.meshgrid(x, y, indexing='ij')
    mask_area_low = (x - crow) ** 2 + (y - ccol) ** 2 <= pass_r ** 2
    mask_low[mask_area_low] = 1

    # 创建高通滤波器
    mask_high = torch.ones((rows, cols), dtype=image_array.dtype, device=image_array.device)
    mask_area_high = (x - crow) ** 2 + (y - ccol) ** 2 <= pass_r ** 2
    mask_high[mask_area_high] = 0

    # 应用低通滤波器
    filtered_fft_image_low = fft_image_shifted * mask_low
    ifft_image_shifted_low = torch.fft.ifftshift(filtered_fft_image_low)
    filtered_image_low = torch.fft.ifft2(ifft_image_shifted_low)
    filtered_image_low = torch.abs(filtered_image_low)

    # 应用高通滤波器
    filtered_fft_image_high = fft_image_shifted * mask_high
    ifft_image_shifted_high = torch.fft.ifftshift(filtered_fft_image_high)
    filtered_image_high = torch.fft.ifft2(ifft_image_shifted_high)
    filtered_image_high = torch.abs(filtered_image_high)

    return filtered_image_high, filtered_image_low
# 只用增强，从弱到强

##########################################3d#########################
def get_two_frequency_domain_3d(images, sep_kernel_range_HF=(13, 25), sep_kernel_range_LF=(9, 23), need_visualization=False, save_path=None):
    B, C, D, H, W = images.shape  # Assuming images is a 5D tensor: (Batch, Channels, Depth, Height, Width)
    HF_view = torch.zeros_like(images)
    LF_view = torch.zeros_like(images)

    for i in range(B):
        for c in range(C):
            image = images[i, c, :, :, :]

            if need_visualization:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                slice_index = image.shape[0] // 2
                plt.imsave(os.path.join(save_path, f'original_img_slice_{i}_{c}.png'), image[slice_index, :, :].cpu().numpy(), cmap='gray')

            view_selection = random.choice([1, 1])

            if view_selection == 0:
                HF_view[i, c, :, :, :] = image
                LF_view[i, c, :, :, :] = image

            elif view_selection == 1:
                kernel_size_HF = sep_kernel_range_HF[0] if sep_kernel_range_HF[0] == sep_kernel_range_HF[1] else random.uniform(sep_kernel_range_HF[0], sep_kernel_range_HF[1])
                kernel_size_LF = sep_kernel_range_LF[0] if sep_kernel_range_LF[0] == sep_kernel_range_LF[1] else random.uniform(sep_kernel_range_LF[0], sep_kernel_range_LF[1])

                original_img, _, high_freq = separate_frequencies_3d(image, kernel_size=kernel_size_HF)
                original_img, low_freq, _ = separate_frequencies_3d(image, kernel_size=kernel_size_LF)

                HF_view[i, c, :, :, :] = high_freq + original_img
                LF_view[i, c, :, :, :] = low_freq + original_img

                if need_visualization:
                    plt.imsave(os.path.join(save_path, f'high_freq_slice_{i}_{c}.png'), high_freq[slice_index, :, :].cpu().numpy(), cmap='gray')
                    plt.imsave(os.path.join(save_path, f'low_freq_slice_{i}_{c}.png'), low_freq[slice_index, :, :].cpu().numpy(), cmap='gray')

            elif view_selection == 2:
                high_freq_noise_img = add_gaussian_noise_3d_2(image, noise_variance=(0, 1.5), p_per_channel=0.5, per_channel=False)
                low_freq_blur_img = gaussian_blur_3d(image, kernel_size=random.choice([1, 3, 5]))

                HF_view[i, c, :, :, :] = high_freq_noise_img
                LF_view[i, c, :, :, :] = low_freq_blur_img

                if need_visualization:
                    plt.imsave(os.path.join(save_path, f'high_freq_noise_img_{i}_{c}.png'), high_freq_noise_img[slice_index, :, :].cpu().numpy(), cmap='gray')
                    plt.imsave(os.path.join(save_path, f'low_freq_blur_img_{i}_{c}.png'), low_freq_blur_img[slice_index, :, :].cpu().numpy(), cmap='gray')

    return HF_view.float(), LF_view.float()