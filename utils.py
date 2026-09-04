import numpy as np
import math, os, torch
import matplotlib.pyplot as plt


# reference from https://github.com/jianzhangcs/ISTA-Net-PyTorch
def my_zero_pad(img, block_size=32):
    old_h, old_w = img.shape
    delta_h = (block_size - np.mod(old_h, block_size)) % block_size
    delta_w = (block_size - np.mod(old_w, block_size)) % block_size
    img_pad = np.concatenate((img, np.zeros([old_h, delta_w])), axis=1)
    img_pad = np.concatenate((img_pad, np.zeros([delta_h, old_w + delta_w])), axis=0)
    new_h, new_w = img_pad.shape
    return img, old_h, old_w, img_pad, new_h, new_w


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# reference from https://github.com/cszn
def H(img, mode, inv=False):
    if inv:
        mode = [0, 1, 2, 5, 4, 3, 6, 7][mode]
    if mode == 0:
        return img
    elif mode == 1:
        return img.rot90(1, [2, 3]).flip([2])
    elif mode == 2:
        return img.flip([2])
    elif mode == 3:
        return img.rot90(3, [2, 3])
    elif mode == 4:
        return img.rot90(2, [2, 3]).flip([2])
    elif mode == 5:
        return img.rot90(1, [2, 3])
    elif mode == 6:
        return img.rot90(2, [2, 3])
    elif mode == 7:
        return img.rot90(3, [2, 3]).flip([2])


def save_loss_plot(loss, path):
    plt.figure(figsize=(10, 6))
    plt.plot(loss, label='Loss', color='blue', linewidth=2)

    plt.title('Loss Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)

    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(path, 'loss.png'))
    plt.close()  


def add_gaussian_noise(image, noise_std):
    if isinstance(noise_std, (int, float)):
        noise_std = torch.tensor(noise_std, device=image.device, dtype=image.dtype)
    noise = torch.randn_like(image) * noise_std
    return torch.clamp(image + noise, 0, 1)
