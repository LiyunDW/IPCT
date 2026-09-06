import numpy as np
import os, glob, cv2, argparse, warnings
from time import time
from skimage.metrics import structural_similarity as ssim
from utils import *
from model import *

warnings.filterwarnings("ignore")


def main():
    global args
    args = parser.parse_args()

    if args.model == 'base':
        args.layer = 16
        args.result_dir = 'result/base'
    elif args.model == 'base+':
        args.layer = 25
        args.result_dir = 'result/base+'

    layer, cs_ratio, dim, epochs = args.layer, args.cs_ratio, args.dim, args.epochs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.enabled = True
    model = IPCT(layer, cs_ratio, dim)
    model = nn.DataParallel(model).to(device)

    para = sum(p.numel() for p in model.parameters())
    phi = model.module.Phiweight.numel()
    print("Net para num: %d" % (para - phi))

    model_dir = "./pth/%s/ratio_%.2f_layer_%d_lr_0.00020" % (args.model, cs_ratio, layer)
    checkpoint = torch.load("%s/net_params_%d.pkl" % (model_dir, epochs), map_location=device)
    model.load_state_dict(checkpoint)
    with torch.no_grad():
        for ipath in args.test_name:
            test_image_paths = glob.glob(os.path.join('./data', ipath) + '/*')
            test_image_num = len(test_image_paths)
            PSNR_list, SSIM_list, TIME_list = [], [], []
            for i in range(test_image_num):
                test_image = cv2.imread(test_image_paths[i], 1)
                test_image_ycrcb = cv2.cvtColor(test_image, cv2.COLOR_BGR2YCrCb)
                img, old_h, old_w, img_pad, new_h, new_w = my_zero_pad(test_image_ycrcb[:, :, 0])
                img_pad = img_pad.reshape(1, 1, new_h, new_w) / 255.0
                x_input = torch.from_numpy(img_pad)
                x_input = x_input.type(torch.FloatTensor).to(device)

                start = time()
                x_output = model(x_input)
                run_time = time() - start
                TIME_list.append(run_time)

                x_output = x_output.cpu().data.numpy().squeeze()
                x_output = np.clip(x_output[:old_h, :old_w], 0, 1).astype(np.float64) * 255.0
                PSNR = psnr(x_output, img)
                SSIM = ssim(x_output, img, data_range=255)
                PSNR_list.append(PSNR)
                SSIM_list.append(SSIM)
                name = os.path.split(test_image_paths[i])[-1].split(".")[0]
                print(f"[{i + 1:02d}/{test_image_num:02d}] "
                      f"Run time for {name}: {run_time:.4f}, "
                      f"PSNR: {PSNR:.2f}, SSIM: {SSIM:.4f}")
                if args.save_flag:
                    test_image_ycrcb[:, :, 0] = x_output
                    im_rec_rgb = cv2.cvtColor(test_image_ycrcb, cv2.COLOR_YCrCb2BGR)
                    im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)
                    save_path = os.path.join(args.result_dir, ipath, str(args.cs_ratio))
                    os.makedirs(save_path, exist_ok=True)
                    cv2.imwrite("%s/%s/%s/%s_PSNR_%.2f_SSIM_%.4f.png" % (
                        args.result_dir, ipath, str(args.cs_ratio), name, PSNR, SSIM), im_rec_rgb)
                    del x_output
            log_data = 'CS Ratio: %.2f, %s: PSNR: %.2f, SSIM: %.4f, time: %1.4f.\n' % (
                args.cs_ratio, ipath, float(np.mean(PSNR_list)), float(np.mean(SSIM_list)), float(np.mean(TIME_list)))
            print(log_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='base+', help='model_version (base or base+)')
    parser.add_argument('--cs_ratio', type=float, default=0.1, help='set cs_ratio, {0.01,0.04,0.1,0.25,0.3,0.4,0.5}')
    parser.add_argument('--block_size', type=int, default=32, help='block size (default: 32)')
    parser.add_argument('--save_dir', type=str, default='pth', help='The directory used to save models')
    parser.add_argument('--layer', type=int, default=25, help='phase number (16 or 25)')
    parser.add_argument('--dim', type=int, default=32, help='initial dim')
    parser.add_argument('--save_flag', type=bool, default=True, help='save the test result')
    parser.add_argument('--epochs', type=int, default=100, help='epoch')
    parser.add_argument('--test_name', type=str, default=["Set11"], help='test dataset')
    parser.add_argument('--result_dir', type=str, default='result', help='test result directory')
    parser.add_argument('--lr', '--learning_rate', default=2e-4, type=float, help='initial lr')
    main()
