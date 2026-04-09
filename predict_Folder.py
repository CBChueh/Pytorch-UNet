import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageOps
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--inputFolder', '-i', default="./", metavar='INPUT', nargs='+', help='Folder of input images')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    
    return parser.parse_args()




def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')
    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')


    folder = args.inputFolder[0]

    targetFolder="OCT_images"
    for root, dirs, files in os.walk(folder):
        if not targetFolder in dirs:
            continue
        subfolder=os.path.join(root,targetFolder)
        folder_new=subfolder+'_mask_predict'
        folder_new2=subfolder+'_mask_predict_overlay'
        os.makedirs(folder_new, exist_ok=True)
        os.makedirs(folder_new2, exist_ok=True)
        for i, in_file in enumerate(os.listdir(subfolder)):
            if not in_file.endswith('.jpg'):
                continue
            img = Image.open(os.path.join(subfolder,in_file))
            mask = predict_img(net=net,
                            full_img=img,
                            scale_factor=args.scale,
                            out_threshold=args.mask_threshold,
                            device=device)
            result = mask_to_image(mask, mask_values).convert('L')
            print(result.mode)

            black_channel = Image.new('L', result.size, 0)
            result_colorful = Image.merge("RGB", ( ImageOps.invert(result), black_channel,result))
            result_blend=Image.blend(result_colorful, img.convert('RGB'), 1.0 / 2)
            out_filename=os.path.join(folder_new,in_file)
            out_filename2=os.path.join(folder_new2,in_file)
            result.save(out_filename)
            result_blend.save(out_filename2)
            logging.info(f'{i} Mask saved to {out_filename}')


