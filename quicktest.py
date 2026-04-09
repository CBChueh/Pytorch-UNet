# %%
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageOps

maskpath="/home/cbchueh/Documents/Pytorch-UNet/output_picture/x_-9mm_y_0mm_20251218_175305/OCT_images_mask_predict/x_-9mm_y_0mm_20251218_175305_frame_086.jpg"
imgpath="/home/cbchueh/Documents/Pytorch-UNet/output_picture/x_-9mm_y_0mm_20251218_175305/OCT_images/x_-9mm_y_0mm_20251218_175305_frame_086.jpg"

img = Image.open(imgpath)
mask = Image.open(maskpath)
black_channel = Image.new('L', mask.size, 0)
blue_3channel = Image.merge("RGB", ( ImageOps.invert(mask), black_channel,mask))

fig, ax = plt.subplots(1, 3)
ax[0].imshow(img,cmap='gray')
ax[1].imshow(Image.blend(blue_3channel, img.convert('RGB'), 1.0 / 2))
ax[2].imshow(blue_3channel)

plt.show()
np.array(mask).max()