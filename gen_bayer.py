import numpy as np
from PIL import Image
from bayer import bayer
import argparse
import os

def splat(values, y, x):
    h, w = values.shape
    output = np.zeros((h*2, w*2), dtype=np.int16)
    output[y:h*2:2, x:w*2:2] = values
    return output

def select(values, y, x):
    h, w = values.shape
    output = np.zeros((h, w), dtype=np.int16)
    output[y:h:2, x:w:2] = values[y:h:2, x:w:2]
    return output

def gather(values, y, x):
    h, w = values.shape
    output = values[y:h:2, x:w:2]
    return output

def center(values, conv_k, y, x):
    h, w = values.shape
    left_XPAD = conv_k/2 - x
    top_YPAD = conv_k/2 - y
    right_XPAD = conv_k/2 - (1-x)
    bottom_YPAD = conv_k/2 - (1-y)

    XPAD = left_XPAD + right_XPAD
    YPAD = top_YPAD + bottom_YPAD

    centered = np.zeros((h+YPAD, w+XPAD), dtype=np.int16)
    centered[top_YPAD:-bottom_YPAD, left_XPAD:-right_XPAD] = values
    return centered

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Provide image directory')
    parser.add_argument('--input_dir', type=str, help='the path of the input image directory')
    parser.add_argument('--output_dir', type=str, help='the path of the output image directory')
    parser.add_argument('--n', type=int, help='number of images to transform')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    input_shape = (64,64)

    count = 0
    for filename in os.listdir(args.input_dir):
        count += 1
        file_prefix = filename.split(".")[0]
        out_basedir = os.path.join(args.output_dir, file_prefix)
        if not os.path.exists(out_basedir):
            os.mkdir(out_basedir)

        pic = Image.open(os.path.join(args.input_dir,filename))
        im = np.array(pic)
        im = np.transpose(im, (2, 0, 1))

        bayer_mosaic = bayer(im, flat=True)
        bayer_mosaic.tofile(os.path.join(out_basedir, "bayer.data"))
        im.tofile(os.path.join(out_basedir, "image.data"))

        gr_m, r_m, b_m, gb_m = bayer(im, return_masks=True) 

        if (args.debug):
            print(bayer_mosaic[0,0:4,0:4])
            print("R")
            print(im[0,0:6,0:6])
            print("G")
            print(im[1,0:6,0:6])
            print("B")
            print(im[2,0:6,0:6])
            
        print(count)
        if count >= args.n:
            break      

