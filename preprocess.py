import os
import argparse
import nrrd
import pandas
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from config import config
from filenames import IMAGE, SEGMENTATION, T1

def normalization(CT):
    img = np.true_divide(CT,26453.0)
    return img

def window(x, wl, ww, convert_8bit=True):
    x = np.clip(x, wl-ww/2, wl+ww/2)
    if convert_8bit:
      x = x - np.min(x)
      x = x / np.max(x)
      x = (x * 255).astype('uint8')
    return x

def run(files, out, use_n4_bias=False):
    f = pandas.read_pickle(files)
    for index, row in f.iterrows():
        print(row)
        try:
            print("working on {} {}".format(index, "-" * 40))
            t1 = row.to_frame().loc["path", config.MODE,IMAGE][0]
            print(t1)
            t1_seg = row.to_frame().loc["path", config.MODE,SEGMENTATION][0]
            print(t1_seg)

            t1_nrrd, _ = nrrd.read(t1)
            t1_seg_nrrd, _ = nrrd.read(t1_seg)

            if config.MODE == "CT":
                out_t1 = window(t1_nrrd,-400,1500)
            else:
                out_t1 = t1_nrrd
            #out_t2_image, out_t2_seg = out_t2[0]

            #
            nrrd.write(os.path.join(out, "{}-{}-{}".format(index, config.MODE, IMAGE)), out_t1)
            nrrd.write(os.path.join(out, "{}-{}-{}".format(index, config.MODE, SEGMENTATION)), t1_seg_nrrd)

        except Exception as e:
            print()
            print("#" * 80)
            print("Exception occured for: {}\n{}".format(index, e))
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n4', action='store_true', help="use n4 bias correction")
    parser.add_argument(
        '--preprocess',
        type=str,
        default=config.PREPROCESS,
        help='preprocess file')
    parser.add_argument(
        '--out',
        type=str,
        default=config.PREPROCESSED_DIR,
        help='output')
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS.n4)
    run(FLAGS.preprocess, FLAGS.out, False)
