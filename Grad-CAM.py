import tensorflow as tf
from tensorflow import keras
from config import config
import os
import matplotlib.cm as cm
# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from keras.models import load_model
import nrrd
import h5py
import efficientnet.tfkeras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from skimage.color import rgb2gray
from PIL import Image
import csv
from shutil import rmtree
from collections import defaultdict
from keras.preprocessing.image import ImageDataGenerator, Iterator
from keras import backend as K
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from calculate_features import normalize_column
from filenames import IMAGE, MANUAL_SEGMENTATION, AUTO_SEGMENTATION, T1
from segmentation import calculate_percentile_slice, select_slice, bounding_box, crop, resize, calculate_volume
import pandas as pd
from tensorflow.keras.models import Model
import cv2
import numpy as np
from keras.models import Model
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
import keras.backend as K
from PIL import Image

def load_image(image_path, segmentation_path, verbose=False, dimension=2):
    image, _ = nrrd.read(image_path)
    segmentation, _ = nrrd.read(segmentation_path)
    image_shape = image.shape
    segmentation_shape = segmentation.shape
    if image.shape != segmentation.shape:
        print("shapes not equal!" + str(image_path))
    if image.shape != segmentation.shape:
        print("shapes not equal!")
        width = segmentation.shape[0]
        height = segmentation.shape[1]
        img_stack_sm = np.zeros((width, height, segmentation.shape[2]))
        for idx in range(segmentation.shape[2]):
            img = image[:, :, idx]
            img_sm = resize(img, (width, height))
            img_stack_sm[:, :, idx] = img_sm
        image = img_stack_sm
    if verbose:
        print("""
        image: {}
        seg: {}
""".format(image.shape, segmentation.shape))
    return[mask_image_percentile(image, segmentation, 100, a) for a in (2, 2, 2)]

def mask_image_percentile(image, segmentation, percentile=100, axis=2):
    plane = calculate_percentile_slice(segmentation, percentile, axis)
    image, segmentation = select_slice(image, segmentation, plane, axis)

    bounds = bounding_box(segmentation)
    image, segmentation = crop(image, segmentation, bounds)

    masked = image
    masked = resize(masked, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    return masked

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def generate_gradcam(img_tensor, model, class_index, activation_layer):
    model_input = model.input
    y_c = model.outputs[0].op.inputs[0][0, class_index]
    A_k = model.get_layer(activation_layer).output

    get_output = K.function([model_input], [A_k, K.gradients(y_c, A_k)[0]])
    [conv_output, grad_val] = get_output([img_tensor])

    conv_output = conv_output[0]
    grad_val = grad_val[0]

    weights = np.mean(grad_val, axis=(0, 1))
    grad_cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
    for k, w in enumerate(weights):
        grad_cam += w * conv_output[:, :, k]

    grad_cam = np.absolute(grad_cam)
    return grad_cam, weights

def generate_cam(img_tensor, model, class_index, activation_layer):
    model_input = model.input

    A_k = model.get_layer(activation_layer).output
    get_output = K.function([model_input], [A_k])
    [conv_output] = get_output([img_tensor])

    conv_output = conv_output[0]
    weights = model.layers[-1].get_weights()[0][:, class_index]

    cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
    for k, w in enumerate(weights):
        cam += w * conv_output[:, :, k]
    return cam, weights

model_directory = config.MODEL_DIR
img_direc = "./Grad_cam_fig"
CT_manual = "CT_manual_Repeat"
PET_manual = "PET_manual"
CT_automated = "CT_automated"
PET_automated = "PET_automated"
CT_manual_2 = "New_CT_manual"

MODE = "CT"

model_id = "***"

model_file_path = os.path.join(model_directory, model_id+"-enet.h5")

f = h5py.File(model_file_path,'r+')
data_p = f.attrs['training_config']
data_p = data_p.decode().replace("learning_rate","lr").encode()
f.attrs['training_config'] = data_p
f.close()

model = load_model(model_file_path)
model.summary()
config1 = model.get_config()
print(config1["layers"][0]["config"]["batch_input_shape"]) # return
last_conv_layer_name = "top_conv"

img_width = 224
img_height = 224

test_set = pd.read_csv("***")
for row in tqdm(test_set.iterrows(), total=len(test_set)):
    patient_row = row[1]
    t1_image_file = os.path.join(config.PREPROCESSED_DIR_CT, "{}-{}-{}".format(patient_row["accession"], MODE, IMAGE))
    t1_seg_file = os.path.join(config.PREPROCESSED_DIR_CT,"{}-{}-{}".format(patient_row["accession"], MODE, MANUAL_SEGMENTATION))
    try:
        image, _ = nrrd.read(t1_image_file)
    except:
        print("Img Path doesn't exist")
        t1_image_file = os.path.join(config.PREPROCESSED_DIR_PET,
                                     "{}-{}-{}".format(patient_row["accession"], MODE, IMAGE))
    try:
        segmentation, _ = nrrd.read(t1_seg_file)
    except:
        print("Seg Path doesn't exist")
        t1_seg_file = os.path.join(config.PREPROCESSED_DIR_PET,
                                   "{}-{}-{}".format(patient_row["accession"], MODE, MANUAL_SEGMENTATION))
        segmentation, _ = nrrd.read(t1_seg_file)
    seg_sum = np.sum(segmentation)
    if seg_sum == 0:
        t1_seg_file = os.path.join(config.PREPROCESSED_DIR_PET,
                                   "{}-{}-{}".format(patient_row["accession"], MODE, MANUAL_SEGMENTATION))
    #######
    image_shape = image.shape
    segmentation_shape = segmentation.shape

    t1_masked = load_image(t1_image_file, t1_seg_file, verbose=True, dimension=1)
    t1_masked_array = np.array(t1_masked)
    final_array = np.reshape(t1_masked_array, (224, 224, 3))
    final_array_exp = np.expand_dims(final_array, axis=0)

    top_image = t1_masked[2]


    img = final_array_exp
    preds = model.predict(img)
    predicted_class = preds.argmax(axis=1)[0]

    print("predicted top1 class:", predicted_class)
    conv_name = 'top_conv'
    grad_cam, grad_val = generate_gradcam(img, model, predicted_class, conv_name)
    grad_cam = cv2.resize(grad_cam, (img_width, img_height))

    img_direc = "./Grad_cam_fig"

    plt.imshow(top_image, cmap='gray')
    plt.imshow(grad_cam, cmap="jet", alpha=0.3)
    plt.savefig(os.path.join(img_direc, "New_CT_manual", (patient_row["accession"] + ".png")))
    plt.clf()