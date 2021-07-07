import os
import numpy as np
import math
from keras import applications
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input, concatenate
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.regularizers import l1_l2

from config import config
from data import data, INPUT_FORM_PARAMETERS

MODEL_NAME = "v1"

OPTIMIZERS = {
    "sgd-01-0.9": lambda: optimizers.SGD(lr=0.01, momentum=0.9),
    "sgd-001-0.9": lambda: optimizers.SGD(lr=0.001, momentum=0.9),
    "sgd-0001-0.9": lambda: optimizers.SGD(lr=0.0001, momentum=0.9),
    "sgd-01-0.9-nesterov": lambda: optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
    "sgd-001-0.9-nesterov": lambda: optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True),
    "sgd-0001-0.9-nesterov": lambda: optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True),
    "adam": lambda: "adam",
    "nadam": lambda: "nadam",
}
