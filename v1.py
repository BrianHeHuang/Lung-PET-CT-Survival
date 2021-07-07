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

def model(input_form="all", aux_size=0, hyperparameters=dict()):
    print("using the following hyperparameters: {}".format(hyperparameters))

    if input_form == "features":
        return features_model(aux_size, hyperparameters)

    parameters = INPUT_FORM_PARAMETERS[input_form]

    inputs = list()
    outputs = list()

    #retreiving the hyperparameters
    DROPOUT = hyperparameters.get("dropout", 0.5)
    OPTIMIZER = hyperparameters.get("optimizer", "sgd-0001-0.9")
    DEEP_DENSE_TOP = hyperparameters.get("deep-dense-top", True)

    #skip for now
    '''
    if parameters["t2"]:
        convnet = applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
        )
        for layer in convnet.layers:
            layer.name = "{}_t2".format(layer.name)
        out = convnet.output
        out = Flatten()(out)
        inputs.append(convnet.input)
        outputs.append(out)
    '''
    if parameters["t1"]:
        # init ResNet
        convnet = applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
        )
        out = convnet.output
        out = Flatten()(out)
        inputs.append(convnet.input)
        outputs.append(out)

    if len(outputs) > 1:
        out = concatenate(outputs)
    else:
        out = outputs[0]

    out = Dense(256, activation="relu", kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(out)

    if DEEP_DENSE_TOP:
        out = Dropout(DROPOUT)(out)
        out = Dense(128, activation="relu", kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(out)
        out = Dropout(DROPOUT)(out)
        out = Dense(64, activation="relu", kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(out)
        out = Dropout(DROPOUT)(out)
        out = Dense(32, activation="relu", kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(out)
        out = Dropout(DROPOUT)(out)

    if parameters["features"]:
        aux_input = Input(shape=(aux_size,), name='aux_input')
        inputs.append(aux_input)
        out = concatenate([out, aux_input])

    out = Dense(16, activation="relu", kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(out)
    predictions = Dense(1, activation="sigmoid", kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(out)

    # creating the final model
    if len(inputs) > 1:
        model = Model(inputs=inputs, outputs=predictions)
    else:
        model = Model(inputs=inputs[0], outputs=predictions)

    # compile the model
    model.compile(
        loss="binary_crossentropy",
        optimizer=OPTIMIZERS[OPTIMIZER](),
        metrics=["accuracy"])

    return model

def features_model(aux_size, hyperparameters):
    OPTIMIZER = hyperparameters.get("optimizer", "sgd-0001-0.9")
    aux_input = Input(shape=(aux_size,), name='aux_input')
    reg=l1_l2(l1=0.00, l2=0.01)
    predictions = Dense(1, kernel_regularizer=reg, activation="sigmoid")(aux_input)
    model = Model(inputs=aux_input, outputs=predictions)
    model.compile(
        loss="binary_crossentropy",
        optimizer=OPTIMIZERS[OPTIMIZER](),
        metrics=["accuracy"])
    return model

def class_weight(training):
    unique, counts = np.unique(training.classes, return_counts=True)
    raw_counts = dict(zip(unique, counts))
    return { k: len(training.classes)/v for k, v in raw_counts.items() }

def train(model, training, validation, run_id, monitor):
    # callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(
            config.MODEL_DIR,
            "{}-{}.h5".format(
                str(run_id),
                MODEL_NAME,
            ),
        ),
        monitor=monitor,
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1,
    )
    early = EarlyStopping(
        monitor=monitor,
        min_delta=0,
        patience=config.PATIENCE,
        verbose=1,
        mode='auto',
    )
    # Train the model - fit_generator from keras
    history = model.fit_generator(
        training,
        steps_per_epoch=training.n / config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=validation,
        validation_steps=math.ceil(validation.n / config.BATCH_SIZE),
        class_weight=class_weight(training),
        callbacks=[checkpoint, early],
    )
    return history.history


# first called by the main run function
def run(run_id=None, mode='normal', loaded_data=None, split_id=None, input_form=config.INPUT_FORM,  label_form="outcome", hyperparameters=dict()):
    if run_id is None:
        run_id = int(datetime.utcnow().timestamp())
    if split_id is None:
        split_id = run_id

    if mode == 'normal':
        if loaded_data is None:
            # create the data objects
            training, validation, test = data(split_id, input_form=input_form, label_form=label_form)
        else:
            training, validation, test = loaded_data
        model_instance = model(input_form, aux_size=training.features_size, hyperparameters=hyperparameters)
        # return trained model
        return train(model_instance, training, validation, run_id, 'val_loss')
    elif mode == 'cross':
        # training, validation, test, holdout_test = loaded_data
        training, validation, test = loaded_data
        model_instance = model(input_form, aux_size=training.features_size, hyperparameters=hyperparameters)
        return train(model_instance, training, validation, run_id, 'val_loss')


if __name__ == '__main__':
    run()
