import numpy as np
from keras layers, models, optimizers
from keras import backend as K
from keras.utils import to_cateforical
import matplotlib.plot as plt
from utils import combine_images
from PIL import combine_images
from CapsuleLayer import CapsuleLayer, PrimaryCap, Length, Mask

K.set_image_data_format('channels_last')

def margin_loss(y_true, y_pred):
    """
    """
    L = y_true * K.square(K.maximun(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * K.square(K.maximun(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def CapsNet(intput_shape, n_class, num_routing):
    """
    :params input_shape [None, width, height, channels]
    :params n_class
    :params num_routing number of routing iterations
    :return A Keras model with 2 inputs (image, label)
            and 2 outputs (capsule otuput and reconstruct image)
    """
    # Image
    x = layers.Input(shape = input_shape)

    # ReLu Conv1
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')

    # PrimaryCap: Conv2D layer with squash activation
    # reshape to [None, num of capsules, dim of vector]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # DigitCap: capsule layer, route-by-agreement
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='digitcaps')(primarycaps)

    # Decoder
    y = layers.Input(shape(n_class,))
    masked_by_y = Mask()([digitcaps, y]) # using capsule masked by the true label, for trianing
    masked = Mask()(digitcaps) # using the capsule with the maximal length, for prediction
    # Share Decoder model in training and predicting
    decoder = model.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Different models for training and prediction
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder[masked]])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))

    return train_model, eval_model, manipulate_model

def train(model, data, args):
    """
    :params model
    :params data: tuple containing training and testing data pairs
                    ((x_train, y_train), (x_test, y_test))
    :return trained model
    """
    # unpack
    (x_train, y_train), (x_test, y_test) = data
