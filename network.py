import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Input, Dropout, concatenate
from tensorflow.keras.models import Model


def unet(input_shape):
    inputs = Input(input_shape)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer="he_uniform")(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer="he_uniform")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer="he_uniform")(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer="he_uniform")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer="he_uniform")(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer="he_uniform")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer="he_uniform")(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer="he_uniform")(conv4)
    drop4 = Dropout(0.3)(conv4)
    pool4 = MaxPooling2D((2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer="he_uniform")(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer="he_uniform")(conv5)
    drop5 = Dropout(0.3)(conv5)

    up6 = UpSampling2D((2, 2))(drop5)
    merge6 = concatenate([drop4, up6], axis=-1)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer="he_uniform")(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer="he_uniform")(conv6)

    up7 = UpSampling2D((2, 2))(conv6)
    merge7 = concatenate([conv3, up7], axis=-1)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer="he_uniform")(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer="he_uniform")(conv7)

    up8 = UpSampling2D((2, 2))(conv7)
    merge8 = concatenate([conv2, up8], axis=-1)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer="he_uniform")(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer="he_uniform")(conv8)

    up9 = UpSampling2D((2, 2))(conv8)
    merge9 = concatenate([conv1, up9], axis=-1)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer="he_uniform")(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer="he_uniform")(conv9)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)

    return model
