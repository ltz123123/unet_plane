import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.model_selection import train_test_split
from network import unet
from tensorflow.keras.optimizers import Adam


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def get_data():
    df = pd.read_csv("annotation.csv")
    img_size = [256, 256, 3]
    imgs = np.zeros([len(df)] + img_size)
    masks = np.zeros([len(df)] + img_size[:-1])

    for idx, row in df.iterrows():
        imgs[idx] = img_to_array(load_img("plane_data/" + row["image_name"]))

        for row_idx in range(2, 2 + row["counts"] * 4, 4):
            coordinates = row[row_idx: row_idx + 4]  # x1, y1, x2, y2
            masks[idx, coordinates[1]: coordinates[3], coordinates[0]: coordinates[2]] = 1

    return train_test_split(imgs, masks, test_size=0.3)


def data_gen(x, y, seed=1):
    img_gen = ImageDataGenerator(
        rotation_range=0.2, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, vertical_flip=True,
        horizontal_flip=True, fill_mode="nearest"
    )
    mask_gen = ImageDataGenerator(
        rotation_range=0.2, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, vertical_flip=True,
        horizontal_flip=True, fill_mode="nearest"
    )

    img_gen.fit(x)
    mask_gen.fit(y)

    img_data_gen = img_gen.flow(x, batch_size=1, seed=seed)
    mask_data_gen = mask_gen.flow(y, batch_size=1, seed=seed)

    for img, mask in zip(img_data_gen, mask_data_gen):
        yield img, mask


x_train, x_test, y_train, y_test = get_data()
model1 = unet([256, 256, 3])
model2 = unet([256, 256, 3])
model1.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
model2.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])


model1.fit(
    data_gen(x_train/255.0, y_train[..., np.newaxis]),
    epochs=5,
    steps_per_epoch=len(x_train),
    validation_data=(x_test/255.0, y_test[..., np.newaxis]),
    validation_steps=len(x_test),
    verbose=1
)

model1.save_weights("model1_weights.h5")

model2.fit(
    x_train/255.0, y_train[..., np.newaxis],
    epochs=5,
    batch_size=1,
    validation_data=(x_test/255.0, y_test[..., np.newaxis]),
    verbose=1
)
model2.save_weights("model2_weights.h5")

