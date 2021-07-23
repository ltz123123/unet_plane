from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from network import unet


img_size = (256, 256, 3)
models = [unet(img_size), unet(img_size)]
models[0].load_weights("model1_weights.h5")
models[1].load_weights("model2_weights.h5")
img_dir = "test_img1"
n_img = len(os.listdir(img_dir))


fig, ax = plt.subplots(n_img, 5, figsize=(10, n_img*2))
for i, img_name in enumerate(os.listdir(img_dir)):
    img = img_to_array(
        load_img(os.path.join(img_dir, img_name), target_size=img_size[:2]),
    )
    ax[i, 0].imshow(img.astype(np.int))
    ax[i, 0].axis("off")

    for j in range(2):
        predicted = models[j](img[np.newaxis, ...]/255.0, training=False)[0]

        ax[i, 1 + j * 2].imshow(predicted, cmap="gray")
        ax[i, 1 + j * 2].axis("off")

        ax[i, 2 + j * 2].imshow(img.astype(int))
        ax[i, 2 + j * 2].imshow(predicted, alpha=0.4, cmap="Blues")
        ax[i, 2 + j * 2].axis("off")

ax[0, 0].set_title("Inputs")
ax[0, 1].set_title("with aug")
ax[0, 3].set_title("without aug")

fig.tight_layout()
fig.savefig("result.jpg")
plt.show()






