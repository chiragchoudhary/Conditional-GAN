import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import keras
from keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical, image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import CGAN
from utils import wasserstein_loss
from callbacks import CustomCallback


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype=np.float32)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10, dtype=np.float32)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


def mnist_transform(x):
    return (tf.cast(x, tf.float32) / 127.5) - 1.0


epochs = 50
batch_size = 128

generator_optimizer = Adam(learning_rate=2e-4)
discriminator_optimizer = Adam(learning_rate=1e-4)

loss_fn = wasserstein_loss

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
                .map(lambda x, y: (mnist_transform(x), y)) \
                .batch(batch_size, drop_remainder=True) \
                .shuffle(1024, reshuffle_each_iteration=True)

callback = CustomCallback()

cgan = CGAN()

cgan.compile_model(discriminator_optimizer, generator_optimizer, loss_fn)
cgan.fit(train_data,
         epochs=30,
         callbacks=[callback]
         )

fig, ax = plt.subplots(10, 10, figsize=(20, 20))
for i in range(10):
    labels = tf.keras.utils.to_categorical(np.arange(10), num_classes=10, dtype=np.float32)
    z = tf.random.normal((10, z_dim))
    generated_digit = cgan.generator((z, labels), training=False)
    for j in range(10):
        ax[i][j].imshow(np.clip(generated_digit[j]*127.5 + 127.5, 0, 255).astype(int))
        ax[i][j].axis('off')
fig.show()
