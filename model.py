import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Conv2D, Dense, ReLU, BatchNormalization, Input, Flatten, Concatenate, Reshape, Conv2DTranspose
from keras import Model
from keras.metrics import Mean


class CGAN(keras.Model):
    def __init__(self):
        super().__init__()
        self.discriminator = self._build_discriminator()
        self.generator = self._build_generator()

    def compile_model(self, discriminator_optimizer, generator_optimizer, loss_fn):
        super().compile(run_eagerly=False)
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.loss_fn = loss_fn
        self.discriminator_loss_metric = Mean(name='d_loss')
        self.generator_loss_metric = Mean(name='g_loss')

    def _build_generator(self):
        img = Input((z_dim,))
        labels = Input((num_classes,))

        z = Concatenate()([img, labels])

        z = Dense(7 * 7 * 128, activation='relu')(z)
        z = BatchNormalization()(z)
        z = ReLU()(z)

        z = Reshape((7, 7, 128))(z)

        z = Conv2DTranspose(
            filters=128,
            kernel_size=3,
            strides=2,
            padding='same',
        )(z)
        z = ReLU()(z)

        z = Conv2DTranspose(
            filters=256,
            kernel_size=3,
            strides=2,
            padding='same',
        )(z)
        z = ReLU()(z)

        output = Conv2D(
            filters=img_shape[-1],
            kernel_size=3,
            strides=1,
            padding='same',
            activation='tanh',
        )(z)

        return Model(inputs=[img, labels], outputs=output, name='Generator')

    def _build_discriminator(self):
        img = Input(img_shape)
        labels = Input((num_classes,))
        z = Conv2D(
            filters=64,
            kernel_size=3,
            strides=2,
            padding='same',
        )(img)
        z = ReLU()(z)

        z = Conv2D(
            filters=128,
            kernel_size=3,
            strides=2,
            padding='same',
        )(z)
        z = ReLU()(z)

        z = Conv2D(
            filters=256,
            kernel_size=3,
            strides=2,
            padding='same',
        )(z)
        z = ReLU()(z)

        z = Flatten()(z)
        z = Concatenate()([z, labels])
        z = Dense(128, activation='relu')(z)
        z = Dense(64, activation='relu')(z)
        output = Dense(1)(z)

        return Model(inputs=[img, labels], outputs=output, name='Discriminator')

    def train_step(self, batch_data):
        img, labels = batch_data
        batch_size = img.shape[0]
        real_target_labels = tf.ones((batch_size, 1))
        fake_target_labels = -1.0 * tf.ones((batch_size, 1))

        # Discriminator Loss
        for _ in range(3):
            with tf.GradientTape() as tape:
                z = tf.random.normal((batch_size, z_dim))
                fake_img = self.generator((z, labels))
                combined_img = tf.concat([img, fake_img], axis=0)
                combined_labels = tf.concat([real_target_labels, fake_target_labels], axis=0)
                # labels += 0.05 * tf.random.uniform(tf.shape(labels))
                predictions = self.discriminator((combined_img, tf.concat([labels, labels], axis=0)))
                d_loss = 0.5 * self.loss_fn(combined_labels, predictions)

                d_loss = d_loss + self.gp_loss(img, fake_img, labels)

            gradients = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.discriminator_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_weights))

        with tf.GradientTape() as tape:
            # Generator Loss
            z = tf.random.normal((batch_size, z_dim))
            fake_img = self.generator((z, labels))
            fake_predictions = self.discriminator((fake_img, labels))
            g_loss = self.loss_fn(real_target_labels, fake_predictions)

        gradients = tape.gradient(g_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(gradients, self.generator.trainable_weights))

        self.discriminator_loss_metric.update_state(d_loss)
        self.generator_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.discriminator_loss_metric, self.generator_loss_metric]

    def gp_loss(self, real_img, fake_img, labels):
        alpha = tf.random.uniform(fake_img.shape, 0.0, 1.0)
        interpolated_img = alpha * real_img + (1 - alpha) * fake_img

        with tf.GradientTape() as tape:
            tape.watch(interpolated_img)
            predictions = self.discriminator((interpolated_img, labels))

        gradients = tape.gradient(predictions, [interpolated_img])
        gradient_l2_norm = K.sqrt(K.sum(K.square(gradients), axis=[1, 2, 3]))

        gradient_penalty = K.mean(K.square(1.0 - gradient_l2_norm))
        return gradient_penalty

