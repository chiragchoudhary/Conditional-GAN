class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.d_losses = []
        self.g_losses = []

    def on_batch_end(self, batch, logs):
        self.d_losses.append(logs['d_loss'])
        self.g_losses.append(logs['g_loss'])

    def on_epoch_end(self, epoch, logs):
        fig, ax = plt.subplots(10, 10, figsize=(20, 20))
        for i in range(10):
            labels = tf.keras.utils.to_categorical(np.arange(10), num_classes=10, dtype=np.float32)
            z = tf.random.normal((10, z_dim))
            generated_digits = self.model.generator((z, labels), training=False)
            for j in range(10):
                ax[i][j].matshow(np.clip(generated_digits[j] * 127.5 + 127.5, 0, 255).astype(int), cmap='viridis')
                ax[i][j].axis('off')
        fig.savefig(f"../data/tmp/mnist/generated_digits_{epoch:03}.png")
        plt.close()

    def on_train_end(self, logs):
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(self.d_losses, label='Discriminator loss')
        ax.plot(self.g_losses, label='Generator loss')
        ax.legend()
        fig.savefig(f"../data/tmp/mnist/training_losses.png")
        plt.close()