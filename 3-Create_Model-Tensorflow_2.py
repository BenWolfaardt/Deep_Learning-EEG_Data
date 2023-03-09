"""
Created on Mon Oct 01 22:10:52 2018
Updated on Sun Aug 14 18:22:12 2022
Authors: Ben Wolfaardt

Inspiration for architecture taken from: 
- https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/tpu.ipynb#scrollTo=mgUC6A-zCMEr
- https://www.tensorflow.org/tutorials/distribute/custom_training
Multiple GPUs, Machines, TPUs implementation: 
- https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/distribute/custom_training.ipynb
- https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/tpu.ipynb#scrollTo=mgUC6A-zCMEr
Using Tensorboard
- https://www.tensorflow.org/tensorboard/get_started
"""

import numpy as np
import pickle
import tensorflow as tf
import datetime

from tensorflow.python.keras import Model, layers

EXPERIMENT = "Siobhan"
EPOCHS = 50
DIRECTORY_PICKLE_DATA_INPUT = "/Users/james.wolfaardt/code/__ben/Code/Deep_Learning-EEG_Data/outputs/pickles"
participant = [1]


# Load preprocessed training/test pickles
def load_data():
    with open(f"{DIRECTORY_PICKLE_DATA_INPUT}/X-{participant}-Training.pickle", 'rb') as f:
        X = pickle.load(f)  # Shape: (1369, 63, 450, 1)
    with open(f"{DIRECTORY_PICKLE_DATA_INPUT}/y-{participant}-Training.pickle", 'rb') as f:
        y = pickle.load(f)
        y = np.transpose(y)  # Shape: (1369,)
    with open(f"{DIRECTORY_PICKLE_DATA_INPUT}/X-{participant}-Test.pickle", 'rb') as f:
        X_val = pickle.load(f)  # Shape: (158, 63, 450, 1)
    with open(f"{DIRECTORY_PICKLE_DATA_INPUT}/y-{participant}-Test.pickle", 'rb') as f:
        y_val = pickle.load(f)
        y_val = np.transpose(y_val)  # Shape: (158,)

    train_ds = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(10).batch(64)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)

    return train_ds, val_ds


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
        self.conv1 = layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            input_shape=(128, 257, 1),
            padding='same',
            activation=tf.nn.relu
        )
        self.conv2 = layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding='same', activation=tf.nn.relu
        )
        self.conv3 = layers.Conv2D(
            filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu
        )
        self.conv4 = layers.Conv2D(
            filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu
        )
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        self.dense1 = layers.Dense(16, activation=tf.nn.relu)
        self.dense2 = layers.Dense(8, activation=tf.nn.relu)
        self.dense3 = layers.Dense(2, activation=tf.nn.sigmoid)
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten
        self.flatten = layers.Flatten()
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D
        self.maxpooling = layers.MaxPool2D(pool_size=(2, 2), padding='same')
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
        self.dropout = layers.Dropout(0.2)

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpooling(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpooling(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.maxpooling(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return self.dense3(x)


def main():
    training_ds, validation_ds = load_data()
    model = MyModel()

    # Selected loss and optimizer:
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam()

    # Define our metrics
    training_loss = tf.keras.metrics.Mean(name='training_loss', dtype=tf.float32)
    training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    validation_loss = tf.keras.metrics.Mean(name='test_loss', dtype=tf.float32)
    validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def training_step(model, optimizer, x_train, y_train):  # where x = epochs, and y = labes
        with tf.GradientTape() as tape:
            predictions = model(x_train, training=True)
            loss = loss_object(y_train, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        training_loss(loss)
        training_accuracy(y_train, predictions)

    @tf.function
    def validation_step(model, x_test, y_test):
        predictions = model(x_test)
        loss = loss_object(y_test, predictions)

        validation_loss(loss)
        validation_accuracy(y_test, predictions)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    training_log_dir = 'logs/gradient_tape/' + current_time + '/training'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(training_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    for epoch in range(EPOCHS):
        for (x_train, y_train) in training_ds:
            training_step(model, optimizer, x_train, y_train)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', training_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', training_accuracy.result(), step=epoch)

        for (x_val, y_val) in validation_ds:
            validation_step(model, x_val, y_val)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', validation_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', validation_accuracy.result(), step=epoch)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(
            epoch + 1,
            training_loss.result(),
            training_accuracy.result() * 100,
            validation_loss.result(),
            validation_accuracy.result() * 100
        ))

        # Reset metrics every epcoh
        training_loss.reset_states()
        training_accuracy.reset_states()
        validation_loss.reset_states()
        validation_accuracy.reset_states()


if __name__ == '__main__':
    main()
    print("Done")
