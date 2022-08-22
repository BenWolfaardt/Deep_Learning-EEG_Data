"""
Created on Mon Oct 01 22:10:52 2018
Updated on Sun Aug 14 18:22:12 2022
Authors: Ben & William

Inspiration for architecture taken from: 
- https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/tpu.ipynb#scrollTo=mgUC6A-zCMEr
- https://www.tensorflow.org/tutorials/distribute/custom_training
Multiple GPUs, Machines, TPUs implementation: 
- https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/distribute/custom_training.ipynb
- https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/tpu.ipynb#scrollTo=mgUC6A-zCMEr
"""

import numpy as np
import pickle
import tensorflow as tf

from tensorflow.python.keras import Model, layers

EXPERIMENT = "Purple"
EPOCHS=5

# Load preprocessed training/test pickles
def load_data():
    with open(f"{EXPERIMENT}-X-Training.pickle", 'rb') as f:
        X = pickle.load(f) # Shape: (1369, 63, 450, 1)
    with open(f"{EXPERIMENT}-Y-Training.pickle", 'rb') as f:
        y = pickle.load(f)
        y = np.transpose(y) # Shape: (1369,)
    with open(f"{EXPERIMENT}-X-Test.pickle", 'rb') as f:
        X_val = pickle.load(f) # Shape: (158, 63, 450, 1)
    with open(f"{EXPERIMENT}-Y-Test.pickle", 'rb') as f:
        y_val = pickle.load(f)
        y_val = np.transpose(y_val) # Shape: (158,)

    train_ds = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(10).batch(64)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)

    return train_ds, val_ds


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
        self.conv1 = layers.Conv2D(
            filters=512,
            kernel_size=(3,3),
            input_shape=(63, 450, 1),
            padding='same',
            activation=tf.nn.relu
        )
        self.conv2 = layers.Conv2D(
            filters=512, kernel_size=(3,3), padding='same', activation=tf.nn.relu
        )
        self.conv3 = layers.Conv2D(
            filters=256, kernel_size=(3,3), padding='same', activation=tf.nn.relu
        )
        self.conv4 = layers.Conv2D(
            filters=48, kernel_size=(3,3), padding='same', activation=tf.nn.relu
        )
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        self.dense1 = layers.Dense(36, activation=tf.nn.relu)
        self.dense2 = layers.Dense(18, activation=tf.nn.relu)
        self.dense3 = layers.Dense(3, activation=tf.nn.softmax)
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

# Main function
def main():
    train_ds, val_ds = load_data()
    model = MyModel()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam()

    training_loss = tf.keras.metrics.Mean(name='training_loss', dtype=None)
    training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    test_loss = tf.keras.metrics.Mean(name='test_loss', dtype=None)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy', dtype=None)

    @tf.function
    def training_step(epochs, labels):
        with tf.GradientTape() as tape:
            predictions = model(epochs)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        training_loss(loss)
        training_accuracy(labels, predictions)

    @tf.function
    def test_step(epochs, labels):
        predictions = model(epochs)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    for epoch in range(EPOCHS):
        training_loss.reset_states()
        training_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for epochs, labels in train_ds:
            training_step(epochs, labels)

        for test_epochs, test_labels in val_ds:
            test_step(test_epochs, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(
            epoch+1,
            training_loss.result(),
            training_accuracy.result()*100,
            test_loss.result(),
            test_accuracy.result()*100
        ))

if __name__ == '__main__':
    main()
    print("Done")
