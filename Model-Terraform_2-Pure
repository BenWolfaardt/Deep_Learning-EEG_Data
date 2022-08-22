"""
Created on Mon Oct 01 22:10:52 2018
Updated on Sun Aug 14 18:22:12 2022
Authors: William & Ben
"""
from ast import AsyncFunctionDef
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
# import tensorflow_datasets as tfds

from keras.callbacks import EarlyStopping
from keras.models import load_model, Sequential
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras import Model, layers, losses, metrics, optimizers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from typing import Tuple, List

EXPERIMENT = "Purple"
CATEGORIES = ['Lavendar','Full Screen','Word']
PARTICIPANTS = {0: "All", 1: "Single"}
NAME = f"{EXPERIMENT}-{PARTICIPANTS}_participant(s)"

# TODO increase number, bigger is better: https://stackoverflow.com/questions/46654424/how-to-calculate-optimal-batch-size
BATCH_SIZE=10 
STEPS_PER_EPOCH=1
EPOCHS=5
VERBOSE=1

KFOLD_SPLITS=2

# Load preprocessed training/test pickles
# y = ndarray
#def load_data() -> Tuple[tf.data.Dataset(), tf.data.Dataset()]:
def load_data():
    with open(f"{EXPERIMENT}-X-Training.pickle", 'rb') as f:
        X = pickle.load(f) # (1369, 63, 450, 1)
    with open(f"{EXPERIMENT}-Y-Training.pickle", 'rb') as f:
        y = pickle.load(f)
        y = np.transpose(y) # (1369,)
    with open(f"{EXPERIMENT}-X-Test.pickle", 'rb') as f:
        X_val = pickle.load(f) # (158, 63, 450, 1)
    with open(f"{EXPERIMENT}-Y-Test.pickle", 'rb') as f:
        y_val = pickle.load(f)
        y_val = np.transpose(y_val) # (158,)

    # X = X[..., tf.newaxis] # (1369, 63, 450, 1, 1)
    # X_val = X_val[..., tf.newaxis] # (158, 63, 450, 1, 1)

    train_ds = tf.data.Dataset.from_tensor_slices(
    (X, y)).shuffle(10).batch(32) # (63, 450, 1, 1), ()
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(10) # (63, 450, 1, 1), ()

    return train_ds, val_ds


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
        self.conv1 = layers.Conv2D(filters=512, kernel_size=(3,3), input_shape=(63, 450, 1), padding='same', activation=tf.nn.relu)
        self.conv2 = layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        self.conv3 = layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        self.conv4 = layers.Conv2D(filters=48, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
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

model = MyModel()


# compile(), train the model with model.fit(), or use the model to do prediction with model.predict().

"""
# Create model 
def create_model() -> Sequential():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.Input(shape=(63,450,1)))

    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), activation=tf.nn.relu))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), activation=tf.nn.relu))
    
    model.add(layers.Conv2D(filters=256, kernel_size=(3,3), activation=tf.nn.relu))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv2D(filters=48, kernel_size=(3,3), activation=tf.nn.relu))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
        
    model.add(layers.Flatten())   
    model.add(layers.Dense(36, activation=tf.nn.relu))
    model.add(layers.Dense(18, activation=tf.nn.relu))
    model.add(layers.Dropout(0.2))
    # Last dense 1ayers must have number of classes in data in the parenthesis
    # Also must be softmax
    model.add(layers.Dense(3, Activation=tf.nn.softmax))
"""


"""
# Compile model
def compile_model(model: Sequential()) -> Sequential():
    model.compile(
        optimizer=optimizers.Adam(),
        # To reduce Python overhead and maximize the performance of your TPU, pass in the steps_per_execution 
        # argument to Keras Model.compile. In this example, it increases throughput by about 50%:
        steps_per_execution=50,
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=metrics.Accuracy(name='accuracy', dtype=None),
        # TODO: metrics=['sparse_categorical_accuracy']
    )
    return model
"""

"""
# Train model 
def train_model(
        model: Sequential(),
        X_train: List,
        y_train: List,
        X_val: List,
        y_val: List,
    ):
    # This is the callback for writing checkpoints during training
    path_checkpoint = 'checkpoint.keras'
    callback_checkpoint = ModelCheckpoint(
        filepath=path_checkpoint,
        monitor='val_loss',
        verbose=1,
        save_weights_only=True,
        save_best_only=True
    )
    # This is the callback for stopping the optimization when performance worsens on the validation-set
    callback_early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5, 
        verbose=1
    )
    # This is the callback for writing the TensorBoard log during training.
    callback_tensorboard = TensorBoard(
        log_dir='./logs/',
        histogram_freq=0,
        write_graph=False
    )
    # This callback reduces the learning-rate for the optimizer if the validation-loss has not improved 
    # since the last epoch (as indicated by patience=0). The learning-rate will be reduced by multiplying 
    # it with the given factor. We set a start learning-rate of 1e-3 above, so multiplying it by 0.1 gives 
    # a learning-rate of 1e-4. We don't want the learning-rate to go any lower than this.
    callback_reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        min_lr=1e-4,
        patience=0,
        verbose=1
    )

    callbacks = [
        callback_early_stopping,
        callback_checkpoint,
        callback_tensorboard,
        callback_reduce_lr
    ]

    # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=VERBOSE,
        callbacks=callbacks,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=(X_val, y_val), 
    )
"""

# Main function
def main():
    train_ds, val_ds = load_data()


    model = MyModel()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam()

    # commented out ones are from original model.fit (give error with name parameter)
    # training_loss = losses.sparse_categorical_crossentropy(name='training_loss', dtype=None)
    training_loss = tf.keras.metrics.Mean(name='training_loss', dtype=None)
    # training_accuracy = metrics.accuracy(name='train_accuracy')
    training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

    # test_loss = losses.sparse_categorical_crossentropy(name='test_loss', dtype=None)
    test_loss = tf.keras.metrics.Mean(name='test_loss', dtype=None)
    # test_accuracy = metrics.accuracy(name='test_accuracy')
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

        for epochs, labels in train_ds: # (63, 450, 1, 1), () in (63, 450, 1, 1), ()
            training_step(epochs, labels)

        for test_epochs, test_labels in val_ds:
            test_step(test_epochs, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch+1,
                                training_loss.result(),
                                training_accuracy.result()*100,
                                test_loss.result(),
                                test_accuracy.result()*100))

    # kfold = StratifiedKFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=42)
    # model = kfold_cross_validation_earlystopping(model,X,y,Xtest,ytest,kfold)

if __name__ == '__main__':
    main()
    print("Done")

# ----------------------------------------------------------------------------------------------------------------------
# Miscelaneous fiddling
# ----------------------------------------------------------------------------------------------------------------------

# Load model from disk
def load_model():
    model = load_model(EXPERIMENT)
    print("Loaded model from disk")
    return model

# ----------------------------------------------------------------------------------------------------------------------

# Multiple GPUs, Machines, TPUs implementation: 
# * https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/distribute/custom_training.ipynb
# * https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/tpu.ipynb#scrollTo=mgUC6A-zCMEr


# """




# """