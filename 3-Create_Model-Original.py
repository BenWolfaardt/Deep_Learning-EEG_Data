"""
Created on 01/10/2018
Updated on 07/09/2022
Authors: William & Ben
"""
import numpy as np
import pickle

from keras.callbacks import EarlyStopping,  ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from nptyping import NDArray
from sklearn.model_selection import StratifiedKFold

EXPERIMENT = "Siobhan"
TRIGGERS = ['Left','Right']
COMPARISON = {0: "All", 1: "Single"}
PARTICIPANTS = ["5"]
# PARTICIPANTS = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
NAME = f"{EXPERIMENT}_{COMPARISON[1]}-{PARTICIPANTS[0]}"
DIRECTORY_PICKLE_DATA_INPUT = "/Users/james.wolfaardt/code/__ben/Code/Deep_Learning-EEG_Data/outputs/pickles"
DIRECTORY_MODEL_DATA_OUTPUT = "/Users/james.wolfaardt/code/__ben/Code/Deep_Learning-EEG_Data/outputs/models"
# TODO note duplicate
participant = ["5"]
KFOLD_SPLITS=5
EPOCHS=20

class Create:
    def __init__(self) -> None:
        pass

    # Load preprocessed training/test pickles
    def load_data(self)  -> tuple[NDArray, NDArray]:
        with open(f"{DIRECTORY_PICKLE_DATA_INPUT}/X-{str(participant[0])}-Training.pickle", 'rb') as f:
            X = pickle.load(f) # shape: (1369, 63, 450, 1)
            X = np.asarray(X)
        with open(f"{DIRECTORY_PICKLE_DATA_INPUT}/y-{str(participant[0])}-Training.pickle", 'rb') as f:
            y = pickle.load(f) # shape: (
            y = np.transpose(y)
        return X, y

    # Create model 
    def create_model(self, X) -> Sequential():
        model = Sequential()
        model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:])) # input_shape: (63, 450, 1)
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu')) 
        
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
            
        model.add(Flatten())   
        
        model.add(Dense(32))
        model.add(Activation('relu'))
        
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dropout(0.2)) 
        
        # Last dense 1ayers must have number of classes in data in the parenthesis
        model.add(Dense(len(TRIGGERS)))
        # For a binary classification you should have a sigmoid last layer
        # See 2nd answer for more details: https://stackoverflow.com/questions/68776790/model-predict-classes-is-deprecated-what-to-use-instead
        # softmax for other experiment
        # TODO add automatic logic for this
        model.add(Activation('sigmoid'))

        model.compile(loss='sparse_categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

        # print(model.summary())

        return model

    # Train model with K-fold cross validation with early stopping
    def kfold_cross_validation(
        self, 
        model: Sequential(),
        X,
        y,
        kfold: StratifiedKFold,
    ) -> Sequential():
        # This is the callback for writing checkpoints during training
        path_checkpoint = 'checkpoint.keras'
        callback_checkpoint = ModelCheckpoint(
            filepath=path_checkpoint,
            monitor='val_loss',
            verbose=1,
            save_weights_only=True,
            save_best_only=True,
        )
        # This is the callback for stopping the optimization when performance worsens on the validation-set
        callback_early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3, 
            verbose=1, 
            # restore_best_weights=True,
        )
        # This is the callback for writing the TensorBoard log during training.
        callback_tensorboard = TensorBoard(
            log_dir='./logs/original/',
            histogram_freq=0,
            write_graph=False,
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
            verbose=1,
        )

        callbacks = [
            callback_early_stopping,
            callback_checkpoint,
            callback_tensorboard,
            # callback_reduce_lr
        ]

        # tqdm_callback = TqdmCallback()
        cvscores = []
        for index, (train_index, validation_index) in enumerate(kfold.split(X, y)):
            print(f"\nTraining on fold: {str(index+1)}/{KFOLD_SPLITS}\n")
            
            #Generate batches
            Xtrain, Xval = X[train_index], X[validation_index]
            ytrain, yval = y[train_index], y[validation_index]

            model.fit(Xtrain,ytrain, epochs=EPOCHS, batch_size=10, verbose=1, validation_data=(Xval,yval), callbacks=callbacks, shuffle=True)
            # history = model.fit(Xtrain,ytrain, epochs=EPOCHS, batch_size=10, verbose=1, validation_data=(Xval,yval), callbacks=[early_stopping], shuffle=True)
            # accuracy_history = history.history['accuracy']
            # val_accuracy_history = history.history['val_accuracy']

            scores = model.evaluate(Xval, yval, verbose=1)
            cvscores.append(scores[1] * 100)

        print(f"\nEvaluated acuracy: {np.mean(cvscores):.2f}% (+/- {np.std(cvscores):.2f}%) on {KFOLD_SPLITS} k-folds")

        return model

    # Save model to disk
    def save_model(self, model: Sequential()) -> None:
        model.save(f"{DIRECTORY_MODEL_DATA_OUTPUT}/{NAME}.h5")
    
    def setup_and_test_data(self) -> None:
        X, y, = self.load_data()
        model = self.create_model(X)
        kfold = StratifiedKFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=42)
        model = self.kfold_cross_validation(model, X, y, kfold)
        self.save_model(model)

if __name__ == '__main__':
    app = Create()
    app.setup_and_test_data()
