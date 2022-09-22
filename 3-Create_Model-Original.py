"""
Created on 01/10/2018
Updated on 07/09/2022
Authors: William & Ben
"""
import numpy as np
import pickle

from envyaml import EnvYAML
from keras.backend import clear_session
from keras.callbacks import EarlyStopping,  ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from nptyping import NDArray
from sklearn.model_selection import StratifiedKFold

class Create:
    def __init__(self) -> None:
        # config
        self.os: str = None
        self.config: EnvYAML = None
        self.experiment: str = None
        # paths
        self.pickles: str = None
        self.models: str = None
        # experiment details
        self.name: str = None
        self.participants: list[int] = []
        self.triggers: list[str] = []
        self.version: str = None
        self.comparison: int = None
        # model
        self.kfold: int = None
        self.epochs: int = None

        def callbacks(self) -> None:
            self.callbacks: list = []
            
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

            self.callbacks = [
                # callback_early_stopping,
                # callback_checkpoint,
                # callback_tensorboard,
                # callback_reduce_lr
            ]

        callbacks(self)

    def load_yaml(self) -> None:
        self.config = EnvYAML("./setup/config.yaml", strict=False)

    def populate_config(self) -> None:
        # TODO parse in as cli command from make file
        self.os = "mac_m1"
        self.experiment = "libet"

        self.pickles = self.config[f"os.{self.os}.io_paths.pickle_files"]
        self.models = self.config[f"os.{self.os}.io_paths.model_files"]
        # TODO use name for folder or something
        self.name = self.config[f"experiment.details.{self.experiment}.name"]
        self.participants = self.config[f"experiment.details.{self.experiment}.participants"]
        self.triggers = self.config[f"experiment.details.{self.experiment}.triggers"]
        self.version = self.config[f"experiment.details.{self.experiment}.version"]
        self.kfolds = self.config[f"model_parameters.kfolds"]
        self.epochs = self.config[f"model_parameters.epochs"]
        self.comparison = self.config[f"model_parameters.comparison"]

    # Load preprocessed training/test pickles
    def load_data(self) -> None:
        self.X: NDArray = None
        self.y: NDArray = None
        
        with open(f"{self.pickles}/{self.version}/X-{str(self.participant)}-Training.pickle", 'rb') as f:
            self.X = pickle.load(f) # shape: (1369, 63, 450, 1)
            self.X = np.asarray(self.X)
        with open(f"{self.pickles}/{self.version}/y-{str(self.participant)}-Training.pickle", 'rb') as f:
            self.y = pickle.load(f) # shape: (
            self.y = np.transpose(self.y)

    # Create model 
    def create_model(self) -> None:
        self.model: Sequential = Sequential()
              
        self.model.add(Conv2D(256, (3, 3), input_shape=self.X.shape[1:])) # input_shape: (63, 450, 1)
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        
        self.model.add(Conv2D(128, (3, 3)))
        self.model.add(Activation('relu')) 
        
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
            
        self.model.add(Flatten())   
        
        self.model.add(Dense(32))
        self.model.add(Activation('relu'))
        
        self.model.add(Dense(16))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2)) 
        
        # Last dense 1ayers must have number of classes in data in the parenthesis
        self.model.add(Dense(len(self.triggers)))
        # For a binary classification you should have a sigmoid last layer
        # See 2nd answer for more details: https://stackoverflow.com/questions/68776790/model-predict-classes-is-deprecated-what-to-use-instead
        # softmax for other experiment
        # TODO add automatic logic for this
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='sparse_categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

        # print(model.summary())

    # Train model with K-fold cross validation with early stopping
    def kfold_cross_validation(self) -> None:
        kfold: StratifiedKFold = None
        cvscores: list = []

        kfold = StratifiedKFold(n_splits=self.kfolds, shuffle=True, random_state=42)

        print(f"\nTraining model for particiapnt: {self.participant}/{len(self.participants)}")

        for index, (train_index, validation_index) in enumerate(kfold.split(self.X, self.y)):
            print(f"Training on fold: {str(index+1)}/{self.kfolds}\n")
            
            #Generate batches
            Xtrain, Xval = self.X[train_index], self.X[validation_index]
            ytrain, yval = self.y[train_index], self.y[validation_index]

            self.model.fit(Xtrain,ytrain, epochs=self.epochs, batch_size=10, verbose=1, validation_data=(Xval,yval), callbacks=self.callbacks, shuffle=True)
            # history = model.fit(Xtrain,ytrain, epochs=EPOCHS, batch_size=10, verbose=1, validation_data=(Xval,yval), callbacks=[early_stopping], shuffle=True)
            # accuracy_history = history.history['accuracy']
            # val_accuracy_history = history.history['val_accuracy']

            scores = self.model.evaluate(Xval, yval, verbose=1)
            cvscores.append(scores[1] * 100)

            print()

        print(f"\nEvaluated accuracy: {np.mean(cvscores):.2f}% (+/- {np.std(cvscores):.2f}%) on {self.kfolds} k-folds")

    # Save model to disk
    def save_model(self) -> None:
        # TODO don't save coparison as number but rather as value of dict {0: "All", 1: "Single"}
        self.filename = f"{self.experiment}_{self.comparison}-{self.participant}"
        self.model.save(f"{self.models}/{self.version}/{self.filename}.h5")
        clear_session()
    
    def setup_and_test_data(self) -> None:
        for self.participant in self.participants:
            self.load_data()
            self.create_model()
            self.kfold_cross_validation()
            self.save_model()

if __name__ == '__main__':
    app = Create()
    app.load_yaml()
    app.populate_config()
    app.setup_and_test_data()
