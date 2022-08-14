# DeepLearning-EEGData
Deep Learning applied to EEG signals of semantic categorization.

The code for preprocessing as well as the Deep Learning models can be found here.

--- 
## Links

### Reminders
* [Clean kfold code example](https://towardsdatascience.com/stratified-k-fold-what-it-is-how-to-use-it-cf3d107d3ea2)

### GitHub
* [William's Thesis Experiment](https://github.com/WilliamGeuns/MachineLearningDeepLearning)
* [Siobhan's Thesis Experiment](https://github.com/smhall97/Masters_EEG_FreeWill)

### Setup
* [Using Keras & Tensorflow with AMD GPU](https://stackoverflow.com/questions/37892784/using-keras-tensorflow-with-amd-gpu)
* [A shell script to install AMDGPU-PRO OpenCL driver](https://gist.github.com/kytulendu/3351b5d0b4f947e19df36b1ea3c95cbe)

## Configuration

* **Save libraries and dependancies to `requirements.txt`**
    ```shell
        pip freeze > requirements.txt
    ```
* **`Conda` create environment from `requirements.txt`**
    ```shell
        pip freeze > requirements.txt
    ```
* **`Conda` install `requirements.txt` into current environment**
    ```shell
        pip freeze > requirements.txt
    ```

---

The below are notes I made whilst refactoring the project

# Code Steps
1. SET to CSV
	* William - [Ben CSV](https://github.com/WilliamGeuns/MachineLearningDeepLearning/blob/master/Ben%20CSV.py)
	* William - [SET to CSV](https://github.com/WilliamGeuns/MachineLearningDeepLearning/blob/master/SET%20to%20CSV.ipynb)
		* Not sure why this exists
		* Much fewer lines of code

2. Create Pickles
	* William - [LoadingDataV7](https://github.com/WilliamGeuns/MachineLearningDeepLearning/blob/master/loadingDataV7.py)
		* Not sure what format of data it reads data in

3. Train Model
    * 0 - `William thank you.ipynb` / `<semantic_category>.ipynb`
    * Ben - `TrainingModel.py`
        * Takes the pickles and trains the model

4. Create Confusion Matrix
    * Ben - `Model.py`
        * Loads the model
        * Then generates the Confusion Matrix
---

# Plan of action

0. Plug in HDD
    * Copy Pickles over

> Everything being run should be reading and saving info on new HDD  
> At least initially to determine file size and to see if we can run things on the NVMe (much faster) - View Disk usage

1. Run `<semantic_category>.ipynb`
    * Time how long it takes
    * Should do the confusion Matrix as well
        * **If not do this next**
        > Confirm Trigers relationships
2. Run Set to CSV
3. Create Pickles
4. Refactor code
    * Model code
        * Siobhan insperation
        * `f` format for strings
        * Variables everywhere
        * All same file    
            * Create Model
            * Train
            * Test
            * Load model
            * Confusion Matrix 
        * Test
    * Set to CSV
        * Test
        * **Add in if __mian__**
    * Create Pickles
        * Test
        **Add in if __mian__**
    * Potentially combine more things together
        * Don't want things too big
            * But keep def functions clean
    * In CNN file
        * Just checking if the model that we are loading is the same as that I was training  
        just a double check :)  
        model.summary()  
        model.get_weights()
    * Find comanality in the different files
        * Consider creating a yaml config file
            * Example for class split
                * Purple, Ball, Pen, etc..

5. Consider creating a basic CLI 
    * FxF for insperation
6. Setup and run all on Linux to train all code

# Triggers

| Type    | Example | Question | Screen | Word |
|---------|---------|----------|----------|------|
| Colours | Purple  | T1       | T2       | T7   |
| Colours | Red     | T4       | T3       | T5   |

| Type    | Example | Digit |  Word | Roman Numeral |
|---------|---------|-------|-------|---------------|
| Numbers | Seven   | T14   | T11   | T9            |
| Numbers | Two     | T15   | T13   | T10           |

| Type    | Example | Phrase | Picture | Word |
|---------|---------|--------|---------|------|
| Objects | Ball    | T17    | T18     | T16  |
| Objects | Pen     | T8     | T6      | T12  |

# Pickles

| Person  | Number |
|---------|--------|
| Tristan | 20     |
| Harry   | 21     |
| Keanu   | 22     |
| Natasha | 23     |


7
1.22.26

```bash
(skripsie) E:\Skripsie\Code\Original>python 7.py
Using TensorFlow backend.
Training on fold: 1/3
Training new iteration on 240 training samples, 122 validation samples, this may be a while...
Train on 240 samples, validate on 122 samples
Epoch 1/8
2022-07-08 18:49:25.808559: I C:\tf_jenkins\workspace\rel-win\M\windows\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
240/240 [==============================] - 198s 826ms/step - loss: 0.7343 - acc: 0.6792 - val_loss: 0.0179 - val_acc: 1.0000
Epoch 2/8
240/240 [==============================] - 198s 825ms/step - loss: 0.0226 - acc: 0.9917 - val_loss: 3.1375e-06 - val_acc: 1.0000     
Epoch 3/8
240/240 [==============================] - 197s 821ms/step - loss: 0.0393 - acc: 0.9875 - val_loss: 3.7791e-05 - val_acc: 1.0000     
Epoch 4/8
240/240 [==============================] - 200s 833ms/step - loss: 0.0106 - acc: 1.0000 - val_loss: 8.9043e-06 - val_acc: 1.0000     
Epoch 5/8
240/240 [==============================] - 195s 814ms/step - loss: 0.0192 - acc: 0.9958 - val_loss: 3.3208e-06 - val_acc: 1.0000
Epoch 6/8
240/240 [==============================] - 197s 819ms/step - loss: 0.0254 - acc: 0.9917 - val_loss: 5.0627e-04 - val_acc: 1.0000
Epoch 7/8
240/240 [==============================] - 196s 818ms/step - loss: 0.0056 - acc: 1.0000 - val_loss: 2.4672e-06 - val_acc: 1.0000
Epoch 8/8
240/240 [==============================] - 196s 816ms/step - loss: 5.7771e-04 - acc: 1.0000 - val_loss: 2.8825e-07 - val_acc: 1.0000
Last training accuracy: 1.0, last validation accuracy: 1.0
Training on fold: 2/3
Training new iteration on 241 training samples, 121 validation samples, this may be a while...
Train on 241 samples, validate on 121 samples
Epoch 1/8
241/241 [==============================] - 196s 815ms/step - loss: 0.6824 - acc: 0.7261 - val_loss: 0.0936 - val_acc: 1.0000
Epoch 2/8
241/241 [==============================] - 196s 814ms/step - loss: 0.2296 - acc: 0.9046 - val_loss: 0.0029 - val_acc: 1.0000
Epoch 3/8
241/241 [==============================] - 196s 813ms/step - loss: 0.1310 - acc: 0.9627 - val_loss: 0.0201 - val_acc: 1.0000
Epoch 4/8
241/241 [==============================] - 196s 812ms/step - loss: 0.1083 - acc: 0.9544 - val_loss: 1.1573e-05 - val_acc: 1.0000
Epoch 5/8
241/241 [==============================] - 196s 814ms/step - loss: 0.0314 - acc: 0.9876 - val_loss: 5.8589e-04 - val_acc: 1.0000
Epoch 6/8
241/241 [==============================] - 196s 812ms/step - loss: 0.0324 - acc: 0.9917 - val_loss: 1.7274e-04 - val_acc: 1.0000
Epoch 7/8
241/241 [==============================] - 197s 818ms/step - loss: 0.0863 - acc: 0.9751 - val_loss: 0.0021 - val_acc: 1.0000
Epoch 8/8
241/241 [==============================] - 197s 816ms/step - loss: 0.0361 - acc: 0.9959 - val_loss: 1.1198e-04 - val_acc: 1.0000
Last training accuracy: 0.9958506214173503, last validation accuracy: 1.0
Training on fold: 3/3
Training new iteration on 243 training samples, 119 validation samples, this may be a while...
Train on 243 samples, validate on 119 samples
Epoch 1/8
243/243 [==============================] - 197s 812ms/step - loss: 0.5746 - acc: 0.7613 - val_loss: 0.0714 - val_acc: 0.9832
Epoch 2/8
243/243 [==============================] - 197s 812ms/step - loss: 0.0882 - acc: 0.9712 - val_loss: 0.0021 - val_acc: 1.0000
Epoch 3/8
243/243 [==============================] - 197s 810ms/step - loss: 0.0174 - acc: 0.9918 - val_loss: 4.8086e-05 - val_acc: 1.0000
Epoch 4/8
243/243 [==============================] - 197s 810ms/step - loss: 6.4367e-04 - acc: 1.0000 - val_loss: 4.7283e-07 - val_acc: 1.0000
Epoch 5/8
243/243 [==============================] - 197s 809ms/step - loss: 8.1133e-04 - acc: 1.0000 - val_loss: 3.4661e-07 - val_acc: 1.0000
Epoch 6/8
243/243 [==============================] - 197s 810ms/step - loss: 0.0023 - acc: 1.0000 - val_loss: 2.4443e-07 - val_acc: 1.0000
Epoch 7/8
243/243 [==============================] - 197s 809ms/step - loss: 1.4324e-05 - acc: 1.0000 - val_loss: 2.3842e-07 - val_acc: 1.0000
Epoch 8/8
243/243 [==============================] - 197s 809ms/step - loss: 5.6312e-04 - acc: 1.0000 - val_loss: 2.3842e-07 - val_acc: 1.0000
Last training accuracy: 1.0, last validation accuracy: 1.0
100.00% (+/- 0.00%)
[1. 1. 1.]
[1.0000000e+00 3.3988107e-10 1.9392887e-26]
[1.0000000e+00 7.8724999e-11 4.0537253e-28]
[1.0000000e+00 5.1808648e-11 1.6126082e-28]
[1.0000000e+00 7.1265943e-10 2.5077134e-25]
[1.0000000e+00 9.1078975e-09 2.2543007e-22]
[2.6691160e-10 1.0000000e+00 1.8465737e-13]
[1.5857763e-09 1.0000000e+00 3.7929394e-15]
[1.2745904e-10 1.0000000e+00 5.5758078e-15]
[3.1649053e-10 1.0000000e+00 1.0126757e-15]
[4.148320e-10 1.000000e+00 5.844798e-16]
[1.85090596e-18 1.03052024e-07 9.99999881e-01]
[0.0000000e+00 1.4195783e-17 1.0000000e+00]
[5.2113050e-28 1.7813436e-11 1.0000000e+00]
[3.2906028e-37 4.4432854e-15 1.0000000e+00]
[5.534547e-38 2.198336e-15 1.000000e+00]
0
0
0
0
0
1
1
1
1
1
2
2
2
2
2
Confusion matrix, without normalization
[[5 0 0]
 [0 5 0]
 [0 0 5]]
Normalized confusion matrix
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_9 (Conv2D)            (None, 55, 448, 512)      5120      
_________________________________________________________________
activation_15 (Activation)   (None, 55, 448, 512)      0
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 27, 224, 512)      0
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 25, 222, 512)      2359808   
_________________________________________________________________
activation_16 (Activation)   (None, 25, 222, 512)      0
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 23, 220, 256)      1179904   
_________________________________________________________________
activation_17 (Activation)   (None, 23, 220, 256)      0
_________________________________________________________________
max_pooling2d_8 (MaxPooling2 (None, 11, 110, 256)      0
_________________________________________________________________
dropout_7 (Dropout)          (None, 11, 110, 256)      0
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 9, 108, 48)        110640
_________________________________________________________________
activation_18 (Activation)   (None, 9, 108, 48)        0
_________________________________________________________________
max_pooling2d_9 (MaxPooling2 (None, 4, 54, 48)         0
_________________________________________________________________
dropout_8 (Dropout)          (None, 4, 54, 48)         0
_________________________________________________________________
flatten_3 (Flatten)          (None, 10368)             0
_________________________________________________________________
dense_7 (Dense)              (None, 36)                373284
_________________________________________________________________
activation_19 (Activation)   (None, 36)                0
_________________________________________________________________
dense_8 (Dense)              (None, 18)                666
_________________________________________________________________
activation_20 (Activation)   (None, 18)                0
_________________________________________________________________
dropout_9 (Dropout)          (None, 18)                0
_________________________________________________________________
dense_9 (Dense)              (None, 3)                 57
_________________________________________________________________
activation_21 (Activation)   (None, 3)                 0
=================================================================
Total params: 4,029,479
Trainable params: 4,029,479
Non-trainable params: 0
_________________________________________________________________
```


# Methodology Steps
1. Record Data
2. Convert Brain Products to Matlab readible
3. Pre-process Data
4. Convert data to CSV for each epoch of every stimulus 
5. first feature scaled – the standardization of the range of independent variables or features of data.
6. Fed into DL... (wow, really Ben...)
	* Training
	* Validation
	* Run test data (unseen) through the model
	* (Above performed using k-fold)
7. Confusion matrix generated
8. 24x models created
	* A CNN model was trained for each semantic object for every participant
	* [(2+2+2)*4] = 24
	> Note if model is run on validation (seen) data or testing data (unseen)

> Each model was trained on the pre-processed EEG data representing a particular semantic object, with the goal of correctly classifying that object into its three modalities.

> It should be noted that these accuracies are with respect to the training and validation of the model, and not based on the models performance on unseen data