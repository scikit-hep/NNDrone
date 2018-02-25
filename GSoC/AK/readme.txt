###############ReadMe########################
File Description:

######### Python Files ########
main.py     : it rus all the model for training and saving
models.py   : contrains the model created in keras.
utils.py    : for handling the data (read/write)

######### Documantation #########
Results and Observation.pdf: Contain the Analysis of

######## Trained keras models #####
task1(feature 2,3)_epoch20.h5       : the trained Original model for 20 iteration
task2(drone feature 2,3)_epoch20.h5 : the trained Drone Model for 20 iteration


####### Final RUN Terminal Output ###########
could be run and reproduced. (just need to have data file in same directory)




abhinav@linux:~/Desktop/Drone/GuruDrone$ python main.py
/home/abhinav/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
/home/abhinav/anaconda3/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.6
  return f(*args, **kwds)
2018-02-25 23:20:08.956047: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
Shape of X:  (20000, 6)
Shape of Y:  (20000, 1)
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_layer (InputLayer)     (None, 2)                 0
_________________________________________________________________
layer1 (Dense)               (None, 20)                60
_________________________________________________________________
layer2 (Dense)               (None, 10)                210
_________________________________________________________________
layer3 (Dense)               (None, 10)                110
_________________________________________________________________
output_layer (Dense)         (None, 1)                 11
=================================================================
Total params: 391
Trainable params: 391
Non-trainable params: 0
_________________________________________________________________
None
Train on 15000 samples, validate on 5000 samples
Epoch 1/20
15000/15000 [==============================] - 1s 79us/step - loss: 0.4560 - val_loss: 0.3066
Epoch 2/20
15000/15000 [==============================] - 1s 37us/step - loss: 0.2749 - val_loss: 0.2860
Epoch 3/20
15000/15000 [==============================] - 1s 35us/step - loss: 0.2665 - val_loss: 0.2805
Epoch 4/20
15000/15000 [==============================] - 1s 37us/step - loss: 0.2640 - val_loss: 0.2784
Epoch 5/20
15000/15000 [==============================] - 1s 35us/step - loss: 0.2625 - val_loss: 0.2771
Epoch 6/20
15000/15000 [==============================] - 1s 37us/step - loss: 0.2618 - val_loss: 0.2756
Epoch 7/20
15000/15000 [==============================] - 1s 35us/step - loss: 0.2616 - val_loss: 0.2798
Epoch 8/20
15000/15000 [==============================] - 1s 36us/step - loss: 0.2612 - val_loss: 0.2760
Epoch 9/20
15000/15000 [==============================] - 1s 35us/step - loss: 0.2607 - val_loss: 0.2773
Epoch 10/20
15000/15000 [==============================] - 1s 35us/step - loss: 0.2606 - val_loss: 0.2723
Epoch 11/20
15000/15000 [==============================] - 1s 36us/step - loss: 0.2602 - val_loss: 0.2766
Epoch 12/20
15000/15000 [==============================] - 1s 35us/step - loss: 0.2590 - val_loss: 0.2729
Epoch 13/20
15000/15000 [==============================] - 1s 37us/step - loss: 0.2602 - val_loss: 0.2726
Epoch 14/20
15000/15000 [==============================] - 1s 35us/step - loss: 0.2594 - val_loss: 0.2778
Epoch 15/20
15000/15000 [==============================] - 1s 37us/step - loss: 0.2593 - val_loss: 0.2708
Epoch 16/20
15000/15000 [==============================] - 1s 35us/step - loss: 0.2593 - val_loss: 0.2702
Epoch 17/20
15000/15000 [==============================] - 1s 36us/step - loss: 0.2590 - val_loss: 0.2713
Epoch 18/20
15000/15000 [==============================] - 1s 35us/step - loss: 0.2588 - val_loss: 0.2695
Epoch 19/20
15000/15000 [==============================] - 1s 36us/step - loss: 0.2590 - val_loss: 0.2694
Epoch 20/20
15000/15000 [==============================] - 1s 35us/step - loss: 0.2587 - val_loss: 0.2705

Evaluating on Training Data
accuracy(%):  90.03999999999999
precision:  0.8729385307346327
recall:  0.9360932475884244
F1Score:  0.9034134988363072


Evaluating on Testing Data
accuracy(%):  89.42
precision:  0.8754208754208754
recall:  0.9227129337539433
F1Score:  0.8984449990401229




_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_layer (InputLayer)     (None, 2)                 0
_________________________________________________________________
layer1 (Dense)               (None, 300)               900
_________________________________________________________________
output_layer (Dense)         (None, 1)                 301
_________________________________________________________________
activation_1 (Activation)    (None, 1)                 0
=================================================================
Total params: 1,201
Trainable params: 1,201
Non-trainable params: 0
_________________________________________________________________
None
Train on 15000 samples, validate on 5000 samples
Epoch 1/1
15000/15000 [==============================] - 1s 38us/step - loss: 0.0321 - val_loss: 0.0023



Train on 15000 samples, validate on 5000 samples
Epoch 1/1
15000/15000 [==============================] - 0s 33us/step - loss: 0.0018 - val_loss: 0.0017



Train on 15000 samples, validate on 5000 samples
Epoch 1/1
15000/15000 [==============================] - 1s 36us/step - loss: 0.0017 - val_loss: 0.0018



Train on 15000 samples, validate on 5000 samples
Epoch 1/1
15000/15000 [==============================] - 0s 33us/step - loss: 0.0016 - val_loss: 0.0016



Train on 15000 samples, validate on 5000 samples
Epoch 1/1
15000/15000 [==============================] - 0s 31us/step - loss: 0.0015 - val_loss: 0.0016



Train on 15000 samples, validate on 5000 samples
Epoch 1/1
15000/15000 [==============================] - 0s 33us/step - loss: 0.0014 - val_loss: 0.0015



Train on 15000 samples, validate on 5000 samples
Epoch 1/1
15000/15000 [==============================] - 0s 33us/step - loss: 0.0013 - val_loss: 0.0013



Train on 15000 samples, validate on 5000 samples
Epoch 1/1
15000/15000 [==============================] - 0s 33us/step - loss: 0.0012 - val_loss: 0.0011



Train on 15000 samples, validate on 5000 samples
Epoch 1/1
15000/15000 [==============================] - 0s 32us/step - loss: 0.0011 - val_loss: 0.0010



Train on 15000 samples, validate on 5000 samples
Epoch 1/1
15000/15000 [==============================] - 0s 32us/step - loss: 9.2341e-04 - val_loss: 8.6696e-04



Train on 15000 samples, validate on 5000 samples
Epoch 1/1
15000/15000 [==============================] - 0s 31us/step - loss: 7.7370e-04 - val_loss: 6.8161e-04



Train on 15000 samples, validate on 5000 samples
Epoch 1/1
15000/15000 [==============================] - 0s 32us/step - loss: 5.9781e-04 - val_loss: 4.9041e-04



Train on 15000 samples, validate on 5000 samples
Epoch 1/1
15000/15000 [==============================] - 0s 33us/step - loss: 3.9300e-04 - val_loss: 2.7997e-04



Train on 15000 samples, validate on 5000 samples
Epoch 1/1
15000/15000 [==============================] - 0s 32us/step - loss: 2.0480e-04 - val_loss: 1.1539e-04



Train on 15000 samples, validate on 5000 samples
Epoch 1/1
15000/15000 [==============================] - 0s 32us/step - loss: 6.6426e-05 - val_loss: 2.5565e-05



Train on 15000 samples, validate on 5000 samples
Epoch 1/1
15000/15000 [==============================] - 0s 32us/step - loss: 2.1543e-05 - val_loss: 1.4857e-05



Train on 15000 samples, validate on 5000 samples
Epoch 1/1
15000/15000 [==============================] - 0s 33us/step - loss: 1.5507e-05 - val_loss: 1.2371e-05



Train on 15000 samples, validate on 5000 samples
Epoch 1/1
15000/15000 [==============================] - 0s 32us/step - loss: 1.2848e-05 - val_loss: 1.2054e-05



Train on 15000 samples, validate on 5000 samples
Epoch 1/1
15000/15000 [==============================] - 0s 33us/step - loss: 1.1869e-05 - val_loss: 1.0210e-05



Train on 15000 samples, validate on 5000 samples
Epoch 1/1
15000/15000 [==============================] - 0s 32us/step - loss: 1.4358e-05 - val_loss: 2.1975e-05
Drone Extended



_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_layer (InputLayer)     (None, 2)                 0
_________________________________________________________________
layer1 (Dense)               (None, 301)               903
_________________________________________________________________
output_layer (Dense)         (None, 1)                 302
_________________________________________________________________
activation_2 (Activation)    (None, 1)                 0
=================================================================
Total params: 1,205
Trainable params: 1,205
Non-trainable params: 0
_________________________________________________________________
None

Evaluating on Training Data for Drone
accuracy(%):  90.06
precision:  0.8719641300286461
recall:  0.9379689174705252
F1Score:  0.9037629897373007


Evaluating on Testing Data for Drone
accuracy(%):  89.44
precision:  0.8737900223380491
recall:  0.9254731861198738
F1Score:  0.8988893144389123
