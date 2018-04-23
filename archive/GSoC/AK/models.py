import numpy as np
from keras.layers import Input,Dense,Activation
from keras.models import Model
import keras.backend as K
import tensorflow as tf
np.random.seed(1)                               #for reproducibility


def feedForwardAll(input_features):
    '''Arguments:
        input_features: the number of input feature in wach example

       Output:
        model: a keras model after compiling the computation graph
    '''
    #Declaring the input place holder
    X_input=Input(input_features,name='input_layer')

    #Building the computation graph(our architecture)
    X=Dense(20,activation='relu',name='layer1',kernel_initializer='glorot_uniform')(X_input)
    X=Dense(10,activation='relu',name='layer2',kernel_initializer='glorot_uniform')(X)
    X=Dense(10,activation='relu',name='layer3',kernel_initializer='glorot_uniform')(X)
    X=Dense(1,activation='sigmoid',name='output_layer',kernel_initializer='glorot_uniform')(X)

    #saving computation Graph in a Model
    model=Model(inputs=X_input,outputs=X,name="AllFeaturesV1")

    #Compiling the model
    model.compile(optimizer='adam',loss='binary_crossentropy')
    # TO -DO
    # binary cross-entropy will only work when or prediction is between 0,1, cuz loss was as by Ng sir
    # do it for tanh by custom accuracy function

    return model

def feedForward23_better(input_features=(2,)):
    X_input=Input(input_features,name='input_layer')
    X=Dense(100,activation='relu',name='layer1',kernel_initializer='glorot_uniform')(X_input)
    X=Dense(50,activation='relu',name='layer2',kernel_initializer='glorot_uniform')(X)
    X=Dense(10,activation='relu',name='layer3',kernel_initializer='glorot_uniform')(X)
    X=Dense(1,activation='sigmoid',name='output_layer',kernel_initializer='glorot_uniform')(X)

    #saving computation Graph in a Model
    model=Model(inputs=X_input,outputs=X,name="23FeatureV1")

    #Compiling the Model
    model.compile(optimizer='adam',loss='binary_crossentropy')

    return model

def DroneNetwork(input_features=(2,)):
    '''
        Arguments:
            input_features: the total number of input features out of 6
            ##teacher_model : the output from trained Original from which drone will learn
        Output:
            model: a Drone model which take learns from the output of
                    same data as of "wise"- Teacher/original model
    '''
    #Input to the drone
    X_input=Input(input_features,name='input_layer')

    #Creating the computation graph of drone
    X=Dense(300,activation='relu',kernel_initializer='glorot_uniform',name="layer1")(X_input)
    X=Dense(1,kernel_initializer='glorot_uniform',name='output_layer')(X)
    X=Activation('sigmoid')(X)

    #saving the computation graph in a Model
    model=Model(inputs=X_input,outputs=X,name="Eklavya")

    #Compiling the model(this part is the real trick where Drone will learns from Teacher)
    model.compile(optimizer='adam',loss=DroneLoss)

    return model

def DroneLoss(Y_teacher,Y_student):
    '''Arguments:
        Y_teacher: this is Y_true(prediction of Teacher) what we will give Drone while fitting
        Y_student: this is Y_pred of out Drone Network

      Note:
        This function will automatically called by Keras by passing the arguments in this
        sequence only.
    '''
    loss=tf.reduce_mean(tf.square(Y_student-Y_teacher))
    return loss

def extendDroneNetwork(oldDrone,input_features=(2,)):
    '''Arguments:
        input_features: the number of features in input layer
        oldDrone      : the old DroneNetwork model

      Output:
        This function currenly creates a new model utilizing the
        previous model weights and extending the weights so that,
        loss function remains continuous.
        We have to find a better way to do this by popping layers
        rather than creating new network each time
    '''

    #Reading the required config of last model
    hidden_layer=oldDrone.get_layer("layer1")
    h_weights=hidden_layer.get_weights()
    # print(len(h_weights))           #2
    # print(h_weights[0].shape)       #(6,300)
    # print(h_weights[1].shape)       #(300,)
    output_layer=oldDrone.get_layer("output_layer")
    o_weights=output_layer.get_weights()
    # print(len(o_weights))           #2
    # print(o_weights[0].shape)       #(300,1)
    # print(o_weights[1].shape)       #(1,)

    #Creating new weight array to be set to layer
    #for hidden layer(no need for bias, broadcasted)
    add_W=np.zeros((h_weights[0].shape[0],1),dtype=np.float32)
    h_W=np.append(h_weights[0],add_W,axis=1)
    add_b=np.zeros((1,),dtype=np.float32)
    h_b=np.append(h_weights[1],add_b,axis=0)
    h_weights=[h_W,h_b]

    #for output layer
    add_W=np.zeros((1,1),dtype=np.float32)
    o_W=np.append(o_weights[0],add_W,axis=0)
    o_weights=[o_W,o_weights[1]]

    #Now our new weights are ready so, we will finally
    #build computation graph of new Model

    #Creating the input placeholder for the new model
    X_input=Input(input_features,name='input_layer')

    #Creating the computation graph of drone
    X=Dense(h_W.shape[1],activation='relu',kernel_initializer='glorot_uniform',name="layer1")(X_input)
    X=Dense(1,kernel_initializer='glorot_uniform',name='output_layer')(X)
    X=Activation('sigmoid')(X)

    #saving the computation graph in a Model
    model=Model(inputs=X_input,outputs=X,name="Eklavya")

    #Compiling the model(this part is the real trick where Drone will learns from Teacher)
    model.compile(optimizer='adam',loss=DroneLoss)

    #Setting new weights of weights of model(to be safe its done after compiling)
    hidden_layer=model.get_layer("layer1")
    hidden_layer.set_weights(h_weights)
    output_layer=model.get_layer("output_layer")
    output_layer.set_weights(o_weights)

    return model
