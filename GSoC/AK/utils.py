import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
np.random.seed(1)   #for any reproducibility of results

#Function handle to get the Training data
def getDataAsNumpy():
    bData=None
    sData=None
    #reading the data stored as binary from sklearn's joblib
    with open("background_data.p",'rb') as f:
        bData=joblib.load(f)
    with open("signal_data.p",'rb') as f:
        sData=joblib.load(f)

    #Converting to nummpy array
    sX=np.asarray(sData)
    bX=np.asarray(bData)
    sY=np.ones((sX.shape[0],1))
    bY=np.ones((bX.shape[0],1))*(0)        #-1 for background label(we'll use tanh)

    X=np.concatenate((sX,bX),axis=0)        #adding them side together
    Y=np.concatenate((sY,bY),axis=0)

    #Shuffling the data based on random permutation
    perm=np.random.permutation(X.shape[0])
    X=X[perm,:]
    Y=Y[perm,:]

    return X,Y

def plot_training_losses(train_history,i,j,epochs):
    loss=train_history.history['loss']
    val_loss=train_history.history['val_loss']

    plt.plot(loss)
    plt.plot(val_loss)
    plt.xlabel('epochs')
    plt.ylabel('binary_crossentropy')
    plt.legend(['loss','val_loss'])
    #plt.savefig(str(i)+str(j)+"ep"+str(epochs)+".png")
    #plt.show()

def evaluate_accuracy(Y_pred,Y_actual):
    Y_pred=(Y_pred>0.5).astype(int)
    Y_actual=Y_actual.astype(int)   #for safety

    #comparing both the prediction and actual value
    comparison=Y_pred==Y_actual
    accuracy=np.mean(comparison)*100

    #Calculating precision and recall
    true_pos=0
    false_pos=0
    false_neg=0
    for i in range(Y_pred.shape[0]):
        if(Y_pred[i,0]==1 and Y_actual[i,0]==0):
            false_pos+=1
        if(Y_pred[i,0]==1 and Y_actual[i,0]==1):
            true_pos+=1
        if(Y_pred[i,0]==0 and Y_actual[i,0]==1):
            false_neg+=1

    precision=(true_pos)/(true_pos+false_pos)
    recall=(true_pos)/(true_pos+false_neg)
    F1Score=(2*precision*recall)/(precision+recall)

    return accuracy,precision,recall,F1Score
