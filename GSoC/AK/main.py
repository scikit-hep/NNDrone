import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

from utils import getDataAsNumpy,evaluate_accuracy,plot_training_losses
from models import feedForwardAll,feedForward23_better,DroneNetwork,extendDroneNetwork


        ############## Getting the data set ###############
#(we may have to normalize them)
X,Y=getDataAsNumpy()
print("Shape of X: ",X.shape)
print("Shape of Y: ",Y.shape)

n=X.shape[1]                                 #number of total features
m=X.shape[0]                                 #total datapoints
#creating the test train and validation split(random permutation is done)
select_feature=[2,3]                        #for selecting certain feature to train on
X_train=X[0:int(0.75*m),select_feature]      #taking the 75% data for trainig
Y_train=Y[0:int(0.75*m),:]
X_test=X[int(0.75*m):,select_feature]
Y_test=Y[int(0.75*m):,:]

###############################################################################
###############################################################################
###############################################################################
    ####### Creating the Model and training data ########
input_shape=(len(select_feature),)
model=feedForwardAll(input_shape)
#model=feedForward23_better()
print(model.summary())
epochs=20
train_history=model.fit(x=X_train,y=Y_train,
                        epochs=epochs,validation_data=(X_test,Y_test))
model.save("task1(feature 2,3)_epoch20.h5")

     ######## MODEL EVALUATION ON Train/Test DATA ########
plot_training_losses(train_history,select_feature[0],select_feature[1],epochs)
Y_drone_train=None  #initializing as later Drone will need it to learn
Y_drone_test=None

print("\nEvaluating on Training Data")
Y_drone_train=Y_pred=model.predict(X_train)
accuracy,precision,recall,F1Score=evaluate_accuracy(Y_pred,Y_train)
print("accuracy(%): ",accuracy)
print("precision: ",precision)
print("recall: ",recall)
print("F1Score: ",F1Score)

print("\n")
print("Evaluating on Testing Data")
Y_drone_test=Y_pred=model.predict(X_test)
accuracy,precision,recall,F1Score=evaluate_accuracy(Y_pred,Y_test)
print("accuracy(%): ",accuracy)
print("precision: ",precision)
print("recall: ",recall)
print("F1Score: ",F1Score)
print("\n\n\n")

#########################################################################
#########################################################################
#########################################################################
       ######### Testing the Drone Network ############
droneModel=DroneNetwork(input_shape)
print(droneModel.summary())
epochs=20
diff_trigger=np.zeros((epochs,4))
last_loss=0                 #loss of the last iteration
last_update_loss=0          #loss at the last update
kappa=-0.02                 #the threshold after which the net will be updated

for i in range(epochs):
    train_history=droneModel.fit(x=X_train,y=Y_drone_train,epochs=1,
                            validation_data=(X_test,Y_drone_test))
    #new_loss=droneModel.evaluate(x=X_train,y=Y_drone_train)(dont evaluate each time)
    new_loss=train_history.history["loss"][-1]
    val_loss=train_history.history["val_loss"][-1]
    relative_diff=(new_loss-last_loss)/new_loss
    diff_trigger[i,0]=new_loss
    diff_trigger[i,1]=relative_diff
    diff_trigger[i,3]=val_loss

    if(i==0):
        last_update_loss=new_loss
    else:
        avg_loss=(last_update_loss+last_loss)/2
        if(relative_diff>kappa): #or new_loss<avg_loss): THINK
            droneModel=extendDroneNetwork(droneModel,input_shape)
            last_update_loss=new_loss
            print("Drone Extended")
            diff_trigger[i,2]=1

    last_loss=new_loss
    print("\n\n")
#checking the final configuration of Drone Network
print(droneModel.summary())

############## MODEL EVALUATION ON Train/Test DATA #############
plot_training_losses(train_history,select_feature[0],select_feature[1],epochs)
#printing the Loss Function
plt.clf()
plt.plot(diff_trigger[:,0])
plt.plot(diff_trigger[:,3])
plt.xlabel('epochs')
plt.ylabel('mean squared loss')
plt.legend(['loss','val_loss'])
plt.show()
#printing the trigger location
plt.clf()
plt.plot(diff_trigger[:,1])
for i in range(diff_trigger.shape[0]):
    if(diff_trigger[i,2]!=0):
        plt.plot(i,diff_trigger[i,2]*diff_trigger[i,1],'o')
plt.xlabel("epochs/iteration")
plt.ylabel("Relative Difference (Criteria 1)")
plt.show()

print("\nEvaluating on Training Data for Drone")
Y_pred=droneModel.predict(X_train)
accuracy,precision,recall,F1Score=evaluate_accuracy(Y_pred,Y_train)

model.save("task2(drone feature 2,3)_epoch20.h5")
#see we are now testing on real data how much drone had learned
#has it become more wise or equal
print("accuracy(%): ",accuracy)
print("precision: ",precision)
print("recall: ",recall)
print("F1Score: ",F1Score)

print("\n")
print("Evaluating on Testing Data for Drone")
Y_pred=droneModel.predict(X_test)
accuracy,precision,recall,F1Score=evaluate_accuracy(Y_pred,Y_test)
print("accuracy(%): ",accuracy)
print("precision: ",precision)
print("recall: ",recall)
print("F1Score: ",F1Score)
print("\n")
