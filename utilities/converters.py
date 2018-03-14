import numpy as np
import pickle
import math

try:
    from utilities import dot_loss, next_batch
except ImportError:
    from utilities.utilities import dot_loss, next_batch


class BasicConverter:
    def __init__(self):
        # training history
        self._updatedLoss = 1000.0
        self._diffs = []
        self._losses = []
        self._updates = []
        # training control
        self._alpha = 0.05
        self._batchSize = 1
        self._num_epochs = 300
        self._threshold = 0.02

    def save_history(self, fname):
        f_train = open(fname, 'wb')
        training_data = [self._losses, self._diffs, self._updates]
        pickle.dump(training_data, f_train)
        f_train.close()

    def convert_model(self, in_model, base_model, datapoints, scaler=None):
        # Create the list of outputs for the base model
        refs = []
        flattened = []
        for b in datapoints:
            if scaler:
                b = scaler.transform([b])
            prob = base_model.predict_proba(b)[0][0]
            b = b[0].flatten().tolist()
            refs.append(prob)
            flattened.append(b)
        for q in range(self._num_epochs):
            # initialize the total loss for the epoch
            epochloss = []

            # loop over our data in batches
            for (batchX, batchY) in next_batch(datapoints, refs, self._batchSize):
                if batchX.shape[0] != self._batchSize:
                    print 'Batch size insufficient (%s), continuing...' % batchY.shape[0]
                    continue

            # Find current output and calculate loss for our graph
            preds = in_model.evaluate_total(batchX, debug=False)
            loss, error = dot_loss(preds, batchY)
            epochloss.append(loss)

            # Update the model
            in_model.update(batchX, batchY, self._alpha)

            avloss = np.average(epochloss)
            diff = 0.0
            if q > 0:
                # Is the fractional difference less than the threshold
                diff = math.fabs(avloss-self._losses[-1])/avloss
                self._diffs.append(diff)
                self._losses.append(avloss)
                update = 0
                modify = True if (diff < self._threshold) else False
                if modify:
                    # If it is less than the threshold, is it below
                    # where we last updated
                    modify = (avloss < (self._updatedLoss-(diff*avloss)))
                if modify:
                    update = 1
                    print 'Model conversion not sufficient, updating...'
                    print 'Last updated loss: %s' % self._updatedLoss
                    self._updatedLoss = avloss
                    in_model.expand_layer_dynamic(0)
                self._updates.append(update)

            print('Epoch: %s, loss %s, diff %.5f, last updated loss %.5f' % (q, avloss, diff, self._updatedLoss))
            # update our loss history list by taking the average loss
            # across all batches
            self._losses.append(avloss)
