import numpy as np
import pickle
import math

try:
    from utilities import dot_loss, next_batch
except ImportError:
    from utilities.utilities import dot_loss, next_batch


class BasicConverter(object):
    def __init__(self, alpha = 0.05, batch_size = 1, num_epochs = 300, threshold = 0.02, add_layer_dynamic = False, layer_to_expand = 0):
        # training control
        self._alpha = alpha
        self._batchSize = batch_size
        self._num_epochs = num_epochs
        self._threshold = threshold
        self._add_layer_dynamic = add_layer_dynamic
        self._layer_to_expand = int(layer_to_expand)
        # training history
        self._updatedLoss = 1000.0
        self._diffs = []
        self._losses = []
        self._updates = []


    def losses(self):
        return self._losses


    def diffs(self):
        return self._diffs


    def updates(self):
        return self._updates


    def save_history(self, fname):
        f_train = open(fname, 'wb')
        training_data = [self._losses, self._diffs, self._updates]
        pickle.dump(training_data, f_train)
        f_train.close()


    def convert_model(self, drone_model, base_model, datapoints, scaler = None, conv_1d = False, conv_2d = False):
        # Create the list of outputs for the base model
        if conv_1d and conv_2d:
            print('ERROR: conv_1d and conv_2d are mutually exclusive')
            return None
        refs = []
        flattened = []
        for point in datapoints:
            spoint = point
            if scaler and not conv_2d:
                point = scaler.transform([point])
            prob = 0.0
            if conv_1d:
                prob = base_model.predict_proba(np.expand_dims(np.expand_dims(spoint, axis = 2), axis = 0))[0][0]
            elif conv_2d:
                # this will match if original model was trained with correct dimensionality
                prob = base_model.predict_proba(np.expand_dims(spoint, axis = 0))
            else:
                prob = base_model.predict_proba(spoint.reshape(1, -1))[0][0]
            if conv_2d:
                spoint = spoint.flatten().tolist()
            else:
                spoint = spoint[0].flatten().tolist()
            refs.append(prob)
            flattened.append(spoint)
        inflate = 0  # to inflate the learning without change iterations
        q = 0
        avloss = 0
        datapoints_for_drone = None
        if conv_2d:
            # BaseModel only accepts vector objects in N-D space
            datapoints_for_drone = np.asarray([np.asarray(point.flatten()) for point in datapoints])
        else:
            datapoints_for_drone = datapoints
        # convert until min epochs are passed and leave only if loss at minima
        while (q < self._num_epochs) or (self._updatedLoss < avloss):
            # initialize the total loss for the epoch
            epochloss = []

            # loop over our data in batches
            for (batchX, batchY) in next_batch(datapoints_for_drone, refs, self._batchSize):
                batchY = np.array(batchY)
                if batchX.shape[0] != self._batchSize:
                    print('Batch size insufficient (%s), continuing...' % batchY.shape[0])
                    continue

                # Find current output and calculate loss for our graph
                preds = drone_model.evaluate_total(batchX, debug = False)
                loss, error = dot_loss(preds, batchY)
                epochloss.append(loss)

                # Update the model
                drone_model.update(batchX, batchY, self._alpha)

            avloss = np.average(epochloss)
            diff = 0.0
            if q > 0:
                # is the relative improvement of the loss too small, smaller than threshold
                diff = math.fabs(avloss-self._losses[-1])/avloss
                self._diffs.append(diff)
                self._losses.append(avloss)
                update = 0
                modify = True if (diff < self._threshold) else False
                if modify:
                    # If it is less than the threshold, is it below
                    # where we last updated, has the drone learned enough
                    #
                    # - skip checks if we have never updated before
                    # - do at least 6 learning iterations before attempting new update
                    # - use asymptotic exponential to push model to learn
                    #   until its loss is far enough away from previous update,
                    inflate += 1  # iterate inflating
                    modify = True if self._updatedLoss == 1000.0 else (avloss < (self._updatedLoss - (50.0 * (1.0 - np.exp(-0.04 * inflate)) * diff * avloss))) and (inflate > 5)
                if modify:
                    update = 1
                    inflate = 0
                    print('Model conversion not sufficient, updating...')
                    print('Last updated loss: %s' % self._updatedLoss)
                    self._updatedLoss = avloss
                    if self._add_layer_dynamic:
                        drone_model.add_layer_dynamic()
                    else:
                        drone_model.expand_layer_dynamic(self._layer_to_expand)
                    print('Model structure is now:')
                    drone_model.print_layers()
                self._updates.append(update)

            print('Epoch: %s, loss %s, diff %.5f, last updated loss %.5f' % (q, avloss, diff, self._updatedLoss))
            # update our loss history list by taking the average loss
            # across all batches
            if q == 0:  # be consistent at the first epoch
                self._losses.append(avloss)
                self._diffs.append(math.fabs(avloss - self._updatedLoss) / avloss)
                self._updates.append(0)
            q += 1
        return drone_model
