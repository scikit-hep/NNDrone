import numpy as np
import pickle
import math

try:
    from wrappers import check_model
except ImportError as e:
    from nndrone.wrappers import check_model

try:
    from utilities import dot_loss, next_batch
except ImportError as e:
    from utilities.utilities import dot_loss, next_batch

class DontCacheRef(Exception):
    pass

class BasicConverter(object):
    def __init__(self, learning_rate = 0.05, batch_size = 1, num_epochs = 300, threshold = 0.02,
                 add_layer_dynamic = False, layer_to_expand = 0, base_model = None):
        # training control
        self._learning_rate = learning_rate
        self._batchSize = batch_size
        self._num_epochs = num_epochs
        self._threshold = threshold
        self._add_layer_dynamic = add_layer_dynamic
        self._layer_to_expand = int(layer_to_expand)
        if base_model is not None:
            self.set_base_model(base_model)
        # training history
        self._updatedLoss = 1000.0
        self._diffs = []
        self._losses = []
        self._updates = []
        self._epoch = 0


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


    def set_base_model(self, base_model):
        check = check_model(base_model)
        if check is not None:
            self._base_model = check
        else:
            raise ValueError('Base model passed to converter is not compatible')


    def base_model(self):
        try:
            return self._base_model
        except AttributeError as e:
            print('Base model was not supplied')
            raise e

    def get_refs(self, datapoints, scaler = None, cache_data = True):
        try:
            if not cache_data:
                raise DontCacheRef()
            # Return the cached list of reference outputs for the base model
            return (self.__datapoints, self.__refs)
        except (DontCacheRef, AttributeError) as e:
            # Create the list of reference outputs for the base model
            dpoints = np.asarray(datapoints)
            if scaler is not None:
                dpoints = scaler.transform(dpoints)
            refs = self.base_model().evaluate(dpoints)
            self.__datapoints = dpoints.reshape(dpoints.shape[0], -1)
            self.__refs = np.asarray(refs)
            return (self.__datapoints, self.__refs)


    def convert_model(self, drone_model, datapoints, base_model = None, scaler = None,
                      cache_data = True, epoch_reset = False):
        if base_model is not None:
            self.set_base_model(base_model)
        # Get the list of reference outputs for the base model
        datapoints_for_drone, refs = self.get_refs(datapoints, scaler, cache_data)
        inflate = 0  # to inflate the learning without change iterations
        if epoch_reset:
            self._epoch = 0
        avloss = 0
        # convert until min epochs are passed and leave only if loss at minima
        while (self._epoch < self._num_epochs) or (self._updatedLoss < avloss):
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
                drone_model.update(batchX, batchY, self._learning_rate)
            avloss = np.average(epochloss)
            diff = 0.0
            if self._epoch > 0:
                # is the relative improvement of the loss too small, smaller than threshold
                diff = math.fabs(avloss - self._losses[-1]) / avloss
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
            print('Epoch: %s, loss %s, diff %.5f, last updated loss %.5f' % (self._epoch, avloss, diff, self._updatedLoss))
            # update our loss history list by taking the average loss
            # across all batches
            if self._epoch == 0:  # be consistent at the first epoch
                self._losses.append(avloss)
                self._diffs.append(math.fabs(avloss - self._updatedLoss) / avloss)
                self._updates.append(0)
            self._epoch += 1
        return drone_model



class AdvancedConverter(object):
    def __init__(self, learning_rate = 0.05, batch_size = 1, num_epochs = 300, threshold = 0.02,
                 add_layer_dynamic = False, layer_to_expand = None, base_model = None):
        # training control
        self._learning_rate = learning_rate
        self._batchSize = batch_size
        self._num_epochs = num_epochs
        self._threshold = threshold
        self._add_layer_dynamic = add_layer_dynamic
        self.__round_robin = False
        if layer_to_expand is None:
            self.__round_robin = True
        self._layer_to_expand = int(layer_to_expand) if layer_to_expand is not None else None
        if base_model is not None:
            self.set_base_model(base_model)
        # training history
        self._updatedLoss = 1000.0
        self._diffs = []
        self._losses = []
        self._updates = []
        self._epoch = 0
        self.__rr_begin = 0
        self.__rr_last = 0

    def losses(self):
        return self._losses


    def diffs(self):
        return self._diffs


    def updates(self):
        return self._updates


    def round_robin(self, num_layers):
        self.__rr_last = self.__rr_begin
        self.__rr_begin = np.random.ranint(0, num_layers - 1)  # careful, expanding last layer will change output number
        return self.__rr_last


    def save_history(self, fname):
        f_train = open(fname, 'wb')
        training_data = [self._losses, self._diffs, self._updates]
        pickle.dump(training_data, f_train)
        f_train.close()


    def set_base_model(self, base_model):
        check = check_model(base_model)
        if check is not None:
            self._base_model = check
        else:
            raise ValueError('Base model passed to converter is not compatible')


    def base_model(self):
        try:
            return self._base_model
        except AttributeError as e:
            print('Base model was not supplied')
            raise e


    def get_refs(self, datapoints, scaler = None, cache_data = True):
        try:
            if not cache_data:
                raise DontCacheRef()
            # Return the cached list of reference outputs for the base model
            return (self.__datapoints, self.__refs)
        except(DontCacheRef, AttributeError) as e:
            # Create the list of reference outputs for the base model
            dpoints = np.asarray(datapoints)
            if scaler:
                dpoints = scaler.transform(dpoints)
            refs = self.base_model().evaluate(dpoints, raw = True)
            self.__datapoints = dpoints
            self.__refs = refs
            return (self.__datapoints, self.__refs)

    def convert_model(self, drone_model, datapoints, base_model = None, scaler = None,
                      cache_data = True, epoch_reset = False):
        if base_model is not None:
            self.set_base_model(base_model)
        # Get the list of reference outputs for the base model
        datapoints_for_drone, refs = self.get_refs(datapoints, scaler, cache_data)
        inflate = 0  # to inflate the learning without change iterations
        if epoch_reset:
            self._epoch = 0
        avloss = 0
        # convert until min epochs are passed and leave only if loss at minima
        while (self._epoch < self._num_epochs) or (self._updatedLoss < avloss):
            # initialize the total loss for the epoch
            epochloss = []
            # loop over our data in batches
            for (batchX, batchY) in next_batch(datapoints_for_drone, refs, self._batchSize):
                batchY = np.array(batchY)
                if batchX.shape[0] != self._batchSize:
                    print('Batch size insufficient ({}), continuing...'.format(batchY.shape[0]))
                    continue
                # Find current output and calculate loss for our graph
                preds = drone_model.evaluate_total(batchX, debug = False)
                loss, error = dot_loss(preds, batchY)
                epochloss.append(loss)
                # Update the model
                drone_model.update(batchX, batchY, self._learning_rate)
            avloss = np.average(epochloss)
            diff = 0.0
            if self._epoch > 0:
                # is the relative improvement of the loss too small, smaller than threshold
                diff = math.fabs(avloss - self._losses[-1]) / avloss
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
                    elif self._layer_to_expand is not None:
                        drone_model.expand_layer_dynamic(self._layer_to_expand)
                    else:
                        drone_model.expand_layer_dynamic(self.round_robin(drone_model.num_layers()))
                    print('Model structure is now:')
                    drone_model.print_layers()
                self._updates.append(update)
            print('Epoch: %s, loss %s, diff %.5f, last updated loss %.5f' % (self._epoch, avloss, diff, self._updatedLoss))
            # update our loss history list by taking the average loss
            # across all batches
            if self._epoch == 0:  # be consistent at the first epoch
                self._losses.append(avloss)
                self._diffs.append(math.fabs(avloss - self._updatedLoss) / avloss)
                self._updates.append(0)
            self._epoch += 1
        return drone_model
